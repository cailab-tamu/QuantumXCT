"""
run_hardware_postopt.py
=======================
Post-optimization "assert quality" run of the final QuantumXCT circuit on a real
IBM quantum computer (hardware validation).

For the saved ovarian result (results/ovarian_cocultured/run.json) it runs BOTH
the baseline circuit (0 gates) and the optimized circuit, in three settings:

  1. exact       : noiseless statevector (reference)
  2. fake-noisy  : Aer with the noise model of the chosen device (no queue, free)
  3. hardware    : the real device, with DD + gate/measure twirling, and readout
                   correction by a tensored calibration (only if --hardware)

and reports KL(baseline), KL(optimized) and the KL reduction for each, plus the
transpiled depth / two-qubit gate count. Output: results/hardware/.

Usage (from QuantumXCT/python, qiskit-env; IBM account saved with
QiskitRuntimeService.save_account(...)):
    python run_hardware_postopt.py                      # exact + fake-noisy only (dry run)
    python run_hardware_postopt.py --hardware           # + real device (least busy)
    python run_hardware_postopt.py --hardware --backend ibm_brisbane --shots 8000

Endianness: counts come back keyed MSB-first with qubit 0 rightmost, identical to
the Aer path; verified in tests/test_review_integrity.py (also after a
layout-changing transpilation to a 127-qubit device).
"""
import argparse
import json
import os
import time
import warnings

import numpy as np

from qxct import QuantumXct, RunConfig, MitigationConfig
from qxct.exact import ExactScorer

warnings.filterwarnings("ignore", category=DeprecationWarning)

GL_CT1 = ["PDGFB", "STAT3", "HES4", "TGFB1"]
GL_CT2 = ["PDGFRB", "TGFBR1", "TGFBR2"]


def load_counts():
    from analysis_tools.data import load_ovarian
    return load_ovarian()

ap = argparse.ArgumentParser()
ap.add_argument("--run-json", default="results/ovarian_cocultured/run.json")
ap.add_argument("--hardware", action="store_true", help="submit to a real IBM device")
ap.add_argument("--backend", default=None, help="device name (default: least busy)")
ap.add_argument("--fake", default="FakeBrisbane", help="fake backend for the noisy dry run")
ap.add_argument("--shots", type=int, default=8000)
ap.add_argument("--out", default="results/hardware")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

run = json.load(open(a.run_json))
assert run["gene_list"] == GL_CT1 + GL_CT2, run["gene_list"]
topo = [tuple(x) for x in run["cnot_sequence"]]
angles = run["crx_angles"]
H = load_counts()

qx = QuantumXct(RunConfig(), gene_list=GL_CT1 + GL_CT2)   # norm='l1' (default), as in run.json
qx.encode(H["ct1_mo"], H["ct2_mo"], H["ct1_co"], H["ct2_co"])
sc = ExactScorer(qx.initial1, qx.initial2, qx.target1, qx.target2)

report = {"topology": [f"{qx.gene_list[c]}->{qx.gene_list[t]}" for c, t in topo],
          "angles": angles, "shots": a.shots,
          "exact": {"baseline_kl": sc.baseline(), "final_kl": sc.kl(topo, angles)}}
report["exact"]["reduction_pct"] = 100 * (1 - report["exact"]["final_kl"] / report["exact"]["baseline_kl"])

def _run(backend, mitigation, tag):
    out = {}
    for name, (t, th) in {"baseline": ([], []), "optimized": (topo, angles)}.items():
        qx.result.topology, qx.result.angles = t, th
        t0 = time.time()
        r = qx.assert_on_hardware(backend=backend, mitigation=mitigation)
        r["wall_s"] = time.time() - t0
        out[name] = {k: v for k, v in r.items() if k not in ("counts_measure1", "counts_measure2")}
        out[name]["counts_measure1"] = r["counts_measure1"]
        out[name]["counts_measure2"] = r["counts_measure2"]
    b, f = out["baseline"]["kl_total"], out["optimized"]["kl_total"]
    out["reduction_pct_raw"] = 100 * (1 - f / b)
    rc_b = (out["baseline"].get("kl_readout_corrected") or {}).get("kl_total")
    rc_f = (out["optimized"].get("kl_readout_corrected") or {}).get("kl_total")
    if rc_b and rc_f:
        out["reduction_pct_readout_corrected"] = 100 * (1 - rc_f / rc_b)
    report[tag] = out
    print(f"[{tag}] KL baseline {b:.4f} -> optimized {f:.4f} ({out['reduction_pct_raw']:.1f}% reduction)")

mit = MitigationConfig.all_on(shots=a.shots)       # DD + twirling + readout correction

# 2) fake-noisy dry run
from qiskit_aer import AerSimulator
import qiskit_ibm_runtime.fake_provider as fp
fake = getattr(fp, a.fake)()
_run(AerSimulator.from_backend(fake, seed_simulator=42), mit, f"fake_noisy_{a.fake}")

# 3) real hardware
if a.hardware:
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    backend = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False)
    print("submitting to", backend.name)
    _run(backend, mit, f"hardware_{backend.name}")

print(f"[exact] KL baseline {report['exact']['baseline_kl']:.4f} -> "
      f"optimized {report['exact']['final_kl']:.4f}")
fn = os.path.join(a.out, "hardware_postopt_report.json")
json.dump(report, open(fn, "w"), indent=1, default=float)
print("saved", fn)
