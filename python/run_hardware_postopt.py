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
transpiled depth / two-qubit gate count. Every run now also records a
DEVICE-CALIBRATION SNAPSHOT (per-qubit T1/T2, 1q gate error, readout error; per
used edge the 2q gate error/duration) under report[tag]["calibration"].

Optional (--zne): a Zero-Noise-Extrapolation / mirror scaffold on the optimized
circuit. It globally folds the optimized unitary U -> U (U^dag U)^n at noise
scales 1,3,5 (identity in the noiseless limit), measures KL at each, and
extrapolates to zero noise. HARDWARE-ONLY and OFF BY DEFAULT, so it never touches
the main submission. It prints a scale=1 self-check: the folded-circuit KL at
scale 1 should match the raw optimized KL of the main run (validates endianness /
target formatting before you trust the extrapolation).

Usage (from QuantumXCT/python, qiskit-env; IBM account saved with
QiskitRuntimeService.save_account(...)):
    python run_hardware_postopt.py                      # exact + fake-noisy only (dry run)
    python run_hardware_postopt.py --hardware           # + real device (least busy)
    python run_hardware_postopt.py --hardware --backend ibm_brisbane --shots 8000
    python run_hardware_postopt.py --hardware --zne     # + ZNE second pass on the real device

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


# --------------------------------------------------------------------------- #
# Device-calibration snapshot: error terms for the qubits/edges actually used. #
# --------------------------------------------------------------------------- #
def device_calibration(backend, qubits):
    """Per-qubit and per-edge error terms for the physical qubits in ``qubits``.

    Pulls from backend.target (gate error/duration, readout error) and, when
    available, backend.properties() (T1, T2, readout error, frequency). Robust to
    fake backends (fields that are missing are simply omitted). Returns a dict
    {"qubits": {q: {...}}, "two_qubit": {"<gate>_<a>_<b>": {...}}, "backend": name}.
    """
    snap = {"backend": getattr(backend, "name", None) if not callable(getattr(backend, "name", None))
            else backend.name(), "qubits": {}, "two_qubit": {}}
    qubits = sorted({int(q) for q in (qubits or [])})
    tgt = getattr(backend, "target", None)
    try:
        props = backend.properties()
    except Exception:
        props = None

    def _instr(name, key):
        try:
            return tgt[name].get(key) if (tgt is not None and name in tgt.operation_names) else None
        except Exception:
            return None

    for q in qubits:
        e = {}
        if props is not None:
            for fld, fn in (("T1_s", props.t1), ("T2_s", props.t2),
                            ("readout_error", props.readout_error), ("freq_hz", props.frequency)):
                try:
                    e[fld] = float(fn(q))
                except Exception:
                    pass
        for g in ("sx", "x", "rz"):
            ip = _instr(g, (q,))
            if ip is not None:
                if getattr(ip, "error", None) is not None:
                    e[f"{g}_error"] = float(ip.error)
                if getattr(ip, "duration", None) is not None:
                    e[f"{g}_duration_s"] = float(ip.duration)
        mp = _instr("measure", (q,))
        if mp is not None and getattr(mp, "error", None) is not None:
            e.setdefault("readout_error", float(mp.error))
        snap["qubits"][q] = e

    qset = set(qubits)
    if tgt is not None:
        for g in ("ecr", "cz", "cx", "rzz"):
            if g not in tgt.operation_names:
                continue
            try:
                items = tgt[g].items()
            except Exception:
                continue
            for pair, ip in items:
                if not pair or len(pair) != 2:
                    continue
                aa, bb = int(pair[0]), int(pair[1])
                if aa in qset and bb in qset:
                    d = {}
                    if getattr(ip, "error", None) is not None:
                        d["error"] = float(ip.error)
                    if getattr(ip, "duration", None) is not None:
                        d["duration_s"] = float(ip.duration)
                    snap["two_qubit"][f"{g}_{aa}_{bb}"] = d
    return snap


ap = argparse.ArgumentParser()
ap.add_argument("--run-json", default="results/ovarian_cocultured/run.json")
ap.add_argument("--hardware", action="store_true", help="submit to a real IBM device")
ap.add_argument("--backend", default=None, help="device name (default: least busy)")
ap.add_argument("--fake", default="FakeBrisbane", help="fake backend for the noisy dry run")
ap.add_argument("--shots", type=int, default=8000)
ap.add_argument("--out", default="results/hardware")
ap.add_argument("--variants", default="raw,readout,dd,twirling,all",
                help="comma-separated mitigation variants to sweep in ONE hardware session: "
                     "raw | readout | dd | twirling | all")
ap.add_argument("--no-zne", action="store_true",
                help="skip the ZNE/mirror pass (otherwise ZNE runs in the same hardware session)")
ap.add_argument("--zne-scales", default="1,3,5",
                help="comma-separated odd noise scale factors for ZNE folding")
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

# Incremental save: write the report to disk after EVERY stage, so a later failure
# (e.g. the ZNE scaffold) can never discard already-collected device results.
REPORT_FN = os.path.join(a.out, "hardware_postopt_report.json")


def save(note=""):
    json.dump(report, open(REPORT_FN, "w"), indent=1, default=float)
    print(f"  saved -> {REPORT_FN}{(' (' + note + ')') if note else ''}", flush=True)


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
    # calibration snapshot for the physical qubits the optimized circuit used
    try:
        out["calibration"] = device_calibration(backend, out["optimized"].get("physical_layout"))
    except Exception as e:
        out["calibration"] = {"error": repr(e)}
    report[tag] = out
    print(f"[{tag}] KL baseline {b:.4f} -> optimized {f:.4f} ({out['reduction_pct_raw']:.1f}% reduction)")
    save(tag)                       # persist immediately after each variant finishes
    return out


# --------------------------------------------------------------------------- #
# ZNE / mirror scaffold (HARDWARE-only, opt-in). Global unitary folding.       #
#                                                                             #
# This reuses the EXACT scoring machinery of assert_quality() so the scale=1   #
# self-check is apples-to-apples with the 'raw'-variant run: same circuit      #
# builder, same Initialize->StatePreparation conversion, same                  #
# get_probability_distribution formatting, and the same target construction    #
# (targets are AMPLITUDE dicts -> _process_target_state_input -> |amp|^2 probs)#
# --------------------------------------------------------------------------- #
def _zne_targets():
    """Target PROBABILITY dicts, built identically to assert_quality().
    qx.target1/target2 are L1 *amplitude* dicts, NOT probabilities -- they must
    pass through _process_target_state_input and get squared, or the KL goes
    negative (feeding raw amplitudes to KL was the original ZNE bug)."""
    from qxct.circuits import _process_target_state_input
    t1, nq1 = _process_target_state_input(qx.target1, None)   # norm=None: already amplitudes
    t2, nq2 = _process_target_state_input(qx.target2, None)
    target1 = {bin(i)[2:].zfill(nq1): float(np.abs(t1[i]) ** 2) for i in range(2 ** nq1)}
    target2 = {bin(i)[2:].zfill(nq2): float(np.abs(t2[i]) ** 2) for i in range(2 ** nq2)}
    return target1, target2, nq1, nq2


def _bare_unitary():
    """The measurement-free optimized circuit U over a flat (n1+n2)-qubit register,
    built with the SAME path as assert_quality (build_optimized_circuit +
    Initialize->StatePreparation), then stripped of measurements/barriers so it is
    invertible for folding. Qubit indices (and therefore endianness) match the
    trusted hardware run exactly."""
    from qiskit import QuantumCircuit
    from qxct.hardware import build_optimized_circuit, _initialize_to_stateprep
    base = build_optimized_circuit(qx.circ1, qx.circ2, topo, angles)
    base = _initialize_to_stateprep(base)                  # StatePreparation is invertible
    n = base.num_qubits
    U = QuantumCircuit(n)
    for inst in base.data:
        if inst.operation.name in ("measure", "barrier"):
            continue
        U.append(inst.operation, [base.find_bit(q).index for q in inst.qubits])
    return U, qx.circ1.num_qubits, qx.circ2.num_qubits


def _fold(U, n_pairs):
    """U (U^dag U)^n_pairs  -> noise scale = 2*n_pairs + 1 (identity in the ideal limit)."""
    qc = U.copy()
    Udg = U.inverse()
    for _ in range(n_pairs):
        qc.barrier(); qc = qc.compose(Udg); qc.barrier(); qc = qc.compose(U)
    return qc


def run_zne(backend, scales, ref_kl=None):
    """SCAFFOLD. Global-folding ZNE of the optimized circuit on real hardware.
    Measures KL(scale) at each odd noise scale and extrapolates to scale -> 0.
    Prints a scale=1 self-check vs ``ref_kl`` (the 'raw'-variant optimized KL from
    this same session). Twirling + DD on. NOTE: raw (no readout correction) KL
    here -- the point is the noise-scale trend. Scoring mirrors assert_quality()."""
    from qiskit import ClassicalRegister, transpile
    from qiskit_ibm_runtime import SamplerV2
    from qxct.circuits import calculate_kl_divergence, get_probability_distribution

    U, n1, n2 = _bare_unitary()
    target1, target2, _, _ = _zne_targets()

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = a.shots
    try:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
    except Exception:
        pass

    rows = []
    for s in scales:
        n_pairs = (int(s) - 1) // 2
        qc = _fold(U, n_pairs)
        c1 = ClassicalRegister(n1, "c_measure1"); c2 = ClassicalRegister(n2, "c_measure2")
        qc.add_register(c1, c2)
        qc.measure(range(n1), c1); qc.measure(range(n1, n1 + n2), c2)
        isa = transpile(qc, backend, optimization_level=3, seed_transpiler=42)
        res = sampler.run([isa]).result()[0]
        p1 = get_probability_distribution(res.data.c_measure1.get_counts(), n1, a.shots)
        p2 = get_probability_distribution(res.data.c_measure2.get_counts(), n2, a.shots)
        kl = calculate_kl_divergence(p1, target1) + calculate_kl_divergence(p2, target2)
        rows.append({"scale": int(s), "fold_pairs": n_pairs, "depth": isa.depth(),
                     "kl_total": float(kl)})
        print(f"[zne] scale {s}: depth {isa.depth()}, KL {kl:.4f}")

    xs = np.array([r["scale"] for r in rows], float)
    ys = np.array([r["kl_total"] for r in rows], float)
    lin = np.polyfit(xs, ys, 1)
    zne = {"scales": rows, "kl_zero_noise_linear": float(np.polyval(lin, 0.0))}
    if len(xs) >= 3:
        zne["kl_zero_noise_quadratic"] = float(np.polyval(np.polyfit(xs, ys, 2), 0.0))
    # self-check: scale=1 folded KL should match the raw-variant optimized KL
    s1 = next((r["kl_total"] for r in rows if r["scale"] == 1), None)
    if ref_kl is not None and s1 is not None:
        zne["scale1_selfcheck"] = {"zne_scale1_kl": s1, "raw_variant_optimized_kl": ref_kl,
                                   "abs_diff": abs(s1 - ref_kl)}
        print(f"[zne] self-check: scale1 KL {s1:.4f} vs raw-variant {ref_kl:.4f} "
              f"(diff {abs(s1-ref_kl):.4f}; should be ~shot noise if endianness/targets match)")
    report["zne"] = zne


# mitigation variants swept together so they share one calibration window
def _variant(name):
    s = a.shots
    table = {
        "raw":      MitigationConfig(shots=s, optimization_level=3),
        "readout":  MitigationConfig(measurement_mitigation=True, shots=s, optimization_level=3),
        "dd":       MitigationConfig(dynamical_decoupling=True, shots=s, optimization_level=3),
        "twirling": MitigationConfig(twirling_gates=True, twirling_measure=True, shots=s, optimization_level=3),
        "all":      MitigationConfig.all_on(shots=s),
    }
    return table[name]

variants = [v.strip() for v in a.variants.split(",") if v.strip()]

# 2) fake-noisy dry run: the 'all' variant, to validate the whole flow offline (no queue)
from qiskit_aer import AerSimulator
import qiskit_ibm_runtime.fake_provider as fp
fake = getattr(fp, a.fake)()
_run(AerSimulator.from_backend(fake, seed_simulator=42),
     MitigationConfig.all_on(shots=a.shots), f"fake_noisy_{a.fake}__all")

# 3) real hardware: sweep every mitigation variant + ZNE in ONE session (same device,
#    same calibration window; each variant also records a calibration snapshot)
if a.hardware:
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService()
    backend = svc.backend(a.backend) if a.backend else svc.least_busy(operational=True, simulator=False)
    print("submitting to", backend.name, "- mitigation variants:", variants)
    for name in variants:
        _run(backend, _variant(name), f"hardware_{backend.name}__{name}")
    if not a.no_zne:
        scales = [int(x) for x in a.zne_scales.split(",") if x.strip()]
        ref = (report.get(f"hardware_{backend.name}__raw", {})
               or report.get(f"hardware_{backend.name}__all", {})).get("optimized", {}).get("kl_total")
        print(f"[zne] ZNE/mirror at scales {scales} on {backend.name} ...")
        # The ZNE scaffold is opt-in and secondary: a failure here must NEVER lose
        # the mitigation-sweep results already saved above. Isolate it.
        try:
            run_zne(backend, scales, ref_kl=ref)
            save("zne")
        except Exception as e:
            import traceback
            report["zne"] = {"error": repr(e), "traceback": traceback.format_exc()}
            print(f"[zne] FAILED (sweep results are safe): {e!r}")
            save("zne-failed")

print(f"[exact] KL baseline {report['exact']['baseline_kl']:.4f} -> "
      f"optimized {report['exact']['final_kl']:.4f}")
save("final")
