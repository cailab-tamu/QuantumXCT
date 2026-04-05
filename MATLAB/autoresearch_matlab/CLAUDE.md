# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autoresearch loop (inspired by karpathy/autoresearch) for QuantumXCT: instead of training a GPT model, it searches for the quantum circuit topology — a set of entanglement (CX/CRY) gates — that best transforms monoculture gene expression into observed co-culture expression, measured by KL divergence.

## Scientific context

Two cell types interact in co-culture; their gene expression shifts. The model encodes each gene as a qubit, encodes monoculture frequencies as initial quantum state amplitudes, then searches all K-link directed inter-cellular gate configurations for the one whose simulated output best matches co-culture observations.

- Qubits 1–3: fibroblast genes (TGFB1, PDGFRB, IL6)
- Qubits 4–7: cancer cell genes (STAT3, IL6RorST, TGFBR1or2, PDGFB)
- Fixed intracellular gate: `CX(4,5)` (STAT3–IL6RorST within cancer cells)

## Running an experiment

Experiments are run via the **MATLAB MCP server** (not the MATLAB IDE directly). The binary is at:

```
C:\Users\jingc\Documents\GitHub\GEOcellar_autogen\matlab-mcp-core-server-win64.exe
```

Use the `evaluate_matlab_code` tool. The MCP server's initial working folder should be set to `scGEAToolbox_dev` so `sc_*` functions are on the MATLAB path — this is required by `prepare.m`.

**Required on MATLAB path before running:**
- `C:\Users\jingc\Documents\GitHub\scGEAToolbox_dev` — provides `sc_*` functions and data-loading utilities called by `prepare.m` (via `s0_merge_subunit_genes.m`)

Add it once per session if not already on the path:

```matlab
addpath('C:\Users\jingc\Documents\GitHub\scGEAToolbox_dev');
cd('C:\Users\jingc\Documents\GitHub\QuantumXCT\MATLAB\autoresearch_matlab');
diary run.log; run('train.m'); diary off
```

These can be sent as separate `evaluate_matlab_code` calls or chained. The MCP server keeps MATLAB state between calls in the same session.

Extract summary metrics:
```bash
grep "^best_cost:\|^n_sampled:\|^gate_type:" run.log
```

**Primary metric**: `best_cost` (lower = better).

## File roles

| File | Status | Purpose |
|---|---|---|
| `train.m` | **Modify freely** | Hyperparameters, gate type, optimizer, search filters |
| `prepare.m` | **Read-only** | Data loading, quantum state init, `costfn`, `extragates` |
| `../+hlp/` | **Read-only** | Fixed evaluation primitives (KL divergence, objective fn, etc.) |
| `results.tsv` | Untracked | Experiment log — append after every run |
| `train_results.mat` | Untracked | Saved output of each run (overwritten) |

`prepare.m` adds `..` to MATLAB path, making `+hlp/` visible automatically.

## Hyperparameters in `train.m`

```matlab
K              = 5;          % number of inter-cellular links
TIME_BUDGET    = 480;        % seconds per run
OPTIMIZER      = 1;          % 1=fminunc  2=fminsearch  3=fmincon
N_RESTARTS     = 1;          % random restarts per configuration
ANGLE_INIT     = 'random';   % 'random'(-pi,pi) or 'pi'(warm start)
GATE_TYPE      = 'cry';      % 'cry' | 'crx' | 'cx'
SKIP_BIDIRECT  = false;
SKIP_NONUNIQUE = true;
```

Other levers: gate type (`cryGate` / `crxGate` / `cxGate`), anti-bidirectional skip filter, unique-qubit filter, bidirectional KL cost terms, EMA beta.

## Experiment loop protocol

1. Create branch: `git checkout -b experiment/<tag>` (tag = date, e.g. `apr3`)
2. First run is always baseline — run `train.m` as-is before making changes
3. Edit `train.m` → `git commit -am "..."` → run → extract metrics → log to `results.tsv`
4. If `best_cost` improved: keep commit. Otherwise: `git reset --hard HEAD~1`

## `results.tsv` format

Tab-separated. Do not commit this file.

```
commit	best_cost	status	description
a1b2c3d	0.201337	keep	baseline K=3 CRY fminunc 5 restarts
```

- `status`: `keep`, `discard`, or `crash`

## Saved outputs

`train_results.mat` contains: `Y` (scores), `Cc` (circuits), `Theta` (optimised angles), `configsK` (all configurations), plus the four distributions and metadata. Load it without re-running to do further analysis or visualization.

## Simplicity criterion

A marginal drop in `best_cost` that requires a complicated workaround is not worth it. Removing code and getting equal or better results is a win.
