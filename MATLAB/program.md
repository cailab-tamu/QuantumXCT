# QuantumXCT

Search for the quantum circuit topology — the set of entanglement gates — that transforms an input quantum state (encoding monoculture gene expression) into an output quantum state that matches the observed co-culture gene expression as closely as possible. The match is measured by KL divergence. Lower is better.

## The scientific problem

Two cell types (fibroblasts, cancer cells) are cultured alone (monoculture) and together (co-culture). When they interact, their gene expression changes. We model that change as a quantum state transformation:

```
|monoculture⟩  ──[entanglement gates]──►  |co-culture⟩
```

- **Qubits 1–3**: fibroblast genes (`TGFB1`, `PDGFRB`, `IL6`)
- **Qubits 4–7**: cancer cell genes (`STAT3`, `IL6RorST`, `TGFBR1or2`, `PDGFB`)
- **Entanglement gates**: CX or CRY gates between the two qubit sets represent ligand-receptor interactions
- **K-link topology**: which K pairs of qubits are connected defines the interaction network

The goal is to find the K-link topology (and optimised gate angles, for parameterised gates) that minimises the total KL divergence between the simulated output state and the observed co-culture state across both cell types.

## Setup

To start a new experiment session:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr3`). The branch `experiment/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b experiment/<tag>` from main.
3. **Read the in-scope files** to get full context:
   - `prepare.m` — fixed: loads the dataset, extracts pattern distributions, initialises quantum states, defines `costfn` and `extragates`. Do not modify.
   - `train.m` — the file you modify. Hyperparameters, gate type, optimizer, number of restarts, search space.
   - `+hlp/i_obj.m` — the gate-angle objective function called by the optimizer. Do not modify.
   - `+hlp/i_kldiverg.m` — the KL divergence metric. Do not modify.
4. **Initialise results.tsv**: create it with just the header row. The baseline will be recorded after the first run.
5. **Confirm and go**.

## Experimentation

Run the training script from the MATLAB command window:

```matlab
run('train.m')
```

**What you CAN do** — modify `train.m` freely:
- Change `K` (number of entanglement links, e.g. 3 or 4)
- Change `OPTIMIZER` (1 = `fminunc`, 2 = `fminsearch`, 3 = `fmincon`)
- Change `N_RESTARTS` (random restarts per configuration)
- Change `ANGLE_INIT` (`'random'` or `'pi'`)
- Switch gate type from `cryGate` to `crxGate` or `cxGate`
- Change or remove the anti-bidirectional skip filter
- Add a unique-qubit filter (`length(unique(configsK{idx})) < 6`)
- Add bidirectional cost terms (forward + reverse KL, see `s9f`)
- Change the EMA beta, output filename, visualisation

**What you CANNOT do:**
- Modify `prepare.m`. It is read-only. It defines the dataset, the quantum state initialisation, `costfn`, and `extragates`.
- Modify any file in `+hlp/`. These are the fixed evaluation primitives.
- Change the dataset or the four target distributions (`pt_f_mo`, `pt_c_mo`, `pt_f_co`, `pt_c_co`).
- Change the ground-truth topology `targettop = [7 2; 1 6; 3 5]`. This is fixed biology.

**The goal**: get the lowest `best_cost` while also getting the biologically ideal configurations (`n_ideal > 0`) to rank near the top. The ideal configurations are those whose link set matches `targettop`. A good result is one where the numerically best configurations and the biologically ideal configurations coincide.

**Simplicity criterion**: all else being equal, simpler is better. A marginal drop in `best_cost` that requires a complicated workaround is not worth it. Removing code and getting equal or better results is a win.

**The first run**: your very first run should always be to establish the baseline — run `train.m` as-is without changes.

## Output format

When the script finishes it prints a summary like this:

```
---
best_cost:      0.201337
ideal_cost:     0.318204
n_configs:      2024
n_valid:        1344
n_ideal:        6
K:              3
optimizer:      1
n_restarts:     5
total_seconds:  183.4
output_file:    train_results.mat
results saved → train_results.mat
```

Extract the key metrics from a saved log:

```bash
grep "^best_cost:\|^ideal_cost:\|^n_ideal:" run.log
```

`best_cost` is the primary metric (lower = better). `ideal_cost` measures how well the known-correct topology scores — it should decrease together with `best_cost`. If `best_cost` drops but `ideal_cost` stays high, the optimizer found a numerically good but biologically meaningless topology.

All results (scores `Y`, circuits `Cc`, angles `Theta`, configurations `configsK`) are saved to `train_results.mat` and can be loaded for further analysis without re-running.

## Logging results

Log every completed run to `results.tsv` (tab-separated — commas break in descriptions). Do not commit this file; leave it untracked.

```
commit	best_cost	ideal_cost	n_ideal	status	description
```

1. `commit` — short git hash (7 chars)
2. `best_cost` — lowest KL cost achieved across all configurations (6 decimal places)
3. `ideal_cost` — KL cost of the best biologically ideal configuration; `NA` if `n_ideal = 0`
4. `n_ideal` — number of configurations matching `targettop` that survived the skip filter
5. `status` — `keep`, `discard`, or `crash`
6. `description` — short plain-text description of what this experiment tried

Example:

```
commit	best_cost	ideal_cost	n_ideal	status	description
a1b2c3d	0.201337	0.318204	6	keep	baseline K=3 CRY fminunc 5 restarts
b2c3d4e	0.187442	0.290115	6	keep	increase N_RESTARTS to 10
c3d4e5f	0.201002	0.321000	6	discard	switch to fminsearch, no improvement
d4e5f6g	0.000000	NA	0	crash	K=4 unique-qubit filter removed all ideal configs
e5f6g7h	0.195881	0.299340	6	keep	random angle init outperforms pi init
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `experiment/apr3`).

LOOP FOREVER:

1. Check the git state: which branch/commit you are on.
2. Modify `train.m` with an experimental idea — hack it directly.
3. `git commit -am "description of change"`
4. Run the experiment and redirect output: in MATLAB, `diary run.log; run('train.m'); diary off`
5. Extract results: `grep "^best_cost:\|^ideal_cost:\|^n_ideal:" run.log`
6. If the grep output is empty, the run crashed. Run `type run.log` to read the MATLAB error and attempt a fix. If the idea is fundamentally broken after a couple of attempts, abandon it.
7. Record results in `results.tsv`.
8. If `best_cost` improved (lower) **and** `ideal_cost` did not worsen, advance the branch (keep the commit).
9. If results are equal or worse, `git reset --hard HEAD~1` to revert.

**Timeout**: MATLAB has no built-in time budget. If a run is taking unexpectedly long (> 30 minutes for K=3), it is likely hung. Kill it, reduce `N_RESTARTS` or switch to a faster optimizer, and retry.

**Crashes**: if the run crashes with a MATLAB error (e.g. dimension mismatch, undefined function), read the error, fix the obvious bug, and re-run. If the underlying idea is broken, log it as `crash` and move on.

**NEVER STOP**: once the experiment loop has begun, do not pause to ask whether to continue. Run experiments autonomously until manually interrupted. If you run out of obvious ideas, consider: trying K=4, bidirectional cost (adding reverse KL terms from `s9f`), different gate types (`crxGate`), constrained angles (`fmincon` with bounds), or combining a fixed target-topology gate layer with a searched layer (as in `s10a`'s `layer_inte1`).
