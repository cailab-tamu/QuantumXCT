# benchmarks

Self-contained benchmarks. Each script and notebook here switches to the `python/` folder
before running, so it finds the datasets and writes to `python/results/` no matter where you
launch it from. They are also steps of `run_all.py`.

| Folder | Run | Writes |
|---|---|---|
| `dm_strategies/` | notebook `dm_strategies_benchmark.ipynb`, or `python benchmarks/dm_strategies/benchmark_dm_strategies.py [--quality]` | `results/dm_strategies/` |
| `scaling/` | `python benchmarks/scaling/scaling_benchmark.py [--exact]` or `scaling_benchmark.ipynb` | `results/scaling/` |
| `hardware/` | `python benchmarks/hardware/run_hardware_postopt.py [--hardware]` or `hardware_validation.ipynb` | `results/hardware/` |

`hardware/` needs the ovarian result `results/ovarian_cocultured/run.json` (run
`qxct_ovarian_cocultured_oo.ipynb` first) and, for a real device, an IBM Quantum account.
