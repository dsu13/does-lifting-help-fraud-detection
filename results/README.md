# results/

The result files behind the paper's tables, one folder per dataset. They hold
aggregates and per-seed scores only (no features, labels or graphs), so they can be
published even though the datasets cannot. The experiment scripts write here directly;
`scripts/09_make_tables.py` reads from here. The command behind each file is in
`main.py` (`python main.py --list`); the table names the stage that writes it.

| File | Written by (`main.py` stage) | Paper |
|---|---|---|
| `<dataset>/stats.json` | `05_compute_stats.py` (`data`) | `tab:datasets` (and the sizes quoted by hand in `tab:dynamic`) |
| `<dataset>/baselines.json` | `06_run_baselines.py` (`static`) | `tab:leaderboard`, `tab:secondary` |
| `<dataset>/ablation_results.json` | `07_train_method.py`, generation ablation (`static`) | `tab:ablation`, the λ=0 row of `tab:selection`, "ours" rows of `tab:leaderboard` and `tab:secondary` |
| `<dataset>/method_results.json` | `07_train_method.py`, selection study (`static`) | `tab:selection` (λ > 0 rows) |
| `synthetic/synthetic_results.json` | `08_synthetic_camouflage.py` (`static`) | `tab:synthetic`, `tab:synthetic_diag` |
| `<dataset>/temporal_crossover.json` | `10_temporal_crossover.py` (`temporal`) | `tab:crossover` and the per-dataset crossover tables, `tab:paired_arms`, statistics macros; static-pool rows of `tab:elliptic_redesign`, 2% rows of `tab:window_sweep`, the arms paired with the variants in `tab:temporal_variants` |
| `<elliptic or ellipticpp>_<xstep or samestep>/temporal_crossover.json` | `10_temporal_crossover.py`, cell pools (`temporal`) | `tab:elliptic_redesign`, cross-step rows of `tab:window_sweep` |
| `<dataset>/temporal_crossover_w<F>.json` | `10_temporal_crossover.py` (`sweep`) | arms B and D at full labels: 1% and 5% rows of `tab:window_sweep`, other-ω rows of `tab:temporal_variants` |
| `<dataset>/temporal_variants*.json` | `10_temporal_crossover.py` (`variants`) | `tab:temporal_variants` |
| `<elliptic or ellipticpp>_<xstep or samestep>/extra_cells.json` | `02b` / `02d --extra-cells` (`data`) | size of the cross-step family (`app:crossover`) |

Every metric block stores the mean, the sample standard deviation (`std_ddof: 1`), the
number of seeds and the per-seed values. Crossover blocks also store the seed order,
with `null` for a seed that failed, and each crossover file records its run
configuration under `_config` (window, seeds, label fractions, split protocol, and the
degeneracy check when the run builds arm D's cells).
