# Does Lifting Help Fraud Detection? Code

**Diego Saldaña-Ulloa** and **Jose Daniel Torres Campos**

Paper: [SSRN 7500858](https://ssrn.com/abstract=7500858), DOI
[10.2139/ssrn.7500858](https://doi.org/10.2139/ssrn.7500858) (see [Citation](#citation)).

Code for the paper *Does Lifting Help Fraud Detection? A Higher-Order Benchmark and a
Three-Regime Study of Camouflaged Fraud*. It builds the fraud graphs and their
candidate-cell (hypergraph) substrate, trains the baselines and the learnable-lifting
model, runs the synthetic D(μ) study and the four-arm temporal crossover, and writes
the paper's generated tables and statistics macros.

> **No data is included.** The scripts read the files you place in `data/raw/`, write
> the graphs derived from them to `data/processed/` (both git-ignored), and write every
> result to `results/` (tracked: aggregates and per-seed scores only). Each dataset is
> subject to its provider's terms; download it from the source listed below.

## Layout

```
CITATION.cff         citation metadata (GitHub's "Cite this repository")
LICENSE              MIT (code only; the datasets keep their own terms)
config.py            paths, seeds, candidate-cell and split defaults (training
                     hyperparameters: MethodConfig in src/method/train.py, script flags)
main.py              runs every step of the paper in order (see Reproducing the paper)
requirements.txt
results/             the result JSONs behind the paper's tables (see results/README.md)
scripts/             the numbered steps, each runnable on its own
  00_download_data.py        raw data: downloads what has a public link, lists what is missing
  01_setup_check.py          package check, data folders
  02_build_ieee_cis.py       IEEE-CIS -> event graph + candidate cells
  02b_build_elliptic.py      Elliptic      (+ --extra-cells xstep | samestep)
  02c_build_dgraph.py        DGraph-Fin    (labeled-node-centered subsample)
  02d_build_ellipticpp.py    Elliptic++    (+ --extra-cells xstep | samestep)
  02e_build_s_ffsd.py        S-FFSD
  02f_build_ethereum.py      Ethereum phishing (XBlock)
  03_load_gad_datasets.py    Amazon, YelpChi
  04_make_splits.py          splits for Amazon / YelpChi / IEEE-CIS
  05_compute_stats.py        dataset statistics
  06_run_baselines.py        baselines
  07_train_method.py         lifting model: generation ablation and selection study
  08_synthetic_camouflage.py synthetic D(mu) study
  09_make_tables.py          the paper's generated tables and statistics macros
  10_temporal_crossover.py   the temporal crossover and its variants
src/
  data/      loaders, splits, statistics, temporal cells (temporal.py)
  graph/     MultiRelationGraph, CandidateCells, the candidate-cell generator
  baselines/ rf, mlp, gcn, bwgnn, fixed_hg, metrics
  method/    CamoLiftNet, objectives, refinement (carving), training, D(mu)
  results_io.py, seeding.py, torch_utils.py   locked result merges, seeds, sparse tensors
```

Every script can be launched from any directory as `python scripts/<name>.py`.

## Install

Use Python 3.11 or 3.12: `numpy<2.0` has no wheels for 3.13 and later.

```bash
py -3.11 -m venv .venv          # Linux/macOS: python3.11 -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.11 and a CUDA 11.8 build of torch on an NVIDIA RTX 4070 (the
code also runs on CPU, much more slowly); the exact versions are listed in
`requirements.txt`. `kaggle` is needed only for the Kaggle downloads of
`00_download_data.py` (see Data).

## Data

No dataset is redistributed here: each stays under its provider's terms. Every file goes
into its own folder under `data/raw/`, as listed below. `python scripts/00_download_data.py`,
the first step of `main.py`, downloads the five datasets that have a public link, skips
those already in place, and prints the steps for the two that have to be fetched by hand.
It stops with an error while any file is missing.

The Kaggle downloads need the `kaggle` package (`pip install kaggle`) and Kaggle API
credentials, set up as https://www.kaggle.com/docs/api describes. For IEEE-CIS, also accept
the competition rules first, at https://www.kaggle.com/c/ieee-fraud-detection/rules, with the
same account.

| Dataset | Files, in `data/raw/` | Source | How |
|---|---|---|---|
| IEEE-CIS | `ieee_cis/train_transaction.csv`, `ieee_cis/train_identity.csv` | https://www.kaggle.com/c/ieee-fraud-detection | step 00, Kaggle API |
| Amazon, YelpChi | `gad/Amazon.mat`, `gad/YelpChi.mat` | https://github.com/YingtongDou/CARE-GNN, `data/Amazon.zip` and `data/YelpChi.zip` | step 00, direct download |
| Elliptic | `elliptic/elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` | https://www.kaggle.com/datasets/ellipticco/elliptic-data-set | step 00, Kaggle API |
| Elliptic++ | `ellipticpp/txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv` | the Google Drive folder linked from https://github.com/git-disl/EllipticPlusPlus | by hand |
| DGraph-Fin | `dgraph/dgraphfin.npz` | https://dgraph.xinye.com (sign-up required) | by hand |
| S-FFSD | `s_ffsd/S-FFSD.csv` | https://github.com/AI4Risk/antifraud, `data/S-FFSD.zip` | step 00, direct download |
| Ethereum | `ethereum/MulDiGraph.pkl` | https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network (the XBlock release) | step 00, Kaggle API |

The two datasets fetched by hand:

- **Elliptic++.** Open the Google Drive folder linked from the Elliptic++ repository, go to
  `Transactions Dataset`, and download `txs_features.csv`, `txs_classes.csv` and
  `txs_edgelist.csv` into `data/raw/ellipticpp/`. The `Actors Dataset` folder is not used.
- **DGraph-Fin.** Sign up at https://dgraph.xinye.com and accept its data-use terms. Then
  download `DGraphFin.zip` from the DGraph-Fin entry of its Datasets page, and unzip
  `dgraphfin.npz` into `data/raw/dgraph/`.

Rerun `python scripts/00_download_data.py`, or `python main.py`, once the files are in place.

The exact file formats each loader expects are documented at the top of its module in
`src/data/`. Amazon's first 3,305 nodes carry no label (the GADBench convention); they
are marked unlabeled and never enter a split.

## Reproducing the paper

With the raw files in place (see Data), one command runs the whole study:

```bash
python main.py
```

`main.py` holds the single list of commands behind the stored results and runs them in
order, each as its own process, stopping at the first failure. Its first step,
`00_download_data.py`, puts the raw data in place (see Data). The run has six stages: `--stages` runs some of them, in pipeline order, and `--list` prints every command
without running it.

```bash
python main.py --list
python main.py --stages temporal sweep variants tables
```

| Stage | Scripts | What it produces |
|---|---|---|
| `data` | 00, 01, 02-02f, 03, 04, 05 | raw data, graphs, candidate pools (with the Elliptic cross-step and same-step pools), splits, dataset statistics |
| `static` | 06, 07, 08 | baselines, the generation ablation, the selection study at λ = 0.5, 1, 2 (its λ = 0 row is the ablation's label run), the D(μ) sweep |
| `temporal` | 10 | the crossover on the six temporal datasets at ω = 2%, then arm D on the Elliptic cell pools |
| `sweep` | 10 | arms B and D at ω = 1% and 5% on IEEE-CIS, DGraph-Fin, S-FFSD and Ethereum |
| `variants` | 10 | the timed gate, past and future cones, and the uncut-pool lifting and columns |
| `tables` | 09 | the LaTeX tables and statistics macros, from `results/` alone (no data or GPU), in `tables/` (git-ignored); `python scripts/09_make_tables.py --out-dir DIR` writes them anywhere |

Seed counts follow the paper's Reproducibility statement: 5 on the static benchmark; 8 on
D(μ), Elliptic and Elliptic++; 16 on Ethereum; 3 on IEEE-CIS, DGraph-Fin and S-FFSD. On
the 49 integer Elliptic steps ω = 2% rounds to one step, and on the Elliptic families
every window of at least one step gives the same cells, so they are not swept.

**GPU and running time.** The torch scripts use a CUDA GPU whenever one is available.
On an RTX 4070 an epoch of the lifting model takes about 0.3 s at IEEE-CIS size and 0.16 s
at Elliptic size, against 11 s and 4.5 s on 8 CPU threads, so the whole reproduction takes
hours on a GPU and days on a CPU. `main.py` runs one job at a time; do the same when you
launch scripts yourself: on Windows, several processes sharing one GPU slowed each other
down 30 to 100 times in our measurements.
Without a GPU, give each job several threads (the default uses them all). Two jobs may
share an output file safely only if they write different keys (06 and 07 merge under a
lock); never run two jobs on the same 10 output file.

| Script | Writes | Paper |
|---|---|---|
| 00 | the raw files of Data, in `data/raw/<folder>/` | all |
| 02*, 03, 04 | `data/processed/<dataset>/`: `graph_meta.json`, `graph_arrays.npz`, `relation_*.npz`, `candidates_*.npz`, `candidates_provenance.json`, `time.npz` (temporal data), `splits/seed*.npz` | all |
| 02b, 02d with `--extra-cells` | `results/<dataset>/extra_cells.json` | size of the cross-step family (`app:crossover`) |
| 05 | `results/<dataset>/stats.json` | `tab:datasets`; sizes quoted in `tab:dynamic` |
| 06 | `results/<dataset>/baselines.json` | `tab:leaderboard`, `tab:secondary` |
| 07 (default) | `results/<dataset>/ablation_results.json` | `tab:ablation`; the λ=0 row of `tab:selection`; "ours" rows of `tab:leaderboard` and `tab:secondary` |
| 07 (selection) | `results/<dataset>/method_results.json` | `tab:selection` (λ > 0 rows) |
| 08 | `results/synthetic/synthetic_results.json` | `tab:synthetic`, `tab:synthetic_diag` |
| 10 (`temporal`, main runs) | `results/<dataset>/temporal_crossover.json` | `tab:crossover` and the per-dataset crossover tables, `tab:paired_arms`, statistics macros; static-pool rows of `tab:elliptic_redesign`, 2% rows of `tab:window_sweep`, the arms paired with the variants in `tab:temporal_variants` |
| 10 (`temporal`, cell pools) | `results/<elliptic or ellipticpp>_<xstep or samestep>/temporal_crossover.json` | `tab:elliptic_redesign`, cross-step rows of `tab:window_sweep` |
| 10 (`sweep`) | `results/<dataset>/temporal_crossover_w*.json` | 1% and 5% rows of `tab:window_sweep`, other-ω rows of `tab:temporal_variants` |
| 10 (`variants`) | `results/<dataset>/temporal_variants*.json` | `tab:temporal_variants` and the cone counts in its caption |

`09_make_tables.py` also writes `crossover_stats*.tex`, the statistics macros the text
and `tab:crossover_summary` quote: for the arm pairs D − A, D − B and B − A of each main
run at full labels, the AP gap, the paired t statistic, its degrees of freedom, the
two-sided p value and the 95% confidence interval. `crossover_stats_variants.tex` holds
the same test for D against the uncut-pool columns, and the cone counts.
`tab_paired_arms.tex` tabulates the gaps and p values of every arm pair in the main run
of each dataset and label fraction, and on Ethereum again on the second half of its
seeds. The paper's other two result tables are typed by hand from these outputs:
`tab:dynamic` from `stats.json`, `tab:crossover_summary` from the macros.

## What the code implements

| Paper | Code |
|---|---|
| multi-relation graph `G` | `MultiRelationGraph` (`src/graph/schema.py`) |
| candidate cells `C(G)`, incidence `B` | `CandidateCells`; `generate_candidate_cells` (`src/graph/candidate_cells.py`): shared-attribute groups, closed relation neighbourhoods (centre always kept when capped), feature-kNN cells `{v} ∪ kNN_k(v)` |
| Hard-Concrete gates, higher-order message passing, feature-preserving readout | `CamoLiftNet` (`src/method/model.py`) |
| objective: reweighted focal BCE + λ·R + β·density | `src/method/objectives.py`, `src/method/train.py` |
| label-guided carving and its count-matched controls | `src/method/refinement.py` |
| temporal-coherence lifting (arm D) | `temporal_coherence_cells` (`src/data/temporal.py`) |
| temporal columns (arm B) | `cell_temporal_features`: computed from arm D's own cells, same ω |
| causal cones, their mirror image and their columns (variants) | `cone_cells` (`direction='past'` or `'future'`), `cone_node_features`; the apex is the only receiver (`CandidateCells.apex`) |
| time-aware gate (variant) | `cell_temporal_descriptor`, appended to the gate scorer's input (`CandidateCells.cell_features`) |
| cross-step family (Elliptic redesign) | `cross_step_knn_cells`, `add_step_knn_family` |
| planted-camouflage model D(μ) | `src/method/synthetic.py` |
| metrics | `src/baselines/metrics.py` |

## Protocol notes

- **Model selection.** Every model is selected on validation AP. The random forest
  picks from a small grid; the MLP, GCN, BWGNN, `fixed_hg` and the lifting model
  early-stop on validation AP. The lifting model and crossover arms A-D select
  checkpoints only from the end of the λ warm-up and ramp (epoch 120 by default), for
  every λ including 0, so a λ > 0 row is a model trained at its target λ and λ
  comparisons share one early-stopping rule.
- **Statistics.** Standard deviations are sample standard deviations (ddof = 1). Arm
  comparisons are paired t-tests over seeds: within a seed every arm shares the split,
  and a seed that fails in one arm is dropped from that pair while the other seeds stay
  paired. Seeds 0-4 use the saved split files; later seeds draw new stratified splits,
  so seeds differ in both split and initialization. The exception is IEEE-CIS, which uses
  one chronological split throughout (the cut does not depend on the seed), so its seeds
  vary only the initialization and, in the crossover, the subsample of positives.
  `--temporal-split` gives any other dataset the same strictly time-ordered split in the
  crossover; no reported run uses it.
- **Temporal cells.** A temporal-coherence cell groups members that co-occur within ω,
  including members up to ω later, so arms B and D use timestamps within ω
  transductively; neither uses labels. The cross-step family is built from past
  neighbours only, but message passing still sends the later anchor's features to its
  earlier cell members, so it is transductive within the window in the same way. The
  causal cones are the exception: a cone holds a node and its strictly earlier
  co-members within ω, and only that node receives the cone's message, so no message
  reaches a node from a later one; the inputs are not causal, since the candidate cells
  the cones are cut from and, on Ethereum, the node features draw on the whole history
  (see Subsamples). The future cones are their mirror image (co-members up to ω later),
  a probe of what later co-members carry, not a deployable model. No dataset's feature
  matrix contains the raw timestamp, so arm A is time-free. `--window-frac F` sets
  ω = F × (node-time span); on integer time it is rounded up to whole units.
- **Elliptic.** No Elliptic edge crosses a time step, so no static cell spans time: on
  `elliptic` arm D is a static higher-order lifting and arm B's columns carry static
  cell counts only. `10_temporal_crossover.py` logs this and records
  `temporal_arms_degenerate` in the result. The `*_xstep` datasets add cells that span
  time; the `*_samestep` datasets are their count-matched same-step control.
- **Subsamples.** DGraph-Fin and Ethereum are subsampled with `--max-nodes`. DGraph-Fin
  takes a labeled-node-centered sample (labeled nodes with their 1-hop neighbourhoods;
  labels are used only as labeled/unlabeled, never by class), with node time taken from
  the full edge stream. Ethereum takes a centred sample: every phishing address, then
  non-phishing addresses in a seeded uniform order, each with its full 1-hop
  neighbourhood, until the budget is spent. Only these centres are labeled; their
  neighbours are unlabeled, so every labeled node of either class keeps all its
  transactions in the graph. Features and first-appearance times are computed on the
  full transaction list. The XBlock data was itself collected outward from phishing
  addresses, a collection bias no sample can remove.
- **Baselines that are not reproductions.** `fixed_hg` is an HGNN-style fixed
  higher-order network on the dataset's own `C(G)`, a representative of the fixed-rule
  hypergraph family (TROPICAL, HCLNet), not their code.
- **Metrics.** AP is primary and ROC-AUC is a diagnostic. Recall@K takes K = the number
  of positives, so it equals precision@K. Best-F1 picks its threshold on the test scores,
  so it is optimistic and secondary.

## Citation

If you use this code, please cite the paper (GitHub's "Cite this repository" button gives
the same reference, from `CITATION.cff`):

> Saldaña-Ulloa, Diego and Torres Campos, Jose Daniel, Does Lifting Help Fraud
> Detection? A Higher-order Benchmark and a Three-regime Study of Camouflaged Fraud
> (September 01, 2026). Available at SSRN: https://ssrn.com/abstract=7500858 or
> http://dx.doi.org/10.2139/ssrn.7500858

```bibtex
@misc{saldanaulloa2026lifting,
  author       = {Salda{\~n}a-Ulloa, Diego and Torres Campos, Jose Daniel},
  title        = {Does Lifting Help Fraud Detection? {A} Higher-order Benchmark and a
                  Three-regime Study of Camouflaged Fraud},
  year         = {2026},
  month        = sep,
  howpublished = {SSRN},
  doi          = {10.2139/ssrn.7500858},
  url          = {https://ssrn.com/abstract=7500858}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The license covers the
code only: the datasets are not included and remain under their providers' terms.
