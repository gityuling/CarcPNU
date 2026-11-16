# CarcPNU

Code for **CarcPNU: Organ-Specific Molecular Carcinogenicity Prediction with Positive–Negative–Unlabeled Learning**.

CarcPNU combines a molecular graph predictor with a Morgan–Tanimoto fingerprint predictor. Provenance-guided PNU learning is applied to the fingerprint branch. The graph predictions remain fixed in the matched supervised/PN-only/PU-only/PNU comparison. Four output scores correspond to **liver, lung, stomach and breast**.

- **Code:** https://github.com/gityuling/CarcPNU
- **Reproduction assets, checkpoints and numerical results:** https://doi.org/10.5281/zenodo.24273180
- **Release:** `v1.0.0`

## Installation

Use Python 3.12 and an NVIDIA GPU. The recorded environment uses PyTorch 2.8.0 with CUDA 12.8 and PyG 2.3.1. Install into a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
```

The package records the reference environment rather than installing a CUDA driver. Choose a compatible PyTorch wheel for other hardware. CPU inference is supported with `--device cpu`; the recorded training and release checks use CUDA.

## Download assets and upstream sources

Download `CarcPNU-reproducibility-v1.0.0.zip` from the Zenodo record and extract it beside this repository. The following commands assume the archive is at `../CarcPNU-reproducibility-v1.0.0`.

```bash
python scripts/fetch_upstream.py
python scripts/verify_release.py --assets ../CarcPNU-reproducibility-v1.0.0 --upstream upstream --full-graph --output verification.json
```

The fetch command obtains the original authors' graph architecture, preprocessing and raw benchmark data directly from [OS-Carcinogenicity](https://github.com/bmil-jnu/OS-Carcinogenicity), pinned to commit `633bbd7294e5b67ae537b88c5f75eb8c53abd454`. It leaves an existing checkout unchanged unless it already matches that commit. The original authors' checkpoint is distinct from the five locally trained graph checkpoints used in CarcPNU.

The verification command checks archive SHA-256 hashes, reconstructs fingerprint predictions from SMILES and fitted parameters, reruns all five local graph checkpoints, and compares scores with the archived predictions. Benchmark molecules use the authors' released graph encodings; novel SMILES use their encoder. This preserves the graph features used in the reported experiments across RDKit versions.

## Refit the four matched fingerprint objectives

Run this in a fresh copy of the repository. Copy only the archive's `inputs/` directory to the repository root; keep archived fitted models and results in the separate asset directory.

```bash
cp -r ../CarcPNU-reproducibility-v1.0.0/inputs .
mkdir -p models predictions tables records logs figures
python scripts/run_ablation.py prepare
python scripts/run_ablation.py train
python scripts/run_ablation.py evaluate
```

The three steps freeze a protocol, fit the four objectives and evaluate them. Existing experiment outputs are protected from overwriting. The original split consists of 435 fingerprint-training parent structures, 57 validation records and 454 test records with partially observed organ labels. The feature basis is trained on training structures only. One deterministic fingerprint fit per objective is combined with each of five fixed graph seeds, 42–46, at graph:fingerprint weights 0.25:0.75.

| Objective | Macro AUROC | Macro AP |
| --- | ---: | ---: |
| Supervised | 0.779741 | 0.931708 |
| PN-only | 0.763403 | 0.912092 |
| PU-only | 0.750633 | 0.916148 |
| PNU | 0.805036 | 0.938051 |

These values are the mean of per-seed, per-organ metrics; computing metrics after averaging predictions is a different aggregation. AUROC and AP do not use a classification threshold. Threshold-based metrics in the original matched experiment use 0.45; manuscript figure data also contain the 0.50 comparison. Complete results and bootstrap intervals are archived under `results/` and `figure_workspace/`.

## Trace annotations and molecular features

```bash
python scripts/rebuild_inputs.py --assets ../CarcPNU-reproducibility-v1.0.0 --upstream upstream --output rebuilt_inputs
```

This reconstructs the original training partitions from upstream CSV files, groups parent structures, assigns P/N/U roles, recomputes fingerprints in the archived feature basis and checks the released snapshots. It exports the source table and row references for every training parent. Role codes are `1=P`, `0=N`, `2=implicit U`, `3=undocumented U`, `-1=conflicting excluded`, and `4=missing excluded`. P/N/U are training roles; the prediction task remains binary organ-specific carcinogenicity scoring.

## Predict new molecules

Prepare a CSV with a `SMILES` column. Other identifier columns are retained.

```bash
python scripts/predict.py --input molecules.csv --output predictions.csv --assets ../CarcPNU-reproducibility-v1.0.0 --upstream upstream
```

The output contains four continuous, five-graph-seed-averaged fusion scores. Use `--fingerprint-only` for the fingerprint branch or `--objective supervised`, `PN_only`, or `PU_only` for the matched alternatives. Invalid structures or structures unsupported by the upstream graph encoder raise an error with a row reference.

## Optional graph retraining

```bash
python scripts/train_graph.py --upstream upstream --seed 42 --output graph_setup_check --check-setup
python scripts/train_graph.py --upstream upstream --seed 42 --output graph_seed42
```

Repeat the second command with seeds 43–46 and distinct output directories if needed. The entry point calls the pinned authors' two-stage training schedule with the local split handling recorded in the archive. The release checked dataset/split construction and the forward pass; the complete five-seed training was performed in the original experiment, whose checkpoints and logs are archived. Exact reported scores are recovered using the archived weights. PNU changes the fingerprint objective, not the graph training objective.

## Figures and experiment records

```bash
python scripts/build_release_figures.py --assets ../CarcPNU-reproducibility-v1.0.0
```

See [FIGURES.md](FIGURES.md) for final panel mapping. The Zenodo archive includes the latest figure source data, per-molecule predictions, fitted fingerprint state, five local graph checkpoints, original graph-training logs, source-row provenance and release verification records. Historical baseline comparison runners are preserved as a separate source archive with their original path layout. Recomputing baseline comparison metrics from saved predictions does not require refitting those comparators.

## Citation and provenance

Use `CITATION.cff` and cite the accompanying Zenodo record. The benchmark is from Song, Choi and Yoo, *Carcinogenicity prediction via multi-task learning of cross-organ representations with attention mechanisms*, **Briefings in Bioinformatics** 2026, 27(3), bbag296, [doi:10.1093/bib/bbag296](https://doi.org/10.1093/bib/bbag296).

Original CarcPNU code is released under MIT. Original release documentation, fitted CarcPNU parameters, and generated results are shared under CC BY 4.0. Third-party data, source code and annotations retain their original rights; see `THIRD_PARTY_NOTICES.md`. Raw upstream source code and original author weights are retrieved from the authors rather than relicensed here.
