# CarcPNU

Research code for **CarcPNU: Organ-Specific Molecular Carcinogenicity Prediction with Positive–Negative–Unlabeled Learning**.

CarcPNU combines predictions from a fixed molecular graph branch and a Morgan–Tanimoto fingerprint branch. The matched ablation compares supervised, PN-only, PU-only, and PNU objectives on the original four-organ benchmark. This repository contains code for reproducing the matched fingerprint-objective experiments from the archived feature and graph-prediction snapshots.

## Contents

- `scripts/run_ablation.py`: prepare, train, and evaluate the four matched configurations.
- `scripts/fitting_core.py`: fingerprint risk and optimizer.
- `scripts/metric_core.py`: per-organ and macro metrics and paired bootstrap calculations.
- `requirements-gpu.txt`: recorded GPU environment; adapt the PyTorch wheel index for your hardware.

The reproduction assets are provided separately in `CarcPNU-reproducibility-v1.zip` (Zenodo record/DOI pending). The archive includes cached training/validation/test features, labels, fixed graph predictions, locally trained graph weights, fingerprint parameters and results. Extract its `inputs/` folder into this repository root. Do not extract the archived `results/` directory into the experiment's output folders.

## Reproduce the four-objective experiment

Use Python 3.12 with CUDA-enabled PyTorch, NumPy and scikit-learn; the recorded environment is `requirements-gpu.txt`. From a fresh copy of this repository with the asset `inputs/` directory in place:

```bash
mkdir -p models predictions tables records logs figures
python scripts/run_ablation.py prepare
python scripts/run_ablation.py train
python scripts/run_ablation.py evaluate
```

The commands require a CUDA-capable GPU. They fit four fingerprint models while reusing five fixed graph-model predictions; they do not retrain graph models or construct features from raw structures. The script refuses to overwrite existing experiment outputs. The paper reports per-seed, per-organ metrics averaged over the five graph seeds and four organs; all five share one fingerprint model for each objective.

On the archived original test set, macro AUROC/AP are supervised 0.779741/0.931708, PN-only 0.763403/0.912092, PU-only 0.750633/0.916148 and PNU 0.805036/0.938051. See saved protocol and results for splits, confidence intervals and full metrics. The released benchmark test resource was examined during development.

## Upstream data and model

The original benchmark, preprocessing notebook, model architecture and data were published separately by Song et al.: [OS-Carcinogenicity](https://github.com/bmil-jnu/OS-Carcinogenicity), source commit `633bbd7294e5b67ae537b88c5f75eb8c53abd454`, DOI [10.1093/bib/bbag296](https://doi.org/10.1093/bib/bbag296). Obtain those upstream materials directly from their authors' repository. The separate asset archive contains derived inputs; permissions and licensing are being checked before public release.

## Citation and licensing

Author list and repository citation metadata are pending confirmation. Source and asset licensing is pending review.
