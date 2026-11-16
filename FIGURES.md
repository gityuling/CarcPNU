# Figure sources

The Zenodo archive preserves the relative data layout required by the original figure scripts. Run `python scripts/build_release_figures.py --assets ../CarcPNU-reproducibility-v1.0.0` from this repository.

| Manuscript element | Released source |
| --- | --- |
| Figure 1, workflow | Editable final workflow SVG and final PDF in the figure workspace |
| Figure 2, benchmark ROC/PR and organ metrics | `build_benchmark_curves.py`; `source_data/benchmark_curve_predictions.npz` |
| Figure 3, structural and objective components | `build_merged_ablation.py`; representation and matched-ablation tables |
| Figure 4, prior/provenance sensitivity | `build_provenance_sensitivity.py`; archived sensitivity results |
| Figure 5, molecular cases | Final editable `fig5_molecular_cases.svg`, final PDF, case tables and individual molecular SVGs |
| Supplementary nine-metric panels | `build_figS1_portrait.py` |
| Supplementary macro benchmark | `build_radar_figures.py` |
| Supplementary organ ROC/PR | `build_figS3_portrait.py` |
| Supplementary structural space/loss panels | `build_merged_ablation.py` and `build_radar_figures.py` |
| TOC graphical abstract | `illustrations/toc_assets_v3/CarcPNU_TOC.svg`, vector PDF, TIFF and AI reference/provenance |

The latest numerical scripts and manuscript-ready assets are included. Older exploratory plotting scripts remain identifiable by their original filenames and are not part of the release figure command. The workflow, molecular-case layout and TOC are editable illustrations; their released SVGs preserve final manual layout and text edits. Their rendering does not require training any model.

The final Figure 2 contains the original DCAMCP logic under a single DCAMCP label. Internal organ keys `mgl`, `mammary`, and `Mammary` in historical arrays refer to the breast task; displayed organ names use Breast.
