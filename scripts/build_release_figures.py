"""Regenerate quantitative panels using the archived figure data workspace."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    a = p.parse_args()
    target = a.assets.resolve() / 'figure_workspace/CarcPNU_JCIM_20261112/figures_v2_20261112'
    if not (target / 'source_data').is_dir():
        raise FileNotFoundError('Extract the complete Zenodo archive first')
    sources = Path(__file__).resolve().parent
    for name in ['build_merged_ablation.py', 'build_provenance_sensitivity.py', 'build_radar_figures.py', 'build_benchmark_curves.py', 'build_figS1_portrait.py', 'build_figS3_portrait.py']:
        shutil.copy2(sources / name, target / name)
        subprocess.run([sys.executable, str(target / name)], check=True)
    print(f'Quantitative panels saved under {target}. The final workflow and molecular-case SVGs are included separately as editable figure sources.')
