"""Predict four organ scores from a CSV with a SMILES column."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model_io import ORGANS, fingerprint_scores, graph_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--upstream', default='upstream', type=Path)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--objective', choices=['supervised', 'PN_only', 'PU_only', 'PNU'], default='PNU')
    p.add_argument('--fingerprint-only', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frame = pd.read_csv(a.input)
    smiles = frame['SMILES'].tolist()
    if not smiles:
        raise ValueError('Input contains no molecules')
    fp = fingerprint_scores(smiles, a.assets, a.objective, a.device)
    scores = fp if a.fingerprint_only else .25 * graph_scores(smiles, a.assets, a.upstream, a.device).mean(0) + .75 * fp
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite prediction')
    for i, organ in enumerate(ORGANS):
        frame[organ + '_score'] = scores[:, i]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.output, index=False)
    print(f'Saved {len(frame)} molecules to {a.output}; output is a continuous score, with no classification threshold applied.')


if __name__ == '__main__':
    main()
