"""Verify archive checksums and checkpoint predictions against saved results."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from model_io import fingerprint_scores, graph_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--upstream', type=Path, default=Path('upstream'))
    p.add_argument('--full-graph', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    count = 0
    checksum = a.assets / 'SHA256SUMS.txt'
    if checksum.exists():
        for line in checksum.read_text().splitlines():
            digest, name = line.split('  ', 1)
            path = a.assets / name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
            count += 1
    d = np.load(a.assets / 'inputs/test_inputs.npz')
    historical = np.load(a.assets / 'inputs/historical_predictions.npz')
    graph = d['graph']
    graph_error = None
    if a.full_graph:
        regenerated = graph_scores(d['smiles'], a.assets, a.upstream)
        graph_error = float(np.abs(regenerated - graph).max())
        assert graph_error < 1e-6, graph_error
    result = {'status': 'passed', 'checked_files': count, 'full_graph_checked': a.full_graph, 'graph_max_absolute_error': graph_error, 'methods': {}}
    for method, key in [('supervised', 'fusion_supervised'), ('PN_only', 'fusion_pn'), ('PU_only', 'fusion_pu'), ('PNU', 'fusion_pnu')]:
        fp = fingerprint_scores(d['smiles'], a.assets, method)
        pred = .25 * graph + .75 * fp[None]
        error = float(np.abs(pred - historical[key]).max())
        assert error < 1e-7, (method, error)
        metrics = []
        for run in pred:
            for i in range(4):
                keep = d['labels'][:, i] >= 0
                y = d['labels'][keep, i]
                metrics.append([roc_auc_score(y, run[keep, i]), average_precision_score(y, run[keep, i])])
        mean = np.mean(metrics, 0)
        result['methods'][method] = {'prediction_max_absolute_error': error, 'macro_AUROC': float(mean[0]), 'macro_AP': float(mean[1])}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
