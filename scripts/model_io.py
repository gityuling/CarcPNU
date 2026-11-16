"""Checkpoint inference and pinned upstream adapters for CarcPNU."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

UPSTREAM_COMMIT = '633bbd7294e5b67ae537b88c5f75eb8c53abd454'
ORGANS = ['liver', 'lung', 'stomach', 'breast']


def upstream_module(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f'Expected upstream {UPSTREAM_COMMIT}; found {commit}')
    sys.path.insert(0, str(root / 'model/multi_task'))
    import utils
    return utils


def notebook_source(path):
    nb = json.loads(Path(path).read_text())
    return '\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')


def graph_dataset(smiles, upstream):
    from rdkit import Chem, RDLogger
    import networkx as nx
    RDLogger.DisableLog('rdApp.warning')
    author = upstream_module(upstream)
    path = Path(upstream) / 'model/multi_task/multi_task_molecular_feature_vector.ipynb'
    tree = ast.parse(notebook_source(path))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    scope = {'np': np, 'Chem': Chem, 'nx': nx}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), scope)
    # Exact benchmark structures use the author's released graph cache. RDKit
    # version changes can alter a small number of atom encodings. New structures
    # are encoded with the pinned author's function in the recorded environment.
    cached = {}
    folder = Path(upstream) / 'data/multi_task/npy_data'
    for prefix in ['multi_refined', 'liv_lun_sto', 'liv_lun_mgl', 'liv_sto_mgl', 'lun_sto_mgl', 'test']:
        smiles_file = folder / ('smiles.txt' if prefix == 'multi_refined' else prefix + '_smiles.txt')
        values = np.load(folder / (prefix + '_data.npy'), allow_pickle=True)
        cached.update(zip(smiles_file.read_text().splitlines(), values))
    graphs = []
    for i, smi in enumerate(smiles):
        graph = cached[smi] if smi in cached else scope['smile_to_graph'](smi)
        if graph is None:
            raise ValueError(f'Row {i + 1} cannot be encoded by the upstream graph encoder: {smi}')
        graphs.append(graph)
    return author.CustomDataset(graphs, np.full((len(smiles), 4), -1), list(smiles))


def graph_scores(smiles, assets, upstream, device='cuda'):
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    author = upstream_module(upstream)
    dataset = graph_dataset(smiles, upstream)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=Batch.from_data_list)
    results = []
    for seed in range(42, 47):
        model = author.MultiTaskGAT(52, 2, 16, .5).to(device).eval()
        state = torch.load(Path(assets) / f'models/graph/seed_{seed}/baseline.pth', map_location=device, weights_only=True)
        model.load_state_dict(state.get('state_dict', state), strict=True)
        with torch.no_grad():
            results.append(np.concatenate([torch.stack(model(batch.to(device), (0, 1, 2, 3))[1], 1).cpu().numpy() for batch in loader]))
    return np.stack(results)


def fingerprint_features(smiles, assets, device='cuda'):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f'Invalid SMILES at row {i + 1}: {smi}')
        fps.append(generator.GetFingerprintAsNumPy(mol))
    state = torch.load(Path(assets) / 'models/fingerprint/features.pt', map_location=device, weights_only=True)
    x = torch.as_tensor(np.stack(fps), dtype=torch.float64, device=device)
    reference = state['train_fingerprints']
    shared = x @ reference.T
    kernel = shared / (x.sum(1)[:, None] + reference.sum(1)[None, :] - shared).clamp_min(1)
    return kernel @ state['projection']


def fingerprint_scores(smiles, assets, objective='PNU', device='cuda'):
    x = fingerprint_features(smiles, assets, device)
    with np.load(Path(assets) / 'models/fingerprint/coefficients.npz') as coefficients:
        w = torch.as_tensor(coefficients[objective], dtype=torch.float64, device=device)
    design = torch.cat([x, torch.ones_like(x[:, :1])], 1)
    return (design @ w).sigmoid().cpu().numpy()
