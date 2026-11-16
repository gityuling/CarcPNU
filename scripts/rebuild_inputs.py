"""Rebuild source-derived labels, P/N/U roles and fingerprints; verify snapshots."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from model_io import fingerprint_features, upstream_module

TASKS = ['liv', 'lun', 'sto', 'mgl']
TABLES = {'multi_refined': 'tissue_merged_data(add_reliability).csv',
          **{s: s + '_data.csv' for s in ['liv_lun_sto', 'liv_lun_mgl', 'liv_sto_mgl', 'lun_sto_mgl']}}


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', required=True, type=Path)
    p.add_argument('--upstream', type=Path, default=Path('upstream'))
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    upstream_module(a.upstream)
    RDLogger.DisableLog('rdApp.warning')
    torch.set_num_threads(2)
    tables = {}
    for name, filename in TABLES.items():
        rows = list(csv.DictReader((a.upstream / 'data/multi_task' / filename).open()))
        for r in rows:
            r['labels'] = [int(float(r[t])) if t in r else -1 for t in TASKS]
            r['reliability'] = [r.get(t + '_reliability', 'unknown') if t in r else 'missing' for t in TASKS]
            mol = rdMolStandardize.ChargeParent(Chem.MolFromSmiles(r['SMILES']))
            r['parent'] = Chem.MolToSmiles(mol, isomericSmiles=False)
        tables[name] = rows
    full = tables['multi_refined']
    tr, va = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(np.zeros(len(full)), np.array([r['labels'] for r in full])))
    heldout = {canonical(full[i]['SMILES']) for i in va}
    selected = [('multi_refined', int(i), full[i]) for i in tr]
    for name in list(TABLES)[1:]:
        rows = tables[name]
        tasks = [TASKS.index(t) for t in name.split('_')]
        ti, _ = next(MultilabelStratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=3).split(np.zeros(len(rows)), np.array([r['labels'] for r in rows])[:, tasks]))
        selected.extend((name, int(i), rows[i]) for i in ti if canonical(rows[i]['SMILES']) not in heldout)
    groups = defaultdict(list)
    for name, index, row in selected:
        groups[row['parent']].append((name, index, row))
    smiles, labels, roles, lineage = [], [], [], []
    for parent, records in sorted(groups.items()):
        smiles.append(records[0][2]['SMILES'])
        yy, rr = [], []
        for t in range(4):
            known = [r for _, _, r in records if r['labels'][t] >= 0]
            values = {r['labels'][t] for r in known}
            evidence = {r['reliability'][t] for r in known}
            yy.append(next(iter(values)) if len(values) == 1 else -1)
            rr.append(-1 if len(values) > 1 else 1 if values == {1} else (0 if 'explicit' in evidence else 2 if 'implicit' in evidence else 3) if values == {0} else 4)
        labels.append(yy); roles.append(rr)
        lineage.append({'parent': parent, 'SMILES': smiles[-1], 'source_rows_zero_based': [{'table': n, 'row': i} for n, i, _ in records], 'labels': yy, 'roles': rr})
    ref = np.load(a.assets / 'inputs/training_inputs.npz')
    assert smiles == ref['smiles'].tolist()
    assert np.array_equal(labels, ref['labels']) and np.array_equal(roles, ref['roles'])
    val_smiles = [full[i]['SMILES'] for i in va]
    val_labels = np.array([full[i]['labels'] for i in va])
    assert np.array_equal(val_labels, ref['validation_labels'])
    x = fingerprint_features(smiles, a.assets).cpu().numpy()
    xv = fingerprint_features(val_smiles, a.assets).cpu().numpy()
    error = max(float(abs(x-ref['train_features']).max()), float(abs(xv-ref['validation_features']).max()))
    assert error < 1e-7, error
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.output / 'training_inputs.npz', **{k: ref[k] for k in ref.files if k not in ['train_features', 'validation_features', 'roles', 'labels']}, train_features=x, validation_features=xv, labels=labels, roles=roles)
    (a.output / 'training_lineage.json').write_text(json.dumps(lineage, indent=2) + '\n')
    report = {'status':'passed', 'training_raw_rows':len(selected), 'training_parents':len(smiles), 'validation_rows':len(va), 'roles_and_labels_match_archive':True, 'feature_max_absolute_error':error, 'role_codes':{'1':'P','0':'N','2':'implicit U','3':'undocumented U','-1':'conflicting excluded','4':'missing excluded'}}
    (a.output / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
