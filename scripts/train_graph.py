"""Optional local graph retraining with pinned upstream training functions.

The main fingerprint-objective comparison reuses the archived graph checkpoints.
This entry point reconstructs the local graph schedule without perturbation data.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Batch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from model_io import notebook_source, upstream_module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=Path('upstream'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, choices=range(42,47), default=42)
    p.add_argument('--check-setup', action='store_true')
    a = p.parse_args()
    author = upstream_module(a.upstream)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('Graph training requires CUDA in the recorded environment')
    torch.cuda.set_per_process_memory_fraction(.15, 0)
    a.output.mkdir(parents=True, exist_ok=False)
    work = a.output.resolve() / 'work'
    shutil.copytree(a.upstream / 'data/multi_task', work / 'data/multi_task')
    target = work / 'model/multi_task'; target.mkdir(parents=True)
    (target / 'param').mkdir()
    script = target / 'training.py'
    source = notebook_source(a.upstream / 'model/multi_task/training_and_gradient_similarity.ipynb')
    source = source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "1"', 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"')
    script.write_text(source)
    scope = {'__name__': 'carcpnu_graph_training', '__file__': str(script)}
    exec(compile(source, str(script), 'exec'), scope)
    orig_seed = scope['seed_all']
    scope['seed_all'] = lambda unused=42: orig_seed(a.seed)
    npydir = a.upstream / 'data/multi_task/npy_data'
    full = author.CustomDataset(np.load(npydir/'multi_refined_data.npy', allow_pickle=True), np.load(npydir/'multi_refined_labels.npy').T, (npydir/'smiles.txt').read_text().splitlines())
    tr, va = next(MultilabelStratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(np.zeros(len(full)),full.data_y))
    canonical = lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s),isomericSmiles=True)
    heldout = {canonical(full.smiles_list[i]) for i in va}
    original_loader = scope['load_combo_dl']
    removed = []

    def filtered_loaders(*args, **kwargs):
        train, val = original_loader(*args, **kwargs)
        for loaders, shuffle in [(train, True),(val,False)]:
            for combo, loader in list(loaders.items()):
                ds, indices = loader.dataset.dataset, loader.dataset.indices
                keep = [i for i in indices if canonical(ds.smiles_list[i]) not in heldout]
                removed.append({'combo':list(combo),'partition':'train' if shuffle else 'validation','removed':len(indices)-len(keep)})
                loaders[combo] = DataLoader(Subset(ds,keep),batch_size=loader.batch_size,shuffle=shuffle,collate_fn=Batch.from_data_list)
        return train,val

    scope['load_combo_dl'] = filtered_loaders
    scope['load_data'] = lambda **unused: full
    report = {'training_seed':a.seed,'split_seed':42,'stage_two_train':tr.tolist(),'stage_two_validation':va.tolist(),'initialization':'random','upstream_commit':'633bbd7294e5b67ae537b88c5f75eb8c53abd454'}
    if a.check_setup:
        from itertools import combinations
        scope['seed_all'](42)
        train,val=filtered_loaders(base_dir=str(work/'data/multi_task'),task_combos=list(combinations(['liv','lun','sto','mgl'],3)),bs=2,vr=.2,seed=3)
        report['stage_one_removed']=removed
        model=author.MultiTaskGAT(52,2,16,.5).cuda().eval()
        with torch.no_grad():
            assert len(model(Batch.from_data_list([full[tr[0]],full[tr[1]]]).cuda(),(0,1,2,3))[1])==4
        report.update(status='setup_passed',optimizer_steps=0)
    else:
        # The notebook loads checkpoints it creates locally during this run.
        original_load=torch.load
        def trusted_training_load(*args,**kwargs):
            kwargs.setdefault('weights_only',False)
            return original_load(*args,**kwargs)
        torch.load=trusted_training_load
        try:
            scope['main']()
        finally:
            torch.load=original_load
        shutil.copy2(target/'param/final_full_retrained.pth',a.output/'baseline.pth')
        report.update(status='completed',stage_one_removed=removed)
    (a.output/'training_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['stage_two_train','stage_two_validation']},indent=2))


if __name__=='__main__':
    main()
