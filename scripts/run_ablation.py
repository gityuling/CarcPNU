"""Four matched, fixed-parameter PNU ablations on the original OS-Carc split.

Standalone replay: inputs contain features and graph scores, no model weights.
Run prepare -> train -> evaluate. Evaluation never selects configurations.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from fitting_core import masses, fit_convex, numerical_retry, predict, tensor
from metric_core import weighted_metrics

RUN=Path(__file__).resolve().parents[1]
TASKS=['liver','lung','stomach','mammary']
METHODS={
    'supervised':{'method':'supervised','penalty':.001},
    'PN_only':{'method':'pn_prior','penalty':.001,'prior':.3,'eta':0.},
    'PU_only':{'method':'pu','penalty':.001,'prior':.3,'eta':1.},
    'PNU':{'method':'pnu','penalty':.001,'prior':.3,'eta':.25}}
METRICS=['AUROC','AP','accuracy','F1','precision','sensitivity','specificity','balanced_accuracy','MCC']


def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def hashes(paths):return {str(p.relative_to(RUN)):sha(p) for p in paths}
def verify(items):
    for path,digest in items.items():assert sha(RUN/path)==digest,path
def csvout(name,rows):
    with (RUN/'tables'/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare():
    assert not (RUN/'protocol.json').exists()
    save(RUN/'protocol.json',{
        'created_utc':now(),'request':'Run ordinary supervision, PN-only, PU-only and full PNU as a matched ablation.',
        'split':'Original 435-parent auxiliary training pool; original 57 validation rows and 454 test rows. No new split.',
        'fixed_configurations':METHODS,'selection':'None. All four configurations fixed from the prior PNU setting before this rerun.',
        'roles':{'P':'consistent observed positive','N':'consistent zero with source-explicit negative annotation',
            'U':'implicit zero or zero with undocumented evidence category','excluded':'conflicting or genuinely missing organ labels'},
        'loss_scope':'These roles control the fingerprint loss only. All models use the same train-only ECFP4/Tanimoto feature map; the map itself includes all 435 training structures. This is a matched loss ablation, not a data-removal study.',
        'graph_scope':'Frozen historical graphs used all original labels in training. Thus PN-only/PU-only describe the fingerprint branch, not an end-to-end PN/PU graph model.',
        'features':'Fixed ECFP4 radius2 2048bit Tanimoto eigenprojection from training structures only; cached features contain no held-out label fitting.',
        'fusion':{'graph':.25,'fingerprint':.75,'graph_seeds':[42,43,44,45,46],'replicate_definition':'One deterministic fingerprint fit shared by five historical graph seeds; no independent fingerprint-training replication.'},
        'endpoints':{'primary':'Four-organ macro AUROC on original454, averaged over graph seeds after computing per-seed metrics.',
            'secondary':'AP, per-organ scores, standalone fingerprint branch; archived432-row canonical de-overlap sensitivity.',
            'threshold_metrics':'Descriptive fixed threshold .45, unchanged from earlier protocol; no calibration or threshold optimization.'},
        'bootstrap':{'replicates':10000,'seed':20261104,'unit':'Whole molecule rows with all four labels; paired across methods; fixed models and graph seeds.',
            'comparisons':['PNU - supervised','PNU - PN_only','PNU - PU_only'],'interpretation':'Pointwise conditional percentile 95% intervals; no historical selection or multiplicity adjustment.'},
        'limitations':'The original test set was repeatedly observed before this rerun. This is exploratory reproducibility/ablation evidence, not new confirmatory evidence. Annotation provenance is not a gold-standard toxicological adjudication.',
        'claims':'A PNU win in this four-way ablation does not establish superiority over simple downweighting, retained in the previous experiments.',
        'inputs_sha256':hashes(list((RUN/'inputs').glob('*'))),
        'sources_sha256':hashes([Path(__file__),RUN/'scripts/fitting_core.py',RUN/'scripts/metric_core.py'])})
    print('Protocol frozen.',flush=True)


def load():
    p=read(RUN/'protocol.json');verify(p['inputs_sha256']);verify(p['sources_sha256']);return p


def train():
    p=load();assert not (RUN/'training_complete.json').exists()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);start=time.monotonic()
    d=np.load(RUN/'inputs/training_inputs.npz');roles=d['roles'];labels=d['labels']
    assert roles.shape==(435,4) and d['validation_labels'].shape==(57,4)
    x,xv=tensor(d['train_features']),tensor(d['validation_features'])
    records={};coefout={};vp={};lineage=[]
    for t,organ in enumerate(TASKS):
        lineage.append({'organ':organ,'P':int((roles[:,t]==1).sum()),'N':int((roles[:,t]==0).sum()),
            'U_implicit':int((roles[:,t]==2).sum()),'U_undocumented':int((roles[:,t]==3).sum()),
            'conflict_excluded':int((roles[:,t]==-1).sum()),'missing_excluded':int((roles[:,t]==4).sum())})
    for name,c in METHODS.items():
        a,b=masses(roles,c)
        pu=((roles==2)|(roles==3));nn=roles==0;pp=roles==1
        if name=='PN_only':assert torch.all(a[tensor(pu).bool()]==0) and torch.all(b[tensor(pu).bool()]==0)
        if name=='PU_only':assert torch.all(a[tensor(nn).bool()]==0) and torch.all(b[tensor(nn).bool()]==0)
        if name=='supervised':
            target=tensor(np.where(labels>=0,labels,0));assert torch.allclose(b,a*target,atol=1e-14,rtol=0)
        # Independently expand the stated PN/PU mixture on random scores.
        g=tensor(np.random.default_rng(711).normal(size=roles.shape));lp=torch.nn.functional.softplus(-g);ln=torch.nn.functional.softplus(g)
        if name!='supervised':
            direct=torch.stack([(1-c['eta'])*(c['prior']*lp[pp[:,t],t].mean()+(1-c['prior'])*ln[nn[:,t],t].mean())+
                c['eta']*(c['prior']*lp[pp[:,t],t].mean()+ln[pu[:,t],t].mean()-c['prior']*ln[pp[:,t],t].mean()) for t in range(4)])
            assert float((direct-(a*ln-b*g).sum(0)).abs().max())<1e-12
        retry=None
        try:coef,diag=fit_convex(x,a,b,c['penalty'])
        except RuntimeError as exc:retry=str(exc);coef,diag=numerical_retry(x,a,b,c['penalty'])
        assert max(v['gradient_max'] for v in diag)<1e-9
        coefout[name]=coef.cpu().numpy();vp[name]=predict(xv,coef)
        records[name]={'configuration':c,'diagnostics':diag,'numerical_retry':retry}
        print(json.dumps({'trained':name,'max_gradient':max(v['gradient_max'] for v in diag)}),flush=True)
    csvout('training_role_counts.csv',lineage)
    np.savez_compressed(RUN/'models/coefficients.npz',**coefout)
    np.savez_compressed(RUN/'predictions/validation.npz',labels=d['validation_labels'],**vp)
    save(RUN/'training_complete.json',{'created_utc':now(),'protocol_sha256':sha(RUN/'protocol.json'),'records':records,
        'artifacts_sha256':hashes([RUN/'models/coefficients.npz',RUN/'predictions/validation.npz']),
        'seconds':time.monotonic()-start,'device':torch.cuda.get_device_name(0),'gpu_peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
        'new_fingerprint_fits':4,'organ_fits':16,'new_graph_training':False,'new_test_scores_loaded_during_training':False,
        'torch':torch.__version__,'numpy':np.__version__,'python':sys.version})
    load();print('Four fixed models fitted and frozen.',flush=True)


def summary(y,pp):
    if pp.ndim==2:pp=pp[None]
    per_seed=np.stack([np.stack([weighted_metrics(y[:,t],pred[:,t],np.ones((1,len(y))))[0] for t in range(4)]) for pred in pp])
    assert np.isfinite(per_seed).all()
    return {'per_organ':per_seed.mean(0).tolist(),'macro':per_seed.mean((0,1)).tolist(),
        'per_seed_macro':per_seed.mean(1).tolist(),'graph_seed_sd':per_seed.mean(1).std(0,ddof=1).tolist() if len(pp)>1 else None}


def evaluate():
    p=load();train=read(RUN/'training_complete.json');verify(train['artifacts_sha256']);assert train['protocol_sha256']==sha(RUN/'protocol.json')
    assert not (RUN/'results.json').exists();torch.set_num_threads(2)
    save(RUN/'records/evaluation_opened.json',{'time':now(),'training_sha256':sha(RUN/'training_complete.json')})
    d=np.load(RUN/'inputs/test_inputs.npz');y=d['labels'];x=tensor(d['features']);graph=d['graph']
    assert y.shape==(454,4) and graph.shape==(5,454,4)
    coefficients=np.load(RUN/'models/coefficients.npz');old=np.load(RUN/'inputs/historical_predictions.npz')
    assert np.array_equal(old['labels'],y) and np.array_equal(old['smiles'],d['smiles'])
    pred={};errors={};oldkeys={'supervised':'fusion_supervised','PN_only':'fusion_pn','PU_only':'fusion_pu','PNU':'fusion_pnu'}
    for name in METHODS:
        pp=predict(x,tensor(coefficients[name]));pred['fingerprint_'+name]=pp[None];pred['fusion_'+name]=.25*graph+.75*pp[None]
        error=float(abs(pred['fusion_'+name]-old[oldkeys[name]]).max());assert error<1e-7;errors[name]=error
    np.savez_compressed(RUN/'predictions/test.npz',labels=y,smiles=d['smiles'],**pred)
    mask=np.load(RUN/'inputs/overlap_mask.npz')['nonoverlap'];assert mask.sum()==432
    report={'created_utc':now(),'metrics':METRICS,'historical_prediction_max_errors':errors,'cohorts':{},'paired':{}}
    rows=[];organ_rows=[];paired_rows=[];max_independent_error=0.
    for cohort,keep in [('original',np.ones(len(y),bool)),('no_overlap',mask)]:
        yy=y[keep];scores={name:summary(yy,pp[:,keep]) for name,pp in pred.items()}
        report['cohorts'][cohort]={'rows':int(keep.sum()),'counts':[[int((yy[:,t]==1).sum()),int((yy[:,t]==0).sum())] for t in range(4)],'methods':scores}
        for name,score in scores.items():
            rows.append({'cohort':cohort,'method':name,**dict(zip(METRICS,score['macro']))})
            for t,organ in enumerate(TASKS):
                organ_rows.append({'cohort':cohort,'method':name,'organ':organ,'positive':int((yy[:,t]==1).sum()),'negative':int((yy[:,t]==0).sum()),**dict(zip(METRICS,score['per_organ'][t]))})
                valid=yy[:,t]>=0
                independently=np.mean([[roc_auc_score(yy[valid,t],pp[valid,t]),average_precision_score(yy[valid,t],pp[valid,t])] for pp in pred[name][:,keep]],axis=0)
                max_independent_error=max(max_independent_error,float(abs(independently-np.array(score['per_organ'][t][:2])).max()))
        # Bootstrap in batches to bound memory. Paired whole-row resampling.
        rng=np.random.default_rng(p['bootstrap']['seed']);dist={name:[] for name in pred}
        for start in range(0,p['bootstrap']['replicates'],500):
            counts=rng.multinomial(len(yy),np.ones(len(yy))/len(yy),size=min(500,p['bootstrap']['replicates']-start))
            for name,pp in pred.items():
                values=np.mean([np.stack([weighted_metrics(yy[:,t],s[:,t],counts)[:,:2] for t in range(4)],axis=1) for s in pp[:,keep]],axis=0)
                dist[name].append(values)
        dist={k:np.concatenate(v) for k,v in dist.items()}
        np.savez_compressed(RUN/'predictions'/f'bootstrap_{cohort}.npz',**dist)
        pairs=[]
        for branch in ['fusion','fingerprint']:
            for base in ['supervised','PN_only','PU_only']:
                a,b=branch+'_PNU',branch+'_'+base;delta=dist[a]-dist[b]
                for t,organ in [(-1,'macro')]+list(enumerate(TASKS)):
                    v=delta.mean(1) if t==-1 else delta[:,t]
                    point=np.array(scores[a]['macro' if t==-1 else 'per_organ'])-np.array(scores[b]['macro' if t==-1 else 'per_organ'])
                    if t!=-1:point=point[t]
                    for j,metric in enumerate(METRICS[:2]):
                        valid=np.isfinite(v[:,j]);lo,hi=np.quantile(v[valid,j],[.025,.975])
                        item={'cohort':cohort,'branch':branch,'method_A':'PNU','method_B':base,'organ':organ,'metric':metric,'delta':float(point[j]),'CI95_low':float(lo),'CI95_high':float(hi),'valid_replicates':int(valid.sum())}
                        pairs.append(item);paired_rows.append(item)
        report['paired'][cohort]=pairs
        print(json.dumps({'evaluated':cohort,'bootstrap_replicates':p['bootstrap']['replicates']}),flush=True)
    assert max_independent_error<1e-12
    csvout('macro_metrics.csv',rows);csvout('per_organ_metrics.csv',organ_rows);csvout('paired_differences.csv',paired_rows)
    save(RUN/'results.json',report)
    checks={'status':'passed','max_independent_AUROC_AP_error':max_independent_error,'historical_prediction_max_errors':errors,
        'all_final_gradients_below_1e9_inverse':True,'PN_excludes_U_from_loss':True,'PU_excludes_N_from_loss':True,
        'PN_PU_PNU_risks_independently_expanded':True,'configuration_selection_on_test':False,'prior_test_exposure':True}
    save(RUN/'records/verification.json',checks);load();verify(train['artifacts_sha256'])
    lines=['FOUR-WAY PNU ABLATION: COLLABORATOR REPORT',f'Generated UTC: {now()}','',
        'Ordinary supervision / PN-only / PU-only / full PNU were independently refitted on CUDA. No configuration search was performed.',
        'All use the original 435-parent training pool, 57 validation rows, 454 test rows and the same ECFP4/Tanimoto structure representation.',
        'PN, PU and PNU share assumed positive prior pi=0.30 and ridge lambda=0.001. eta=0, 1 and 0.25 respectively. Ordinary supervision uses lambda=0.001 and the original observed class frequencies.',
        'U contains implicit zeros and zeros with undocumented evidence category. Conflicting and genuinely missing labels are excluded. Source-explicit N is an annotation category, not independent expert adjudication.',
        '', 'MAIN COMPARISON: fixed 25% graph + 75% fingerprint fusion','Method | macro AUROC | macro AP']
    for name in METHODS:
        m=report['cohorts']['original']['methods']['fusion_'+name]['macro'];lines.append(f'{name} | {m[0]:.6f} | {m[1]:.6f}')
    lines+=['','FINGERPRINT BRANCH ALONE','Method | macro AUROC | macro AP']
    for name in METHODS:
        m=report['cohorts']['original']['methods']['fingerprint_'+name]['macro'];lines.append(f'{name} | {m[0]:.6f} | {m[1]:.6f}')
    lines+=['','PAIRED ORIGINAL-TEST MACRO COMPARISONS: 10,000 molecule bootstrap draws']
    for r in report['paired']['original']:
        if r['organ']=='macro':lines.append(f"{r['branch']}: PNU minus {r['method_B']}, {r['metric']} delta={r['delta']:+.6f}; 95% CI [{r['CI95_low']:+.6f}, {r['CI95_high']:+.6f}]")
    lines+=['','INTERPRETATION AND LIMITS',
        'The four-way table tests the contribution of the PN/PU risk components under matched prior and regularization. A higher point estimate alone is not proof of a statistically reliable improvement.',
        'PN-only/PU-only apply to the fingerprint loss. The frozen graph branch was previously trained on original zero labels. The shared feature map also uses all training structures without labels. Do not describe the entire fused model as never seeing U/N information.',
        'The fingerprint branch is fitted once deterministically. Five frozen graph seeds affect the fusion; they are not five independent datasets or fingerprint fits.',
        'The original test set has already been repeatedly observed. This rerun reproduces old results and does not provide a fresh confirmation. Bootstrap intervals are conditional and pointwise, without historical-selection or multiplicity correction.',
        'The 432-row de-overlap sensitivity is retained in the tables. It is not a new independent dataset.',
        'Threshold-based metrics use the historical fixed .45 threshold. Higher AUROC/AP does not require higher F1 or sensitivity at this threshold.',
        'The existing simple-downweighting controls remain in previous records. This four-method ablation addresses component contributions, not superiority over all competing methods.',
        '', 'FILES AND REPRODUCTION',
        'tables/macro_metrics.csv: all macro metrics, both branches, original and de-overlap cohorts.',
        'tables/per_organ_metrics.csv: per-organ metrics and class counts. tables/paired_differences.csv: AUROC/AP contrasts and intervals.',
        'protocol.json: frozen configuration, scope and hashes. training_complete.json and records/verification.json: fitting and checks.',
        'inputs/: feature/label/graph-score snapshots, no trained weights. scripts/: standalone mathematical and metric implementation.',
        'predictions/: validation/test scores and bootstrap distributions. models/: fitted local coefficients; omitted from collaborator archive.',
        'For replay in a copied directory, keep inputs/ and scripts/. Use a new output tree without protocol.json, training_complete.json, results.json or fitted outputs; create models/, predictions/, tables/, records/, logs/ and figures/.',
        'With CUDA torch, numpy and scikit-learn installed: python scripts/run_ablation.py prepare; python scripts/run_ablation.py train; python scripts/run_ablation.py evaluate.',
        'The script refuses to overwrite completed protocol/training/evaluation files. No API key or external request is required.']
    (RUN/'REPORT_EN.txt').write_text('\n'.join(lines)+'\n')
    save(RUN/'status.json',{'stage':'complete','created_utc':now(),'new_fits':4,'organ_fits':16,'all_previous_results_preserved':True})
    print(json.dumps({'status':'complete','main':{n:report['cohorts']['original']['methods']['fusion_'+n]['macro'][:2] for n in METHODS}}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','train','evaluate']);args=ap.parse_args()
    {'prepare':prepare,'train':train,'evaluate':evaluate}[args.mode]()
