#!/usr/bin/env python3
"""Rebuild themed CarcPNU figures from archived predictions. No fitting/tuning.
Run with OS-Carcinogenicity-baseline/.venv-gpu/bin/python; BLAS threads <= 2.
"""
from pathlib import Path
import json, hashlib, textwrap, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, accuracy_score, f1_score, precision_score,
    recall_score, balanced_accuracy_score, matthews_corrcoef)
from sklearn.decomposition import PCA
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Draw
from rdkit.Chem.Draw import rdMolDraw2D

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent.parent
A=ROOT/'OS-Carcinogenicity-G2CP/phase6_pnu_ablation_20261104'
D=ROOT/'OS-Carcinogenicity-G2CP/phase5_diagnostics_20261104'
B=ROOT/'OS-Carcinogenicity-baseline/full_comparison_20261101/evaluation'
DATA=OUT/'source_data';DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
    'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#9aa9b4','text.color':'#203347','axes.labelcolor':'#203347',
    'savefig.facecolor':'white','figure.facecolor':'white'})
C={'sup':'#347daf','pnu':'#008b7f','pn':'#b88c36','pu':'#9a77ad','graph':'#778694',
   'bad':'#c45b58','train':'#b4bfca','good':'#008b7f','diag':'#bc865b'}
ORG=['Liver','Lung','Stomach','Mammary']; short=['liv','lun','sto','mgl']
MET=['AUROC','AP','Accuracy','F1','Precision','Sensitivity','Specificity','Balanced accuracy','MCC']
TH=.45
paths=[A/'predictions/test.npz',A/'inputs/training_inputs.npz',A/'inputs/historical_predictions.npz',
       A/'tables/paired_differences.csv',A/'tables/macro_metrics.csv',B/'predictions.npz',
       B/'macro_metrics.csv',D/'predictions/real.npz',D/'tables/eta_sensitivity.csv',
       D/'tables/role_permutation.csv',ROOT/'CarcPNU_U与先验复核_20261112/recalculated_results.json']
z=np.load(paths[0],allow_pickle=False); tr=np.load(paths[1],allow_pickle=False)
h=np.load(paths[2],allow_pickle=False); base=np.load(B/'predictions.npz',allow_pickle=False)
diag=np.load(D/'predictions/real.npz',allow_pickle=False)
y=z['labels']; sm=z['smiles']; train_sm=tr['smiles']; roles=tr['roles']
assert np.array_equal(y,h['labels']) and np.array_equal(sm,h['smiles'])
assert np.array_equal(y,base['labels'])
assert np.allclose(z['fusion_supervised'],base['fusion_supervised'])
assert np.allclose(z['fusion_PNU'],base['fusion_pnu'])
valid=y>=0
sup=z['fusion_supervised'].mean(0);pnu=z['fusion_PNU'].mean(0)
gain=np.where(valid,(2*y-1)*(pnu-sup),np.nan)
cs=(sup>=TH)==y; cp=(pnu>=TH)==y
changed=np.where(~cs&cp,1,np.where(cs&~cp,-1,0));changed[~valid]=0
NAMES=[('OS-Carcinogenicity','multi_task_published'),('Single-task GAT','single_task'),
 ('MolBERT','MolBERT'),('CarcGC','CarcGC'),('DCAMCP (corrected)','DCAMCP_corrected'),
 ('DCAMCP (original logic)','DCAMCP'),('XGBoost','XGBoost'),('Random forest','RF'),
 ('Structure + supervised','fusion_supervised'),('CarcPNU','fusion_pnu')]
def scores(yy,pp,threshold=TH):
    hard=pp>=threshold
    return np.array([roc_auc_score(yy,pp),average_precision_score(yy,pp),accuracy_score(yy,hard),
      f1_score(yy,hard,zero_division=0),precision_score(yy,hard,zero_division=0),
      recall_score(yy,hard),recall_score(yy,hard,pos_label=0),balanced_accuracy_score(yy,hard),matthews_corrcoef(yy,hard)])
def organ_scores(pred,threshold=TH):
    return np.mean([[scores(y[valid[:,t],t],p[valid[:,t],t],threshold) for t in range(4)] for p in pred],axis=0)
def figure(title,subtitle,size=(12,9.5)):
 f=plt.figure(figsize=size)
 f.suptitle(title,x=.07,y=.975,ha='left',fontsize=17,fontweight='bold')
 f.text(.07,.91 if size[1]<9 else .936,subtitle,fontsize=9,color='#5d7182',ha='left')
 return f
def panel(ax,letter,title):
 ax.set_title(letter+'  '+title,loc='left',pad=12,fontweight='bold')
 ax.set_axisbelow(True)
def grid(ax,axis='x'):ax.grid(axis=axis,color='#e7edf2',lw=.7)
def save(f,name,caption):

 for ext in ['pdf','svg','png']:f.savefig(OUT/(name+'.'+ext),dpi=220)

 # Clean versions for manuscript placement: captions supply titles and notes.
 for obj in list(f.texts):obj.remove()
 f.savefig(OUT/(name+'_manuscript.pdf'),bbox_inches='tight',pad_inches=.03)
 plt.close(f)
 print('Saved',name,flush=True)
# Figure 6: parameter and provenance robustness with all relevant simple controls.
prior=json.load(open(paths[-1]))['prior_sensitivity'];eta=pd.read_csv(D/'tables/eta_sensitivity.csv').sort_values('eta')
f=figure('Sensitivity to PNU parameters','AUROC and AP across tested settings.',(12.5,5.6))
gs=f.add_gridspec(1,2,left=.10,right=.96,bottom=.18,top=.78,wspace=.48)
ax=f.add_subplot(gs[0,0]);pi=[r['prior'] for r in prior]
for t,org in enumerate(ORG):ax.plot(pi,[r['original']['per_organ'][t][0] for r in prior],'.-',lw=1,label=('Breast' if org=='Mammary' else org))
ax.plot(pi,[r['original']['macro'][0] for r in prior],'o-',color='#24394a',lw=2,label='Macro');ax.axvline(.3,ls=':',color=C['graph']);ax.set_xlabel('Assumed positive prior π');ax.set_ylabel('AUROC');ax.legend(ncol=2,fontsize=7);grid(ax,'y');panel(ax,'A','Organ responses to the assumed prior')
ax=f.add_subplot(gs[0,1]);ax.plot(eta.eta,eta.test_AUROC,'o-',color=C['pnu'],label='Macro AUROC');ax.plot(eta.eta,eta.test_AP,'s-',color=C['sup'],label='Macro AP');ax.axvline(.25,ls=':',color=C['graph']);ax.set_xlabel('PN/PU mixing coefficient η');ax.set_ylabel('Metric');ax.legend(fontsize=8);grid(ax,'y');panel(ax,'B','PN-only → PNU → PU-only')
save(f,'fig6_robustness','A: tested prior values. B: fixed-prior mixing sensitivity.')
