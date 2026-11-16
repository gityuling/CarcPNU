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
# Figure 3: all organ curves; each seed shown separately, no ensemble/mean-metric confusion.
f=figure('3 | Organ-specific discrimination and precision-recall','Each saved graph-seed prediction is plotted; legends report mean per-seed AUROC or AP.',(13,8.8))
gs=f.add_gridspec(2,4,left=.065,right=.98,bottom=.10,top=.86,hspace=.47,wspace=.34)
curve_data={}
for t,org in enumerate(ORG):
 yy=y[valid[:,t],t];n=len(yy);pos=int(yy.sum())
 for j in range(2):
  ax=f.add_subplot(gs[j,t]);panel(ax,'ABCDEFGH'[j*4+t],('Breast' if org == 'Mammary' else org)+f' (n={n}; +{pos}/-{n-pos})')
  for label,pred,col in [('Released OS',h['published'],C['graph']),('Supervised',z['fusion_supervised'],C['sup']),('PNU',z['fusion_PNU'],C['pnu'])]:
   vals=[]
   for s,pred_seed in enumerate(pred):
    pp=pred_seed[valid[:,t],t]
    if j==0:xx,zz,tt=roc_curve(yy,pp);v=roc_auc_score(yy,pp)
    else:zz,xx,tt=precision_recall_curve(yy,pp);v=average_precision_score(yy,pp)
    vals.append(v);ax.plot(xx,zz,color=col,lw=1.2,alpha=.8 if len(pred)==1 else .38,drawstyle='steps-post' if j else 'default')
    curve_data[f'{org}_{label}_{j}_{s}']={'x':xx.tolist(),'y':zz.tolist(),'metric':v}
   ax.plot([],[],color=col,label=f'{label}: {np.mean(vals):.3f}')
  if j==0:ax.plot([0,1],[0,1],ls=':',color='#a6b2bb');ax.set_xlabel('False-positive rate');ax.set_ylabel('True-positive rate')
  else:ax.axhline(pos/n,ls=':',color='#a6b2bb');ax.set_xlabel('Recall');ax.set_ylabel('Precision')
  ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.03);ax.legend(fontsize=7,loc='lower right' if j==0 else 'lower left',framealpha=.9);grid(ax,'both')
f.text(.065,.03,'Dotted PR lines show organ-specific positive fractions. AP is noninterpolated average precision, not trapezoidal PR area.',fontsize=8)
save(f,'fig3_organ_curves','ROC curves (A-D) and precision-recall curves (E-H) on each organ\'s observed labels. For fusion, five individual seed curves are displayed and labels give the mean per-seed metric, matching the benchmark convention. The fingerprint fit is shared by the five graph seeds. Released OS uses one prediction array. PR horizontal lines indicate positive fractions; high AP must be interpreted against class balance. No uncertainty band is inferred from the seed curves.')
(DATA/'curve_coordinates.json').write_text(json.dumps(curve_data),encoding='utf-8')

