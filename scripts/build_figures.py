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
all_scores={key:organ_scores(base[key]) for _,key in NAMES}
rows=[]
for threshold in [TH,.5]:
 for name,key in NAMES:
  os=organ_scores(base[key],threshold)
  for t,organ in enumerate(ORG+['Macro']):
   row={'model':name,'organ':organ,'threshold':threshold,'aggregation':'metrics per graph seed, then mean'}
   row.update(dict(zip(MET,os[t] if t<4 else os.mean(0))));rows.append(row)
pd.DataFrame(rows).to_csv(DATA/'all_nine_metrics_both_thresholds.csv',index=False)
old=pd.read_csv(B/'macro_metrics.csv').set_index('model')
for _,key in NAMES:
 assert np.allclose(all_scores[key].mean(0)[:2],old.loc[key,['auroc','ap']].astype(float),atol=1e-10)
abl=pd.read_csv(A/'tables/macro_metrics.csv'); paired=pd.read_csv(A/'tables/paired_differences.csv')
for key in ['supervised','PN_only','PU_only','PNU']:
 arr=organ_scores(z['fusion_'+key]).mean(0)
 ref=abl[(abl.cohort=='original')&(abl.method=='fusion_'+key)].iloc[0]
 assert np.allclose(arr,ref[['AUROC','AP','accuracy','F1','precision','sensitivity','specificity','balanced_accuracy','MCC']].astype(float),atol=1e-10)

# Structure-only calculations; neither labels nor predictions enter the embedding.
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
train_m=[Chem.MolFromSmiles(s) for s in train_sm];test_m=[Chem.MolFromSmiles(s) for s in sm]
assert all(m is not None for m in train_m+test_m)
tfp=[gen.GetFingerprint(m) for m in train_m]; qfp=[gen.GetFingerprint(m) for m in test_m]
train_bits=np.array(tfp,dtype=np.float64);test_bits=np.array(qfp,dtype=np.float64)
sim=np.array([DataStructs.BulkTanimotoSimilarity(f,tfp) for f in qfp])
nearest=np.argsort(-sim,axis=1,kind='stable')[:,:2]
pca=PCA(n_components=2,svd_solver='full');xytr=pca.fit_transform(train_bits);xy=pca.transform(test_bits)
mean_gain=np.nanmean(gain,axis=1)
pd.DataFrame({'test_row_1based':np.arange(1,len(y)+1),'smiles':sm,'pc1':xy[:,0],'pc2':xy[:,1],
 'nearest_train_row_1based':nearest[:,0]+1,'nearest_Tanimoto':sim[np.arange(len(y)),nearest[:,0]],
 'mean_label_aligned_score_change':mean_gain}).to_csv(DATA/'chemical_space.csv',index=False)
pd.DataFrame({'train_row_1based':np.arange(1,len(train_sm)+1),'smiles':train_sm,'pc1':xytr[:,0],'pc2':xytr[:,1]}).to_csv(DATA/'training_space.csv',index=False)

# Selection rule, not hand-picked chemistry: two threshold corrections and two regressions,
# ranked by absolute label-aligned score change, distinct organ within each group and distinct test row overall.
cases=[];used=set()
for status in [1,-1]:
 candidates=[(abs(gain[i,t]),i,t) for i,t in zip(*np.where(valid&(changed==status)))]
 candidates.sort(key=lambda x:(-x[0],x[1],x[2]));seen_org=set()
 for _,i,t in candidates:
  if i in used or t in seen_org:continue
  cases.append((i,t,status));used.add(i);seen_org.add(t)
  if len(seen_org)==2:break
assert len(cases)==4
role_name={-1:'Conflict',0:'N',1:'P',2:'U-implicit',3:'U-undocumented',4:'Missing'}
case_rows=[];pair_rows=[]
for i in range(len(y)):
 for t in range(4):
  if valid[i,t]:pair_rows.append({'test_row_1based':i+1,'organ':ORG[t],'label':int(y[i,t]),
    'supervised_mean_score':sup[i,t],'PNU_mean_score':pnu[i,t],'signed_score_change':gain[i,t],
    'threshold':TH,'status':{1:'corrected',-1:'regressed',0:'unchanged'}[changed[i,t]]})
for k,(i,t,status) in enumerate(cases):
 for rank,j in enumerate(nearest[i]):
  case_rows.append({'case':'ABCD'[k],'status':'corrected' if status==1 else 'regressed','organ':ORG[t],
    'test_row_1based':i+1,'test_SMILES':sm[i],'label':int(y[i,t]),'supervised':sup[i,t],'PNU':pnu[i,t],
    'neighbor_rank':rank+1,'train_row_1based':j+1,'train_SMILES':train_sm[j],'Tanimoto':sim[i,j],
    **{ORG[o]+'_training_role':role_name[int(roles[j,o])] for o in range(4)}})
pd.DataFrame(case_rows).to_csv(DATA/'molecular_cases.csv',index=False)
pd.DataFrame(pair_rows).to_csv(DATA/'all_prediction_changes.csv',index=False)
trans=[]
for t,org in enumerate(ORG):
 trans.append({'organ':org,'n':int(valid[:,t].sum()),'positive':int((y[:,t]==1).sum()),
 'negative':int((y[:,t]==0).sum()),'correct_both':int((valid[:,t]&cs[:,t]&cp[:,t]).sum()),
 'corrected':int((valid[:,t]&(changed[:,t]==1)).sum()),'regressed':int((valid[:,t]&(changed[:,t]==-1)).sum()),
 'wrong_both':int((valid[:,t]&~cs[:,t]&~cp[:,t]).sum())})
pd.DataFrame(trans).to_csv(DATA/'threshold_transitions.csv',index=False)

captions={};atlas=PdfPages(OUT/'CarcPNU_new_figures_review.pdf')
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
 captions[name]=caption
 for ext in ['pdf','svg','png']:f.savefig(OUT/(name+'.'+ext),dpi=220)
 atlas.savefig(f)
 # Clean versions for manuscript placement: captions supply titles and notes.
 for obj in list(f.texts):obj.remove()
 f.savefig(OUT/(name+'_manuscript.pdf'),bbox_inches='tight',pad_inches=.03)
 plt.close(f)
 print('Saved',name,flush=True)
def heat(ax,v,rows,cols,vmin=0,vmax=1,cmap='YlGnBu',fmt='.3f'):
 im=ax.imshow(v,aspect='auto',vmin=vmin,vmax=vmax,cmap=cmap)
 ax.set_xticks(range(len(cols)),cols);ax.set_yticks(range(len(rows)),rows)
 for i in range(v.shape[0]):
  for j in range(v.shape[1]):
   if np.isfinite(v[i,j]):
    rgb=im.cmap(im.norm(v[i,j]))[:3]; luminance=np.dot(rgb,[.2126,.7152,.0722])
    ax.text(j,i,format(v[i,j],fmt),ha='center',va='center',fontsize=8,
      color='white' if luminance<.5 else '#203347')
 return im
def dot(ax,vals,names,colors,metric,limits=None):
 yy=np.arange(len(vals))[::-1]
 ax.scatter(vals,yy,c=colors,s=42,zorder=3);ax.set_yticks(yy,names);ax.set_xlabel(metric);grid(ax)
 if limits:ax.set_xlim(*limits)
 for x,j in zip(vals,yy):ax.annotate(f'{x:.4f}',(x,j),xytext=(5,0),textcoords='offset points',va='center',fontsize=8)
 ax.set_ylim(-.7,len(vals)-.2)

# Figure 1: broad ranking and organ-level context, with threshold-dependent behavior.
f=figure('1 | Benchmark performance across organs and metrics','Original test annotations; archived AUROC/AP reproduced exactly. Threshold panels use a common 0.45.',(12.5,12))
gs=f.add_gridspec(3,2,left=.22,right=.97,bottom=.11,top=.88,hspace=.56,wspace=.62,height_ratios=[1.1,1.1,1])
names=[n for n,k in NAMES]; colors=[C['pnu'] if k=='fusion_pnu' else C['sup'] if k=='fusion_supervised' else C['graph'] for n,k in NAMES]
mac=np.array([all_scores[k].mean(0) for n,k in NAMES])
for j in range(2):
 ax=f.add_subplot(gs[0,j]);dot(ax,mac[:,j],names if j==0 else ['']*len(names),colors,MET[j],(.50,.88) if j==0 else (.75,.99));panel(ax,'AB'[j],'Overall '+MET[j])
for j in range(2):
 ax=f.add_subplot(gs[1,j]);heat(ax,np.array([all_scores[k][:,j] for n,k in NAMES]),names if j==0 else ['']*len(names),ORG,vmin=.4 if j==0 else .65,vmax=1);panel(ax,'CD'[j],'Organ-specific '+MET[j])
ax=f.add_subplot(gs[2,0]);yy=np.arange(10)[::-1]
ax.scatter(mac[:,7],yy,color=C['sup'],s=32,label='Balanced accuracy');ax.scatter(mac[:,8],yy,color=C['diag'],s=32,marker='s',label='MCC')
ax.set_yticks(yy,names);ax.set_xlim(-.1,1);ax.set_xlabel('Four-organ macro metric');grid(ax);ax.legend(fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.19),frameon=False);panel(ax,'E','Class-balanced classification metrics')
ax=f.add_subplot(gs[2,1]);ax.scatter(mac[:,5],mac[:,6],c=colors,s=45)
offsets={0:(6,-9),5:(6,7),8:(-13,-10),9:(6,8),2:(7,-9),6:(5,7)}
for i,(xx,yy) in enumerate(zip(mac[:,5],mac[:,6])):ax.annotate(str(i+1),(xx,yy),xytext=offsets.get(i,(4,7)),textcoords='offset points',fontsize=8)
ax.set_xlim(-.05,1.05);ax.set_ylim(-.05,1.05);ax.set_xlabel('Macro sensitivity');ax.set_ylabel('Macro specificity');grid(ax,'both');panel(ax,'F','Sensitivity-specificity tradeoff')
f.text(.22,.025,'Panel F numbers follow model order from top to bottom. Published and local implementations have different training histories.',fontsize=8)
save(f,'fig1_benchmark','A-D: four-organ macro and organ-level AUROC/AP. Metrics are computed per saved run and then averaged. E-F: thresholds unified at 0.45 without test-set tuning; MCC and balanced accuracy complement ranking metrics. Number labels in F follow the ten-model order. Different model training pools and implementation adaptations remain; corrected and original-logic DCAMCP are both retained.')

# Figure 2: structural representation contribution and where molecules lie.
branches=[('Local graph',h['graph'],C['graph']),('FP supervised',z['fingerprint_supervised'],C['sup']),
 ('Fusion supervised',z['fusion_supervised'],C['sup']),('FP PNU',z['fingerprint_PNU'],C['pnu']),('Fusion PNU',z['fusion_PNU'],C['pnu'])]
bs=[organ_scores(p) for n,p,c in branches]
f=figure('2 | Structural representation and chemical coverage','A training-fitted projection and exact fingerprint similarities describe the same evaluated molecules.',(12,11.5))
gs=f.add_gridspec(3,2,left=.18,right=.96,bottom=.08,top=.88,hspace=.62,wspace=.48)
for j in range(2):
 vals=[v[:,j].mean() for v in bs]
 ax=f.add_subplot(gs[0,j]);dot(ax,vals,[b[0] for b in branches] if j==0 else ['']*5,[b[2] for b in branches],MET[j],(min(vals)-.025,max(vals)+.055));panel(ax,'AB'[j],'Representation: '+MET[j])
ax=f.add_subplot(gs[1,0]);ax.scatter(xytr[:,0],xytr[:,1],s=12,c=C['train'],alpha=.7,label='Training parents (435)');ax.scatter(xy[:,0],xy[:,1],s=14,facecolors='none',edgecolors=C['sup'],lw=.6,label='Test records (454)');ax.legend(fontsize=7,loc='upper right');panel(ax,'C','Chemical space: structures only')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)');ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)');maplim=(ax.get_xlim(),ax.get_ylim())
ax=f.add_subplot(gs[1,1]);limit=max(abs(mean_gain.min()),abs(mean_gain.max()));im=ax.scatter(xy[:,0],xy[:,1],s=18,c=mean_gain,cmap='BrBG',vmin=-limit,vmax=limit,alpha=.85);f.colorbar(im,ax=ax,pad=.02,shrink=.8,label='Mean label-aligned score change');panel(ax,'D','Where PNU changes predictions');ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_xlim(maplim[0]);ax.set_ylim(maplim[1])
ax=f.add_subplot(gs[2,0]);ax.hist(sim.max(1),bins=np.linspace(0,1,21),color=C['sup'],alpha=.85,edgecolor='white');ax.set_xlabel('Maximum Tanimoto to fingerprint training pool');ax.set_ylabel('Test records');panel(ax,'E','Nearest-neighbor structural coverage');grid(ax,'y')
ax=f.add_subplot(gs[2,1]);v=np.array([(bs[2]-bs[1])[:,0],(bs[4]-bs[3])[:,0]])
heat(ax,v,['Supervised','PNU'],ORG,vmin=-.015,vmax=.015,cmap='BrBG',fmt='+.4f');panel(ax,'F','Fusion minus fingerprint-only AUROC')
f.text(.18,.025,'Projection uses training Morgan bits only. Score change is averaged over available labels; positive means movement toward the recorded label.',fontsize=8)
save(f,'fig2_structure_space','A-B distinguish local graph predictions, fingerprint-only scores and fixed 25% graph / 75% fingerprint fusion. C: PCA fitted only to 435 training fingerprints, with test projection; 2-D proximity is not a substitute for exact structural similarity. D: mean label-aligned score change (2y-1)(PNU-supervised), over available labels and seed-mean predictions. E: exact radius-2 2048-bit Morgan Tanimoto to the fingerprint pool, not all graph-training molecules. F: AUROC changes due to fusion; improvements are small. Maps and scores are descriptive, not biological mechanisms or independent validation.')

# Figure 3: all organ curves; each seed shown separately, no ensemble/mean-metric confusion.
f=figure('3 | Organ-specific discrimination and precision-recall','Each saved graph-seed prediction is plotted; legends report mean per-seed AUROC or AP.',(13,8.8))
gs=f.add_gridspec(2,4,left=.065,right=.98,bottom=.10,top=.86,hspace=.47,wspace=.34)
curve_data={}
for t,org in enumerate(ORG):
 yy=y[valid[:,t],t];n=len(yy);pos=int(yy.sum())
 for j in range(2):
  ax=f.add_subplot(gs[j,t]);panel(ax,'ABCDEFGH'[j*4+t],org+f' (n={n}; +{pos}/-{n-pos})')
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

# Figure 4: matched ablation and predictions that improve or regress.
f=figure('4 | PNU components and changes in prediction behavior','Matched representation and graph predictions. A-D: per-seed metrics; E-F: seed-mean score diagnostics.',(12,11.5))
gs=f.add_gridspec(3,2,left=.14,right=.96,bottom=.12,top=.88,wspace=.42,hspace=.61)
losses=['supervised','PN_only','PU_only','PNU'];labs=['Supervised','PN-only','PU-only','PNU']
for j in range(2):
 ax=f.add_subplot(gs[0,j]);xx=np.arange(4)
 for b,offset,color in [('fingerprint',-.12,C['graph']),('fusion',.12,C['pnu'])]:
  v=[organ_scores(z[b+'_'+k]).mean(0)[j] for k in losses];ax.plot(xx+offset,v,'o-',color=color,label='Fingerprint' if b=='fingerprint' else 'Fusion',lw=1.2)
  for x,zz in zip(xx+offset,v):ax.annotate(f'{zz:.3f}',(x,zz),xytext=(0,7 if b=='fusion' else -13),textcoords='offset points',ha='center',fontsize=8)
 ax.set_xticks(xx,labs);ax.set_ylabel('Macro '+MET[j]);ax.legend(fontsize=8);ax.margins(y=.3);grid(ax,'y');panel(ax,'AB'[j],'Four training objectives: '+MET[j])
for j,metric in enumerate(['AUROC','AP']):
 ax=f.add_subplot(gs[1,j]);ax.axvline(0,ls=':',color=C['graph'])
 for idx,k in enumerate(['supervised','PN_only','PU_only']):
  q=paired[(paired.cohort=='original')&(paired.branch=='fusion')&(paired.method_A=='PNU')&(paired.method_B==k)&(paired.organ=='macro')&(paired.metric==metric)].iloc[0]
  ax.hlines(2-idx,q.CI95_low,q.CI95_high,color=C['pnu'],lw=2);ax.scatter(q.delta,2-idx,color=C['pnu'],s=45)
  ax.text(q.delta,2-idx+.2,f'{q.delta:+.4f}',ha='center',fontsize=8)
 ax.set_yticks([2,1,0],['vs supervised','vs PN-only','vs PU-only']);ax.set_ylim(-.6,2.7);ax.set_xlabel('PNU minus comparator');grid(ax);panel(ax,'CD'[j],metric+': paired 95% intervals')
ax=f.add_subplot(gs[2,0]);left=np.zeros(4)
for key,label,col in [('correct_both','Correct in both','#a4c8bb'),('corrected','Corrected',C['good']),('regressed','Regressed',C['bad']),('wrong_both','Wrong in both','#d5dce2')]:
 vals=np.array([r[key]/r['n'] for r in trans]);ax.barh(np.arange(4),vals,left=left,color=col,label=label,height=.65)
 for i,v in enumerate(vals):
  if v>.04:ax.text(left[i]+v/2,i,str(trans[i][key]),ha='center',va='center',fontsize=8)
 left+=vals
ax.set_yticks(range(4),[f'{o} (n={r["n"]})' for o,r in zip(ORG,trans)]);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Fraction of evaluable organ records');ax.legend(ncol=2,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.25));panel(ax,'E','Classification transitions at 0.45')
ax=f.add_subplot(gs[2,1]);parts=ax.violinplot([gain[valid[:,t],t] for t in range(4)],showmeans=True,showextrema=False)
for b in parts['bodies']:b.set_facecolor(C['pnu']);b.set_alpha(.45)
ax.axhline(0,ls=':',color=C['graph']);ax.set_xticks(range(1,5),ORG);ax.set_ylabel('Label-aligned score change');grid(ax,'y');panel(ax,'F','Continuous prediction changes')
f.text(.14,.018,'E pools classifications within each organ. Counts are not macro-averaged metrics; pooled gains and losses do not directly measure AUROC change.',fontsize=8)
save(f,'fig4_pnu_ablation','A-B: four loss settings, with fingerprint-only and fusion results. C-D: archived 10,000 paired molecule bootstrap intervals, conditional on fitted predictions and not corrected for model selection. E: classifications from seed-mean predictions at 0.45; counts are organ-records, not unique molecules. F: signed score changes toward the recorded label, not calibrated changes in disease risk. Ranking improvement does not guarantee fewer classification errors at a fixed threshold. Missing labels are excluded.')

# Figure 5: real molecular drawings, two neighbors, full organ context.
f=figure('5 | Molecular case studies and local structural neighborhoods','Two corrected and two regressed cases, selected by a recorded rule; no chemical mechanism is inferred.',(14,12))
gs=f.add_gridspec(4,4,left=.045,right=.975,bottom=.07,top=.89,wspace=.37,hspace=.64,width_ratios=[1,1,1,1.45])
(DATA/'molecules').mkdir(exist_ok=True)
for row,(i,t,status) in enumerate(cases):
 label='Corrected' if status==1 else 'Regressed'; col=C['good'] if status==1 else C['bad']
 indices=[i,*nearest[i]]
 for k,idx in enumerate(indices):
  ax=f.add_subplot(gs[row,k]);mol=test_m[idx] if k==0 else train_m[idx]
  drawer=rdMolDraw2D.MolDraw2DSVG(650,350);rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol);drawer.FinishDrawing()
  (DATA/'molecules'/f'case_{"ABCD"[row]}_{"test" if k==0 else "neighbor_"+str(k)}.svg').write_text(drawer.GetDrawingText(),encoding='utf-8')
  img=Draw.MolToImage(mol,size=(650,350));ax.imshow(img);ax.axis('off')
  if k==0:
   ax.set_title(f'{"ABCD"[row]}  {label}: {ORG[t]} | Test #{i+1}',fontsize=10,color=col,loc='left',fontweight='bold',pad=8)
   sub=f'Observed label: {y[i,t]}\nSupervised {sup[i,t]:.3f}  →  PNU {pnu[i,t]:.3f}'
  else:
   ax.set_title(f'Neighbor {k} | Train #{idx+1}\nTanimoto = {sim[i,idx]:.3f}',fontsize=9,pad=5)
   sub=f'{ORG[t]} role: {role_name[int(roles[idx,t])]}\n'+ ' / '.join(role_name[int(v)].replace('U-implicit','Ui').replace('U-undocumented','Ud').replace('Missing','--').replace('Conflict','X') for v in roles[idx])
  ax.text(.5,-.12,sub,transform=ax.transAxes,ha='center',va='top',fontsize=8)
 ax=f.add_subplot(gs[row,3]);v=np.vstack([np.where(valid[i],y[i],np.nan),sup[i],pnu[i]])
 heat(ax,v,['Observed','Supervised','PNU'],['Liv','Lun','Sto','Mam'],vmin=0,vmax=1,cmap='YlGnBu',fmt='.2f')
 for q in np.flatnonzero(~valid[i]):ax.text(q,0,'NA',ha='center',va='center',fontsize=8)
 ax.set_title('Four-organ labels and scores',fontsize=10,pad=10)
f.text(.045,.022,'Neighbor role order: liver / lung / stomach / mammary. Ui = implicit U; Ud = undocumented U; -- = missing; X = conflicting. Rows are 1-based archive IDs.',fontsize=8)
save(f,'fig5_molecular_cases','Two corrected and two regressed organ-records at threshold 0.45, sorted within status by absolute signed score change, requiring different organs within each status and distinct test records. Full selection output and all candidates are saved. Each row shows the actual test structure and its two closest fingerprint-pool training neighbors, exact Tanimoto and recorded training roles. Right panels show recorded binary test labels and seed-mean scores for all four organs. Missing test labels are NA. Similarity and source roles do not establish shared mechanisms or correctness of a U label; examples are exploratory and chosen after observing predictions.')

# Figure 6: parameter and provenance robustness with all relevant simple controls.
prior=json.load(open(paths[-1]))['prior_sensitivity'];eta=pd.read_csv(D/'tables/eta_sensitivity.csv').sort_values('eta');perm=pd.read_csv(D/'tables/role_permutation.csv')
control=[('Supervised','fusion_fixed_supervised',C['sup']),('PNU','fusion_fixed_pnu',C['pnu']),
 ('Implicit-zero weighting','fusion_fixed_downweight_implicit',C['diag']),('All-U weighting','fusion_fixed_downweight_all_u',C['diag']),
 ('Matched-curvature BCE','fusion_matched_curvature_bce',C['diag'])]
ctrl=np.array([organ_scores(diag[k]).mean(0) for n,k,c in control]);actual=ctrl[1]-ctrl[3]
assert np.isclose(actual[0],.005172614010604426)
f=figure('6 | Sensitivity to priors, role assignment and loss alternatives','Previously fitted settings are described without choosing a new test-optimal configuration.',(12.5,11.5))
gs=f.add_gridspec(3,2,left=.19,right=.96,bottom=.08,top=.88,hspace=.65,wspace=.52)
ax=f.add_subplot(gs[0,0]);pi=[r['prior'] for r in prior]
for t,org in enumerate(ORG):ax.plot(pi,[r['original']['per_organ'][t][0] for r in prior],'.-',lw=1,label=org)
ax.plot(pi,[r['original']['macro'][0] for r in prior],'o-',color='#24394a',lw=2,label='Macro');ax.axvline(.3,ls=':',color=C['graph']);ax.set_xlabel('Assumed positive prior π');ax.set_ylabel('AUROC');ax.legend(ncol=2,fontsize=7);grid(ax,'y');panel(ax,'A','Organ responses to the assumed prior')
ax=f.add_subplot(gs[0,1]);ax.plot(eta.eta,eta.test_AUROC,'o-',color=C['pnu'],label='Macro AUROC');ax.plot(eta.eta,eta.test_AP,'s-',color=C['sup'],label='Macro AP');ax.axvline(.25,ls=':',color=C['graph']);ax.set_xlabel('PN/PU mixing coefficient η');ax.set_ylabel('Metric');ax.legend(fontsize=8);grid(ax,'y');panel(ax,'B','PN-only → PNU → PU-only')
for j in range(2):
 ax=f.add_subplot(gs[1,j]);vv=perm.PNU_minus_downweight_AUROC.to_numpy() if j==0 else (perm.PNU_AP-perm.downweight_AP).to_numpy()
 ax.hist(vv,bins=9,color=C['train'],edgecolor='white');ax.axvline(actual[j],color=C['pnu'],lw=2,label=f'Source roles: {actual[j]:+.4f}');ax.axvline(0,ls=':',color=C['graph']);ax.set_xlabel(f'PNU minus all-U weighting: {MET[j]}');ax.set_ylabel('Permutations');ax.legend(fontsize=8);panel(ax,'CD'[j],'Source roles vs 30 shuffled assignments')
for j in range(2):
 ax=f.add_subplot(gs[2,j]);dot(ax,ctrl[:,j],[n for n,k,c in control] if j==0 else ['']*5,[c for n,k,c in control],MET[j],(.765,.835) if j==0 else (.92,.96));panel(ax,'EF'[j],'Fixed loss alternatives: '+MET[j])
f.text(.19,.025,'Role permutation preserves P, missing/conflicting entries and zero-role counts. Source-role contrasts are descriptive, not confirmatory p-values.',fontsize=8)
save(f,'fig6_robustness','A: assumed-prior scan with organ and macro AUROC. B: fixed-prior eta scan with macro AUROC and AP. C-D: 30 within-organ zero-role permutations compared with the actual provenance assignment, using the fixed PNU-minus-all-U-weighting contrast; this is not the implicit-only weighting comparison. E-F retain simple fixed weighting and matched-curvature controls. Similar performance limits claims of unique PNU correction benefit. Settings and test data had previously been examined; this is sensitivity analysis, not validation of prevalence, mislabel rate or an independent external test. The archived overlap-filtered comparison also remains available in the original experiment records.')

# Supplemental figure: all nine metrics, both prespecified archived thresholds.
f=figure('S1 | Complete nine-metric comparison','Dots: threshold 0.45; crosses: threshold 0.50. AUROC/AP do not depend on either threshold.',(16,12))
gs=f.add_gridspec(3,3,left=.16,right=.98,bottom=.07,top=.88,wspace=.18,hspace=.40)
mac50=np.array([organ_scores(base[k],.5).mean(0) for n,k in NAMES])
for j,metric in enumerate(MET):
 ax=f.add_subplot(gs[j//3,j%3]);yy=np.arange(10)[::-1]
 ax.scatter(mac[:,j],yy,c=colors,s=36,zorder=3,label='0.45');ax.scatter(mac50[:,j],yy,c=colors,s=30,marker='x',zorder=3,label='0.50')
 for i in range(10):ax.plot([mac[i,j],mac50[i,j]],[yy[i]]*2,color=colors[i],lw=.8,alpha=.65)
 ax.set_yticks(yy,names if j%3==0 else ['']*10);ax.set_xlim(-.08 if j==8 else 0,1.05);ax.set_xlabel('Four-organ macro '+metric);grid(ax);panel(ax,'ABCDEFGHI'[j],metric)
 if j==2:ax.legend(fontsize=8,loc='lower right')
f.text(.16,.025,'Seven threshold-dependent metrics are recomputed uniformly from saved scores. An untuned common threshold is descriptive, not a claim of calibration.',fontsize=8)
save(f,'figS1_nine_metrics','All nine macro metrics for the ten benchmark methods at common thresholds 0.45 and 0.50, recomputed from archived numerical predictions. The original baseline table used 0.50, while phase-6 ablations used 0.45; this figure resolves that mismatch explicitly. AUROC/AP are threshold-free. Per-organ metrics and exact values for both thresholds are included in the accompanying CSV. Metrics are averaged across graph seeds and then organs; zero-denominator precision/F1 and MCC use zero, matching archived conventions. These scores are not assumed to be calibrated risks.')

# Additional methods diagram, explicitly distinguishing graph labels from fingerprint roles.
f=figure('Methods | CarcPNU prediction workflow','Molecular input, provenance-defined fingerprint training, and fixed graph-fingerprint fusion.',(12,7))
ax=f.add_axes([.04,.13,.92,.72]);ax.set_xlim(0,12);ax.set_ylim(-.2,6);ax.axis('off')
def box(x,yy,w,hh,text,color):
 ax.add_patch(FancyBboxPatch((x,yy),w,hh,boxstyle='round,pad=0.13',facecolor=color,edgecolor='#9babb8',lw=1));ax.text(x+w/2,yy+hh/2,text,ha='center',va='center',fontsize=10)
def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#627c8e',lw=1.6))
box(.15,2.2,2,1.4,'Molecular structures\n(SMILES)','#eaf0f4')
box(3,4,2.7,1.5,'Molecular graph\nOriginal binary annotations\n5 fixed graph seeds','#edf0f4')
box(3,1.9,2.7,1.4,'Morgan fingerprint\nTanimoto feature map\nTrain-only basis','#e5f1f7')
box(3,.05,2.7,1.25,'Annotation provenance\nP / source-explicit N / U\nConflicts & missing excluded','#fbf0df')
box(6.4,1.7,2.3,1.6,'Fingerprint PNU risk\nπ = 0.30; η = 0.25\nOne deterministic fit','#e0f2ed')
box(9.35,2.8,2.1,1.4,'Fixed score fusion\n25% graph + 75% FP','#e0f2ed')
box(9.25,.25,2.3,1.4,'Four organ scores\nLiver · Lung\nStomach · Mammary','#eaf0f4')
arrow((2.3,3.2),(2.85,4.7));arrow((2.3,2.6),(2.85,2.6));arrow((5.85,2.6),(6.25,2.6));arrow((5.85,.7),(6.4,1.7));arrow((8.85,2.6),(9.2,3.3));arrow((5.85,4.75),(10.2,4.35));arrow((10.4,2.65),(10.4,1.8))
f.text(.07,.06,'P/N/U affects the fingerprint loss only. Test labels retain their original binary meaning. U is unresolved provenance, not verified hidden positives.',fontsize=9)
save(f,'fig0_workflow','Conceptual workflow reflecting the implemented model. The graph branch retains original binary training annotations. P/N/U provenance roles affect only the fingerprint loss, with conflicts and genuinely missing labels excluded. The train-only fingerprint basis includes all 435 structures. Five graph-seed predictions share one fingerprint fit; there are not five independently trained end-to-end PNU models. The four test labels remain binary; no G2CP expression data are used in this reported model.')
atlas.close()
for source in [A/'tables/paired_differences.csv',A/'tables/macro_metrics.csv',A/'tables/per_organ_metrics.csv',D/'tables/eta_sensitivity.csv',D/'tables/role_permutation.csv',D/'tables/real_comparison.csv']:
 (DATA/('archived_'+source.name)).write_bytes(source.read_bytes())
(DATA/'prior_sensitivity.json').write_text(json.dumps(prior,indent=2),encoding='utf-8')
pd.DataFrame([{'branch':branches[b][0],'organ':org,**dict(zip(MET,bs[b][t]))} for b in range(len(bs)) for t,org in enumerate(ORG)]).to_csv(DATA/'representation_metrics.csv',index=False)
(OUT/'captions_EN.txt').write_text('\n\n'.join(k+'\n'+v for k,v in captions.items()),encoding='utf-8')
manifest={'models_retrained':False,'thresholds':[.45,.50],'classification_and_cases_threshold':TH,
 'case_aggregation':'mean graph-seed predictions','benchmark_aggregation':'mean of per-seed metrics, then macro over organs',
 'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
 'counts':trans,'case_rule':'Top 2 corrected and top 2 regressed by absolute signed score change, distinct organ within status and distinct test row.',
 'figures':list(captions),'checks':{'archived_AUROC_AP_match':True,'phase6_all_nine_metrics_match':True,'row_alignment_checks':True},
 'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'transitions':trans,'cases':[(int(i+1),ORG[t],int(s)) for i,t,s in cases],'PCA_variance':pca.explained_variance_ratio_.tolist()},indent=2),flush=True)
