"""Figure 2 from saved per-run predictions; no training or threshold selection."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
OUT=Path(__file__).resolve().parent
DATA=OUT/'source_data'
ROOT=OUT.parents[2]
source=ROOT/'OS-Carcinogenicity-baseline/full_comparison_20261101/evaluation/predictions.npz'
local=DATA/'benchmark_curve_predictions.npz'
MODELS=[('OS-Carcinogenicity','multi_task_published','#778694','--'),('Single-task GAT','single_task','#bc9254',':'),('MolBERT','MolBERT','#5565a6','-.'),('CarcGC','CarcGC','#9d71a7','--'),('DCAMCP','DCAMCP','#b6605f',':'),('XGBoost','XGBoost','#6397bd','-.'),('Random forest','RF','#8b764f','--'),('CarcPNU','fusion_pnu','#008b7f','-')]
if not local.exists():
 z=np.load(source)
 np.savez_compressed(local,labels=z['labels'],**{key:z[key] for _,key,_,_ in MODELS})
z=np.load(local);labels=z['labels'];mask=labels>=0
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#9aa9b4','text.color':'#203347','axes.labelcolor':'#203347','savefig.facecolor':'white','figure.facecolor':'white'})
fig=plt.figure(figsize=(11.5,9.4))
axes=[fig.add_axes([.19,.56,.34,.36]),fig.add_axes([.60,.56,.34,.36])]
grid=np.linspace(0,1,2001);reference=pd.read_csv(DATA/'all_nine_metrics_both_thresholds.csv')
records=[];coordinates={};metrics=[]
for name,key,color,style in MODELS:
 pred=z[key];assert pred.ndim==3 and pred.shape[1:]==labels.shape
 organ_values=[];roc_values=[];pr_values=[]
 for t,organ in enumerate(['Liver','Lung','Stomach','Breast']):
  yt=labels[mask[:,t],t];values=[];rs=[];ps=[]
  for run in pred:
   score=run[mask[:,t],t];assert np.isfinite(score).all()
   fpr,tpr,_=roc_curve(yt,score);precision,recall,_=precision_recall_curve(yt,score)
   rs.append(np.interp(grid,fpr,tpr))
   # Noninterpolated PR: precision for the smallest attained recall >= requested recall.
   rr=recall[::-1];pp=precision[::-1]
   ps.append(pp[np.minimum(np.searchsorted(rr,grid,side='left'),len(rr)-1)])
   values.append([roc_auc_score(yt,score),average_precision_score(yt,score)])
  vals=np.mean(values,axis=0);organ_values.append(vals);roc_values.append(np.mean(rs,axis=0));pr_values.append(np.mean(ps,axis=0))
  archived=reference[(reference.model==name)&(reference.threshold==.45)&(reference.organ==('Mammary' if organ=='Breast' else organ))].iloc[0]
  assert np.allclose(vals,archived[['AUROC','AP']].to_numpy(float),atol=1e-8),name
  records.append({'model':name,'organ':organ,'AUROC':vals[0],'AP':vals[1]})
 macro=np.mean(organ_values,axis=0);metrics.append(organ_values)
 curves=[np.mean(roc_values,axis=0),np.mean(pr_values,axis=0)]
 coordinates[name]={'x':grid.tolist(),'roc':curves[0].tolist(),'pr':curves[1].tolist(),'macro_AUROC':float(macro[0]),'macro_AP':float(macro[1])}
 for j,ax in enumerate(axes):
  ax.plot(grid,curves[j],color=color,ls=style,lw=2.4 if name=='CarcPNU' else 1.3,label=f'{name}  {macro[j]:.4f}',zorder=5 if name=='CarcPNU' else 3)
axes[0].plot([0,1],[0,1],ls=':',color='#acb5bd',lw=.9,zorder=1)
prevalence=float(np.mean([labels[mask[:,t],t].mean() for t in range(4)]))
axes[1].axhline(prevalence,ls=':',color='#acb5bd',lw=.9,zorder=1)
for j,ax in enumerate(axes):
 ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xticks(np.linspace(0,1,6));ax.set_yticks(np.linspace(0,1,6))
 ax.set_xlabel('False-positive rate' if j==0 else 'Recall');ax.set_ylabel('True-positive rate' if j==0 else 'Precision')
 ax.set_title(['A  Macro-averaged ROC','B  Macro-averaged precision–recall'][j],loc='left',fontweight='bold',pad=12)
 ax.grid(color='#e8edf1',lw=.6);ax.set_axisbelow(True)
 ax.legend(loc='lower right' if j==0 else 'lower left',fontsize=8,frameon=True,facecolor='white',edgecolor='none',framealpha=.88,labelspacing=.3,handlelength=2.3)
for j,x in enumerate([.19,.60]):
 ax=fig.add_axes([x,.08,.34,.31]);v=np.array(metrics)[:,:,j]
 im=ax.imshow(v,aspect='auto',vmin=.4,vmax=1,cmap='YlGnBu')
 ax.set_xticks(range(4),['Liver','Lung','Stomach','Breast'],fontsize=10)
 ax.set_yticks(range(len(MODELS)),[m[0] for m in MODELS] if j==0 else ['']*len(MODELS),fontsize=10)
 ax.tick_params(length=0,pad=7)
 ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(MODELS),1),minor=True)
 ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',bottom=False,left=False)
 for sp in ax.spines.values():sp.set_visible(False)
 for i in range(len(MODELS)):
  for t in range(4):
   lum=np.dot(im.cmap(im.norm(v[i,t]))[:3],[.2126,.7152,.0722])
   ax.text(t,i,f'{v[i,t]:.3f}',ha='center',va='center',fontsize=10.5,fontweight='bold' if i==7 else 'normal',color='white' if lum<.5 else '#203347')
 ax.add_patch(Rectangle((-.5,6.5),4,1,fill=False,edgecolor='#008b7f',lw=2,clip_on=False))
 if j==0:ax.get_yticklabels()[-1].set_color('#008b7f');ax.get_yticklabels()[-1].set_fontweight('bold')
 ax.set_title(['C  AUROC by organ','D  AP by organ'][j],loc='left',fontweight='bold',pad=14)
cb=fig.colorbar(im,cax=fig.add_axes([.958,.08,.012,.31]),ticks=[.4,.6,.8,1]);cb.ax.tick_params(labelsize=8,length=2);cb.outline.set_visible(False);cb.ax.set_title('Score',fontsize=8,pad=8)
for ext in ['pdf','svg','png']:
 name='fig2_benchmark_curves'+('_manuscript' if ext=='pdf' else '')+'.'+ext
 fig.savefig(OUT/name,bbox_inches='tight',pad_inches=.06,dpi=240)
plt.close(fig)
pd.DataFrame(records).to_csv(DATA/'benchmark_curve_metrics.csv',index=False)
(DATA/'benchmark_macro_curves.json').write_text(json.dumps(coordinates))
(DATA/'benchmark_curve_manifest.json').write_text(json.dumps({'source_sha256':hashlib.sha256(local.read_bytes()).hexdigest(),'aggregation':'per-run curves -> mean within organ -> equal mean across four organs','ROC':'linear interpolation at common FPR grid','PR':'noninterpolated precision at common recall grid','legend':'mean original per-run per-organ AUROC/AP, not area under resampled plotted curve','grid_points':len(grid),'threshold_selected':False,'mean_positive_fraction':prevalence,'archived_organ_metrics_verified':True},indent=2))
print('Figure 2 saved; all eight models verified against archived organ metrics.')
