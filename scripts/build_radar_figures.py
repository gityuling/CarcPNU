"""Four-panel benchmark, supplementary macro scores, and matched loss analyses.
Uses exported figure source data only; no training or metric recalculation.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parent
DATA=OUT/'source_data'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,
 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#9aa9b4','text.color':'#203347','axes.labelcolor':'#203347',
 'savefig.facecolor':'white','figure.facecolor':'white'})
MET=['AUROC','AP','Accuracy','F1','Precision','Sensitivity','Specificity','Balanced accuracy','MCC']
LAB=['AUROC','AP','Accuracy','F1','Precision','Sensitivity','Specificity','Balanced\naccuracy','MCC']
C={'sup':'#347daf','pnu':'#008b7f','pn':'#c28b36','pu':'#9776b6','base':'#7c8b99','rf':'#a57454'}
tbl=pd.read_csv(DATA/'all_nine_metrics_both_thresholds.csv')
tab=tbl[(tbl.threshold==.45)&(tbl.organ=='Macro')].set_index('model')
abl=pd.read_csv(DATA/'archived_macro_metrics.csv');abl=abl[abl.cohort=='original'].set_index('method')
COL=['AUROC','AP','accuracy','F1','precision','sensitivity','specificity','balanced_accuracy','MCC']
pairs=pd.read_csv(DATA/'archived_paired_differences.csv')
trans=pd.read_csv(DATA/'threshold_transitions.csv')
changes=pd.read_csv(DATA/'all_prediction_changes.csv')
ORG=['Liver','Lung','Stomach','Mammary']
NAMES=['OS-Carcinogenicity','Single-task GAT','MolBERT','CarcGC','DCAMCP','XGBoost','Random forest','CarcPNU']

def values(name):return tab.loc[name,MET].to_numpy(dtype=float)
def av(branch,loss):return abl.loc[branch+'_'+loss,COL].to_numpy(dtype=float)
def panel(ax,label,title):ax.set_title(label+'  '+title,loc='left',fontweight='bold',pad=12)
def radar(ax,series,title,label,*,large=False,draw_heading=True):
 theta=np.linspace(0,2*np.pi,len(MET),endpoint=False);closed=np.r_[theta,theta[0]]
 ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
 ax.set_xticks(theta,LAB);ax.tick_params(axis='x',pad=10 if large else 7,labelsize=11 if large else 8)
 if large:
  for angle,tick in zip(theta,ax.get_xticklabels()):
   horizontal=np.sin(angle)
   tick.set_horizontalalignment('left' if horizontal>.2 else 'right' if horizontal<-.2 else 'center')
 ax.set_ylim(0,1);ax.set_yticks([.25,.5,.75,1],['0.25','0.50','0.75','1.00'],fontsize=6.5,color='#778b9c')
 ax.set_rlabel_position(20);ax.grid(color='#d6e0e7',lw=.7)
 ax.spines['polar'].set_color('#c4d1db');ax.set_facecolor('#fafcfd')
 for name,v,color,ls in series:
  assert len(v)==9 and np.isfinite(v).all() and v.min()>=0 and v.max()<=1
  vv=np.r_[v,v[0]]
  ax.plot(closed,vv,color=color,lw=(2.5 if name in ['PNU','CarcPNU'] else 1.7) if large else (1.7 if name in ['PNU','CarcPNU'] else 1.3),ls=ls,marker='o',ms=3.3 if large else 2.7,label=name,zorder=3)
  ax.fill(closed,vv,color=color,alpha=.035 if large else .04)
 if draw_heading:ax.text(-.20,1.24,label+'  '+title,transform=ax.transAxes,fontweight='bold',fontsize=10,va='bottom')
 ax.legend(loc='upper center',bbox_to_anchor=(.5,-.21 if large else -.16),frameon=False,ncol=2,fontsize=10 if large else 7.5,columnspacing=1.1,handlelength=2.4)

radar_specs=[
 ('Carcinogenicity comparators',[
  ('Released OS',values('OS-Carcinogenicity'),C['base'],'--'),
  ('CarcGC',values('CarcGC'),C['pu'],'-.'),
  ('DCAMCP',values('DCAMCP'),C['pn'],':'),
  ('CarcPNU',values('CarcPNU'),C['pnu'],'-')]),
 ('General-purpose comparators',[
  ('MolBERT',values('MolBERT'),'#5772ae','--'),
  ('XGBoost',values('XGBoost'),C['sup'],':'),
  ('Random forest',values('Random forest'),C['rf'],'-.'),
  ('CarcPNU',values('CarcPNU'),C['pnu'],'-')]),
 ('Fingerprint-only ablation',[(n,av('fingerprint',k),C[c],ls) for n,k,c,ls in [
  ('Supervised','supervised','sup','--'),('PN-only','PN_only','pn','-.'),('PU-only','PU_only','pu',':'),('PNU','PNU','pnu','-')]]),
 ('Fusion ablation',[(n,av('fusion',k),C[c],ls) for n,k,c,ls in [
  ('Supervised','supervised','sup','--'),('PN-only','PN_only','pn','-.'),('PU-only','PU_only','pu',':'),('PNU','PNU','pnu','-')]])]
source=[]
for title,series in radar_specs:
 for name,v,col,ls in series:source.append({'panel':title,'method':name,'threshold':.45,**dict(zip(MET,v))})
pd.DataFrame(source).to_csv(DATA/'radar_values.csv',index=False)

def save(f,name,headline,subtitle):
 # Manuscript file has panel letters and legends but no repeated overall title.
 f.savefig(OUT/(name+'_manuscript.pdf'),bbox_inches='tight',pad_inches=.03)
 f.text(.06,.976,headline,fontsize=16,fontweight='bold',va='top')
 f.text(.06,.941,subtitle,fontsize=8.5,color='#64798a')
 for ext in ['png','svg']:f.savefig(OUT/(name+'.'+ext),dpi=240)
 plt.close(f)

# Main Figure 2: combined nine-metric macro heatmap above aligned organ heatmaps.
# All three heatmaps share one absolute score color scale; CarcPNU row is outlined.
PROF=[('OS-Carcinogenicity','carc'),('CarcGC','carc'),('DCAMCP','carc'),
 ('MolBERT','gen'),('XGBoost','gen'),('Random forest','gen'),('CarcPNU','pnu')]
f=plt.figure(figsize=(12.4,9.2))
ax=f.add_axes([.225,.545,.70,.335])
v=np.array([values(n) for n,_ in PROF])
im=ax.imshow(v,aspect='auto',vmin=.4,vmax=1,cmap='YlGnBu')
ax.set_xticks(range(9),LAB,fontsize=10)
ax.set_yticks(range(len(PROF)),[n for n,_ in PROF],fontsize=10.5)
ax.tick_params(axis='both',length=0,pad=8)
ax.set_xticks(np.arange(-.5,9,1),minor=True)
ax.set_yticks(np.arange(-.5,len(PROF),1),minor=True)
ax.grid(which='minor',color='white',lw=.65)
ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():spine.set_visible(False)
for i in range(len(PROF)):
 for k in range(9):
  lum=np.dot(im.cmap(im.norm(v[i,k]))[:3],[.2126,.7152,.0722])
  ax.text(k,i,f'{v[i,k]:.3f}',ha='center',va='center',fontsize=10.5,
   fontweight='bold' if PROF[i][1]=='pnu' else 'normal',
   color='white' if lum<.5 else '#203347')
ax.axhline(2.5,color='white',lw=2.4,zorder=4)  # carcinogenicity-focused | general-purpose
ax.axvline(1.5,color='white',lw=2.4,zorder=4)  # threshold-free | threshold 0.45
ax.add_patch(Rectangle((-.5,len(PROF)-1.5),9,1,fill=False,edgecolor=C['pnu'],lw=2,clip_on=False,zorder=5))
ax.get_yticklabels()[-1].set_color(C['pnu']);ax.get_yticklabels()[-1].set_fontweight('bold')
f.text(.225,.935,'A  Nine-metric macro profiles',fontsize=13,fontweight='bold',va='bottom')
cbA=f.colorbar(im,cax=f.add_axes([.947,.545,.012,.335]),ticks=[.4,.6,.8,1])
cbA.ax.tick_params(labelsize=9,length=2);cbA.outline.set_visible(False)
cbA.ax.set_title('Score',fontsize=9,pad=9)
heat_positions=[.225,.595]
heat_values=[]
for j,x in enumerate(heat_positions):
 ax=f.add_axes([x,.055,.325,.325])
 v=np.array([[tbl[(tbl.threshold==.45)&(tbl.model==n)&(tbl.organ==o)].iloc[0][MET[j]] for o in ORG] for n in NAMES])
 heat_values.append(v)
 im=ax.imshow(v,aspect='auto',vmin=.4,vmax=1,cmap='YlGnBu')
 ax.set_xticks(range(4),[('Breast' if x == 'Mammary' else x) for x in ORG],fontsize=11)
 ax.set_yticks(range(len(NAMES)),NAMES if j==0 else ['']*len(NAMES),fontsize=11)
 ax.tick_params(axis='both',length=0,pad=9)
 ax.set_xticks(np.arange(-.5,4,1),minor=True)
 ax.set_yticks(np.arange(-.5,len(NAMES),1),minor=True)
 ax.grid(which='minor',color='white',lw=.65)
 ax.tick_params(which='minor',bottom=False,left=False)
 for spine in ax.spines.values():spine.set_visible(False)
 for i in range(len(NAMES)):
  for k in range(4):
   lum=np.dot(im.cmap(im.norm(v[i,k]))[:3],[.2126,.7152,.0722])
   ax.text(k,i,f'{v[i,k]:.3f}',ha='center',va='center',fontsize=11.5,
    fontweight='bold' if NAMES[i]=='CarcPNU' else 'normal',
    color='white' if lum<.5 else '#203347')
 ax.add_patch(Rectangle((-.5,len(NAMES)-1.5),4,1,fill=False,edgecolor=C['pnu'],lw=2,clip_on=False,zorder=5))
 if j==0:
  ax.get_yticklabels()[-1].set_color(C['pnu']);ax.get_yticklabels()[-1].set_fontweight('bold')
 f.text(x,.405,'BC'[j]+'  '+MET[j]+' by organ',fontsize=13,fontweight='bold',va='bottom')
cb=f.colorbar(im,cax=f.add_axes([.947,.055,.012,.325]),ticks=[.4,.6,.8,1])
cb.ax.tick_params(labelsize=9,length=2);cb.outline.set_visible(False)
cb.ax.set_title('Score',fontsize=9,pad=9)
for ext in ['pdf','png','svg']:
 name='fig1_benchmark_radar'+('_manuscript' if ext=='pdf' else '')+'.'+ext
 f.savefig(OUT/name,dpi=240,bbox_inches='tight',pad_inches=.08)
plt.close(f)

# Former main panels A/B are retained as Figure S2, with unchanged point values.
f=plt.figure(figsize=(12.4,5.8))
gs=f.add_gridspec(1,2,left=.225,right=.96,bottom=.13,top=.88,wspace=.25)
mac=np.array([values(n) for n in NAMES])
colors=[C['pnu'] if n=='CarcPNU' else C['base'] for n in NAMES]
for j in range(2):
 ax=f.add_subplot(gs[0,j]);yy=np.arange(len(NAMES))[::-1]
 ax.axhspan(-.4,.4,color=C['pnu'],alpha=.07,zorder=0)
 ax.scatter(mac[:,j],yy,c=colors,s=50,zorder=3)
 for v,y in zip(mac[:,j],yy):ax.annotate(f'{v:.4f}',(v,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=11)
 ax.set_yticks(yy,NAMES if j==0 else ['']*len(NAMES),fontsize=11)
 if j==0:ax.get_yticklabels()[-1].set_fontweight('bold');ax.get_yticklabels()[-1].set_color(C['pnu'])
 ax.set_ylim(-.7,len(NAMES)-.3)
 ax.set_xlim(*((.50,.88) if j==0 else (.75,1.015)))
 ax.set_xlabel('Macro '+MET[j],fontsize=12)
 ax.set_xticks([.5,.6,.7,.8] if j==0 else [.75,.8,.85,.9,.95,1])
 ax.tick_params(axis='x',labelsize=10);ax.tick_params(axis='y',length=0,pad=9)
 ax.grid(axis='x',color='#e7edf2');ax.set_axisbelow(True)
 ax.set_title('AB'[j]+'  Macro '+MET[j],loc='left',fontweight='bold',fontsize=13,pad=16)
for ext in ['pdf','png','svg']:
 name='figS2_benchmark_macro'+('_manuscript' if ext=='pdf' else '')+'.'+ext
 f.savefig(OUT/name,dpi=240,bbox_inches='tight',pad_inches=.08)
plt.close(f)

