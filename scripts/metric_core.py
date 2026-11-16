"""Exact archived weighted metric implementation; handles ties and missing labels."""
import numpy as np

def weighted_metrics(y, p, counts):
    keep=y>=0; y=y[keep]; p=p[keep]; w=counts[:,keep].astype(float)
    positives=w@(y==1); negatives=w@(y==0)
    valid=(positives>0)&(negatives>0)
    order=np.argsort(p,kind='stable')
    starts=np.r_[0,np.flatnonzero(np.diff(p[order])!=0)+1]
    pos=np.add.reduceat(w[:,order]*(y[order]==1)[None],starts,axis=1)
    neg=np.add.reduceat(w[:,order]*(y[order]==0)[None],starts,axis=1)
    denom=positives*negatives
    auc=np.divide((pos*(np.cumsum(neg,axis=1)-.5*neg)).sum(1),denom,
                  out=np.full(len(w),np.nan),where=denom>0)
    pos,neg=pos[:,::-1],neg[:,::-1]
    precision_curve=np.divide(np.cumsum(pos,axis=1),np.cumsum(pos+neg,axis=1),
                              out=np.zeros_like(pos),where=np.cumsum(pos+neg,axis=1)>0)
    ap=np.divide((pos*precision_curve).sum(1),positives,out=np.full(len(w),np.nan),where=positives>0)
    ap[~valid]=np.nan
    hard=p>=.45
    tp=w@((y==1)&hard); fn=positives-tp; fp=w@((y==0)&hard); tn=negatives-fp
    div=lambda a,b,zero=0.:np.divide(a,b,out=np.full_like(a,zero),where=b>0)
    sens=div(tp,positives,np.nan); spec=div(tn,negatives,np.nan)
    mcc=div(tp*tn-fp*fn,np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    mcc[~valid]=np.nan
    return np.stack([auc,ap,div(tp+tn,positives+negatives,np.nan),div(2*tp,2*tp+fp+fn),
                     div(tp,tp+fp),sens,spec,.5*(sens+spec),mcc],axis=-1)
