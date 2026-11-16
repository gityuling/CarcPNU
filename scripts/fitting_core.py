"""Exact extracted mathematical functions; original source hashes in manifest."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
TASKS = ['liv','lun','sto','mgl']
def tensor(x):
    return torch.as_tensor(x,dtype=torch.float64,device='cuda')
def fit_convex(x, a, b, penalty):
    """Damped Newton with unpenalized intercept (1e-10 numerical ridge)."""
    design = torch.cat([x, torch.ones_like(x[:, :1])], 1)
    coefficients, diagnostics = [], []
    for t in range(a.shape[1]):
        aa, bb = a[:, t], b[:, t]
        regularizer = torch.eye(design.shape[1], device=x.device, dtype=x.dtype) * penalty
        regularizer[-1, -1] = 1e-10
        w = torch.zeros(design.shape[1], device=x.device, dtype=x.dtype)
        w[-1] = torch.logit(bb.sum() / aa.sum())

        def objective(v):
            logits = design @ v
            return (aa * torch.nn.functional.softplus(logits) - bb * logits).sum() + .5*v@regularizer@v

        for iteration in range(100):
            probability = (design@w).sigmoid()
            gradient = design.T@(aa*probability-bb) + regularizer@w
            if gradient.abs().max().item() < 1e-9:
                break
            hessian = design.T@(design*(aa*probability*(1-probability))[:, None]) + regularizer
            step = torch.linalg.solve(hessian, gradient)
            old, directional = objective(w), gradient@step
            fraction = 1.
            for _ in range(35):
                candidate = w-fraction*step
                if objective(candidate) <= old-1e-4*fraction*directional:
                    w = candidate
                    break
                fraction *= .5
            else:
                raise RuntimeError(f'Newton line search failed at task {t}')
        else:
            raise RuntimeError(f'Newton did not converge at task {t}')
        assert torch.isfinite(w).all()
        logits = design@w
        data_risk = (aa*torch.nn.functional.softplus(logits)-bb*logits).sum()
        coefficients.append(w)
        diagnostics.append({'task': TASKS[t], 'iterations': iteration+1,
            'gradient_max': gradient.abs().max().item(), 'data_risk': data_risk.item(),
            'regularized_objective': objective(w).item(), 'linear_mass': bb.sum().item()})
    return torch.stack(coefficients, 1), diagnostics

def predict(x, coef):
    return (torch.cat([x, torch.ones_like(x[:, :1])], 1)@coef).sigmoid().detach().cpu().numpy()

def masses(roles, c):
    """Mean-risk coefficients; vector priors permit per-organ oracle diagnostics."""
    a=np.zeros(roles.shape);b=np.zeros(roles.shape)
    method=c['method']; p=roles==1;n=roles==0;u=(roles==2)|(roles==3)
    for t in range(4):
        pp,nn,uu=p[:,t],n[:,t],u[:,t]
        assert pp.any() and nn.any()
        if method in ['supervised','downweight_all_u','downweight_implicit','pn_empirical']:
            w=(pp|nn|uu).astype(float)
            if method=='downweight_all_u':w[uu]=c['weight']
            if method=='downweight_implicit':w[roles[:,t]==2]=c['weight']
            if method=='pn_empirical':w[uu]=0
            a[:,t]=w/w.sum();b[:,t]=a[:,t]*pp
        else:
            assert uu.any() or c['eta']==0
            pi=c['prior'][t] if isinstance(c['prior'],list) else c['prior']
            eta=c['eta']
            a[pp,t]=(1-eta)*pi/pp.sum()
            a[nn,t]=(1-eta)*(1-pi)/nn.sum()
            if uu.any():a[uu,t]=eta/uu.sum()
            b[pp,t]=pi/pp.sum()
            if method=='matched_curvature_bce':b[pp,t]=a[pp,t]
    assert np.all(a>=0) and np.allclose(a.sum(0),1)
    assert np.all((b.sum(0)>0)&(b.sum(0)<1))
    return tensor(a),tensor(b)

def metrics(y, predictions):
    if predictions.ndim==2:predictions=predictions[None]
    values=[]
    for pred in predictions:
        organ=[]
        for t in range(4):
            m=y[:,t]>=0
            if len(np.unique(y[m,t]))!=2:organ.append([np.nan,np.nan]);continue
            organ.append([roc_auc_score(y[m,t],pred[m,t]),average_precision_score(y[m,t],pred[m,t])])
        values.append(organ)
    v=np.mean(values,axis=0)
    return v

def macro_auc(y,pp):
    return float(metrics(y,pp)[:,0].mean())

def choose(records, method, **filters):
    eligible=[r for r in records if r['config']['method']==method and all(r['config'].get(k)==v for k,v in filters.items())]
    assert eligible,(method,filters)
    # One score for all four organs; larger ridge then predeclared grid index.
    return max(eligible,key=lambda r:(r['validation_auc'] ,r['config']['penalty'],-r['index']))['index']

def numerical_retry(x,a,b,penalty):
    """Same convex objective, with roundoff allowance in the Armijo test.

    Called only if the historical solver fails; retain its 1e-9 gradient target.
    """
    design=torch.cat([x,torch.ones_like(x[:,:1])],1);coefs=[];diags=[]
    reg=torch.eye(design.shape[1],device=x.device,dtype=x.dtype)*penalty;reg[-1,-1]=1e-10
    for t in range(4):
        aa,bb=a[:,t],b[:,t];w=torch.zeros(design.shape[1],device=x.device,dtype=x.dtype)
        w[-1]=torch.logit(bb.sum()/aa.sum())
        def obj(z):
            g=design@z
            return (aa*torch.nn.functional.softplus(g)-bb*g).sum()+.5*z@reg@z
        for iteration in range(200):
            p=(design@w).sigmoid();grad=design.T@(aa*p-bb)+reg@w
            if float(grad.abs().max())<1e-9:break
            h=design.T@(design*(aa*p*(1-p))[:,None])+reg
            step=torch.linalg.solve(h,grad);old=obj(w);direction=grad@step
            for back in range(50):
                fraction=.5**back;candidate=w-fraction*step
                if obj(candidate)<=old-1e-4*fraction*direction+2e-14*torch.maximum(old.abs(),torch.ones_like(old)):
                    w=candidate;break
            else:raise RuntimeError('Numerical retry line search failed')
        else:raise RuntimeError('Numerical retry did not converge')
        coefs.append(w);diags.append({'gradient_max':float(grad.abs().max()),'iterations':iteration+1})
    return torch.stack(coefs,1),diags

def fit_grid(x,xv,yv,roles,configs,graph=None):
    records=[];coefs=[];valpred=[]
    for i,c in enumerate(configs):
        a,b=masses(roles,c);retry=None
        try:
            coef,diag=fit_convex(x,a,b,c['penalty'])
        except RuntimeError as exc:
            retry=str(exc);coef,diag=numerical_retry(x,a,b,c['penalty'])
            print(json.dumps({'stage':'numerical_retry','config':c,'original_error':retry,'max_gradient':max(d['gradient_max'] for d in diag)}),flush=True)
        pp=predict(xv,coef)
        score=macro_auc(yv,pp if graph is None else .25*graph+.75*pp[None])
        records.append({'index':i,'config':c,'validation_auc':score,'max_gradient':max(d['gradient_max'] for d in diag),'numerical_retry':retry})
        coefs.append(coef.cpu().numpy());valpred.append(pp)
    return records,np.stack(coefs),np.stack(valpred)
