"""Learn zero/nonzero gating and positive residual regression inside grouped CV."""
from pathlib import Path
import sys,json,itertools,time
import numpy as np
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.preprocessing import StandardScaler,SplineTransformer,PolynomialFeatures
from sklearn.model_selection import GroupKFold
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/rmse_research_20261015'


def rmse(y,p):return float(np.sqrt(np.mean((y-np.clip(p,0,5))**2)))


def predict(a,y,b,p):
    scale=StandardScaler().fit(a)
    x,z=scale.transform(a),scale.transform(b)
    # Train a binary model; its output is used identically for held-out and test rows.
    gate=LogisticRegression(C=p['C'],max_iter=2000).fit(x,y>0)
    probability=gate.predict_proba(z)[:,1]
    keep=y>0;target=y[keep]-a[keep,0]
    if p['family']=='hurdle_kernel':
        xx=a[keep];zz=b
        k=np.exp(-p['gamma']*cdist(xx,xx,'sqeuclidean'));q=np.exp(-p['gamma']*cdist(zz,xx,'sqeuclidean'))
        mean=k.mean(0);grand=mean.mean()
        q=q-q.mean(1)[:,None]-mean[None,:]+grand
        k=k-mean[:,None]-mean[None,:]+grand;k.flat[::len(k)+1]+=p['alpha']
        coef=cho_solve(cho_factor(k,lower=True,check_finite=False),target-target.mean(),check_finite=False)
        positive=b[:,0]+target.mean()+q@coef
    else:
        transform=PolynomialFeatures(degree=p['degree'],include_bias=False)
        xx=transform.fit_transform(a[keep]/5);zz=transform.transform(b/5)
        scaler=StandardScaler().fit(xx);xx=scaler.transform(xx);zz=scaler.transform(zz)
        positive=b[:,0]+Ridge(alpha=p['alpha']).fit(xx,target).predict(zz)
    return np.clip(probability*np.clip(positive,0,5),0,5)


def main():
    d=np.load(OUT/'inputs.npz');x,test,y,folds,groups=[d[k] for k in ['x','test','y','folds','groups']]
    def tune(a,y,g,family):
        split=list(GroupKFold(3).split(a,y,g));records=[]
        if family=='hurdle_kernel':params=[dict(family=family,alpha=r,gamma=h,C=c) for r,h,c in itertools.product([.3,3.,30.],[.03,.3],[.1,1.,10.])]
        else:params=[dict(family=family,alpha=r,degree=v,C=c) for r,v,c in itertools.product([10.,100.,1000.],[1,2],[.1,1.,10.])]
        for p in params:
            pred=np.empty(len(y))
            for tr,va in split:pred[va]=predict(a[tr],y[tr],a[va],p)
            records.append((rmse(y,pred),p,pred))
        s,p,inside=min(records,key=lambda r:r[0]);return p,inside,dict(selected=p,inner_rmse=s,candidates=len(records))
    dev={};full={};report={}
    for family in ['hurdle_regression','hurdle_kernel']:
        pred=np.empty(len(y));fits=[]
        for f in sorted(set(folds)):
            tr=np.flatnonzero(folds!=f);va=np.flatnonzero(folds==f)
            start=time.monotonic();p,inside,record=tune(x[tr],y[tr],groups[tr],family);pred[va]=predict(x[tr],y[tr],x[va],p)
            fits.append(record);np.savez(OUT/f'outer{f}_{family}.npz',prediction=pred[va],inside=inside)
            print('FOLD',f,family,rmse(y[va],pred[va]),round(time.monotonic()-start,1),flush=True)
        p,inside,record=tune(x,y,groups,family);t=predict(x,y,test,p)
        np.savez(OUT/f'full_{family}.npz',prediction=t,inside=inside)
        dev[family]=pred;full[family]=t
        report[family]=dict(rmse=rmse(y,pred),outer_fits=fits,full_fit=record)
        print('SCREEN',family,report[family]['rmse'],flush=True)
    np.savez(OUT/'hurdle_development_predictions.npz',**dev);np.savez(OUT/'hurdle_test_predictions.npz',**full)
    (OUT/'hurdle_research_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
