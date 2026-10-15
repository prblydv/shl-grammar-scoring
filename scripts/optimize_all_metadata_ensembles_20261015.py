"""Prediction-only, grouped optimization using every locally available metadata export."""
from pathlib import Path
import hashlib
import itertools
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/all_metadata_ensembles_20261015'
DELIVERY = Path.home() / 'Downloads/shl_all_metadata_ensembles_20261015'
PREVIOUS = ROOT / 'artifacts/optimized_ensembles_20261014'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def metrics(y, p):
    p = np.clip(p, 0., 5.)
    mse = float(np.mean((y-p)**2))
    return dict(mse=mse, rmse=float(np.sqrt(mse)), pearson=float(np.corrcoef(y, p)[0,1]))


def load_sources():
    index = OUT/'discovered_metadata_exports.json'
    if not index.exists():
        shutil.copy2('/tmp/shl_metadata_inventory.json', index)
    inventory = json.loads(index.read_text())
    original = json.loads((ROOT/'artifacts/evolutionary_ensembles_20261014/evolution_report.json').read_text())
    old_order = [s['run_id'] for s in original['sources']]
    documents = []
    for sha, entry in inventory.items():
        frozen = OUT/'source_metadata'/(sha[:12]+'_'+entry['file'])
        path = frozen if frozen.exists() else Path(entry['copies'][0])
        assert digest(path) == sha
        d = json.loads(path.read_text())
        assert len(d['oof_predictions']) == 769 and len(d['submission_predictions']) == 216
        documents.append((sha, entry, d))
    reference = next(d for _,_,d in documents if d['run_id'] == old_order[0])
    names = [r['filename'] for r in reference['oof_predictions']]
    tnames = [r['filename'] for r in reference['submission_predictions']]
    y = np.array([r['true_score'] for r in reference['oof_predictions']])
    folds = np.array([r['fold'] for r in reference['oof_predictions']])
    groups = np.array([r['source_group'] for r in reference['oof_predictions']])
    assert len(set(names)) == 769 and len(set(tnames)) == 216
    # Filenames may repeat across train/test; align each split independently.
    by_oof = {}
    for sha, entry, d in documents:
        rows = {r['filename']:r for r in d['oof_predictions']}
        test = {r['filename']:r for r in d['submission_predictions']}
        assert len(rows) == 769 and len(test) == 216
        assert set(rows) == set(names) and set(test) == set(tnames)
        assert np.array_equal([rows[n]['true_score'] for n in names], y)
        assert np.array_equal([rows[n]['fold'] for n in names], folds)
        assert np.array_equal([rows[n]['source_group'] for n in names], groups)
        p = np.array([rows[n]['predicted_score'] for n in names], dtype=np.float64)
        t = np.array([test[n]['predicted_score'] for n in tnames], dtype=np.float64)
        assert np.isfinite(p).all() and np.isfinite(t).all()
        assert np.all((p >= 0) & (p <= 5)) and np.all((t >= 0) & (t <= 5))
        assert np.isclose(metrics(y,p)['rmse'],d['scores']['pooled_oof_rmse'],atol=1e-7)
        key = hashlib.sha256(p.tobytes()).hexdigest()
        entry = dict(entry, sha256=sha, oof_sha256=key)
        by_oof.setdefault(key, []).append((entry, d, p, t))
    chosen=[]; handled=[]
    for variants in by_oof.values():
        assert len({d['run_id'] for _,d,_,_ in variants})==1
        variants.sort(key=lambda v:({'full_data':0,'cv_submission':1,'best_fold':2}.get(v[0]['action'],3),v[0]['sha256']))
        entry,d,p,t = variants[0]
        chosen.append((entry,d,p,t))
        for e,_,_,_ in variants:
            handled.append(dict(e, disposition='selected' if e['sha256']==entry['sha256'] else 'same OOF run; full-data export preferred', selected_sha256=entry['sha256']))
    chosen.sort(key=lambda v:old_order.index(v[1]['run_id']) if v[1]['run_id'] in old_order else len(old_order))
    assert len(chosen)==12 and chosen[-1][1]['run_id']=='ast_wavlm_baseplus_seed42_run01'
    x=np.column_stack([r[2] for r in chosen]);test=np.column_stack([r[3] for r in chosen])
    assert all(len(set(folds[groups==g]))==1 for g in set(groups))
    source_folder=OUT/'source_metadata';source_folder.mkdir(parents=True,exist_ok=True)
    for sha,e,d in documents:
        name=sha[:12]+'_'+e['file'];target=source_folder/name
        if not target.exists():shutil.copy2(e['copies'][0],target)
        assert digest(target)==sha
    audit=dict(metadata_paths=sum(len(e['copies']) for e in inventory.values()),distinct_exports=len(documents),unique_oof_runs=len(chosen),handled_exports=handled,
               ordered_runs=[dict(file=e['file'],run_id=d['run_id'],action=e['action'],sha256=e['sha256'],base_metrics=metrics(y,p)) for e,d,p,t in chosen],
               policy='Every metadata export checked; byte-identical copies accounted; same-OOF CV/full-data exports are one model, preferring full-data inference. All 12 unique prediction columns are retained by every new family.')
    (OUT/'metadata_inventory.json').write_text(json.dumps(audit,indent=2)+'\n')
    return names,tnames,y,folds,groups,x,test,audit


def feature_design(x, new_scale=1.):
    level=(x[:,0]-2.5)/2.5
    f=np.column_stack([x,level**2,level[:,None]*(x[:,1:]-x[:,[0]])])
    f[:,x.shape[1]-1]*=new_scale
    f[:,-1]*=new_scale
    return f


def raw_prediction(x,y,z,p):
    kind=p['kind'];scale=p['new_scale'];alpha=p['alpha']
    residual=y-x[:,0];offset=residual.mean()
    if kind=='quadratic':
        a=feature_design(x,scale);b=feature_design(z,scale)
        mean=a.mean(0);a=a-mean;b=b-mean
        gram=a.T@a;gram.flat[::len(gram)+1]+=alpha
        coef=cho_solve(cho_factor(gram,lower=True,check_finite=False),a.T@(residual-offset),check_finite=False)
        return np.clip(z[:,0]+offset+b@coef,0,5)
    a=np.column_stack([x[:,0]/2.5,x[:,1:]-x[:,[0]]])
    b=np.column_stack([z[:,0]/2.5,z[:,1:]-z[:,[0]]])
    weights=np.ones(x.shape[1]);weights[-1]=scale
    if p.get('quality',False):
        rms=np.sqrt(np.mean((x-y[:,None])**2,axis=0))
        weights[1:]*=np.clip(rms[0]/rms[1:],.35,1.25)
    a=a*weights;b=b*weights
    dd=cdist(a,a,'sqeuclidean');dt=cdist(b,a,'sqeuclidean');g=p['gamma']
    def kernel(distance):
        if kind=='matern':
            r=np.sqrt(np.maximum(3*g*distance,0));return (1+r)*np.exp(-r)
        if kind in ['multiscale','joint']:
            return sum(np.exp(-g*w*distance) for w in [.3,1.,3.])/3
        return np.exp(-g*distance)
    k=kernel(dd);q=kernel(dt)
    # Center the kernel with training statistics to fit an unpenalized intercept.
    row_mean=k.mean(0);grand=row_mean.mean()
    q=q-q.mean(1)[:,None]-row_mean[None,:]+grand
    k=k-row_mean[:,None]-row_mean[None,:]+grand
    if kind=='joint':
        f=feature_design(x,scale);ff=feature_design(z,scale);mean=f.mean(0)
        f=f-mean;ff=ff-mean
        k=f@f.T+p['kernel_weight']*k
        q=ff@f.T+p['kernel_weight']*q
    k.flat[::len(k)+1]+=alpha
    coef=cho_solve(cho_factor(k,lower=True,check_finite=False),residual-offset,check_finite=False)
    return np.clip(z[:,0]+offset+q@coef,0,5)


def grid(family):
    if family=='quadratic':
        return [dict(kind='quadratic',alpha=a,new_scale=s) for a,s in itertools.product([3.,10.,30.,100.,300.],[.25,1.])]
    if family.startswith('joint'):
        return [dict(kind='joint',quality=family.endswith('quality'),alpha=a,gamma=g,new_scale=s,kernel_weight=w)
                for a,g,s,w in itertools.product([3.,10.,30.,100.],[.03,.1,.3],[.25,1.],[1.,10.,100.])]
    return [dict(kind='rbf' if family in ['rbf','quality_rbf'] else family,quality=family=='quality_rbf',alpha=a,gamma=g,new_scale=s)
            for a,g,s in itertools.product([.3,1.,3.,10.,30.],[.03,.1,.3,1.],[.25,1.])]


def tune(x,y,groups,family):
    split=list(GroupKFold(n_splits=3).split(x,y,groups));records=[];predictions=[]
    for p in grid(family):
        pred=np.empty(len(y))
        for tr,va in split:
            assert not set(groups[tr]).intersection(groups[va])
            pred[va]=raw_prediction(x[tr],y[tr],x[va],p)
        records.append(dict(params=p,mse=metrics(y,pred)['mse']));predictions.append(pred)
    minimum=min(r['mse'] for r in records)
    eligible=[i for i,r in enumerate(records) if r['mse']<=minimum+.0002]
    index=min(eligible,key=lambda i:(-records[i]['params']['alpha'],records[i]['params'].get('kernel_weight',0),records[i]['params']['new_scale'],records[i]['params'].get('gamma',0)))
    return records[index]['params'],predictions[index],dict(selected=records[index],minimum_inner_mse=minimum,candidates=len(records))


def blend_weights(component,y):
    prior=np.repeat(1/component.shape[1],component.shape[1])
    def objective(w):return np.mean((y-component@w)**2)+.01*np.sum((w-prior)**2)
    fit=minimize(objective,prior,method='SLSQP',bounds=[(0.,1.)]*len(prior),constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],options=dict(maxiter=500,ftol=1e-12))
    assert fit.success,fit.message
    return fit.x


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    names,tnames,y,folds,groups,x,test,audit=load_sources()
    print('INVENTORY',audit['distinct_exports'],'exports',audit['unique_oof_runs'],'unique runs',flush=True)
    correlations=np.corrcoef((x-y[:,None]).T)
    print('NEW_MODEL',metrics(y,x[:,-1]),'residual correlation with best AST',correlations[0,-1],flush=True)
    old_dev=pd.read_csv(PREVIOUS/'04_quadratic_rbf_all11_development.csv').set_index('filename').meta_development_prediction.reindex(names).to_numpy()
    old_test=pd.read_csv(PREVIOUS/'04_quadratic_rbf_all11.csv').set_index('filename').label.reindex(tnames).to_numpy()
    assert np.isfinite(old_dev).all() and np.isfinite(old_test).all()
    families=['quadratic','rbf','quality_rbf','multiscale','matern','joint','joint_quality']
    outer=[(np.flatnonzero(folds!=f),np.flatnonzero(folds==f)) for f in sorted(set(folds))]
    dev={f:np.empty(len(y)) for f in families};fits={f:[] for f in families}
    inner_components={};scores=[]
    cache=OUT/'screen.npz';detail=OUT/'screen_details.json'
    if cache.exists() and detail.exists():
        record=json.loads(detail.read_text())
        assert record['source_hashes']==[s['sha256'] for s in audit['ordered_runs']]
        loaded=np.load(cache);dev={f:loaded[f] for f in families};fits=record['fits']
        inner_components={int(k):np.array(v) for k,v in record['inner_components'].items()}
    else:
        for f,(tr,va) in enumerate(outer):
            inner=[]
            for family in families:
                started=time.monotonic();param,inside,summary=tune(x[tr],y[tr],groups[tr],family)
                dev[family][va]=raw_prediction(x[tr],y[tr],x[va],param)
                fits[family].append(summary);inner.append(inside)
                print('FOLD',f+1,family,json.dumps(metrics(y[va],dev[family][va])),'seconds',round(time.monotonic()-started,1),'params',param,flush=True)
            inner_components[f]=np.column_stack(inner)
        np.savez(cache,**dev)
        detail.write_text(json.dumps(dict(source_hashes=[s['sha256'] for s in audit['ordered_runs']],fits=fits,inner_components={str(k):v.tolist() for k,v in inner_components.items()}),indent=2)+'\n')
    for family in families:
        row=dict(name=family,**metrics(y,dev[family]));scores.append(row);print('SCREEN',json.dumps(row),flush=True)
    stack=np.empty(len(y));blend_details=[]
    for f,(tr,va) in enumerate(outer):
        w=blend_weights(inner_components[f],y[tr]);stack[va]=np.column_stack([dev[k][va] for k in families])@w
        blend_details.append(dict(fold=int(folds[va][0]),weights=w.tolist()))
    dev['regularized_multikernel_stack']=stack
    dev['updated_quadratic_rbf']=.5*dev['quadratic']+.5*dev['rbf']
    dev['quadratic_multiscale']=.5*dev['quadratic']+.5*dev['multiscale']
    dev['quadratic_quality_rbf']=.5*dev['quadratic']+.5*dev['quality_rbf']
    dev['smooth_kernel_average']=.5*dev['quadratic']+.5*np.mean([dev[k] for k in ['rbf','quality_rbf','multiscale','matern']],axis=0)
    for k in list(dev):
        scores=[r for r in scores if r['name']!=k]
        scores.append(dict(name=k,**metrics(y,dev[k])))
    scores.sort(key=lambda r:r['mse'])
    baseline=metrics(y,old_dev)
    print('BASELINE',baseline,flush=True)
    print('RANKING',json.dumps(scores),flush=True)
    full={};full_details={};inner=[]
    for family in families:
        started=time.monotonic();p,inside,s=tune(x,y,groups,family)
        full[family]=raw_prediction(x,y,test,p);inner.append(inside);full_details[family]=s
        print('FULL',family,'seconds',round(time.monotonic()-started,1),'params',p,flush=True)
    w=blend_weights(np.column_stack(inner),y)
    full['regularized_multikernel_stack']=np.column_stack([full[k] for k in families])@w
    full['updated_quadratic_rbf']=.5*full['quadratic']+.5*full['rbf']
    full['quadratic_multiscale']=.5*full['quadratic']+.5*full['multiscale']
    full['quadratic_quality_rbf']=.5*full['quadratic']+.5*full['quality_rbf']
    full['smooth_kernel_average']=.5*full['quadratic']+.5*np.mean([full[k] for k in ['rbf','quality_rbf','multiscale','matern']],axis=0)
    np.savez(OUT/'full_predictions.npz',**full)
    np.savez(OUT/'development_predictions.npz',**dev)
    report=dict(inventory=audit,baseline=dict(file='04_quadratic_rbf_all11.csv',observed_public_score=.3166,development=baseline),ranking=scores,
                outer_fits=fits,full_fits=full_details,stack=dict(component_order=families,full_weights=w.tolist(),folds=blend_details),
                validation='Fixed original three grouped outer folds; inner three-fold source-group CV tunes parameters. Convex meta weights use inner cross-fitted component predictions, never outer held-out labels. Features/scales/centering fitted inside each training partition.',
                limitations=['Frozen base OOF predictions are not regenerated in fully nested base-level CV; development measures are not independent performance estimates.',
                             'Families and final candidates reuse the same development data and earlier leaderboard-informed winner. Selection optimism remains.',
                             'Some historic exports use best-fold or full-data test predictions while OOF uses held-out CV models; inference-distribution mismatch remains.',
                             'Same-OOF duplicate exports are not independent models; full-data export preferred, matching earlier winner.',
                             'No test labels, public-score tuning, backbone retraining or Kaggle uploads used. Lower development MSE does not guarantee a better public/private score.'])
    (OUT/'optimization_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('SCREEN_AND_FULL_COMPLETE',flush=True)


def finalize_outputs():
    names,tnames,y,folds,groups,x,test,audit=load_sources()
    report=json.loads((OUT/'optimization_audit.json').read_text())
    full=np.load(OUT/'full_predictions.npz');dev=np.load(OUT/'development_predictions.npz')
    old_file=PREVIOUS/'04_quadratic_rbf_all11.csv'
    previous=json.loads((PREVIOUS/'optimized_report.json').read_text())
    old_hash=next(r['sha256'] for r in previous['results'] if r['file']==old_file.name)
    assert digest(old_file)==old_hash
    old=pd.read_csv(old_file).set_index('filename').label.reindex(tnames).to_numpy()
    old_dev=pd.read_csv(PREVIOUS/'04_quadratic_rbf_all11_development.csv').set_index('filename').meta_development_prediction.reindex(names).to_numpy()
    # Verify that the user's new CSV belongs to the new metadata export.
    uploaded=Path.home()/'Downloads/submission-25.csv'
    if uploaded.exists():
        latest=pd.read_csv(uploaded)
        assert latest.filename.is_unique and set(latest.filename)==set(tnames)
        np.testing.assert_allclose(latest.set_index('filename').label.reindex(tnames),test[:,-1],atol=1e-12,rtol=0)
    recipes=[
        ('01_joint_quadratic_rbf_all12.csv',full['joint_quality'],dev['joint_quality'],
         'Jointly fitted quadratic and multiscale Gaussian residual kernel, with training-only model-quality distance weights. Lowest development MSE in this screen.',
         {'component':'joint_quality','all_unique_base_runs':12}),
        ('02_updated_quadratic_rbf_all12.csv',full['updated_quadratic_rbf'],dev['updated_quadratic_rbf'],
         'Updated version of the successful quadratic/RBF recipe: half regularized quadratic, half centered RBF residual regression, using all 12 runs including WavLM.',
         {'quadratic':.5,'centered_rbf':.5,'all_unique_base_runs':12}),
        ('03_conservative_best_blend_all12.csv',.5*old+.25*full['joint_quality']+.25*full['updated_quadratic_rbf'],
         .5*old_dev+.25*dev['joint_quality']+.25*dev['updated_quadratic_rbf'],
         'Retains 50% of the existing public-best 0.3166 submission, adding 25% of each updated all-model ensemble. Fixed conservative weights.',
         {'retained_04_quadratic_rbf_all11':.5,'joint_quality_all12':.25,'updated_quadratic_rbf_all12':.25}),
    ]
    DELIVERY.mkdir(parents=True,exist_ok=True);results=[]
    _, group_index=np.unique(groups,return_inverse=True)
    group_n=np.bincount(group_index)
    rng=np.random.default_rng(20261015)
    sampled=rng.integers(0,len(group_n),(3000,len(group_n)))
    for filename,p,d,comment,recipe in recipes:
        p=np.asarray(p);d=np.asarray(d)
        assert p.shape==(216,) and d.shape==(769,) and np.isfinite(p).all() and np.isfinite(d).all()
        assert np.all((p>=0)&(p<=5))
        frame=pd.DataFrame(dict(filename=tnames,label=p));target=OUT/filename
        frame.to_csv(target,index=False)
        reread=pd.read_csv(target)
        assert list(reread.columns)==['filename','label'] and len(reread)==216 and reread.filename.is_unique
        assert reread.filename.tolist()==tnames and reread.label.between(0,5).all()
        np.testing.assert_allclose(reread.label,p,atol=1e-12,rtol=0)
        group_delta=np.bincount(group_index,weights=(y-d)**2-(y-old_dev)**2)
        bootstrap=group_delta[sampled].sum(1)/group_n[sampled].sum(1)
        result=dict(file=filename,comment=comment,recipe=recipe,development=metrics(y,d),
            fold_rmse=[metrics(y[folds==f],d[folds==f])['rmse'] for f in sorted(set(folds))],
            positive=metrics(y[y>0],d[y>0]),zero_rmse=metrics(y[y==0],d[y==0])['rmse'],
            paired_group_bootstrap_delta_mse_95_interval=np.quantile(bootstrap,[.025,.975]).tolist(),
            bootstrap_note='Exploratory interval conditional on searched/selected models; does not account for search optimism or base-level OOF leakage.',
            public_score=None,sha256=digest(target),rms_change_from_previous_public_best=float(np.sqrt(np.mean((p-old)**2))))
        results.append(result)
        pd.DataFrame(dict(filename=names,label=y,fold=folds,source_group=groups,meta_development_prediction=d)).to_csv(OUT/filename.replace('.csv','_development.csv'),index=False)
        shutil.copy2(target,DELIVERY/filename)
        assert digest(DELIVERY/filename)==result['sha256']
        print('DELIVERED',filename,json.dumps(result['development']),flush=True)
    assert len({r['sha256'] for r in results})==3
    assert digest(old_file)==old_hash
    report['delivered']=results
    report['previous_public_best_unchanged_sha256']=old_hash
    report['method_references']=['https://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html']
    (OUT/'optimization_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('DELIVERY',DELIVERY,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        if '--finalize-only' not in sys.argv:main()
        finalize_outputs()
