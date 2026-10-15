"""RMSE-only grouped meta-learning research. No backbone training or label lookups."""
from pathlib import Path
import sys
sys.path.insert(0, '/tmp/shl_ensemble_optional_catboost')
import hashlib
import itertools
import json
import time
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler, SplineTransformer, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits
import ensemble_latest_metadata_20261015 as sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/rmse_research_20261015'
DELIVERY = Path.home() / 'Downloads/shl_rmse_research_20261015'


def rmse(y, p):
    return float(np.sqrt(np.mean((y - np.clip(p, 0, 5)) ** 2)))


def representation(a, b, kind):
    # Fit every data-dependent transform inside the current training partition.
    if kind == 'raw':
        return a, b
    if kind == 'relative':
        return np.column_stack([a[:, 0], a[:, 1:] - a[:, [0]]]), np.column_stack([b[:, 0], b[:, 1:] - b[:, [0]]])
    if kind == 'quadratic':
        tr = PolynomialFeatures(degree=2, include_bias=False)
        return tr.fit_transform(a / 5), tr.transform(b / 5)
    if kind == 'spline':
        tr = SplineTransformer(n_knots=4, degree=2, include_bias=False, extrapolation='linear')
        x = tr.fit_transform(a)
        z = tr.transform(b)
        return np.column_stack([a, x, (a[:, [0]]-2.5)*(a[:, 1:]-a[:, [0]])]), np.column_stack([b, z, (b[:, [0]]-2.5)*(b[:, 1:]-b[:, [0]])])
    raise ValueError(kind)


def kernel_prediction(a, y, b, gene):
    # Gene: log regularization, log bandwidth, and four semantic source-group scales.
    buckets = np.array([0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 0, 3])
    weights = 10 ** np.asarray(gene[2:])[buckets]
    x = a * weights
    z = b * weights
    gamma, alpha = 10**gene[1], 10**gene[0]
    k = np.exp(-gamma * cdist(x, x, 'sqeuclidean'))
    q = np.exp(-gamma * cdist(z, x, 'sqeuclidean'))
    avg = k.mean(0)
    grand = avg.mean()
    q = q - q.mean(1)[:, None] - avg[None, :] + grand
    k = k - avg[:, None] - avg[None, :] + grand
    k.flat[::len(k)+1] += alpha
    residual = y-a[:, 0]
    coef = cho_solve(cho_factor(k, lower=True, check_finite=False), residual-residual.mean(), check_finite=False)
    return np.clip(b[:, 0]+residual.mean()+q@coef, 0, 5)


def predict(a, y, b, p):
    family = p['family']
    if family == 'evolved_kernel':
        return kernel_prediction(a, y, b, p['gene'])
    count = p.get('top', a.shape[1])
    idx = np.argsort(np.mean((a-y[:, None])**2, axis=0))[:count]
    # Preserve x0 anchor separately even if model screening excludes it.
    x, z = representation(a[:, idx], b[:, idx], p.get('features', 'raw'))
    residual = p.get('residual', True)
    target = y-a[:, 0] if residual else y
    if family in ['ridge', 'spline', 'svr']:
        scale = StandardScaler().fit(x)
        x, z = scale.transform(x), scale.transform(z)
    if family in ['ridge', 'spline']:
        model = Ridge(alpha=p['alpha'])
    elif family == 'svr':
        model = SVR(C=p['C'], epsilon=p['epsilon'], gamma=p['gamma'], cache_size=300)
    elif family == 'extra_trees':
        model = ExtraTreesRegressor(n_estimators=160, min_samples_leaf=p['leaf'], max_features=p['fraction'], max_depth=p['depth'], random_state=7315, n_jobs=1)
    elif family == 'hist_boost':
        model = HistGradientBoostingRegressor(max_iter=p['iterations'], max_leaf_nodes=p['leaves'], min_samples_leaf=p['leaf'], l2_regularization=p['l2'], learning_rate=.035, early_stopping=False, random_state=7315)
    elif family == 'catboost':
        model = CatBoostRegressor(iterations=p['iterations'], depth=p['depth'], learning_rate=.035, l2_leaf_reg=p['l2'], loss_function='RMSE', boosting_type='Ordered', random_seed=7315, thread_count=1, verbose=False, allow_writing_files=False)
    else:
        raise ValueError(family)
    model.fit(x, target)
    return np.clip(model.predict(z)+(b[:, 0] if residual else 0), 0, 5)


def grids(family):
    if family == 'ridge':
        return [dict(family=family, features=f, alpha=a, top=t) for f, a, t in itertools.product(['raw', 'quadratic'], [1., 10., 100., 1000.], [7, 14])]
    if family == 'spline':
        return [dict(family=family, features='spline', alpha=a, top=t) for a, t in itertools.product([1., 10., 100., 1000.], [4, 7, 14])]
    if family == 'svr':
        return [dict(family=family, features='relative', C=c, gamma=g, epsilon=e, top=t) for c, g, e, t in itertools.product([.1, 1., 10.], [.01, .1], [.05, .2], [7, 14])]
    if family == 'extra_trees':
        return [dict(family=family, leaf=l, fraction=f, depth=d) for l, f, d in itertools.product([10, 30], [.7, 1.], [4, 8])]
    if family == 'hist_boost':
        return [dict(family=family, iterations=n, leaves=l, leaf=30, l2=r) for n, l, r in itertools.product([100, 300], [3, 7], [10., 50.])]
    if family == 'catboost':
        return [dict(family=family, iterations=n, depth=d, l2=r) for n, d, r in itertools.product([300, 700], [2, 4], [10., 50.])]
    raise ValueError(family)


def tune(a, y, groups, family):
    split = list(GroupKFold(n_splits=3).split(a, y, groups))
    if family == 'evolved_kernel':
        def objective(gene):
            pred = np.empty(len(y))
            for tr, va in split:
                pred[va] = kernel_prediction(a[tr], y[tr], a[va], gene)
            # Optimize MSE only. Bounds constrain the model, without a ranking objective.
            return float(np.mean((y-pred)**2))
        opt = differential_evolution(objective, [(-1., 2.), (-2., .3)]+[(-.6, .6)]*4,
                                     seed=20261015, maxiter=30, popsize=6, tol=1e-4, polish=False, workers=1)
        best = dict(family=family, gene=opt.x.tolist())
        score = np.sqrt(opt.fun)
        details = dict(evaluations=opt.nfev, converged=bool(opt.success), message=str(opt.message))
    else:
        records = []
        for p in grids(family):
            pred = np.empty(len(y))
            for tr, va in split:
                assert not set(groups[tr]).intersection(groups[va])
                pred[va] = predict(a[tr], y[tr], a[va], p)
            records.append((rmse(y, pred), p))
        score, best = min(records, key=lambda r: r[0])
        details = dict(candidates=len(records), ranking=[dict(rmse=s, params=p) for s, p in sorted(records, key=lambda r:r[0])])
    # Inner held-out predictions for fitting a combiner without outer-label access.
    inside = np.empty(len(y))
    for tr, va in split:
        inside[va] = predict(a[tr], y[tr], a[va], best)
    return best, inside, dict(selected=best, inner_rmse=float(score), **details)


def genetic_blend(a, y):
    # Evolve a smooth convex blend and small global affine adjustment, using MSE only.
    from scipy.special import softmax
    def prediction(g):
        return np.clip((a@softmax(g[:-2]))*g[-2]+g[-1], 0, 5)
    def objective(g):
        return float(np.mean((y-prediction(g))**2))
    opt = differential_evolution(objective, [(-3., 3.)]*a.shape[1]+[(.9, 1.1), (-.25, .25)], seed=315, maxiter=80, popsize=6, tol=1e-6, polish=True)
    return dict(weights=softmax(opt.x[:-2]).tolist(), slope=float(opt.x[-2]), intercept=float(opt.x[-1]), inner_rmse=float(np.sqrt(opt.fun)), evaluations=opt.nfev)


def apply_blend(a, fit):
    return np.clip((a@np.asarray(fit['weights']))*fit['slope']+fit['intercept'], 0, 5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names, tnames, y, folds, groups, x, test, audit = sources.load_sources()
    # Snapshot immutable inputs for exact reproduction and readback verification.
    np.savez(OUT/'inputs.npz', names=names, test_names=tnames, y=y, folds=folds, groups=groups, x=x, test=test)
    audit['ordered_runs'] = [{k:v for k,v in r.items() if k!='base_metrics'} for r in audit['ordered_runs']]
    (OUT/'source_inventory.json').write_text(json.dumps(audit, indent=2)+'\n')
    families = ['ridge', 'spline', 'svr', 'extra_trees', 'hist_boost', 'catboost', 'evolved_kernel']
    dev = {f:np.empty(len(y)) for f in families}
    fits = {f:[] for f in families}
    inner = {}
    split = [(np.flatnonzero(folds!=f), np.flatnonzero(folds==f)) for f in sorted(set(folds))]
    for fi, (tr, va) in enumerate(split):
        components = []
        for family in families:
            cache = OUT/f'outer{fi}_{family}.npz'
            detail = OUT/f'outer{fi}_{family}.json'
            if cache.exists() and detail.exists():
                cached=np.load(cache); pred=cached['prediction']; inside=cached['inside']; record=json.loads(detail.read_text())
            else:
                start=time.monotonic()
                param, inside, record = tune(x[tr], y[tr], groups[tr], family)
                pred = predict(x[tr], y[tr], x[va], param)
                record['elapsed_seconds']=time.monotonic()-start
                np.savez(cache, prediction=pred, inside=inside)
                detail.write_text(json.dumps(record, indent=2)+'\n')
            dev[family][va]=pred;fits[family].append(record);components.append(inside)
            print('FOLD', fi, family, 'rmse', rmse(y[va],pred), 'seconds', round(record['elapsed_seconds'],1), flush=True)
        inner[fi]=np.column_stack(components)
    # Evolved blend and per-fold-selected best learner, selected within outer train only.
    dev['genetic_blend']=np.empty(len(y));dev['adaptive_family']=np.empty(len(y));blend_fits=[]
    for fi,(tr,va) in enumerate(split):
        fit=genetic_blend(inner[fi],y[tr]);blend_fits.append(fit)
        dev['genetic_blend'][va]=apply_blend(np.column_stack([dev[f][va] for f in families]),fit)
        winner=int(np.argmin(np.mean((inner[fi]-y[tr,None])**2,axis=0)))
        dev['adaptive_family'][va]=dev[families[winner]][va]
    ranking=sorted([dict(family=f,rmse=rmse(y,p),per_fold_rmse=[rmse(y[va],p[va]) for tr,va in split]) for f,p in dev.items()],key=lambda r:r['rmse'])
    print('RANKING',json.dumps(ranking),flush=True)
    full={};full_fits={};inside=[]
    for family in families:
        cache=OUT/f'full_{family}.npz';detail=OUT/f'full_{family}.json'
        if cache.exists() and detail.exists():
            c=np.load(cache);full[family]=c['prediction'];ins=c['inside'];record=json.loads(detail.read_text())
        else:
            param,ins,record=tune(x,y,groups,family)
            full[family]=predict(x,y,test,param)
            np.savez(cache,prediction=full[family],inside=ins);detail.write_text(json.dumps(record,indent=2)+'\n')
        inside.append(ins);full_fits[family]=record
        print('FULL',family,flush=True)
    gfit=genetic_blend(np.column_stack(inside),y)
    full['genetic_blend']=apply_blend(np.column_stack([full[f] for f in families]),gfit)
    winner=min(families,key=lambda f:full_fits[f]['inner_rmse'])
    full['adaptive_family']=full[winner]
    np.savez(OUT/'development_predictions.npz',**dev)
    np.savez(OUT/'test_predictions.npz',**full)
    report=dict(ranking=ranking,outer_fits=fits,full_fits=full_fits,genetic_blend=dict(component_order=families,outer=blend_fits,full=gfit),adaptive_full_family=winner,
                objective='MSE/RMSE only; no Pearson, hardcoded labels, filename features or leaderboard-score fitting.',
                validation='Original fixed three grouped outer folds; three-fold grouped inner tuning. Parameters, feature screening, scaling and splines fit inside each training partition. Meta-learners only: frozen base OOF inputs are not fully nested base validation.',
                limitations=['Base OOF models were not retrained inside outer meta-folds; base-level dependence remains.', 'Repeated searches and ranking on reused development labels introduce selection optimism.', 'Inner cross-fitting for ensemble weighting reuses hyperparameter-selected inner predictions; no outer validation labels enter fitting.', 'Mixed historic CV/full-data/best-fold test predictions may differ in distribution from OOF inputs.', 'Public/private performance unknown; no guarantee of improvement, especially large improvement.'],source_inventory=audit)
    (OUT/'research_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RESEARCH_COMPLETE',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
