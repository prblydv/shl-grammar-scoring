"""Build five prediction-only ensembles; save CSVs and auditable metadata, no weights."""
from pathlib import Path
import hashlib
import json
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / 'Downloads'
OUT = ROOT / 'artifacts/regression_ensembles_20261014'
DELIVERY = DOWNLOADS / 'shl_ensembles_20261014'


def rmse(y, p):
    return float(np.sqrt(np.mean((y - np.clip(p, 0, 5)) ** 2)))


def fit(x, y, family, parameter):
    if family == 'convex':
        target = np.zeros(x.shape[1]); target[0] = 1
        def objective(w):
            residual = x @ w - y
            return np.mean(residual ** 2) + parameter * np.sum((w - target) ** 2)
        def gradient(w):
            return 2 * x.T @ (x @ w - y) / len(y) + 2 * parameter * (w - target)
        result = minimize(objective, target, jac=gradient, method='SLSQP',
                          bounds=[(0, 1)] * len(target),
                          constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1,
                                        'jac': lambda w: np.ones_like(w)}],
                          options={'maxiter': 500, 'ftol': 1e-12})
        if not result.success:
            raise RuntimeError(result.message)
        w = np.maximum(result.x, 0); w /= w.sum()
        return {'weights': w, 'intercept': 0.0}
    if family == 'hillclimb':
        # Forward selection with replacement gives nonnegative sum-to-one weights.
        count = np.zeros(x.shape[1]); total = np.zeros(len(y))
        best_score = np.inf; best_weights = None
        for iteration in range(1, parameter + 1):
            scores = [np.mean(((total + x[:, j]) / iteration - y) ** 2)
                      for j in range(x.shape[1])]
            j = int(np.argmin(scores)); count[j] += 1; total += x[:, j]
            if scores[j] < best_score:
                best_score = scores[j]; best_weights = count.copy() / iteration
        return {'weights': best_weights, 'intercept': 0.0}
    if family == 'ridge':
        # Penalize departures from the newest model, rather than shrink scores to zero.
        model = Ridge(alpha=parameter).fit(x, y - x[:, 0])
        w = model.coef_.copy(); w[0] += 1
        return {'weights': w, 'intercept': float(model.intercept_)}
    if family == 'elasticnet':
        model = ElasticNet(alpha=parameter, l1_ratio=.5, max_iter=50000,
                           tol=1e-8, selection='cyclic').fit(x, y - x[:, 0])
        w = model.coef_.copy(); w[0] += 1
        return {'weights': w, 'intercept': float(model.intercept_)}
    if family == 'huber':
        # Robust residual regression, then partial correction to limit extrapolation.
        alpha, correction = parameter
        model = HuberRegressor(alpha=alpha, epsilon=1.5, max_iter=3000,
                               tol=1e-7).fit(x, y - x[:, 0])
        w = correction * model.coef_.copy(); w[0] += 1
        return {'weights': w, 'intercept': correction * float(model.intercept_)}
    raise ValueError(family)


def predict(model, x):
    return np.clip(x @ model['weights'] + model['intercept'], 0, 5)


def choose(x, y, family, parameters, splits):
    candidates = []
    for parameter in parameters:
        p = np.empty(len(y))
        for training, validation in splits:
            p[validation] = predict(fit(x[training], y[training], family, parameter), x[validation])
        candidates.append({'parameter': parameter, 'rmse': rmse(y, p)})
    # Within a small band, prefer the earlier, more conservative grid setting.
    minimum = min(c['rmse'] for c in candidates)
    chosen = next(c for c in candidates if c['rmse'] <= minimum + .0005)
    return chosen, candidates


def main():
    OUT.mkdir(parents=True, exist_ok=True); DELIVERY.mkdir(parents=True, exist_ok=True)
    source_dir = OUT / 'source_metadata'; source_dir.mkdir(exist_ok=True)
    paths = sorted(DOWNLOADS.glob('run_metadata*.txt'))
    docs = [(p, json.loads(p.read_text())) for p in paths]
    # One OOF feature per run: CV/full-data exports of the same run are duplicates.
    by_run = {}
    for p, d in docs:
        previous = by_run.get(d['run_id'])
        if previous is None or d['approaches']['submission']['action'] == 'full_data':
            by_run[d['run_id']] = (p, d)
    ordered = sorted(by_run.values(), key=lambda item: item[1]['scores']['pooled_oof_rmse'])
    reference = ordered[0][1]
    names = [r['filename'] for r in reference['oof_predictions']]
    test_names = [r['filename'] for r in reference['submission_predictions']]
    assert len(names) == len(set(names)) == 769
    assert len(test_names) == len(set(test_names)) == 216
    y = np.array([r['true_score'] for r in reference['oof_predictions']])
    folds = np.array([r['fold'] for r in reference['oof_predictions']])
    groups = np.array([r['source_group'] for r in reference['oof_predictions']])
    columns = []; tests = []; sources = []
    for p, d in ordered:
        lookup = {r['filename']: r for r in d['oof_predictions']}
        assert len(lookup) == len(d['oof_predictions']) and set(lookup) == set(names)
        rows = [lookup[n] for n in names]
        assert np.array_equal(y, [r['true_score'] for r in rows])
        assert np.array_equal(folds, [r['fold'] for r in rows])
        assert np.array_equal(groups, [r['source_group'] for r in rows])
        column = np.array([r['predicted_score'] for r in rows])
        assert np.isfinite(column).all()
        assert abs(rmse(y, column) - d['scores']['pooled_oof_rmse']) < 1e-8
        test_lookup = {r['filename']: r['predicted_score'] for r in d['submission_predictions']}
        assert len(test_lookup) == len(d['submission_predictions']) and set(test_lookup) == set(test_names)
        test = np.array([test_lookup[n] for n in test_names])
        assert np.isfinite(test).all() and ((test >= 0) & (test <= 5)).all()
        columns.append(column); tests.append(test)
        sources.append({'filename': p.name, 'run_id': d['run_id'],
                        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                        'action': d['approaches']['submission']['action'],
                        'pooled_oof_rmse': rmse(y, column)})
        shutil.copy2(p, source_dir / p.name)
    x = np.stack(columns, axis=1); test = np.stack(tests, axis=1)
    for group in set(groups):
        assert len(set(folds[groups == group])) == 1
    latest = pd.read_csv(DOWNLOADS / 'submission-23.csv').set_index('filename')
    assert np.allclose(latest.label.reindex(test_names), test[:, 0], atol=1e-12)
    residual_correlations = np.corrcoef((x - y[:, None]).T)
    pd.DataFrame(residual_correlations, index=[s['run_id'] for s in sources],
                 columns=[s['run_id'] for s in sources]).to_csv(OUT / 'residual_correlations.csv')
    outer = [(np.flatnonzero(folds != fold), np.flatnonzero(folds == fold))
             for fold in sorted(set(folds))]
    specifications = [
        ('convex', [.1, .03, .01, .003, 0.0], '01_convex_regression.csv',
         'Conservative nonnegative regression blend; weights sum to one, no score calibration.'),
        ('ridge', [100., 30., 10., 3., 1., .3], '02_ridge_regression.csv',
         'Regularized linear correction around the newest model; can correct shared score bias.'),
        ('elasticnet', [.03, .01, .003, .001, .0003], '03_elasticnet_regression.csv',
         'Sparse regularized correction; reduces reliance on redundant models.'),
        ('huber', [(a, c) for c in [.5, 1.] for a in [100., 10., 1.]], '04_huber_regression.csv',
         'Robust linear correction; limits the influence of extreme training residuals.'),
        ('hillclimb', [10, 25, 50, 100], '05_hillclimb_regression_blend.csv',
         'Greedy selection with replacement; nonnegative average emphasizing complementary models.'),
    ]
    results = []
    for family, parameters, filename, comment in specifications:
        development = np.empty(len(y)); fold_reports = []
        for training, validation in outer:
            inner = list(GroupKFold(n_splits=3).split(x[training], y[training], groups[training]))
            selected, _ = choose(x[training], y[training], family, parameters, inner)
            model = fit(x[training], y[training], family, selected['parameter'])
            development[validation] = predict(model, x[validation])
            fold_reports.append({'fold': int(folds[validation][0]),
                                 'parameter': selected['parameter'],
                                 'rmse': rmse(y[validation], development[validation])})
        selected, candidates = choose(x, y, family, parameters, outer)
        model = fit(x, y, family, selected['parameter'])
        predictions = predict(model, test)
        target = OUT / filename
        pd.DataFrame({'filename': test_names, 'label': predictions}).to_csv(target, index=False)
        readback = pd.read_csv(target)
        assert list(readback.columns) == ['filename', 'label']
        assert len(readback) == 216 and readback.filename.tolist() == test_names
        assert readback.filename.is_unique and readback.label.between(0, 5).all()
        assert np.isfinite(readback.label).all()
        assert np.allclose(readback.label, predictions, atol=1e-12)
        shutil.copy2(target, DELIVERY / filename)
        row = {'family': family, 'file': filename, 'comment': comment,
               'meta_development_rmse': rmse(y, development),
               'all_oof_fit_rmse_not_validation': rmse(y, predict(model, x)),
               'parameter': selected['parameter'], 'intercept': model['intercept'],
               'weights': {s['run_id']: float(w) for s, w in zip(sources, model['weights'])},
               'fold_reports': fold_reports, 'candidate_diagnostics': candidates,
               'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        results.append(row)
        pd.DataFrame({'filename': names, 'label': y, 'fold': folds,
                      'meta_development_prediction': development}).to_csv(OUT / f'{family}_development.csv', index=False)
        print(json.dumps({k: row[k] for k in ['family', 'meta_development_rmse', 'all_oof_fit_rmse_not_validation', 'parameter']}), flush=True)
    assert len({r['sha256'] for r in results}) == 5
    report = {'sources': sources, 'train_rows': 769, 'test_rows': 216,
              'latest_model_oof_rmse': rmse(y, x[:, 0]),
              'equal_average_oof_rmse': rmse(y, x.mean(axis=1)),
              'leaderboard_scores_used_to_fit_regression': False,
              'deduplicated_runs': len(sources), 'results': results,
              'limitations': [
                  'These are meta-level development checks, not fully nested base-model validation: base models for meta-training rows may have trained on outer meta-validation labels.',
                  'Model-family comparison and prior experiments reuse the development labels; reported development improvements can be optimistic.',
                  'OOF features come from held-out CV folds; test features mix CV ensembles, full-data refits, and selected-best-fold exports. Their distributions can differ.',
                  'No leaderboard improvement is guaranteed; supplied public leaderboard scores were not used as regression targets or to optimize weights.',
                  'Downloaded submission.csv differs from the archived neural ensemble; neither is included as an unverified 0.3316-scoring feature.',
              ]}
    report_path = OUT / 'ensemble_report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(report_path, DELIVERY / report_path.name)
    ranked = sorted(results, key=lambda r: r['meta_development_rmse'])
    text = '# Five regression ensemble candidates\n\n'
    for i, row in enumerate(ranked, 1):
        text += f"{i}. `{row['file']}` — {row['comment']} Meta-development RMSE: {row['meta_development_rmse']:.6f}; all-OOF fit: {row['all_oof_fit_rmse_not_validation']:.6f} (in-sample).\n\n"
    text += '6. Development diagnostics are optimistic and are not an independent or fully nested estimate of leaderboard performance. Full-data and CV test outputs are mixed. All CSVs contain only filename,label and 216 finite predictions bounded to [0,5]. No submissions were uploaded.\n'
    (DELIVERY / 'COMMENTS.md').write_text(text)
    print('DELIVERY', DELIVERY, flush=True)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        main()
