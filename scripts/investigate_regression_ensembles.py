"""Investigate frozen ensemble inputs; produce three candidates without base training."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from create_regression_ensembles import fit, predict, rmse

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'artifacts/regression_ensembles_20261014'
OUT = ROOT / 'artifacts/regression_ensembles_followup_20261014'
DELIVERY = Path.home() / 'Downloads/shl_ensembles_followup_20261014'


def bagged_fit(x, y, groups):
    generator = np.random.default_rng(20261014)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    fits = []
    for _ in range(40):
        sampled = generator.choice(unique, len(unique), replace=True)
        rows = np.concatenate([indices[g] for g in sampled])
        fits.append(fit(x[rows], y[rows], 'elasticnet', .003))
    return {'weights': np.mean([m['weights'] for m in fits], axis=0),
            'intercept': float(np.mean([m['intercept'] for m in fits]))}


def quadratic_features(x):
    level = (x[:, 0] - 2.5) / 2.5
    differences = x[:, 1:] - x[:, [0]]
    return np.column_stack([x, level ** 2, level[:, None] * differences])


def quadratic_fit(x, y, alpha):
    model = Ridge(alpha=alpha).fit(quadratic_features(x), y - x[:, 0])
    return model


def quadratic_predict(model, x):
    return np.clip(x[:, 0] + model.predict(quadratic_features(x)), 0, 5)


def main():
    OUT.mkdir(parents=True, exist_ok=True); DELIVERY.mkdir(parents=True, exist_ok=True)
    original = json.loads((PREVIOUS / 'ensemble_report.json').read_text())
    documents = []
    for source in original['sources']:
        p = PREVIOUS / 'source_metadata' / source['filename']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == source['sha256']
        documents.append(json.loads(p.read_text()))
    rows = documents[0]['oof_predictions']; names = [r['filename'] for r in rows]
    test_names = [r['filename'] for r in documents[0]['submission_predictions']]
    y = np.array([r['true_score'] for r in rows]); folds = np.array([r['fold'] for r in rows])
    groups = np.array([r['source_group'] for r in rows])
    x = np.column_stack([[{r['filename']: r['predicted_score'] for r in d['oof_predictions']}[n]
                         for n in names] for d in documents])
    test = np.column_stack([[{r['filename']: r['predicted_score'] for r in d['submission_predictions']}[n]
                            for n in test_names] for d in documents])
    outer = [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in sorted(set(folds))]
    results = []

    def save(filename, values, development, comment, detail):
        target = OUT / filename
        pd.DataFrame({'filename': test_names, 'label': values}).to_csv(target, index=False)
        reread = pd.read_csv(target)
        assert list(reread.columns) == ['filename', 'label']
        assert len(reread) == 216 and reread.filename.tolist() == test_names and reread.filename.is_unique
        assert np.isfinite(reread.label).all() and reread.label.between(0, 5).all()
        assert np.allclose(values, reread.label, atol=1e-12)
        shutil.copy2(target, DELIVERY / filename)
        result = {'file': filename, 'comment': comment, 'meta_development_rmse': rmse(y, development),
                  'positive_label_rmse': rmse(y[y > 0], development[y > 0]),
                  'zero_label_rmse': rmse(y[y == 0], development[y == 0]),
                  'fold_rmse': [rmse(y[v], development[v]) for _, v in outer],
                  'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'details': detail}
        results.append(result)
        pd.DataFrame({'filename': names, 'label': y, 'fold': folds,
                      'meta_development_prediction': development}).to_csv(OUT / filename.replace('.csv', '_development.csv'), index=False)
        print(json.dumps({k: result[k] for k in ['file', 'meta_development_rmse', 'positive_label_rmse', 'zero_label_rmse']}), flush=True)

    elastic = pd.read_csv(PREVIOUS / '03_elasticnet_regression.csv').set_index('filename').label.reindex(test_names).to_numpy()
    ridge = pd.read_csv(PREVIOUS / '02_ridge_regression.csv').set_index('filename').label.reindex(test_names).to_numpy()
    elastic_dev = pd.read_csv(PREVIOUS / 'elasticnet_development.csv').set_index('filename').meta_development_prediction.reindex(names).to_numpy()
    ridge_dev = pd.read_csv(PREVIOUS / 'ridge_development.csv').set_index('filename').meta_development_prediction.reindex(names).to_numpy()
    save('01_elastic_ridge_equal_blend.csv', .5 * elastic + .5 * ridge, .5 * elastic_dev + .5 * ridge_dev,
         'Equal blend of the two best observed public-scoring candidates; fixed 50/50 weights, no leaderboard weight search.',
         {'elasticnet_weight': .5, 'ridge_weight': .5,
          'selection_note': 'The component choice uses the user-reported leaderboard ranking; this is leaderboard-informed selection.'})

    bag_dev = np.empty(len(y))
    for train, valid in outer:
        model = bagged_fit(x[train], y[train], groups[train]); bag_dev[valid] = predict(model, x[valid])
    model = bagged_fit(x, y, groups)
    save('02_bagged_elasticnet.csv', predict(model, test), bag_dev,
         'Average of 40 Elastic Net regressions bootstrapped by source group, to stabilize regression coefficients.',
         {'alpha': .003, 'l1_ratio': .5, 'bootstrap_fits': 40, 'seed': 20261014,
          'weights': model['weights'].tolist(), 'intercept': model['intercept'],
          'note': 'Alpha is frozen from the previous batch, not an independently selected hyperparameter.'})

    alphas = [100., 30., 10.]
    quad_dev = np.empty(len(y)); choices = []
    for train, valid in outer:
        inner = list(GroupKFold(n_splits=3).split(x[train], y[train], groups[train]))
        scores = []
        for alpha in alphas:
            p = np.empty(len(train))
            for inside, heldout in inner:
                m = quadratic_fit(x[train][inside], y[train][inside], alpha)
                p[heldout] = quadratic_predict(m, x[train][heldout])
            scores.append(rmse(y[train], p))
        minimum = min(scores); alpha = next(a for a, s in zip(alphas, scores) if s <= minimum + .0005)
        quad_dev[valid] = quadratic_predict(quadratic_fit(x[train], y[train], alpha), x[valid]); choices.append(alpha)
    scores = []
    for alpha in alphas:
        p = np.empty(len(y))
        for train, valid in outer:
            p[valid] = quadratic_predict(quadratic_fit(x[train], y[train], alpha), x[valid])
        scores.append(rmse(y, p))
    minimum = min(scores); alpha = next(a for a, s in zip(alphas, scores) if s <= minimum + .0005)
    model = quadratic_fit(x, y, alpha)
    save('03_smooth_quadratic_regression.csv', quadratic_predict(model, test), quad_dev,
         'Experimental regularized score-dependent correction: a quadratic level term and level-by-model-disagreement interactions.',
         {'alpha': alpha, 'inner_selected_alpha_by_fold': choices,
          'coefficient': model.coef_.tolist(), 'intercept': float(model.intercept_),
          'feature_recipe': 'original predictions; ((newest-2.5)/2.5)^2; ((newest-2.5)/2.5)*(other-newest)'})

    # A small diversity ablation compares held-out meta predictions with a fixed ridge penalty.
    ablations = []
    for subset in [[0], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6], list(range(x.shape[1]))]:
        p = np.empty(len(y))
        for train, valid in outer:
            p[valid] = predict(fit(x[train][:, subset], y[train], 'ridge', 10.), x[valid][:, subset])
        ablations.append({'models': len(subset), 'source_indices': subset, 'meta_development_rmse': rmse(y, p)})
    correlations = np.corrcoef((x - y[:, None]).T)
    report = {'previous_batch_public_scores': {
        '03_elasticnet_regression.csv': .3204, '02_ridge_regression.csv': .3214,
        '04_huber_regression.csv': .3242, '01_convex_regression.csv': .3263,
        '05_hillclimb_regression_blend.csv': .3267},
        'leaderboard_evidence': 'User confirmation and screenshot, 14 October 2026.',
        'strong_four_error_correlations': correlations[:4, :4].tolist(),
        'source_count': len(documents), 'source_hashes': original['sources'],
        'ridge_diversity_ablations': ablations, 'results': results,
        'limitations': original['limitations'] + [
            'The observed 0.3204 remains the best verified public score; follow-up scores are unknown.',
            'Candidates reuse the same development set; nonlinear model search can further overfit.',
            'Bagging alpha comes from prior development selection. The equal blend components were chosen after observing public scores.',
            'No new base model training, model checkpoint writing or Kaggle upload was performed.']}
    (OUT / 'investigation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    shutil.copy2(OUT / 'investigation_report.json', DELIVERY / 'investigation_report.json')
    (DELIVERY / 'COMMENTS.md').write_text('\n\n'.join(
        f"{i}. {r['file']}: {r['comment']} Meta-development RMSE {r['meta_development_rmse']:.6f}. Public score unknown."
        for i, r in enumerate(results, 1)) + '\n\n4. Retain the verified 0.3204 Elastic Net candidate. These development diagnostics are not independent or fully nested validation; no improvement is guaranteed.\n')
    print('ABLATIONS', json.dumps(ablations), flush=True)
    print('DELIVERY', DELIVERY, flush=True)


if __name__ == '__main__':
    main()
