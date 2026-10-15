"""Prediction-only ensembles from all downloaded metadata, with frozen source audits."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import optimize_all_metadata_ensembles_20261015 as engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/latest_metadata_ensembles_20261015'
DELIVERY = Path.home() / 'Downloads/shl_latest_ensembles_20261015'
OLD = ROOT / 'artifacts/all_metadata_ensembles_20261015/metadata_inventory.json'


def load_sources():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = OUT / 'source_metadata'
    frozen.mkdir(exist_ok=True)
    order = [r['run_id'] for r in json.loads(OLD.read_text())['ordered_runs']]
    docs = []
    for path in sorted((Path.home() / 'Downloads').glob('run_metadata*.txt')):
        sha = engine.digest(path)
        target = frozen / (sha[:12] + '_' + path.name)
        if not target.exists():
            shutil.copy2(path, target)
        assert engine.digest(target) == sha
        d = json.loads(target.read_text())
        docs.append((path, sha, d))
    ref = next(d for _, _, d in docs if d['run_id'] == order[0])
    names = [r['filename'] for r in ref['oof_predictions']]
    tnames = [r['filename'] for r in ref['submission_predictions']]
    assert len(set(names)) == len(names) == 769
    assert len(set(tnames)) == len(tnames) == 216
    y = np.array([r['true_score'] for r in ref['oof_predictions']])
    folds = np.array([r['fold'] for r in ref['oof_predictions']])
    groups = np.array([r['source_group'] for r in ref['oof_predictions']])
    variants = {}
    for path, sha, d in docs:
        rows = {r['filename']: r for r in d['oof_predictions']}
        ts = {r['filename']: r for r in d['submission_predictions']}
        assert len(rows) == len(d['oof_predictions']) == 769
        assert len(ts) == len(d['submission_predictions']) == 216
        assert set(rows) == set(names) and set(ts) == set(tnames)
        for key, expected in [('true_score', y), ('fold', folds), ('source_group', groups)]:
            assert np.array_equal([rows[n][key] for n in names], expected), (path, key)
        p = np.array([rows[n]['predicted_score'] for n in names], dtype=float)
        t = np.array([ts[n]['predicted_score'] for n in tnames], dtype=float)
        assert np.isfinite(p).all() and np.isfinite(t).all()
        assert ((p >= 0) & (p <= 5)).all() and ((t >= 0) & (t <= 5)).all()
        assert np.isclose(engine.metrics(y, p)['rmse'], d['scores']['pooled_oof_rmse'], atol=1e-9)
        action = d.get('approaches', {}).get('submission', {}).get('action')
        # Export records explicitly identify the selected inference action.
        old_inventory = json.loads(OLD.read_text())
        historic = next((r for r in old_inventory['handled_exports'] if r.get('sha256') == sha), None)
        if historic:
            action = historic['action']
        elif action:
            pass
        elif d.get('full_data_run'):
            action = 'full_data'
        else:
            action = 'cv_submission'
        entry = dict(file=path.name, run_id=d['run_id'], action=action, sha256=sha,
                     base_metrics=engine.metrics(y, p))
        key = hashlib.sha256(p.tobytes()).hexdigest()
        variants.setdefault(key, []).append((entry, p, t))
    chosen = []
    handled = []
    for same in variants.values():
        assert len({v[0]['run_id'] for v in same}) == 1
        same.sort(key=lambda v: ({'full_data': 0, 'cv_submission': 1, 'best_fold': 2}.get(v[0]['action'], 3), v[0]['sha256']))
        chosen.append(same[0])
        for e, _, _ in same:
            handled.append(dict(e, selected=e['sha256'] == same[0][0]['sha256']))
    chosen.sort(key=lambda v: order.index(v[0]['run_id']) if v[0]['run_id'] in order else len(order))
    assert all(len(set(folds[groups == g])) == 1 for g in set(groups))
    audit = dict(distinct_exports=len(docs), unique_oof_runs=len(chosen),
                 handled_exports=handled, ordered_runs=[c[0] for c in chosen],
                 new_runs=[c[0] for c in chosen if c[0]['run_id'] not in order])
    (OUT / 'metadata_inventory.json').write_text(json.dumps(audit, indent=2) + '\n')
    return names, tnames, y, folds, groups, np.column_stack([c[1] for c in chosen]), np.column_stack([c[2] for c in chosen]), audit


def deliver():
    names, tnames, y, folds, groups, x, test, audit = load_sources()
    report = json.loads((OUT / 'optimization_audit.json').read_text())
    dev = dict(np.load(OUT / 'development_predictions.npz'))
    full = dict(np.load(OUT / 'full_predictions.npz'))
    prior_root = ROOT / 'artifacts/all_metadata_ensembles_20261015'
    prior_dev = np.load(prior_root / 'development_predictions.npz')['joint_quality']
    prior_full = np.load(prior_root / 'full_predictions.npz')['joint_quality']
    # The previous run uses the same reference order; verify explicitly.
    prior_rows = pd.read_csv(prior_root / '01_joint_quadratic_rbf_all12_development.csv')
    assert prior_rows.filename.tolist() == names
    prior_test_rows = pd.read_csv(prior_root / '01_joint_quadratic_rbf_all12.csv')
    assert prior_test_rows.filename.tolist() == tnames
    np.testing.assert_allclose(prior_rows.meta_development_prediction, prior_dev, atol=1e-12)
    recipes = {}
    for label, d, t, source in [
        ('prior_joint_half_new', prior_dev, prior_full, str(prior_root / 'optimization_audit.json')),
        ('public_best_half_new', pd.read_csv(engine.PREVIOUS / '04_quadratic_rbf_all11_development.csv').set_index('filename').meta_development_prediction.reindex(names).to_numpy(),
         pd.read_csv(engine.PREVIOUS / '04_quadratic_rbf_all11.csv').set_index('filename').label.reindex(tnames).to_numpy(), str(engine.PREVIOUS / 'optimized_report.json')),
    ]:
        dev[label] = .5 * d + .5 * dev['updated_quadratic_rbf']
        full[label] = .5 * t + .5 * full['updated_quadratic_rbf']
        recipes[label] = dict(prior_weight=.5, new_updated_quadratic_rbf_weight=.5,
                              prior_audit=source, prior_audit_sha256=engine.digest(source))
    ranking = sorted([dict(name=k, **engine.metrics(y, v)) for k, v in dev.items()], key=lambda r: r['rmse'])
    report['ranking'] = ranking
    report['fixed_blend_recipes'] = recipes
    report['limitations'].append('Fixed 50/50 comparisons with previously searched ensembles reuse their development predictions; ranking these additional recipes adds selection optimism. Public-best recipe selection is historically leaderboard-informed.')
    selected = [r['name'] for r in ranking[:3]]
    best_pearson = max(ranking, key=lambda r: r['pearson'])['name']
    if best_pearson not in selected:
        selected.append(best_pearson)
    DELIVERY.mkdir(parents=True, exist_ok=True)
    results = []
    for i, family in enumerate(selected, 1):
        p, oof = full[family], dev[family]
        filename = f'{i:02d}_{family}_all{audit["unique_oof_runs"]}.csv'
        frame = pd.DataFrame(dict(filename=tnames, label=p))
        path = DELIVERY / filename
        frame.to_csv(path, index=False)
        check = pd.read_csv(path)
        assert list(check.columns) == ['filename', 'label'] and len(check) == 216
        assert check.filename.tolist() == tnames and check.filename.is_unique
        assert np.isfinite(check.label).all() and check.label.between(0, 5).all()
        np.testing.assert_allclose(check.label, p, rtol=0, atol=1e-12)
        score = engine.metrics(y, oof)
        row = dict(file=filename, family=family, development=score,
                   mean_fold_rmse=float(np.mean([engine.metrics(y[folds == f], oof[folds == f])['rmse'] for f in sorted(set(folds))])),
                   per_fold=[dict(fold=int(f), **engine.metrics(y[folds == f], oof[folds == f])) for f in sorted(set(folds))],
                   positive=engine.metrics(y[y > 0], oof[y > 0]),
                   zero_rmse=engine.metrics(y[y == 0], oof[y == 0])['rmse'],
                   sha256=engine.digest(path), public_rmse=None, public_pearson=None)
        metadata = dict(schema_version=1, run_id=filename[:-4], selected_family=family,
                        scores=row, source_inventory=audit,
                        validation=report['validation'], limitations=report['limitations'],
                        fit_details=report['full_fits'], stack=report['stack'],
                        fixed_blend_recipe=recipes.get(family),
                        oof_predictions=[dict(filename=n, true_score=float(a), predicted_score=float(b), fold=int(f), source_group=str(g)) for n, a, b, f, g in zip(names, y, oof, folds, groups)],
                        submission_predictions=[dict(filename=n, predicted_score=float(v)) for n, v in zip(tnames, p)],
                        submission_csv_text=path.read_text(),
                        reproduction_script=Path(__file__).read_text(),
                        estimator_source=(ROOT / 'scripts/optimize_all_metadata_ensembles_20261015.py').read_text())
        meta_name = f'{i:02d}_{family}_run_metadata_{score["rmse"]:.6f}.txt'
        (DELIVERY / meta_name).write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        row['metadata_file'] = meta_name
        results.append(row)
        print('DELIVERED', filename, score, flush=True)
    report['delivered'] = results
    (OUT / 'optimization_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pd.DataFrame([dict(file=r['file'], pooled_oof_rmse=r['development']['rmse'], mean_fold_rmse=r['mean_fold_rmse'], oof_pearson=r['development']['pearson'], public_rmse='', public_pearson='') for r in results]).to_csv(DELIVERY / 'ensemble_scores.csv', index=False)
    (DELIVERY / 'README.txt').write_text('Ranked by grouped cross-fitted meta-development RMSE. Pearson is recorded for every candidate. Submit only the numbered submission CSV files. Matching TXT files contain OOF/test predictions, provenance and fit details. Public/private performance is unknown. Frozen base OOF inputs are not fully nested; repeated model selection can overfit development results. Previous public-best files are unchanged.\n')


if __name__ == '__main__':
    engine.OUT = OUT
    engine.load_sources = load_sources
    with threadpool_limits(limits=1):
        engine.main()
        deliver()
