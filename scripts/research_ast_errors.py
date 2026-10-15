"""Read-only error audit of frozen OOF metadata, existing transcript/audio summaries."""
from pathlib import Path
import hashlib
import html
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/ast_error_research_20261014'
SOURCE = ROOT / 'artifacts/regression_ensembles_20261014/source_metadata'


def metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    return {'n': len(e), 'rmse': float(np.sqrt(np.mean(e ** 2))),
            'mae': float(np.mean(abs(e))), 'bias': float(e.mean()),
            'errors_ge075': int((abs(e) >= .75).sum()),
            'errors_ge1': int((abs(e) >= 1).sum())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = SOURCE / 'run_metadata_0.494861.txt'
    d = json.loads(path.read_text()); c = d['run_reports']['resolved_config.json']
    frame = pd.DataFrame(d['oof_predictions']).set_index('filename')
    frame = frame.rename(columns={'true_score': 'label', 'predicted_score': 'ast'})
    assert len(frame) == 769 and frame.index.is_unique
    labels = pd.read_csv(ROOT / 'data/labels/train.csv').set_index('filename').label.reindex(frame.index)
    assert np.array_equal(labels, frame.label)
    assert abs(metrics(frame.label, frame.ast)['rmse'] - d['scores']['pooled_oof_rmse']) < 1e-10
    for name, file in [('cnn', '0.515594-2'), ('fusion_deep', '0.504259'), ('fusion_shallow', '0.515180')]:
        other = json.loads((SOURCE / f'run_metadata_{file}.txt').read_text())
        table = pd.DataFrame(other['oof_predictions']).set_index('filename').reindex(frame.index)
        assert np.array_equal(table.true_score, frame.label)
        assert np.array_equal(table.fold, frame.fold)
        assert np.array_equal(table.source_group, frame.source_group)
        frame[name] = table.predicted_score
    for name, file in [('elastic', ROOT / 'artifacts/regression_ensembles_20261014/elasticnet_development.csv'),
                       ('quadratic', ROOT / 'artifacts/regression_ensembles_followup_20261014/03_smooth_quadratic_regression_development.csv')]:
        table = pd.read_csv(file).set_index('filename').reindex(frame.index)
        assert np.array_equal(table.label, frame.label)
        frame[name] = table.meta_development_prediction
    previous = pd.read_csv(ROOT / 'artifacts/architecture_error_analysis/all_recordings.csv').set_index('filename').reindex(frame.index)
    assert np.array_equal(previous.label, frame.label)
    features = ['whisper_transcript', 'crisper_transcript', 'crisper_error', 'duration_s',
                'whisper_words', 'crisper_words', 'words_per_second', 'asr_agreement',
                'crisper_word_ratio', 'frame_dynamic_db', 'relative_quiet_fraction', 'rms_dbfs']
    frame = frame.join(previous[features])
    spec = pd.DataFrame(d['run_reports']['spectrogram_report.json']['rows'])
    spec = spec[spec.split == 'train'].set_index('filename').reindex(frame.index)
    frame['spectrogram_frames'] = spec.frames
    frame['chunks'] = np.ceil(spec.frames / 1024).astype(int)
    frame['padded_frame_fraction'] = 1 - spec.frames / (frame.chunks * 1024)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'artifacts/deberta-tokenizer-check', local_files_only=True, use_fast=False)
    lengths = [len(tokens) for tokens in tokenizer(frame.whisper_transcript.fillna('').tolist(), truncation=False, add_special_tokens=False)['input_ids']]
    frame['text_tokens'] = lengths
    frame['text_truncated_256'] = frame.text_tokens > 254
    models = ['ast', 'cnn', 'fusion_deep', 'fusion_shallow', 'elastic', 'quadratic']
    for model in models:
        frame[model + '_error'] = frame[model] - frame.label
    frame['ast_abs_error'] = abs(frame.ast_error)
    frame['ast_squared_error'] = frame.ast_error ** 2
    frame['shared_failure075'] = (frame[[m + '_error' for m in models[:4]]].abs() >= .75).all(axis=1)
    frame['ast_vs_cnn_squared_gain'] = frame.cnn_error ** 2 - frame.ast_error ** 2
    summary = {m: metrics(frame.label, frame[m]) for m in models}
    slices = []
    bins = [('zero', frame.label == 0), ('1–2', (frame.label > 0) & (frame.label <= 2)),
            ('2–3', (frame.label > 2) & (frame.label <= 3)),
            ('3–4', (frame.label > 3) & (frame.label <= 4)), ('4–5', frame.label > 4)]
    for name, mask in bins:
        for model in models:
            slices.append({'slice': name, 'model': model, **metrics(frame.label[mask], frame[model][mask])})
    slice_table = pd.DataFrame(slices); slice_table.to_csv(OUT / 'score_slice_metrics.csv', index=False)
    label_metrics = []
    for label, subset in frame.groupby('label'):
        label_metrics.append({'label': label, **metrics(subset.label, subset.ast),
                              'squared_error_share': float(subset.ast_squared_error.sum() / frame.ast_squared_error.sum())})
    pd.DataFrame(label_metrics).to_csv(OUT / 'label_metrics.csv', index=False)
    histories = []; fold_metrics = []
    for report in d['fold_training_reports']:
        mask = frame.fold == report['fold']
        reproduced = metrics(frame.label[mask], frame.ast[mask])
        assert abs(reproduced['rmse'] - report['best_rmse']) < 1e-8
        selected = next(h for h in report['history'] if h['epoch'] == report['best_epoch'])
        fold_metrics.append({'fold': report['fold'], 'training_rows': report['training_rows'],
                             'best_epoch': report['best_epoch'], 'epochs_completed': len(report['history']),
                             'selected_training_mse': selected['train_loss'],
                             'sqrt_selected_training_mse_not_eval_rmse': np.sqrt(selected['train_loss']),
                             **reproduced})
        histories.extend(dict(h, fold=report['fold'], selected=h['epoch'] == report['best_epoch']) for h in report['history'])
    pd.DataFrame(histories).to_csv(OUT / 'training_history.csv', index=False)
    pd.DataFrame(fold_metrics).to_csv(OUT / 'fold_metrics.csv', index=False)
    associations = []
    positive = frame[frame.label > 0].copy()
    for feature in ['duration_s', 'whisper_words', 'text_tokens', 'words_per_second', 'asr_agreement',
                    'crisper_word_ratio', 'frame_dynamic_db', 'relative_quiet_fraction', 'rms_dbfs', 'padded_frame_fraction']:
        data = positive[[feature, 'ast_abs_error', 'label']].dropna()
        centered = data[[feature, 'ast_abs_error']] - data.groupby('label')[[feature, 'ast_abs_error']].transform('mean')
        associations.append({'feature': feature, 'n': len(data),
                             'spearman_raw': float(spearmanr(data[feature], data.ast_abs_error).statistic),
                             'spearman_label_centered': float(spearmanr(centered[feature], centered.ast_abs_error).statistic)})
    pd.DataFrame(associations).to_csv(OUT / 'feature_associations.csv', index=False)
    # Exploratory paired source-group bootstrap of AST versus older CNN fusion.
    grouped = frame.assign(ast_sse=frame.ast_error ** 2, cnn_sse=frame.cnn_error ** 2).groupby('source_group').agg(
        ast_sse=('ast_sse', 'sum'), cnn_sse=('cnn_sse', 'sum'), n=('label', 'size'))
    rng = np.random.default_rng(20261014)
    idx = rng.integers(len(grouped), size=(3000, len(grouped)))
    n = grouped.n.to_numpy()[idx].sum(axis=1)
    diff = np.sqrt(grouped.ast_sse.to_numpy()[idx].sum(axis=1) / n) - np.sqrt(grouped.cnn_sse.to_numpy()[idx].sum(axis=1) / n)
    paired = {'rmse_ast_minus_cnn': summary['ast']['rmse'] - summary['cnn']['rmse'],
              'exploratory_group_bootstrap_95_interval': np.quantile(diff, [.025, .975]).tolist(),
              'limitations': 'After model/experiment selection; not independent confirmatory inference. AST and audio hidden width changed together.'}
    concentration = {}
    for count in [10, 20, 50, 100]:
        concentration[str(count)] = float(frame.ast_squared_error.nlargest(count).sum() / frame.ast_squared_error.sum())
    worst = frame.sort_values('ast_abs_error', ascending=False).head(30)
    worst.to_csv(OUT / 'worst_30_recordings.csv')
    frame.sort_values('ast_vs_cnn_squared_gain', ascending=False).head(20).to_csv(OUT / 'ast_rescues_vs_cnn.csv')
    frame.sort_values('ast_vs_cnn_squared_gain').head(20).to_csv(OUT / 'ast_regressions_vs_cnn.csv')
    frame.to_csv(OUT / 'all_recording_errors.csv')
    correlations = frame[[m + '_error' for m in models[:4]]].corr()
    correlations.to_csv(OUT / 'base_error_correlations.csv')
    truncation = {str(value): metrics(subset.label, subset.ast) for value, subset in frame.groupby('text_truncated_256')}
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes[0, 0].scatter(frame.label, frame.ast, alpha=.4, s=15); axes[0, 0].plot([0, 5], [0, 5], 'k--')
    axes[0, 0].set(title='AST best-epoch OOF predictions', xlabel='Label', ylabel='Prediction')
    for model in ['ast', 'elastic', 'quadratic']:
        table = slice_table[slice_table.model == model]
        axes[0, 1].plot(table['slice'], table.bias, 'o-', label=model)
    axes[0, 1].axhline(0, color='black', linewidth=.7); axes[0, 1].legend(); axes[0, 1].set(title='Bias by label range', ylabel='Prediction − label')
    hist = pd.DataFrame(histories)
    for fold, h in hist.groupby('fold'):
        axes[0, 2].plot(h.epoch, h.validation_rmse, label=f'Fold {fold+1}')
        best = h[h.selected]; axes[0, 2].scatter(best.epoch, best.validation_rmse, marker='*', s=130)
        axes[1, 0].plot(h.epoch, h.train_loss, label=f'Fold {fold+1}')
    axes[0, 2].legend(); axes[0, 2].set(title='Validation RMSE; selected epochs starred', xlabel='Epoch')
    axes[1, 0].legend(); axes[1, 0].set(title='Training MSE during updates', xlabel='Epoch')
    ordered = frame.ast_squared_error.sort_values(ascending=False).to_numpy()
    axes[1, 1].plot(np.arange(1, len(ordered)+1), np.cumsum(ordered)/ordered.sum()); axes[1, 1].set(title='Error concentration', xlabel='Worst recordings included', ylabel='Fraction of total squared error')
    for model in models[:4]:
        axes[1, 2].hist(frame[model + '_error'], bins=35, histtype='step', label=model)
    axes[1, 2].legend(); axes[1, 2].set(title='Base-model error distributions', xlabel='Prediction − label')
    figure.tight_layout(); figure.savefig(OUT / 'diagnostics.png', dpi=160); plt.close(figure)
    # Browser listening aid; listening is for human verification, not performed by this script.
    page = '<!doctype html><meta charset="utf-8"><title>AST error review</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}section{border-top:1px solid #ccc;padding:18px}pre{white-space:pre-wrap}</style><h1>AST error review: largest 30 OOF misses</h1><p>Transcripts can be inaccurate; labels are unchanged. Listen before inferring grammar, accent, or label problems.</p>'
    for name, row in worst.iterrows():
        page += f'<section><h2>{html.escape(name)} — label {row.label:g}, AST {row.ast:.3f}, quadratic development {row.quadratic:.3f}</h2><audio controls preload="none" src="../competition_audio_full/train/{html.escape(name)}"></audio><h3>Whisper</h3><pre>{html.escape(str(row.whisper_transcript))}</pre><h3>Crisper</h3><pre>{html.escape(str(row.crisper_transcript))}</pre></section>'
    (OUT / 'listening_review.html').write_text(page)
    result = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'run_id': d['run_id'], 'models': summary, 'fold_metrics': fold_metrics,
              'slice_metrics': slices, 'error_concentration': concentration,
              'common_base_misses075': int(frame.shared_failure075.sum()),
              'ast_vs_cnn': paired, 'feature_associations': associations, 'text_truncation': truncation,
              'trainable_parameters': d['fold_training_reports'][0]['trainable_parameters'],
              'full_data_epochs': d['full_data_run']['training_plan.json']['epochs'],
              'configuration': {k: c[k] for k in ['spectrogram_lr', 'audio_hidden_dim', 'audio_num_layers', 'max_length', 'spectrogram_chunk_frames']}}
    (OUT / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
