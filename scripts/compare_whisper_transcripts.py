#!/usr/bin/env python3
"""Rerun the locked six-fold baseline on Large-v3 transcripts and compare errors."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import nbformat
import numpy as np
import pandas as pd
import sklearn
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from finish_transcription import validate

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'artifacts/whisper_comparison/reference'


def paired_comparison(small, large, seed=42, repetitions=20000):
    """Conditional paired-record bootstrap of fixed out-of-fold predictions."""
    paired = small.merge(large, on='filename', suffixes=('_small', '_large'),
                         how='outer', validate='one_to_one', indicator=True)
    if not paired['_merge'].eq('both').all():
        raise ValueError('Models were evaluated on different recordings.')
    if not np.array_equal(paired.label_small, paired.label_large):
        raise ValueError('Labels differ between experiments.')
    if not np.array_equal(paired.fold_small, paired.fold_large):
        raise ValueError('Fold assignments differ between experiments.')
    actual = paired.label_small.to_numpy()
    small_errors = np.square(actual - paired.prediction_small.to_numpy())
    large_errors = np.square(actual - paired.prediction_large.to_numpy())
    if not np.isfinite(small_errors).all() or not np.isfinite(large_errors).all():
        raise ValueError('Nonfinite out-of-fold errors.')
    small_rmse = float(np.sqrt(small_errors.mean()))
    large_rmse = float(np.sqrt(large_errors.mean()))
    rng = np.random.default_rng(seed)
    differences = []
    # Batch resamples so the confidence interval does not require a large matrix.
    for start in range(0, repetitions, 512):
        indices = rng.integers(0, len(actual), size=(min(512, repetitions-start), len(actual)))
        differences.extend(np.sqrt(large_errors[indices].mean(axis=1))
                           - np.sqrt(small_errors[indices].mean(axis=1)))
    interval = np.quantile(differences, [0.025, 0.975]).tolist()
    fold_rows = []
    for fold in sorted(paired.fold_small.unique()):
        mask = paired.fold_small.eq(fold).to_numpy()
        a, b = float(np.sqrt(small_errors[mask].mean())), float(np.sqrt(large_errors[mask].mean()))
        fold_rows.append({'fold': int(fold), 'rows': int(mask.sum()),
                          'small_rmse': a, 'large_v3_rmse': b, 'delta_large_minus_small': b-a})
    report = {
        'rows': len(actual), 'small_oof_rmse': small_rmse, 'large_v3_oof_rmse': large_rmse,
        'delta_rmse_large_minus_small': large_rmse-small_rmse,
        'paired_record_bootstrap_95_percent_interval': interval,
        'bootstrap_repetitions': repetitions, 'seed': seed,
        'interpretation': 'Negative delta favors Large-v3; positive delta favors Small.',
        'limitation': 'Approximate uncertainty conditional on these fixed OOF predictions. '
                      'Rows may share speakers, and cross-validation training sets overlap. '
                      'This is not proof of a universal model-size effect or a test of ASR word accuracy.',
    }
    return report, pd.DataFrame(fold_rows), paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transcripts', type=Path, required=True,
                        help='Downloaded transcripts_large_v3.parquet')
    args = parser.parse_args()
    for name, expected in json.loads((REFERENCE/'sha256.json').read_text()).items():
        if hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Locked baseline reference changed: {name}')
    original = json.loads((REFERENCE/'experiment.json').read_text())
    for name, current in [('sklearn', sklearn.__version__), ('pandas', pd.__version__), ('numpy', np.__version__)]:
        if original[name] != current:
            raise ValueError(f'{name} changed from {original[name]} to {current}; restore the original environment.')
    large = validate(args.transcripts)
    if 'model_id' not in large or not large.model_id.eq('openai/whisper-large-v3').all():
        raise ValueError('Expected Large-v3 transcripts, identified by model_id.')
    small = pd.read_parquet(REFERENCE/'small_transcripts.parquet')
    if set(zip(large['split'], large.audio_id)) != set(zip(small['split'], small.audio_id)):
        raise ValueError('Large-v3 and Small transcripts cover different recordings.')
    destination = ROOT/'artifacts/transcription_large_v3'
    destination.mkdir(parents=True, exist_ok=True)
    copied = destination/'transcripts_large_v3.parquet'
    if args.transcripts.resolve() != copied.resolve():
        shutil.copy2(args.transcripts, copied)
    notebook = nbformat.read(REFERENCE/'baseline_notebook.ipynb', as_version=4)
    config = original['config'].copy()
    for key in ['word_ngram_range', 'char_ngram_range']:
        config[key] = tuple(config[key])
    config['experiment_name'] = 'tfidf_ridge_large_v3'
    notebook.cells[2].source = ('CONFIG = ' + repr(config)
        + '\nLOCAL_LABELS_DIR = ' + repr(str(REFERENCE))
        + '\nLOCAL_TRANSCRIPTS_PATH = ' + repr(str(copied)))
    notebook.cells[0].source = '# Whisper Large-v3 — locked TF-IDF/Ridge comparison\n\n' + notebook.cells[0].source
    for cell in notebook.cells:
        cell.source = cell.source.replace(
            'shl-2026-whisper-small-transcripts', 'shl-2026-whisper-large-v3-transcript-data')
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    folder = ROOT/'experiments/tfidf_ridge_large_v3'
    folder.mkdir(parents=True, exist_ok=True)
    notebook_path = folder/'tfidf_ridge_large_v3.ipynb'
    nbformat.write(notebook, notebook_path)
    with tempfile.TemporaryDirectory() as tmp:
        spec_folder = Path(tmp)/'shl-comparison'
        spec_folder.mkdir()
        (spec_folder/'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'SHL comparison', 'language': 'python'}))
        manager = KernelManager(kernel_name='shl-comparison',
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[tmp]))
        client = NotebookClient(notebook, km=manager, timeout=600,
                                resources={'metadata': {'path': str(ROOT)}})
        try:
            client.execute()
        finally:
            nbformat.write(notebook, notebook_path)
    results = ROOT/'artifacts/baseline/tfidf_ridge_large_v3'
    new_manifest = json.loads((results/'experiment.json').read_text())
    new_config = new_manifest['config'].copy()
    new_config['experiment_name'] = original['config']['experiment_name']
    assert new_config == original['config'], 'Scoring hyperparameters changed.'
    report, folds, paired = paired_comparison(
        pd.read_csv(REFERENCE/'oof_predictions.csv'), pd.read_csv(results/'oof_predictions.csv'))
    report['new_transcripts_sha256'] = hashlib.sha256(copied.read_bytes()).hexdigest()
    report['changed_training_transcripts'] = int(paired.transcript_small.ne(paired.transcript_large).sum())
    (results/'paired_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    folds.to_csv(results/'paired_fold_rmse.csv', index=False)
    paired.to_csv(results/'paired_oof_predictions.csv', index=False)
    notebook.cells.extend([
        nbformat.v4.new_markdown_cell('## Paired Small vs Large-v3 results\n\n'
            'These use identical folds and scoring hyperparameters. Negative RMSE delta favors Large-v3. '
            'The bootstrap interval is conditional on fixed OOF predictions, not a universal proof.'),
        nbformat.v4.new_code_cell('# Recorded comparison from this completed run.\n'
            'comparison = '+repr(report)+'\ndisplay(pd.Series(comparison).to_frame("value"))')])
    nbformat.write(notebook, notebook_path)
    print(json.dumps(report, indent=2))
    print(folds.to_string(index=False))
    print('Executed comparison notebook:', notebook_path)
    print('Submission generated:', results/'submission.csv')


if __name__ == '__main__':
    main()
