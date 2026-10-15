"""Stages: bundle -> inspect -> download -> analyze -> folds -> train -> export -> backup."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_utils as integrity
import telemetry
import audio_features as audio


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def bundle(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    transcripts = pd.read_csv(root / 'artifacts/transcripts-large-v3-dataset/transcripts_large_v3.csv')
    if transcripts.duplicated(['split', 'audio_id']).any():
        raise ValueError('Duplicate transcript keys')
    for split in ['train', 'test']:
        labels = pd.read_csv(root / f'data/labels/{split}.csv')
        if labels.filename.duplicated().any():
            raise ValueError('Duplicate label IDs')
        text = transcripts.loc[transcripts.split.eq(split)].rename(columns={'audio_id': 'filename', 'transcript': 'text'})
        if set(labels.filename) != set(text.filename):
            raise ValueError(f'{split}: transcript/label ID sets differ')
        joined = labels.merge(text[['filename', 'text', 'error']], on='filename', validate='one_to_one', how='left')
        if joined.text.isna().any() or joined.text.str.strip().eq('').any() or joined.error.fillna('').ne('').any():
            raise ValueError('Missing, empty or failed transcript')
        if split == 'train' and (not np.isfinite(joined.label).all() or not joined.label.between(0, 5).all()):
            raise ValueError('Invalid training labels')
        joined[['filename', 'label', 'text']].to_csv(output / f'{split}.csv', index=False)
    sample = pd.read_csv(root / 'data/labels/sample_submission.csv')
    test = pd.read_csv(output / 'test.csv')
    if list(sample.columns) != ['filename', 'label'] or sample.filename.duplicated().any():
        raise ValueError('Official sample schema or IDs differ')
    shutil.copy2(root / 'data/labels/sample_submission.csv', output / 'sample_submission.csv')
    template_matches = set(sample.filename) == set(test.filename)
    order = sample[['filename']] if template_matches else test[['filename']]
    order.assign(label=-1).to_csv(output / 'submission_template.csv', index=False)
    print(f'Submission order: {len(order)} actual test IDs; official template matches: {template_matches}', flush=True)
    pairs = pd.read_csv(root / 'artifacts/zero_label_analysis/matched_source_fold_audit.csv')
    pairs[['zero_filename', 'positive_filename']].to_csv(output / 'source_pairs.csv', index=False)
    write_json(output / 'manifest.json', {
        'transcript_source': 'saved Whisper Large-v3; unchanged text and labels',
        'submission_policy': 'Use official schema and order if IDs match; otherwise actual test.csv IDs/order, consistent with previously accepted submissions',
        'source_pair_scope': '31 waveform-supported pairs; remaining speaker/source identities unresolved',
        'rows': {s: len(pd.read_csv(output / f'{s}.csv')) for s in ['train', 'test']},
        'files': {p.name: integrity.file_sha256(p) for p in sorted(output.glob('*.csv'))}})
    print(f'Text-only bundle ready: {output}', flush=True)


def verify_bundle(data):
    data = Path(data)
    for name, digest in read_json(data / 'manifest.json')['files'].items():
        if integrity.file_sha256(data / name) != digest:
            raise ValueError(f'Data bundle changed: {name}')


def inspect(output):
    import torch
    import transformers
    report = {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
              'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
              'disk_free_gib': shutil.disk_usage(output).free / 2**30,
              'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                        'vram_gib': torch.cuda.get_device_properties(i).total_memory / 2**30}
                       for i in range(torch.cuda.device_count())]}
    write_json(Path(output) / 'environment.json', report)
    print(json.dumps(report, indent=2))
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(Path(output) / 'installed-requirements.txt').open('w'), check=True)


def download(config, output):
    if not audio.uses_text(config):
        directory = Path(output) / 'backbone'
        directory.mkdir(exist_ok=True)
        write_json(directory / 'config.json', {'audio_backbone': config['audio_backbone'], 'pretrained': False})
        (Path(output) / 'tokenizer').mkdir(exist_ok=True)
        write_json(Path(output) / 'model_provenance.json', {'model_id': config['audio_backbone'], 'revision': 'trained_from_scratch'})
        return
    from huggingface_hub import HfApi, snapshot_download
    model_id = config['model_id']
    provenance = Path(output) / 'model_provenance.json'
    if provenance.exists():
        saved = read_json(provenance)
        if saved['model_id'] != model_id:
            raise ValueError('Use a new output directory for a different backbone')
        revision = saved['revision']
        if config.get('revision') and config['revision'] != revision:
            raise ValueError('Requested revision differs from the saved revision')
    else:
        revision = HfApi().model_info(model_id, revision=config.get('revision') or 'main').sha
    directory = Path(output) / 'backbone'
    if not directory.exists():
        # A shared cache stores one copy per pinned revision across experiments.
        files = HfApi().list_repo_files(model_id, revision=revision)
        weights = ['*.safetensors'] if any(f.endswith('.safetensors') for f in files if '/' not in f) else ['pytorch_model*.bin']
        cached = snapshot_download(model_id, revision=revision,
                          cache_dir=str(Path(output).resolve().parent.parent / 'model_cache'),
                          allow_patterns=['*.json', '*.txt', '*.model', *weights],
                          ignore_patterns=['*optimizer*', '*training_args*', '*/*'])
        directory.symlink_to(Path(cached).resolve(), target_is_directory=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(directory, use_fast=False, local_files_only=True).save_pretrained(Path(output) / 'tokenizer')
    write_json(provenance, {'model_id': model_id, 'revision': revision, 'sha256': integrity.model_fingerprint(directory)})
    print(f'Downloaded {model_id} at {revision}', flush=True)


def analyze(data, output):
    from transformers import AutoConfig, AutoTokenizer
    verify_bundle(data)
    data, output = Path(data), Path(output)
    tokenizer = AutoTokenizer.from_pretrained(output / 'backbone', use_fast=False, local_files_only=True)
    architecture = AutoConfig.from_pretrained(output / 'backbone')
    # This initial pipeline supports encoder backbones with ordinary absolute positions.
    # RoBERTa reserves positions below padding_idx + 1.
    native = int(architecture.max_position_embeddings)
    if architecture.model_type == 'roberta':
        native -= int(architecture.pad_token_id) + 1
    if tokenizer.model_max_length < 1_000_000:
        native = min(native, int(tokenizer.model_max_length))
    specials = tokenizer.num_special_tokens_to_add(pair=False)
    rows, report = [], {'native_limit': native, 'special_tokens': specials,
                        'provenance': read_json(output / 'model_provenance.json'), 'splits': {}}
    candidates = sorted({x for x in [256, 384, 512, 640, 1024, 2048, native] if specials < x <= native})
    for split in ['train', 'test']:
        frame = pd.read_csv(data / f'{split}.csv', keep_default_na=False)
        tokens = tokenizer(frame.text.tolist(), add_special_tokens=False, truncation=False, padding=False)['input_ids']
        content = np.array([len(x) for x in tokens])
        total = content + specials
        summaries = []
        for limit in candidates:
            removed = np.maximum(content - (limit-specials), 0)
            summaries.append({'limit': limit, 'fraction_truncated': float(np.mean(total > limit)),
                              'fraction_content_tokens_discarded': float(removed.sum() / max(1, content.sum()))})
        report['splits'][split] = {'median': float(np.median(total)),
                                  **{f'p{p}': float(np.percentile(total, p)) for p in [90, 95, 99]},
                                  'maximum': int(total.max()), 'candidates': summaries}
        for row, count in zip(frame.itertuples(), total):
            rows.append({'split': split, 'filename': row.filename, 'total_tokens': int(count)})
        lost = []
        for i in np.argsort(total)[-10:][::-1]:
            lost.append({'filename': frame.iloc[i].filename, 'tokens': int(total[i]),
                         'discarded_suffix_at_native_limit': tokenizer.decode(tokens[i][native-specials:])})
        write_json(output / f'{split}_truncation_examples.json', lost)
    eligible = [r['limit'] for r in report['splits']['train']['candidates'] if r['fraction_truncated'] <= .05]
    report['recommended_limit'] = min(eligible) if eligible else native
    report['native_covers_95_percent'] = bool(eligible)
    report['selection_status'] = 'Recommendation only; GPU feasibility and user selection required before training.'
    pd.DataFrame(rows).to_csv(output / 'token_counts.csv', index=False)
    write_json(output / 'length_report.json', report)
    print(json.dumps(report, indent=2))


def assign_folds(data, output, config):
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    if config['validation_policy'] != 'grouped_multilabel_score_bins':
        raise ValueError('Validation policy is not specified/approved')
    verify_bundle(data)
    data, output = Path(data), Path(output)
    train, test = [pd.read_csv(data / f'{s}.csv', keep_default_na=False) for s in ['train', 'test']]
    parents = {name: name for name in train.filename}
    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x
    for pair in pd.read_csv(data / 'source_pairs.csv').itertuples():
        if pair.zero_filename not in parents or pair.positive_filename not in parents:
            raise ValueError('Source-pair ID absent from training')
        a, b = find(pair.zero_filename), find(pair.positive_filename)
        parents[max(a, b)] = min(a, b)
    train['source_group'] = train.filename.map(find)
    # Zero separate; positive bins (0,2], (2,3], (3,4], (4,5].
    train['score_bin'] = np.where(train.label.eq(0), 0, np.digitize(train.label, [2, 3, 4], right=True)+1)
    indicators = pd.crosstab(train.source_group, train.score_bin).gt(0).astype(int)
    groups = indicators.index.to_numpy()
    assignments = {}
    n_folds=config.get('n_folds',3)
    if type(n_folds) is not int or not 2<=n_folds<=len(groups): raise ValueError('Invalid number of folds')
    splitter = MultilabelStratifiedKFold(n_splits=config.get("n_folds",3), shuffle=True, random_state=config.get('fold_seed',config['seed']))
    for fold, (_, valid) in enumerate(splitter.split(np.zeros((len(groups), 1)), indicators.to_numpy())):
        assignments.update({g: fold for g in groups[valid]})
    train['fold'] = train.source_group.map(assignments)
    if train.groupby('source_group').fold.nunique().max() != 1:
        raise ValueError('Source group crosses folds')
    existing = output / 'folds.csv'
    folds = train[['filename', 'label', 'source_group', 'score_bin', 'fold']]
    if existing.exists() and existing.read_text() != folds.to_csv(index=False):
        raise ValueError('Frozen fold assignment differs; use a new validation directory')
    folds.to_csv(existing, index=False)
    train['split'], test['split'], test['fold'] = 'train', 'test', -1
    frame = pd.concat([train, test], ignore_index=True)
    frame.to_parquet(output / 'data.parquet', index=False)
    write_json(output / 'fold_report.json', {'policy': config['validation_policy'], 'seed': config['seed'], 'n_folds':config.get('n_folds',3),'fold_seed':config.get('fold_seed',config['seed']),
        'bin_edges': [0, 2, 3, 4, 5], 'original_labels_preserved': True,
        'fold_counts': train.groupby('fold').size().to_dict(),
        'score_counts': train.groupby(['fold', 'label']).size().rename('count').reset_index().to_dict('records'),
        'limitations': 'Only verified pairs are grouped; other shared speakers/sources may remain. Rare score levels cannot be evenly distributed.'})


def resolved_config(config, output):
    if not config.get('design_confirmed'):
        raise ValueError('First-run design choices are awaiting user answers')
    required = ['max_length', 'batch_size', 'eval_batch_size', 'gradient_accumulation', 'output_mode']
    if any(config.get(k) is None for k in required):
        raise ValueError(f'Complete measured/confirmed settings: {required}')
    report = read_json(Path(output) / 'length_report.json')
    if type(config['max_length']) is not int or not report['special_tokens'] < config['max_length'] <= report['native_limit']:
        raise ValueError('Unsupported sequence length')
    if math.ceil(config['max_length'] / 8) * 8 > report['native_limit']:
        raise ValueError('Rounded dynamic padding exceeds native context; choose a supported multiple of eight')
    for key in ['batch_size', 'eval_batch_size', 'gradient_accumulation', 'epochs', 'patience']:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    if config.get('max_runtime_hours') is not None and (not math.isfinite(config['max_runtime_hours']) or config['max_runtime_hours'] <= 0):
        raise ValueError('Specify a positive runtime budget')
    if config['output_mode'] not in ['linear', 'bounded']:
        raise ValueError('Choose linear or bounded regression output')
    from pooling import SUPPORTED
    if config['loss'] != 'mse' or config['pooling'] not in SUPPORTED:
        raise ValueError('Use MSE loss and a supported pooling method')
    if config.get('precision','bf16')!='bf16': raise ValueError('GPU probe/training supports BF16 only')
    for key in ['head_dropout','backbone_dropout']:
        if not 0<=config.get(key,0.)<1: raise ValueError('Dropout must be in [0,1)')
    result = dict(config, model_revision=read_json(Path(output) / 'model_provenance.json')['revision'])
    return result


def jobs_for(config, output):
    output = Path(output)
    frame = pd.read_parquet(output / 'data.parquet')
    return integrity.make_jobs({'id': config['run_id'], 'loss': config['loss'], 'pooling': config['pooling']},
        config['seed'], frame, output / 'data.parquet', output / 'backbone', output, config,
        worker_path=HERE / 'train_worker.py')


def train(config, output, smoke=False):
    import train_worker as worker
    import torch
    output = Path(output)
    resolved = resolved_config(config, output)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the Vast.ai CUDA GPU')
    frame = pd.read_parquet(output / 'data.parquet')
    if smoke:
        if audio.uses_audio(resolved):
            from batch_probe import probe
            texts = frame.loc[frame.split.eq('train'), 'text'].tolist()
            report = {'reports': [probe(resolved, output / 'backbone', texts, mode, resolved[key])
                      for mode, key in [('train', 'batch_size'), ('eval', 'eval_batch_size')]]}
            if not all(item['accepted'] for item in report['reports']):
                raise RuntimeError('Multimodal smoke test exceeded memory margin')
        else:
            report = worker.smoke_test(resolved, output / 'backbone', frame.loc[frame.split.eq('train') & frame.text.str.strip().ne(''), 'text'], gpu_ids=[0])
        report['config'] = resolved
        write_json(output / 'smoke.json', report)
        return
    if read_json(output / 'smoke.json').get('config') != resolved:
        raise ValueError('Run GPU smoke test on the exact resolved configuration first')
    write_json(output / 'resolved_config.json', resolved)
    # Session budget is cumulative across invocations; checkpoints pause at epoch boundaries.
    budget_path = output / 'runtime_budget.json'
    spent = read_json(budget_path)['used_seconds'] if budget_path.exists() else 0.0
    remaining = math.inf if resolved.get('max_runtime_hours') is None else resolved['max_runtime_hours'] * 3600 - spent
    if remaining <= 0:
        raise RuntimeError('Cumulative runtime budget exhausted')
    start = time.time()
    try:
        print('RESOLVED CONFIGURATION '+json.dumps(resolved, sort_keys=True), flush=True)
        for job in jobs_for(resolved, output):
            telemetry.context(fold=job['fold'], candidate_id=job['candidate_id'], epoch=None, batch=None)
            telemetry.emit('fold_start', fold_rows=int(frame.loc[frame.split.eq('train') & frame.fold.eq(job['fold'])].shape[0]))
            if time.time()-start >= remaining:
                print('Budget reached; resume files retained', flush=True)
                return
            job['deadline_epoch_time'] = start + remaining
            report = worker.run_job(job)
            if report.get('paused'):
                return
        export(resolved, output)
    finally:
        write_json(budget_path, {'used_seconds': spent + time.time()-start,
                                'limit_hours': resolved['max_runtime_hours'], 'boundary': 'epoch; may overrun by one epoch'})


def export(config, output):
    output = Path(output)
    frame = pd.read_parquet(output / 'data.parquet')
    oof, test, score, reports = integrity.collect_predictions(jobs_for(config, output), frame)
    sample = pd.read_csv(output / 'input' / 'submission_template.csv')
    submission = sample[['filename']].merge(test, on='filename', how='left', validate='one_to_one').rename(columns={'prediction': 'label'})
    if submission.filename.tolist() != sample.filename.tolist() or not np.isfinite(submission.label).all() or set(test.filename) != set(sample.filename):
        raise ValueError('Invalid submission alignment')
    if not submission.label.between(0, 5).all():
        raise ValueError('Predictions exceed [0,5]; specify a clipping experiment before export')
    oof.to_csv(output / 'oof_predictions.csv', index=False)
    test.to_csv(output / 'test_predictions.csv', index=False)
    temporary = output / 'submission.csv.tmp'
    submission.to_csv(temporary, index=False)
    temporary.replace(output / 'submission.csv')
    # Leakage-free named comparison on the exact same folds.
    baseline = np.empty(len(oof))
    for fold in sorted(oof.fold.unique()):
        mask = oof.fold.eq(fold).to_numpy()
        baseline[mask] = oof.loc[~mask, 'label'].mean()
    baseline_rmse = float(np.sqrt(np.mean((oof.label-baseline)**2)))
    oof.assign(prediction=baseline).to_csv(output / 'fold_mean_baseline_oof.csv', index=False)
    write_json(output / 'metrics.json', {'cv_rmse': score, 'baseline_name': 'training_fold_mean',
        'baseline_cv_rmse': baseline_rmse, 'cv_delta': score-baseline_rmse,
        'zero_rmse': float(np.sqrt(np.mean((oof.loc[oof.label.eq(0), 'prediction'])**2))),
        'positive_rmse': float(np.sqrt(np.mean((oof.loc[oof.label.gt(0), 'label']-oof.loc[oof.label.gt(0), 'prediction'])**2))),
        'folds': reports, 'submission_rows': len(submission), 'public_metric': 'unverified',
        'public_score': None, 'public_score_delta': None})
    telemetry.emit('submission_ready', cv_rmse=score, submission_rows=len(submission))
    print(f'Submission ready: {output / "submission.csv"}; CV RMSE {score:.6f}', flush=True)


def backup(output, destination):
    source, destination = Path(output).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Backup must be outside run directory')
    files = {str(p.relative_to(source)): integrity.file_sha256(p) for p in source.rglob('*')
             if p.is_file() and '.cache' not in p.relative_to(source).parts and p.name != 'backup_receipt.json'}
    shutil.copytree(source, destination, dirs_exist_ok=False, ignore=shutil.ignore_patterns('.cache'))
    for name, digest in files.items():
        if integrity.file_sha256(destination / name) != digest:
            raise ValueError(f'Backup verification failed: {name}')
    write_json(source / 'backup_receipt.json', {'destination': str(destination), 'verified_files': files,
        'note': 'Copy to independent persistent storage before Vast termination; this path may be on the same instance.'})
    print(f'Backup verified: {destination}', flush=True)



def measure_batches(config, output):
    output = Path(output)
    measured = output / 'measured_config.json'
    if measured.exists():
        saved = read_json(measured)
        if any(saved.get(k) != v for k,v in config.items() if k not in ['batch_size','eval_batch_size','gradient_accumulation']):
            raise ValueError('Saved batch measurements belong to another configuration')
        return saved
    results = []
    selected = dict(config)
    for mode, sizes in [('train',[config['batch_size']] if config.get('batch_size') else [x for x in [8,4,2,1] if config['effective_batch_size']%x==0]),
                        ('eval',[config['eval_batch_size']] if config.get('eval_batch_size') else [16,8,4,2,1])]:
        for size in sizes:
            telemetry.context(probe_mode=mode, probe_size=size)
            telemetry.emit('batch_probe_start')
            probe_config = output / 'probe_config.json'
            write_json(probe_config, selected)
            report = output / f'probe_{mode}_{size}.json'
            command = [sys.executable,'-u',str(HERE / 'batch_probe.py'),'--config',str(probe_config),
                       '--model',str(output / 'backbone'),'--data',str(output / 'data.parquet'),
                       '--mode',mode,'--size',str(size),'--report',str(report)]
            subprocess.run(command, check=True)
            result = read_json(report)
            results.append(result)
            write_json(output / 'batch_measurements.json', results)
            print('BATCH PROBE '+json.dumps(result), flush=True)
            if result['accepted']:
                selected['batch_size' if mode == 'train' else 'eval_batch_size'] = size
                break
        else:
            raise RuntimeError(f'No feasible {mode} batch; see batch_measurements.json')
    if config.get('gradient_accumulation') is not None:
        selected['gradient_accumulation']=config['gradient_accumulation']
        if selected['gradient_accumulation']*selected['batch_size']!=selected['effective_batch_size']:
            raise ValueError('Microbatch times accumulation must equal effective_batch_size')
    else:
        if selected['effective_batch_size']%selected['batch_size']:
            raise ValueError('Effective batch must be divisible by selected microbatch')
        selected['gradient_accumulation']=selected['effective_batch_size']//selected['batch_size']
    write_json(measured,selected)
    return selected


def write_backup_manifest(output, include_live=False):
    output = Path(output)
    telemetry.close()
    excluded = {'backup_manifest.json','.run.lock','progress.json','latest_status.json',
                'heartbeats.jsonl','terminal_status.json','train.log','events.jsonl','error.json'}
    if include_live:
        excluded={'backup_manifest.json','.run.lock'}
    files = {str(p.relative_to(output)):integrity.file_sha256(p) for p in output.rglob('*')
             if p.is_file() and p.name not in excluded and '.cache' not in p.relative_to(output).parts}
    write_json(output / 'backup_manifest.json', {'files':files,'excluded_live_files':sorted(excluded),
               'created_at':time.time(),'scope':'All immutable run artifacts including backbone, best/resume checkpoints and TensorBoard'})


def run_all(config, output):
    output = Path(output)
    telemetry.emit('inspect')
    inspect(output)
    telemetry.emit('verify_input')
    verify_bundle(output / 'input')
    telemetry.emit('download')
    download(config, output)
    telemetry.emit('token_analysis')
    if audio.uses_text(config):
        analyze(output / 'input', output)
    else:
        # Preserve downstream report interfaces without loading a text tokenizer.
        config = dict(config, max_length=8)
        write_json(output / 'length_report.json', {'modality': 'audio', 'native_limit': 8,
                   'special_tokens': 0, 'recommended_limit': 8, 'splits': {}, 'token_analysis': 'not applicable'})
    if config.get('max_length') is None:
        report=read_json(output/'length_report.json')
        limit=report['recommended_limit']
        limit=min(math.ceil(limit/8)*8,(report['native_limit']//8)*8)
        config=dict(config,max_length=limit)
        print(f'Training-token coverage selected max_length={limit}',flush=True)
    telemetry.emit('fold_assignment')
    assign_folds(output / 'input', output, config)
    if audio.uses_audio(config):
        telemetry.emit('audio_features')
        config = audio.prepare(config, output)
    print('FOLD REPORT '+json.dumps(read_json(output / 'fold_report.json')),flush=True)
    source=output / 'source'
    source.mkdir(exist_ok=True)
    for path in HERE.iterdir():
        if path.is_file(): shutil.copy2(path,source / path.name)
    config = measure_batches(config, output)
    telemetry.context(probe_mode=None,probe_size=None)
    # Completed folds retain only best weights; one active fold needs its
    # model/Adam resume and space for an atomic replacement, plus a reserve.
    count = next(r['parameter_count'] for r in read_json(output / 'batch_measurements.json') if r.get('accepted'))
    required = count * 4 * (config.get('n_folds',3)+6) + 2*2**30
    available = shutil.disk_usage(output).free
    if available < required:
        raise RuntimeError(f'Insufficient free disk for checkpoints: {required/2**30:.1f} GiB required')
    telemetry.emit('smoke')
    train(config, output, smoke=True)
    telemetry.emit('training_start')
    train(config, output)
    telemetry.emit('backup_manifest')
    write_backup_manifest(output)
    telemetry.emit('complete', submission=str(output / 'submission.csv'))


def status(output):
    output = Path(output)
    value = read_json(output / 'latest_status.json')
    try:
        os.kill(value['pid'],0)
        value['pid_alive'] = True
    except (OSError,KeyError):
        value['pid_alive'] = False
    value['heartbeat_age_seconds'] = max(0,time.time()-value['heartbeat_at'])
    print(json.dumps(value,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['bundle', 'inspect', 'download', 'analyze', 'folds', 'smoke', 'train', 'export', 'backup','run','status','seal'])
    parser.add_argument('--config', type=Path, default=HERE / 'first_run.json')
    parser.add_argument('--output', type=Path, default=Path('runs/first_model'))
    parser.add_argument('--root', type=Path, default=HERE.parents[1])
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.stage == 'status':
        status(args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('SHL_RUN_DIR', str(args.output.resolve()))
    config = read_json(args.config)
    data = args.output / 'input'
    def execute():
        if args.stage == 'bundle': bundle(args.root, data)
        elif args.stage == 'inspect': inspect(args.output)
        elif args.stage == 'download': download(config, args.output)
        elif args.stage == 'analyze': analyze(data, args.output)
        elif args.stage == 'folds':
            if not config.get('design_confirmed'):
                raise ValueError('Awaiting validation-design confirmation')
            assign_folds(data, args.output, config)
        elif args.stage == 'smoke': train(config, args.output, smoke=True)
        elif args.stage == 'train': train(config, args.output)
        elif args.stage == 'export': export(resolved_config(config, args.output), args.output)
        elif args.stage == 'run': run_all(config,args.output)
        elif args.stage == 'seal': write_backup_manifest(args.output,include_live=True)
        elif args.stage == 'backup':
            if args.destination is None: parser.error('--destination is required')
            backup(args.output, args.destination)
    if os.environ.get('SHL_MANAGED_RUN') == '1':
        execute()
    else:
        with telemetry.run_lock(args.output):
            execute()


if __name__ == '__main__':
    main()
