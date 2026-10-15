"""Scheduling, integrity checks and fold aggregation shared by the notebook/tests."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd


_FILE_DIGEST_CACHE = {}


def file_sha256(path):
    """Hash file contents, reusing a digest only while file metadata is unchanged."""
    path = Path(path).resolve()
    before = path.stat()
    key = (str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns,
           before.st_ino)
    if key in _FILE_DIGEST_CACHE:
        return _FILE_DIGEST_CACHE[key]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError(f'File changed while hashing: {path}')
    result = digest.hexdigest()
    _FILE_DIGEST_CACHE[key] = result
    return result


def implementation_fingerprint(worker_path=None):
    """Identify the actual worker and sibling utilities copied into a notebook run."""
    worker = Path(worker_path) if worker_path is not None else Path(__file__).with_name('train_worker.py')
    result = {'worker_sha256': file_sha256(worker),
              'utils_sha256': file_sha256(worker.with_name('sweep_utils.py'))}
    for name in ['pipeline.py','telemetry.py','runner.py','batch_probe.py','pooling.py','audio_features.py','spectrogram_features.py']:
        path = worker.with_name(name)
        if path.exists(): result[name] = file_sha256(path)
    return result


def model_fingerprint(model_dir):
    """Identify backbone/tokenizer content independently of its local mount path."""
    directory = Path(model_dir)
    if not directory.is_dir():
        raise ValueError(f'Missing local backbone/tokenizer directory: {directory}')
    files = sorted(p for p in directory.rglob('*') if p.is_file()
                   and not any(part.startswith('.') for part in p.relative_to(directory).parts))
    if not files:
        raise ValueError(f'Empty local backbone/tokenizer directory: {directory}')
    entries = [(p.relative_to(directory).as_posix(), file_sha256(p)) for p in files]
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def fingerprint_frame(frame):
    columns = ['split', 'filename', 'label', 'fold', 'text']
    ordered = frame[columns].sort_values(['split', 'filename']).to_csv(index=False)
    return hashlib.sha256(ordered.encode()).hexdigest()


def ordered_frame_fingerprint(frame):
    columns = ['split', 'filename', 'label', 'fold', 'text']
    return hashlib.sha256(frame[columns].to_csv(index=False).encode()).hexdigest()


def make_jobs(candidate, seed, frame, data_path, model_dir, output_dir, base_config,
              worker_path=None):
    config = dict(base_config, loss=candidate['loss'], pooling=candidate['pooling'])
    payload = {'config': copy.deepcopy(config), 'seed': int(seed),
               'data_sha256': fingerprint_frame(frame),
               'data_order_sha256': ordered_frame_fingerprint(frame),
               'model_revision': base_config['model_revision'],
               'model_sha256': model_fingerprint(model_dir),
               'implementation_sha256': implementation_fingerprint(worker_path)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    jobs = []
    for fold in sorted(frame.loc[frame['split'].eq('train'), 'fold'].unique()):
        directory = Path(output_dir) / 'models' / candidate['id'] / f'seed_{seed}_{digest}' / f'fold_{int(fold)}'
        jobs.append({'candidate_id': candidate['id'], 'seed': int(seed), 'fold': int(fold),
                     'config': copy.deepcopy(config), 'job_dir': str(directory), 'data_path': str(data_path),
                     'model_dir': str(model_dir),
                     'signature_payload': dict(copy.deepcopy(payload), fold=int(fold))})
    return jobs


def verify_job_inputs(job, worker_path=None):
    """Verify loaded inputs and executable code before accepting a cache or resume."""
    import audio_features
    audio_features.verify(job['config'])
    payload = job.get('signature_payload', {})
    required = {'config', 'seed', 'fold', 'data_sha256', 'data_order_sha256',
                'model_revision', 'model_sha256', 'implementation_sha256'}
    if not required.issubset(payload):
        raise ValueError('Job is missing required provenance fingerprints.')
    if (job['config'] != payload['config'] or job['seed'] != payload['seed']
            or job['fold'] != payload['fold']
            or job['config'].get('model_revision') != payload['model_revision']):
        raise ValueError('Job configuration, seed, fold or revision differs from its provenance.')
    frame = pd.read_parquet(job['data_path'])
    if (fingerprint_frame(frame) != payload['data_sha256']
            or ordered_frame_fingerprint(frame) != payload['data_order_sha256']):
        raise ValueError('Job data content or row order differs from its provenance.')
    if job['fold'] not in set(frame.loc[frame['split'].eq('train'), 'fold']):
        raise ValueError('Assigned fold is absent from the job data.')
    if model_fingerprint(job['model_dir']) != payload['model_sha256']:
        raise ValueError('Backbone or tokenizer contents differ from the job provenance.')
    if implementation_fingerprint(worker_path) != payload['implementation_sha256']:
        raise ValueError('Training implementation differs from the job provenance.')


def expected_job_signature(job, torch_version, transformers_version):
    content = {'job': job['signature_payload'], 'torch': torch_version,
               'transformers': transformers_version}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def _runtime_versions(torch_version, transformers_version):
    if torch_version is None:
        import torch
        torch_version = torch.__version__
    if transformers_version is None:
        import transformers
        transformers_version = transformers.__version__
    return torch_version, transformers_version


def validate_completed(job, torch_version=None, transformers_version=None,
                       worker_path=None, verify_inputs=True):
    if verify_inputs:
        verify_job_inputs(job, worker_path)
    torch_version, transformers_version = _runtime_versions(torch_version, transformers_version)
    folder = Path(job['job_dir'])
    report = json.loads((folder / 'complete.json').read_text())
    if report.get('signature') != expected_job_signature(job, torch_version, transformers_version):
        raise ValueError('Completed fold belongs to a different configuration or environment.')
    for key in ['candidate_id', 'seed', 'fold']:
        if report.get(key) != job[key]:
            raise ValueError(f'Completed fold has inconsistent {key}.')
    names = {'best.pt', 'validation_predictions.csv', 'test_predictions.csv'}
    checksums = report.get('artifact_sha256', {})
    if set(checksums) != names:
        raise ValueError('Completed fold lacks the required artifact checksums.')
    for name in sorted(names):
        path = folder / name
        if not path.is_file() or file_sha256(path) != checksums[name]:
            raise ValueError(f'Completed artifact is missing or changed: {path}')
    return report


def _flatten_jobs(value):
    if isinstance(value, dict):
        if 'job_dir' in value:
            return [value]
        return [job for item in value.values() for job in _flatten_jobs(item)]
    return [job for item in value for job in _flatten_jobs(item)]


def incremental_storage_bytes(screen_jobs, confirmation_jobs_by_candidate,
                              bytes_per_checkpoint, parallel_workers):
    """Conservative *additional* storage, accounting for cached/resumable folds.

    Screen all configurations, but retain confirmation capacity only for the most
    expensive possible winner. An optimizer resume is estimated at three delta
    checkpoints. Four further deltas per active worker cover atomic best/last
    replacements. Already existing files receive credit only after provenance
    validation; queued checkpoint deletions are never assumed to happen early.
    """
    import torch
    import transformers
    unit = int(bytes_per_checkpoint)
    if unit <= 0 or parallel_workers < 1:
        raise ValueError('Checkpoint bytes and parallel workers must be positive.')
    screening = _flatten_jobs(screen_jobs)
    branches = [_flatten_jobs(value) for value in confirmation_jobs_by_candidate.values()]
    if not branches:
        branches = [[]]
    observed = {}
    for job in _flatten_jobs([screening, *branches]):
        identity = str(Path(job['job_dir']).resolve())
        if identity in observed:
            continue
        verify_job_inputs(job)
        directory = Path(job['job_dir'])
        if (directory / 'complete.json').exists():
            validate_completed(job, torch.__version__, transformers.__version__, verify_inputs=False)
            observed[identity] = (True, 0, 0)
            continue
        signature = expected_job_signature(job, torch.__version__, transformers.__version__)
        sizes = []
        for name in ['best.pt', 'last.pt']:
            path = directory / name
            if path.exists():
                checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
                if checkpoint.get('signature') != signature:
                    raise ValueError(f'Existing checkpoint has inconsistent provenance: {path}')
                sizes.append(path.stat().st_size)
                del checkpoint
            else:
                sizes.append(0)
        observed[identity] = (False, *sizes)
    estimates = []
    for branch in branches:
        plan = {str(Path(job['job_dir']).resolve()): job for job in [*screening, *branch]}
        pending = [observed[identity] for identity in plan if not observed[identity][0]]
        slots = min(int(parallel_workers), len(pending))
        permanent = sum(max(unit - best, 0) for _, best, _ in pending)
        # Assume the first active jobs have the smallest existing resumes: a
        # conservative bound regardless of scheduling order or future deletions.
        existing_last = sorted(last for _, _, last in pending)[:slots]
        resume = sum(max(3 * unit - size, 0) for size in existing_last)
        estimates.append(permanent + resume + 4 * unit * slots)
    return int(max(estimates, default=0))


def archive_submissions(output_dir, global_submission=None):
    """Move previous visible exports aside before a potentially incomplete run."""
    directory = Path(output_dir)
    paths = sorted(p for p in directory.glob('*submission*.csv') if p.is_file())
    if global_submission is not None:
        global_path = Path(global_submission)
        if global_path.is_file() and all(global_path.resolve() != p.resolve() for p in paths):
            paths.append(global_path)
    if not paths:
        return []
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    archive = directory / 'export_history' / timestamp
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for index, path in enumerate(paths):
        target = archive / path.name
        if target.exists():
            target = archive / f'{index}_{path.name}'
        shutil.move(str(path), str(target))
        moved.append(target)
    return moved


def run_jobs(jobs, worker_path, gpu_ids, deadline, poll_seconds=10, dashboard=False):
    """Each process sees only its assigned GPU; stop launching jobs at the deadline."""
    if len({str(Path(job['job_dir']).resolve()) for job in jobs}) != len(jobs):
        raise ValueError('Duplicate job directories would race on the same checkpoints.')
    if jobs and not gpu_ids:
        raise ValueError('At least one worker device is required.')
    queue, finished = [], []
    for job in jobs:
        if (Path(job['job_dir']) / 'complete.json').exists():
            finished.append(validate_completed(job, worker_path=worker_path))
            print(f"Verified cached {job['candidate_id']} seed={job['seed']} fold={job['fold']}", flush=True)
        else:
            queue.append(job)
    active, last_lines, measured_durations = {}, {}, []
    stage_started, heartbeat_tick = time.monotonic(), 0
    launched_at = {}
    panel = None
    if dashboard:
        from IPython.display import display, HTML
        import html
        panel = display(HTML('<b>Starting GPU workers…</b>'),display_id=True)
    last_heartbeat = float('-inf')
    try:
        while queue or active:
            for gpu in gpu_ids:
                if gpu in active or not queue or time.time() >= deadline:
                    continue
                job = queue.pop(0)
                folder = Path(job['job_dir'])
                folder.mkdir(parents=True, exist_ok=True)
                job = dict(job, deadline_epoch_time=deadline)
                path = folder / 'job.json'
                path.write_text(json.dumps(job, indent=2) + '\n')
                log_path = folder / 'worker.log'
                log = log_path.open('a')
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                                   TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2')
                previous_line_count = len(log_path.read_text(errors='replace').splitlines())
                process = subprocess.Popen([sys.executable, '-u', str(worker_path), '--job', str(path)],
                                           env=environment, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (process, log, job, log_path)
                launched_at[gpu] = time.monotonic()
                last_lines[str(log_path)] = previous_line_count
                print(f"GPU {gpu}: {job['candidate_id']} seed={job['seed']} fold={job['fold']}", flush=True)
            for gpu, (process, log, job, log_path) in list(active.items()):
                lines = log_path.read_text(errors='replace').splitlines()
                offset = last_lines.get(str(log_path),0)
                for line in lines[offset:]:
                    print(f'GPU {gpu}: {line}', flush=True)
                last_lines[str(log_path)] = len(lines)
                code = process.poll()
                if code is None:
                    continue
                log.close()
                del active[gpu]
                if code:
                    raise RuntimeError(f"Worker failed ({code}): {log_path}\n" + '\n'.join(lines[-20:]))
                completed = Path(job['job_dir']) / 'complete.json'
                if completed.exists():
                    report = validate_completed(job, worker_path=worker_path)
                    finished.append(report)
                    measured_durations.append(report['elapsed_seconds'])
                    mean_duration = sum(measured_durations) / len(measured_durations)
                    eta = mean_duration * (len(queue) + len(active)) / max(len(gpu_ids), 1) / 60
                    print(f'Completed {len(finished)}/{len(jobs)} fold jobs; rough ETA {eta:.1f} min', flush=True)
                else:
                    print('Fold paused with an epoch checkpoint. Rerun this cell to resume.', flush=True)
            heartbeat_tick += 1
            eta = ('warming up; waiting for first completed fold' if not measured_durations else
                   f"{sum(measured_durations)/len(measured_durations)*(len(queue)+len(active))/max(len(gpu_ids),1)/60:.1f} min (rough)")
            summary=(f"{'|/-+'[heartbeat_tick % 4]} HEARTBEAT {time.strftime('%H:%M:%S')} | "
                     f"completed {len(finished)}/{len(jobs)} | active GPUs {list(active)} | "
                     f"queued {len(queue)} | elapsed {(time.monotonic()-stage_started)/60:.1f} min | ETA {eta}")
            details=[]
            for gpu, (_, _, job, _) in active.items():
                status_path=Path(job['job_dir'])/'live_progress.json'
                message=f"GPU {gpu}: {job['candidate_id']} | seed {job['seed']} | fold {job['fold']} | loading / waiting for batch status"
                if status_path.exists():
                    status=json.loads(status_path.read_text())
                    age=time.time()-status['updated_at']
                    message=(f"GPU {gpu}: {job['candidate_id']} | seed {job['seed']} | fold {job['fold']} | "
                             f"epoch {status['epoch']}/{status['epochs']} | batch {status['batch']}/{status['batches']} | "
                             f"train loss {status['train_loss']:.5f} | last batch update {age:.0f}s ago")
                details.append(message)
            if not dashboard or time.monotonic()-last_heartbeat>=60:
                print(summary,flush=True)
                for message in details:print(message,flush=True)
                last_heartbeat=time.monotonic()
            if panel is not None:
                percent=100*len(finished)/max(len(jobs),1)
                rows=''.join('<div style="padding:8px;background:#eef2ff;margin:6px 0;border-radius:6px">'+html.escape(message)+'</div>' for message in details)
                state='PAUSED AT BUDGET' if queue and not active and time.time()>=deadline else ('DONE' if not queue and not active else 'ACTIVE')
                panel.update(HTML('<div style="padding:16px;border:1px solid #cbd5e1;border-radius:10px"><b>'+state+' — '+html.escape(summary)+'</b>'+
                                 f'<progress value="{len(finished)}" max="{max(len(jobs),1)}" style="width:100%;height:18px"></progress>'+rows+
                                 '<small>Monitor alive ≠ batches advancing. Batch timestamps show the last measured training update. Full logs are saved to worker.log.</small></div>'))
            if queue and not active and time.time() >= deadline:
                break
            if queue or active:
                time.sleep(poll_seconds)
        return len(finished) == len(jobs)
    finally:
        # Interrupting the cell must stop both GPU workers, including on a worker error.
        for process, log, _, _ in active.values():
            if process.poll() is None:
                process.terminate()
        for process, log, _, _ in active.values():
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


def collect_predictions(jobs, frame):
    folds, tests, reports = [], [], []
    expected_folds = set(frame.loc[frame['split'].eq('train'), 'fold'].astype(int))
    if {job['fold'] for job in jobs} != expected_folds or len(jobs) != len(expected_folds):
        raise ValueError('Need exactly one completed job for each fold.')
    for job in jobs:
        folder = Path(job['job_dir'])
        reports.append(validate_completed(job))
        valid = pd.read_csv(folder / 'validation_predictions.csv')
        test = pd.read_csv(folder / 'test_predictions.csv')
        if not valid.fold.eq(job['fold']).all() or valid.filename.duplicated().any():
            raise ValueError('Invalid fold predictions.')
        expected = frame[frame['split'].eq('train') & frame.fold.eq(job['fold'])]
        check = expected[['filename', 'label']].merge(valid, on='filename', how='outer',
                                                     suffixes=('_expected', ''), validate='one_to_one', indicator=True)
        if not check['_merge'].eq('both').all() or not np.array_equal(check.label_expected, check.label):
            raise ValueError('Validation IDs or labels differ from the assigned fold.')
        expected_test = frame.loc[frame['split'].eq('test'), 'filename'].tolist()
        if test.filename.tolist() != expected_test or not np.isfinite(test.prediction).all():
            raise ValueError('Test prediction IDs/order/values differ.')
        folds.append(valid)
        tests.append(test.prediction.to_numpy())
    oof = pd.concat(folds, ignore_index=True)
    expected_train = frame[frame['split'].eq('train')][['filename']]
    oof = expected_train.merge(oof, on='filename', validate='one_to_one', how='left')
    if not np.isfinite(oof.prediction).all():
        raise ValueError('Missing or nonfinite OOF predictions.')
    test = frame[frame['split'].eq('test')][['filename']].copy()
    test['prediction'] = np.mean(tests, axis=0)
    rmse = float(np.sqrt(np.mean((oof.label - oof.prediction) ** 2)))
    return oof, test, rmse, reports


def average_seed_predictions(pairs):
    reference_oof, reference_test = pairs[0]
    for oof, test in pairs[1:]:
        if (oof.filename.tolist() != reference_oof.filename.tolist()
                or test.filename.tolist() != reference_test.filename.tolist()
                or not np.array_equal(oof.label, reference_oof.label)
                or not np.array_equal(oof.fold, reference_oof.fold)):
            raise ValueError('Seed ensembles must use the same IDs, labels and folds.')
    oof, test = reference_oof.copy(), reference_test.copy()
    oof['prediction'] = np.mean([a.prediction.to_numpy() for a, _ in pairs], axis=0)
    test['prediction'] = np.mean([b.prediction.to_numpy() for _, b in pairs], axis=0)
    return oof, test


def paired_rmse_interval(labels, prediction_a, prediction_b, seed=42, repetitions=10000):
    first = np.square(np.asarray(labels) - np.asarray(prediction_a))
    second = np.square(np.asarray(labels) - np.asarray(prediction_b))
    random = np.random.default_rng(seed)
    differences = []
    for start in range(0, repetitions, 256):
        index = random.integers(0, len(first), (min(256, repetitions-start), len(first)))
        differences.extend(np.sqrt(second[index].mean(1)) - np.sqrt(first[index].mean(1)))
    return {'rmse_a': float(np.sqrt(first.mean())), 'rmse_b': float(np.sqrt(second.mean())),
            'delta_b_minus_a': float(np.sqrt(second.mean()) - np.sqrt(first.mean())),
            'conditional_95_percent_interval': np.quantile(differences, [.025, .975]).tolist(),
            'note': 'Conditional on fixed OOF predictions. Candidate/epoch selection, '
                    'overlapping training folds and shared speakers are not accounted for.'}
