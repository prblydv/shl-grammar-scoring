#!/usr/bin/env python3
"""Finish the README workflow without starting another transcription run."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def kaggle(*arguments):
    result = subprocess.run(
        [str(ROOT / 'kaggle'), *arguments],
        cwd=ROOT, text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def validate(path):
    frame = pd.read_parquet(path)
    required = {'audio_id', 'split', 'transcript', 'error'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    counts = frame['split'].value_counts().to_dict()
    if counts != {'train': 769, 'test': 216}:
        raise ValueError(f'Unexpected recording counts: {counts}')
    if frame[['split', 'audio_id']].isna().any().any():
        raise ValueError('Missing recording identifiers.')
    if frame.duplicated(['split', 'audio_id']).any():
        raise ValueError('Duplicate recording identifiers.')
    if frame['error'].notna().any():
        raise ValueError(f"Transcription errors: {frame['error'].notna().sum()}")
    valid_text = frame['transcript'].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    )
    if not valid_text.all():
        raise ValueError('Missing or empty transcripts.')
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true', help='Wait for the active run.')
    parser.add_argument('--upload', action='store_true', help='Create the private dataset.')
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'kernel-metadata.json').read_text())
    kernel = metadata['id']
    while True:
        status = kaggle('kernels', 'status', kernel)
        print(status, flush=True)
        if '"KernelWorkerStatus.COMPLETE"' in status:
            break
        if any(f'"KernelWorkerStatus.{state}"' in status for state in ('RUNNING', 'QUEUED')):
            if not args.wait:
                print('Run still active. Use --wait --upload to finish automatically.')
                return
            time.sleep(30)
            continue
        raise RuntimeError(f'Notebook did not complete successfully: {status}')

    output = ROOT / 'artifacts' / 'transcription'
    print(kaggle('kernels', 'output', kernel, '-p', str(output), '--force'), flush=True)
    frame = validate(output / 'transcripts.parquet')
    print(f'Validated {len(frame)} transcripts with no errors.', flush=True)
    dataset_id = kernel.split('/')[0] + '/shl-2026-whisper-small-transcripts'
    dataset_folder = ROOT / 'artifacts' / 'transcripts-dataset'
    dataset_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output / 'transcripts.parquet', dataset_folder / 'transcripts.parquet')
    dataset_metadata = {
        'id': dataset_id,
        'title': 'SHL 2026 Whisper Small Transcripts',
        'licenses': [{'name': 'other'}],
        'description': 'Private transcripts of the SHL Hiring Assessment 2026 '
                       'competition recordings. Original competition data terms apply.',
    }
    (dataset_folder / 'dataset-metadata.json').write_text(
        json.dumps(dataset_metadata, indent=2) + '\n'
    )
    if args.upload:
        # Kaggle datasets create defaults to private; never pass --public.
        print(kaggle('datasets', 'create', '-p', str(dataset_folder), '--keep-tabular'), flush=True)
        print(f'Private dataset: https://www.kaggle.com/datasets/{dataset_id}', flush=True)
    else:
        print(f'Dataset prepared in {dataset_folder}. Use --upload to create it.')


if __name__ == '__main__':
    main()
