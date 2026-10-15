"""Explicit pretrained catalog; download only the selected, revision-pinned model."""
import json
import os
from pathlib import Path

SPEECH_REPOS = {
    'wavlm_base_plus': 'microsoft/wavlm-base-plus',
    'wavlm_large': 'microsoft/wavlm-large',
    'hubert_base': 'facebook/hubert-base-ls960',
    'hubert_large': 'facebook/hubert-large-ls960-ft',
    'hubert_xlarge': 'facebook/hubert-xlarge-ls960-ft',
    'wav2vec2_base': 'facebook/wav2vec2-base-960h',
    'wav2vec2_large': 'facebook/wav2vec2-large-960h',
    'wav2vec2_large_lv60': 'facebook/wav2vec2-large-lv60',
    'wav2vec2_xlsr300m': 'facebook/wav2vec2-xls-r-300m',
}

AST_REPOS = {
    'ast_audioset': 'MIT/ast-finetuned-audioset-10-10-0.4593',
    'ast_audioset_v2': 'MIT/ast-finetuned-audioset-10-10-0.448-v2',
    'ast_audioset_stride12': 'MIT/ast-finetuned-audioset-12-12-0.447',
    'ast_audioset_stride14': 'MIT/ast-finetuned-audioset-14-14-0.443',
    'ast_audioset_stride16': 'MIT/ast-finetuned-audioset-16-16-0.442',
    'ast_speech_commands': 'MIT/ast-finetuned-speech-commands-v2',
    'ast_distilled': 'bookbot/distil-ast-audioset',
}


def cache_root(output):
    runtime = Path(os.environ.get('SHL_RUNTIME_ROOT', str(Path(output).resolve().parents[1]))).resolve()
    if runtime == Path('/workspace') or Path('/workspace') in runtime.parents:
        raise ValueError('Model/runtime caches must stay outside /workspace')
    directory = runtime / 'model_cache'
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def selected_download(repo, revision, output, report_name):
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig
    from sweep_utils import model_fingerprint
    from telemetry import atomic_json
    report_path = Path(output) / report_name
    prior = json.loads(report_path.read_text()) if report_path.exists() else None
    if prior and (prior['model_id'] != repo or (revision and revision != prior['revision'])):
        raise ValueError('Pretrained model changed: use a new experiment name')
    pinned = prior['revision'] if prior else HfApi().model_info(repo, revision=revision or 'main').sha
    files = HfApi().list_repo_files(repo, revision=pinned)
    weights = ['*.safetensors'] if any('/' not in p and p.endswith('.safetensors') for p in files) else ['pytorch_model*.bin']
    path = snapshot_download(repo, revision=pinned, cache_dir=str(cache_root(output)),
                             allow_patterns=['*.json', *weights], ignore_patterns=['*/*'])
    architecture = AutoConfig.from_pretrained(path, local_files_only=True).to_dict()
    provenance = dict(model_id=repo, revision=pinned, model_sha256=model_fingerprint(path))
    if prior and prior != provenance:
        raise ValueError('Cached pretrained assets changed')
    atomic_json(report_path, provenance)
    return path, json.loads(json.dumps(architecture)), provenance
