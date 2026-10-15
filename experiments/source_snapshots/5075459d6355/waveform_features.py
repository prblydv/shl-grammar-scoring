"""Trainable speech encoders on all raw-audio chunks; cached PCM stays outside workspace."""
import copy
import functools
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backbone_registry import SPEECH_REPOS, cache_root, selected_download


def uses(config):
    return config.get('audio_backbone') in SPEECH_REPOS


def validate(config):
    for key in ['audio_chunk_seconds', 'audio_chunk_batch_size']:
        value = config.get(key, 8 if key == 'audio_chunk_seconds' else 8)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid ' + key)
    if type(config.get('audio_chunk_batch_size', 8)) is not int:
        raise ValueError('AUDIO_CHUNK_BATCH_SIZE must be an integer')
    if round(config.get('audio_chunk_seconds', 8) * 16000) < 1:
        raise ValueError('Audio chunk duration must contain at least one sample')
    layers = config.get('audio_trainable_last_n_layers', 4)
    if layers is not None and (type(layers) is not int or layers < 0):
        raise ValueError('AUDIO_TRAINABLE_LAST_N_LAYERS must be None or a nonnegative integer')


def prepare(config, output):
    import soundfile as sf
    import pandas as pd
    from scipy.signal import resample_poly
    from sweep_utils import file_sha256
    from telemetry import atomic_json
    validate(config)
    output = Path(output)
    source = Path(config['audio_waveform_root']).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError('Set AUDIO_WAVEFORM_ROOT to the original full WAV dataset: ' + str(source))
    frame = pd.read_parquet(output / 'data.parquet')
    lookup = {}
    for split in ['train', 'test']:
        for p in (source / split).rglob('*.wav'):
            if '.ipynb_checkpoints' in p.parts:
                continue
            key = (split, p.name)
            if key in lookup:
                raise ValueError('Duplicate source waveform: ' + str(key))
            lookup[key] = p
    cache = cache_root(output).parent / 'waveform_cache'
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for row in frame.itertuples():
        key = (str(row.split), str(row.filename))
        if key not in lookup:
            raise FileNotFoundError('Missing source WAV: ' + str(key))
        path = lookup[key]; digest = file_sha256(path)
        target = cache / (digest + '_mono16k_v1.npy')
        if not target.exists():
            values, rate = sf.read(path, dtype='float32', always_2d=True)
            values = values.mean(axis=1)
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                values = resample_poly(values, 16000 // divisor, rate // divisor).astype(np.float32)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError('Empty/nonfinite source audio: ' + str(path))
            temporary = target.with_suffix('.tmp.npy')
            np.save(temporary, values, allow_pickle=False); temporary.replace(target)
        values = np.load(target, mmap_mode='r', allow_pickle=False)
        if values.ndim != 1 or not len(values):
            raise ValueError('Invalid waveform cache: ' + str(target))
        records.append(dict(split=key[0], filename=key[1], source_path=str(path), source_sha256=digest,
                            pcm_path=str(target), pcm_sha256=file_sha256(target), samples=len(values),
                            label=float(row.label) if row.split == 'train' else None))
    index = output / 'waveform_index.json'
    report = dict(sample_rate=16000, records=records, channel_policy='mean original channels; polyphase resample; no silence removal',
                  windows='all nonoverlapping windows plus final tail; preserve original start times',
                  chunk_seconds=config.get('audio_chunk_seconds', 8), normalization='per valid chunk if pretrained processor requests it')
    if index.exists() and json.loads(index.read_text()) != report:
        raise ValueError('Source audio changed: choose a new experiment name')
    atomic_json(index, report)
    directory, architecture, provenance = selected_download(SPEECH_REPOS[config['audio_backbone']],
        config.get('audio_model_revision'), output, 'audio_pretrained.json')
    if architecture['model_type'] not in {'wavlm', 'hubert', 'wav2vec2'}:
        raise ValueError('Unsupported speech encoder architecture')
    processor = Path(directory) / 'preprocessor_config.json'
    processor = json.loads(processor.read_text()) if processor.exists() else {'do_normalize': True, 'sampling_rate': 16000}
    if processor.get('sampling_rate', 16000) != 16000:
        raise ValueError('This waveform path requires a 16kHz speech encoder')
    result = dict(config, audio_data_path=str(index.resolve()), audio_data_sha256=file_sha256(index),
                  audio_input_dim=int(architecture['hidden_size']), audio_model_dir=directory,
                  audio_model_config=architecture, audio_model_id=provenance['model_id'],
                  audio_model_revision=provenance['revision'], audio_model_sha256=provenance['model_sha256'],
                  audio_do_normalize=bool(processor.get('do_normalize', True)))
    atomic_json(output / 'audio_report.json', dict(model=provenance, input='all raw waveform chunks', rows=len(records),
        sample_rate=16000, chunk_seconds=config.get('audio_chunk_seconds', 8),
        augmentation='none; runtime LayerDrop and SpecAugment disabled',
        trainable_last_n_layers=config.get('audio_trainable_last_n_layers', 4),
        freeze_feature_extractor=config.get('audio_freeze_feature_extractor', True), cache=str(cache)))
    return result


def verify(config):
    import sweep_utils
    if sweep_utils.file_sha256(config['audio_data_path']) != config['audio_data_sha256']:
        raise ValueError('Waveform index changed')
    if sweep_utils.model_fingerprint(config['audio_model_dir']) != config['audio_model_sha256']:
        raise ValueError('Pretrained speech encoder assets changed')


class Store:
    def __init__(self, config):
        verify(config); self.config = config
        report = json.loads(Path(config['audio_data_path']).read_text())
        self.lookup = {(r['split'], r['filename']): r for r in report['records']}
        self.load = functools.lru_cache(maxsize=8)(self._load)

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != 'load'}

    def __setstate__(self, state):
        self.__dict__.update(state); self.load = functools.lru_cache(maxsize=8)(self._load)

    def _load(self, split, filename):
        from sweep_utils import file_sha256
        row = self.lookup[(split, filename)]
        if file_sha256(row['pcm_path']) != row['pcm_sha256']:
            raise ValueError('Cached PCM changed: ' + filename)
        return np.load(row['pcm_path'], allow_pickle=False)

    def rows(self, frame):
        # Metadata only: decoding happens lazily in CPU data-loader workers.
        return [(str(r.split), str(r.filename)) for r in frame.itertuples()]

    def item(self, split, filename):
        values = self.load(str(split), str(filename)); step = int(round(self.config.get('audio_chunk_seconds', 8) * 16000))
        count = math.ceil(len(values) / step)
        chunks = np.zeros((count, step), np.float32)
        lengths = np.minimum(step, len(values) - np.arange(count) * step).astype(np.int64)
        for i, length in enumerate(lengths):
            chunks[i, :length] = values[i*step:i*step+length]
        return dict(waveform=torch.from_numpy(chunks), waveform_lengths=torch.from_numpy(lengths),
                    waveform_times=torch.arange(count, dtype=torch.float32) * (step / 16000))


class Encoder(nn.Module):
    def __init__(self, config, random_backbone=False):
        super().__init__(); self.config = config
        from transformers import AutoConfig, AutoModel
        architecture = copy.deepcopy(config['audio_model_config'])
        kind = architecture.pop('model_type')
        architecture = AutoConfig.for_model(kind, **architecture)
        # No stochastic layer skipping, especially when only final blocks are trainable.
        architecture.layerdrop = 0.
        self.encoder = AutoModel.from_config(architecture) if random_backbone else AutoModel.from_pretrained(
            config['audio_model_dir'], config=architecture, local_files_only=True)
        # Preserve the original parameter schema, but disable SpecAugment at runtime:
        # masking can fail on short final tails, and every original sample is retained.
        self.encoder.config.mask_time_prob = 0.
        self.encoder.config.mask_feature_prob = 0.
        layers = self.encoder.encoder.layers
        count = config.get('audio_trainable_last_n_layers', 4)
        if count is not None:
            if count > len(layers):
                raise ValueError('Requested more trainable speech blocks than this backbone contains')
            self.encoder.requires_grad_(False)
            if count:
                for layer in layers[-count:]: layer.requires_grad_(True)
        if config.get('audio_freeze_feature_extractor', True):
            self.encoder.feature_extractor.requires_grad_(False)
        if not any(p.requires_grad for p in self.encoder.feature_extractor.parameters()):
            # HF's frontend also has a flag that otherwise forces a gradient graph
            # through the frozen prefix when gradient checkpointing is enabled.
            self.encoder.feature_extractor._freeze_parameters()
        if config.get('gradient_checkpointing') and any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        hidden = architecture.hidden_size; width = config['audio_output_dim']
        self.project = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, width), nn.GELU(), nn.Dropout(config['audio_dropout']))
        self.position = nn.Linear(2, width, bias=False)
        self.attention = nn.Linear(width, 1)
        self.norm = nn.LayerNorm(width)
        receptive = 1; stride = 1
        for k, s in zip(architecture.conv_kernel, architecture.conv_stride):
            receptive += (k-1)*stride; stride *= s
        self.minimum_samples = receptive
        self.parameter_report = dict(model_id=config.get('audio_model_id', config['audio_backbone']),
            encoder_total_parameters=sum(p.numel() for p in self.encoder.parameters()),
            encoder_trainable_parameters=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad),
            trainable_last_n_layers=count, blocks=len(layers),
            branch_total_parameters=sum(p.numel() for p in self.parameters()),
            branch_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, inputs):
        raw = inputs['waveform']; lengths = inputs['waveform_lengths']; starts = inputs['waveform_times']; owner = inputs['waveform_owner']
        embeddings = [None] * len(raw)
        step = self.config.get('audio_chunk_batch_size', 8)
        # Trim each group to true length: convolutional group normalization must not see padded tails.
        for length in sorted(set(lengths.detach().cpu().tolist())):
            indices = (lengths == length).nonzero(as_tuple=True)[0]
            for first in range(0, len(indices), step):
                selected = indices[first:first+step]; values = raw[selected, :length].float()
                if self.config.get('audio_do_normalize', True):
                    values = (values-values.mean(1, keepdim=True)) * torch.rsqrt(values.var(1, unbiased=False, keepdim=True)+1e-7)
                if values.shape[1] < self.minimum_samples:
                    values = torch.nn.functional.pad(values, (0, self.minimum_samples-values.shape[1]))
                hidden = self.encoder(input_values=values).last_hidden_state
                projected = self.project(hidden.mean(1))
                for index, value in zip(selected.detach().cpu().tolist(), projected): embeddings[index] = value
        embedding = torch.stack(embeddings)
        center = starts + lengths.float() / 32000
        timed = torch.stack([torch.sin(center/10), torch.cos(center/10)], dim=-1)
        embedding = self.norm(embedding+self.position(timed))
        scores = self.attention(embedding).squeeze(-1).float()+lengths.float().log()
        return torch.stack([(embedding[owner==i] * scores[owner==i].softmax(0)[:,None].to(embedding.dtype)).sum(0)
                            for i in range(int(owner.max())+1)])
