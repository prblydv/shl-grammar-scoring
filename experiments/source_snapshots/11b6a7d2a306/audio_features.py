"""Selected fixed audio channels, aligned by split/filename, with fold-local scaling."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def uses_text(config):
    return config.get('num_backbones', 1) == 2 or config.get('single_backbone_modality', 'text') == 'text'


def uses_audio(config):
    return config.get('num_backbones', 1) == 2 or config.get('single_backbone_modality', 'text') == 'audio'


def validate(config):
    if type(config['num_backbones']) is not int or config['num_backbones'] not in (1, 2):
        raise ValueError('NUM_BACKBONES must be 1 or 2')
    if config['single_backbone_modality'] not in ('text', 'audio'):
        raise ValueError('SINGLE_BACKBONE_MODALITY must be text or audio')
    if not uses_audio(config):
        return
    names = config['audio_encodings']
    if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
        raise ValueError('AUDIO_ENCODINGS must be a nonempty list of unique channel names')
    if config['audio_backbone'] not in ('mlp', 'residual_mlp', 'gated_mlp'):
        raise ValueError('Choose mlp, residual_mlp or gated_mlp for saved fixed vectors')
    if config['audio_normalization'] not in ('standard', 'none'):
        raise ValueError('Choose standard or none audio normalization')
    if config['fusion'] != 'concat':
        raise ValueError('Only concat fusion is implemented')
    for key in ('audio_hidden_dim', 'audio_num_layers', 'audio_output_dim', 'fusion_hidden_dim'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Invalid ' + key)
    for key in ('audio_dropout', 'fusion_dropout'):
        if not 0 <= config[key] < 1:
            raise ValueError('Invalid ' + key)
    if not np.isfinite(config['audio_lr']) or config['audio_lr'] <= 0:
        raise ValueError('AUDIO_LR must be positive and finite')


def prepare(config, output):
    """Freeze only selected columns; never fit transforms on validation/test rows."""
    import h5py
    import pandas as pd
    import sweep_utils as integrity
    output = Path(output)
    source = Path(config['audio_feature_file']).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError('Set AUDIO_FEATURE_FILE to your extracted shl_all_channels.h5: ' + str(source))
    with h5py.File(source, 'r') as handle:
        channels = json.loads(handle.attrs['channels_json'])
        unknown = set(config['audio_encodings']) - set(channels)
        if unknown:
            raise ValueError(f'Unknown fixed channels: {sorted(unknown)}. Available: {list(channels)}')
        matrices, widths = [], {}
        for name in config['audio_encodings']:
            definition = channels[name]
            start, stop = definition['start'], definition['stop']
            values = np.asarray(handle['X'][:, start:stop], dtype=np.float32)
            available = np.asarray(handle['available'][:, start:stop], dtype=bool)
            values[~available | ~np.isfinite(values)] = np.nan
            matrices.append(values)
            widths[name] = stop - start
        values = np.concatenate(matrices, axis=1)
        keys = list(zip(handle['metadata/split'].asstr()[:], handle['metadata/filename'].asstr()[:]))
        labels = np.asarray(handle['metadata/label'][:], dtype=float)
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate split/filename keys in audio artifact')
    lookup = {key: i for i, key in enumerate(keys)}
    frame = pd.read_parquet(output / 'data.parquet')
    wanted = list(zip(frame.split.astype(str), frame.filename.astype(str)))
    missing = set(wanted) - set(lookup)
    if missing:
        raise ValueError(f'Audio artifact is missing recordings: {sorted(missing)[:10]}')
    indices = [lookup[key] for key in wanted]
    known = frame.split.eq('train').to_numpy()
    if not np.array_equal(labels[indices][known], frame.loc[known, 'label'].to_numpy(dtype=float)):
        raise ValueError('Audio artifact training labels differ from notebook labels')
    selected = values[indices]
    # Reject a completely absent selected channel, while retaining legitimate
    # per-recording missing values for fold-local imputation.
    for name, matrix in zip(config['audio_encodings'], matrices):
        if not np.isfinite(matrix[indices][known]).any():
            raise ValueError('Selected channel has no available training features: ' + name)
    target = output / 'audio_features.npz'
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if (saved['keys'].tolist() != [list(key) for key in wanted] or
                    not np.array_equal(saved['values'], selected, equal_nan=True)):
                raise ValueError('Selected audio features changed; choose a new EXPERIMENT_NAME')
    else:
        temporary = output / 'audio_features.tmp.npz'
        np.savez_compressed(temporary, values=selected, keys=np.asarray(wanted, dtype=str))
        temporary.replace(target)
    result = dict(config, audio_input_dim=int(selected.shape[1]), audio_channel_dimensions=widths,
                  audio_data_path=str(target.resolve()), audio_data_sha256=integrity.file_sha256(target))
    from telemetry import atomic_json
    atomic_json(output / 'audio_report.json', dict(channels=widths, input_dim=selected.shape[1],
        source=str(source), missing_entries=int((~np.isfinite(selected)).sum()),
        normalization='Fit imputation/scaling on each training fold only; stored in model buffers'))
    return result


def verify(config):
    if uses_audio(config):
        import sweep_utils as integrity
        if integrity.file_sha256(config['audio_data_path']) != config['audio_data_sha256']:
            raise ValueError('Frozen audio features changed')


class FeatureStore:
    def __init__(self, config):
        verify(config)
        with np.load(config['audio_data_path'], allow_pickle=False) as data:
            self.values = data['values'].copy()
            self.lookup = {tuple(key): i for i, key in enumerate(data['keys'].tolist())}

    def rows(self, frame):
        return self.values[[self.lookup[(str(row.split), str(row.filename))] for row in frame.itertuples()]]


class AudioDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, features, labelled=True):
        self.audio = features
        self.encoded = tokenizer(frame.text.tolist(), truncation=True, max_length=max_length, padding=False) if tokenizer else {}
        self.labels = frame.label.to_numpy(dtype=np.float32) if labelled else None

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        row = {key: values[index] for key, values in self.encoded.items()}
        row['audio_features'] = self.audio[index]
        if self.labels is not None:
            row['labels'] = float(self.labels[index])
        return row


class AudioCollator:
    def __init__(self, tokenizer):
        from transformers import DataCollatorWithPadding
        self.text = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8, return_tensors='pt') if tokenizer else None

    def __call__(self, rows):
        audio = torch.from_numpy(np.stack([row['audio_features'] for row in rows]))
        text = [{key: value for key, value in row.items() if key != 'audio_features'} for row in rows]
        if self.text:
            batch = self.text(text)
        else:
            batch = {}
            if 'labels' in text[0]:
                batch['labels'] = torch.tensor([row['labels'] for row in text], dtype=torch.float32)
        batch['audio_features'] = audio
        return batch


class VectorBlock(nn.Module):
    def __init__(self, width, dropout, kind):
        super().__init__()
        self.kind = kind
        self.network = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Linear(width, width) if kind == 'gated_mlp' else None

    def forward(self, values):
        transformed = self.network(values)
        if self.gate is not None:
            gate = self.gate(values).sigmoid()
            return gate * transformed + (1 - gate) * values
        return values + transformed if self.kind == 'residual_mlp' else transformed


class AudioGrammarModel(nn.Module):
    def __init__(self, config, text_model, initial_mean):
        super().__init__()
        self.config = config
        self.backbone = text_model.backbone if text_model else None
        if text_model:
            self.pooler = text_model.pooler
        width, hidden, output = config['audio_input_dim'], config['audio_hidden_dim'], config['audio_output_dim']
        self.register_buffer('audio_mean', torch.zeros(width))
        self.register_buffer('audio_scale', torch.ones(width))
        self.audio_backbone = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
            *[VectorBlock(hidden, config['audio_dropout'], config['audio_backbone']) for _ in range(config['audio_num_layers'])],
            nn.LayerNorm(hidden), nn.Linear(hidden, output), nn.GELU())
        combined = output + (self.pooler.output_dim if text_model else 0)
        if text_model:
            self.fusion = nn.Sequential(nn.LayerNorm(combined), nn.Linear(combined, config['fusion_hidden_dim']),
                nn.GELU(), nn.Dropout(config['fusion_dropout']))
            combined = config['fusion_hidden_dim']
        else:
            self.fusion = nn.Identity()
        self.norm = nn.LayerNorm(combined)
        self.dropout = nn.Dropout(config['head_dropout'])
        self.head = nn.Linear(combined, 1)
        nn.init.normal_(self.head.weight, std=.01)
        rate = np.clip(initial_mean / config['prediction_max'], .01, .99)
        nn.init.constant_(self.head.bias, float(np.log(rate / (1-rate))) if config['output_mode'] == 'bounded' else initial_mean)
        self.layer_report = dict(text_model.layer_report if text_model else {},
            modalities=['text', 'audio'] if text_model else ['audio'], audio_backbone=config['audio_backbone'],
            audio_input_dim=width, audio_output_dim=output,
            total_parameters=sum(p.numel() for p in self.parameters()),
            trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def fit_audio_normalization(self, values):
        finite = np.isfinite(values)
        count = finite.sum(axis=0)
        mean = np.divide(np.where(finite, values, 0).astype(np.float64).sum(axis=0), count,
                         out=np.zeros(values.shape[1], dtype=np.float64), where=count > 0)
        filled = np.where(finite, values, mean).astype(np.float64)
        scale = filled.std(axis=0)
        scale[scale < 1e-6] = 1.
        if self.config['audio_normalization'] == 'none':
            scale[:] = 1.
        self.audio_mean.copy_(torch.as_tensor(mean, dtype=torch.float32, device=self.audio_mean.device))
        self.audio_scale.copy_(torch.as_tensor(scale, dtype=torch.float32, device=self.audio_scale.device))

    def forward(self, inputs):
        values = inputs['audio_features'].float()
        values = torch.where(torch.isfinite(values), values, self.audio_mean)
        if self.config['audio_normalization'] == 'standard':
            values = values - self.audio_mean
        audio = self.audio_backbone(values / self.audio_scale)
        if self.backbone is not None:
            text = {key: value for key, value in inputs.items() if key != 'audio_features'}
            output = self.backbone(**text, output_hidden_states=True) if self.pooler.needs_layers else self.backbone(**text)
            audio = torch.cat([self.pooler(output, text['attention_mask']), audio], dim=-1)
        logits = self.head(self.dropout(self.norm(self.fusion(audio)))).float().squeeze(-1)
        predictions = self.config['prediction_max'] * logits.sigmoid() if self.config['output_mode'] == 'bounded' else logits
        return predictions, logits
