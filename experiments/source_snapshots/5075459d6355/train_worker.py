"""One resumable Vast.ai fold; derived from the tested DeBERTa sweep worker."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
import telemetry
from pooling import Pooling
from transformers import AutoConfig, AutoModel, AutoTokenizer, DataCollatorWithPadding
from transformers import get_linear_schedule_with_warmup
import audio_features as audio
import spectrogram_features as spectrogram
import waveform_features as waveform
import loader_utils


def sweep_helpers():
    """Load the utility beside this worker, including notebook-written copies."""
    path = Path(__file__).with_name('sweep_utils.py')
    spec = importlib.util.spec_from_file_location('shl_worker_sweep_utils', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_save(value, path):
    path = Path(path)
    if Path('/workspace') in path.resolve().parents:
        raise ValueError('Training checkpoints must stay outside /workspace; use /opt/shl-runtime/runs')
    telemetry.emit('checkpoint_save',checkpoint=path.name)
    temporary = path.with_suffix('.tmp')
    storages = {}
    def measure(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif isinstance(item, dict):
            for child in item.values(): measure(child)
        elif isinstance(item, (list, tuple)):
            for child in item: measure(child)
    measure(value)
    required = sum(storages.values()) + 2 * 2**30
    if shutil.disk_usage(path.parent).free < required:
        raise RuntimeError(f'Checkpoint needs {required/2**30:.1f} GiB free including a 2 GiB reserve; previous checkpoint preserved')
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length, labelled=True):
        self.encoded = tokenizer(frame.text.tolist(), truncation=True,
                                 max_length=max_length, padding=False)
        self.labels = frame.label.to_numpy(dtype=np.float32) if labelled else None

    def __len__(self):
        return len(self.encoded['input_ids'])

    def __getitem__(self, index):
        row = {key: value[index] for key, value in self.encoded.items()}
        if self.labels is not None:
            row['labels'] = float(self.labels[index])
        return row


class TextGrammarModel(nn.Module):
    def __init__(self, config, model_dir, initial_mean=2.5, random_backbone=False):
        super().__init__()
        if random_backbone:  # Used only for the small CPU integration test.
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
        else:
            self.backbone = AutoModel.from_pretrained(model_dir, local_files_only=True)
        hidden = getattr(self.backbone.config,'hidden_size',getattr(self.backbone.config,'dim',None))
        self.pooling = config['pooling']
        blocks = backbone_blocks(self.backbone)
        self.pooler = Pooling(config,hidden,len(blocks))
        block_names={id(module):name for name,module in self.backbone.named_modules()}
        self.loss_name = config['loss']
        self.maximum = float(config['prediction_max'])
        self.run_output_mode = config['output_mode']
        self.ordinal_steps = int(round(self.maximum / 0.5))
        self.norm = nn.LayerNorm(self.pooler.output_dim)
        self.dropout = nn.Dropout(config.get('head_dropout',0.0))
        self.head = nn.Linear(self.pooler.output_dim, self.ordinal_steps if self.loss_name == 'ordinal' else 1)
        nn.init.normal_(self.head.weight, std=0.01)
        with torch.no_grad():
            if self.loss_name == 'ordinal':
                # Ordered cumulative logits: first logit minus positive increments.
                self.head.bias[0] = initial_mean - 0.25
                self.head.bias[1:] = math.log(math.expm1(0.5))
            else:
                rate = np.clip(initial_mean / self.maximum, 0.01, 0.99)
                self.head.bias.fill_(math.log(rate / (1 - rate)) if config['output_mode'] == 'bounded' else initial_mean)
        blocks = backbone_blocks(self.backbone)
        count = len(blocks)
        frozen = config['freeze_top_n_layers']
        reset = config['re_init_top_n_layers']
        for name, value in [('freeze_top_n_layers', frozen), ('re_init_top_n_layers', reset)]:
            if type(value) is not int or not 0 <= value <= count:
                raise ValueError(f'{name} must be an integer in [0, {count}]')
        if frozen and reset:
            raise ValueError('Top frozen and reinitialized blocks overlap; choose separate trials.')
        for block in list(blocks)[count-reset:] if reset else []:
            block.apply(self.backbone._init_weights)
        for block in list(blocks)[count-frozen:] if frozen else []:
            for parameter in block.parameters():
                parameter.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, nn.Dropout):
                module.p = config.get('backbone_dropout',0.0)
        self.layer_report = {
            'backbone': config.get('model_id'), 'total_blocks': count, 'pooling':self.pooling, 'pooling_output_dim':self.pooler.output_dim,
            'head_dropout':config.get('head_dropout',0.),'backbone_dropout':config.get('backbone_dropout',0.),
            'frozen_indices': list(range(count-frozen, count)),
            'frozen_names': [block_names[id(blocks[i])] for i in range(count-frozen,count)],
            'reinitialized_indices': list(range(count-reset, count)),
            'reinitialized_names': [block_names[id(blocks[i])] for i in range(count-reset,count)],
            'trainable_blocks': count-frozen,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)}
        print(json.dumps(self.layer_report), flush=True)
        if config['gradient_checkpointing']:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})

    def forward(self, inputs):
        output = self.backbone(**inputs,output_hidden_states=True) if self.pooler.needs_layers else self.backbone(**inputs)
        pooled = self.pooler(output,inputs['attention_mask'])
        raw = self.head(self.dropout(self.norm(pooled))).float()
        if self.loss_name == 'ordinal':
            logits = torch.cat([raw[:, :1], raw[:, :1] - F.softplus(raw[:, 1:]).cumsum(1)], dim=1)
            prediction = 0.5 * logits.sigmoid().sum(1)
        else:
            logits = raw.squeeze(-1)
            prediction = self.maximum * logits.sigmoid() if self.run_output_mode == 'bounded' else logits
        return prediction, logits


def GrammarModel(config, model_dir, initial_mean=2.5, random_backbone=False):
    text = TextGrammarModel(config, model_dir, initial_mean, random_backbone) if audio.uses_text(config) else None
    if spectrogram.uses(config): return spectrogram.GrammarModel(config,text,initial_mean,random_backbone)
    return audio.AudioGrammarModel(dict(config,_random_audio_backbone=random_backbone), text, initial_mean) if audio.uses_audio(config) else text


def objective(prediction, logits, labels, config):
    maximum = config['prediction_max']
    if config['loss'] == 'mse':
        return F.mse_loss(prediction, labels)
    if config['loss'] == 'bce':
        return F.binary_cross_entropy_with_logits(logits, labels / maximum)
    if config['loss'] == 'huber':
        return F.smooth_l1_loss(prediction / maximum, labels / maximum, beta=config['huber_beta'])
    if config['loss'] == 'ordinal':
        thresholds = torch.arange(logits.shape[1], device=labels.device).float() * 0.5 + 0.25
        target = (labels[:, None] > thresholds).float()
        return F.binary_cross_entropy_with_logits(logits, target)
    raise ValueError(f"Unknown loss: {config['loss']}")


def backbone_blocks(backbone):
    if hasattr(backbone,'encoder') and hasattr(backbone.encoder,'layer'):
        return backbone.encoder.layer
    if hasattr(backbone,'transformer') and hasattr(backbone.transformer,'layer'):
        return backbone.transformer.layer
    raise ValueError('Supported encoder families: DeBERTa, BERT, RoBERTa and DistilBERT. Encoder-decoder/audio/SVR require a separate pipeline.')


def parameter_groups(model, config):
    blocks=backbone_blocks(model.backbone) if model.backbone is not None else [];layers=len(blocks)
    block_ids={id(p):i for i,block in enumerate(blocks) for p in block.parameters()}
    groups={}
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        if id(parameter) in block_ids:
            exponent=layers-1-block_ids[id(parameter)]
            rate=config['encoder_lr']*config['layer_lr_decay']**exponent
        elif name.startswith('backbone.'):
            rate=config['encoder_lr']*config['layer_lr_decay']**layers
        elif name.startswith('spectrogram_backbone.'):
            rate=config['spectrogram_lr']
        elif name.startswith('audio_backbone.encoder.'):
            rate=config.get('audio_encoder_lr',1e-5)
        elif name.startswith('audio_backbone.'):
            rate=config.get('audio_lr', config['head_lr'])
        else: rate=config['head_lr']
        decay=0. if parameter.ndim==1 or name.endswith('bias') else config['weight_decay']
        groups.setdefault((rate,decay),[]).append(parameter)
    return [{'params':p,'lr':lr,'weight_decay':decay} for (lr,decay),p in groups.items()]


def trainable_state(model):
    # Full scoring model includes frozen weights and buffers for self-contained folds.
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def load_delta(model, state):
    model.load_state_dict(state, strict=True)


def move_batch(batch, device):
    labels = batch.pop('labels', None)
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    return inputs, None if labels is None else labels.to(device, non_blocking=True)


@torch.no_grad()
def predict(model, loader, device, amp_dtype, amp_enabled, stage="prediction"):
    started, last_log = time.monotonic(), 0.0
    model.eval()
    predictions = []
    telemetry.emit(stage,batch=0,batches=len(loader))
    for step, batch in enumerate(loader):
        inputs, _ = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output, _ = model(inputs)
        predictions.extend(output.float().cpu().tolist())
        telemetry.emit(stage,batch=step+1,batches=len(loader))
        now = time.monotonic()
        if now-last_log >= 15 or step+1 == len(loader):
            eta = (now-started)/(step+1)*(len(loader)-step-1)
            print(f"{stage}: batch {step+1}/{len(loader)} | ETA {eta/60:.1f} min", flush=True)
            last_log = now
    predictions = np.asarray(predictions, dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError('Nonfinite predictions.')
    return predictions


def _smoke_variant(config, model_dir, tokenizer, encoded, device, seed):
    """Exercise an actual optimizer update at the configured maximum sequence size."""
    set_seed(seed)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    if amp_enabled:
        torch.cuda.reset_peak_memory_stats(device)
    model = GrammarModel(config, model_dir).to(device)
    padded_length = math.ceil(config['max_length'] / 8) * 8
    if padded_length > model.backbone.config.max_position_embeddings:
        raise ValueError('Padded max_length exceeds the backbone position limit.')
    optimizer = torch.optim.AdamW(parameter_groups(model, config), eps=config['adam_epsilon'], betas=tuple(config['adam_betas']))
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    order = sorted(range(len(encoded['input_ids'])),
                   key=lambda index: len(encoded['input_ids'][index]), reverse=True)
    def examples(size):
        return [{key: values[order[index % len(order)]] for key, values in encoded.items()}
                for index in range(size)]
    def padded_batch(size, maximum=False):
        rows = examples(size)
        return tokenizer.pad(rows, padding='max_length' if maximum else True,
                             max_length=config['max_length'] if maximum else None,
                             pad_to_multiple_of=8, return_tensors='pt')
    # Ordinary dynamic padding and a worst-case padded batch both use real texts.
    dynamic = padded_batch(config['batch_size'])
    training_batch = padded_batch(config['batch_size'], maximum=True)
    inputs = {key: value.to(device) for key, value in training_batch.items()}
    labels = torch.linspace(0, config['prediction_max'], config['batch_size'], device=device)
    before = model.head.weight.detach().clone()
    model.train()
    for attempt in range(8):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            prediction, logits = model(inputs)
            loss = objective(prediction, logits, labels, config)
        if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
            raise ValueError('Nonfinite smoke-test training output/loss; change precision before the sweep.')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite_gradients = bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients)
        if finite_gradients:
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=not scaler.is_enabled())
        elif not scaler.is_enabled():
            raise ValueError('Nonfinite or missing smoke-test gradients.')
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if finite_gradients and scaler.get_scale() >= old_scale and attempt >= 1:
            break
    else:
        raise ValueError('AMP could not produce a finite optimizer update after eight scale adjustments.')
    del gradients
    if torch.equal(before, model.head.weight.detach()):
        raise ValueError('Smoke-test optimizer did not update the prediction head.')
    head_update = float(torch.linalg.vector_norm(model.head.weight.detach() - before).cpu())
    optimizer.zero_grad(set_to_none=True)
    eval_batch = padded_batch(config['eval_batch_size'], maximum=True)
    eval_inputs = {key: value.to(device) for key, value in eval_batch.items()}
    model.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        evaluation, _ = model(eval_inputs)
        dynamic_inputs = {key: value.to(device) for key, value in dynamic.items()}
        dynamic_output, _ = model(dynamic_inputs)
        same_examples, _ = model(inputs)
    if not torch.isfinite(evaluation).all() or not torch.isfinite(dynamic_output).all():
        raise ValueError('Nonfinite smoke-test evaluation predictions.')
    torch.testing.assert_close(dynamic_output, same_examples, rtol=2e-3, atol=2e-3)
    if config['output_mode'] == 'bounded' and not ((evaluation >= 0) & (evaluation <= config['prediction_max'])).all():
        raise ValueError('Smoke-test predictions are outside the configured score range.')
    if amp_enabled:
        torch.cuda.synchronize(device)
    report = {'device': str(device), 'gpu_name': torch.cuda.get_device_name(device) if amp_enabled else None,
              'precision': str(amp_dtype) if amp_enabled else 'torch.float32',
              'loss': config['loss'], 'pooling': config['pooling'], 'training_loss': float(loss.detach().cpu()),
              'optimizer_attempts': attempt + 1, 'head_update_norm': head_update,
              'train_shape': list(inputs['input_ids'].shape), 'eval_shape': list(eval_inputs['input_ids'].shape),
              'dynamic_shape': list(dynamic['input_ids'].shape),
              'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20 if amp_enabled else None,
              'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20 if amp_enabled else None}
    return report


def smoke_test(config, model_dir, texts, losses=('mse',),
               poolings=None, gpu_ids=None, allow_cpu_test=False, seed=42):
    """Verify every configured loss/pool on real weights before scheduling folds.

    The test uses actual longest texts, max-length train/eval batches, checkpointing
    and the same CUDA autocast/scaler as training. It writes no fold checkpoints,
    restores caller RNG/settings and returns a JSON-serializable memory report.
    """
    texts = list(texts)
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Smoke test requires nonempty transcript strings.')
    if min(config['batch_size'], config['eval_batch_size'], config['max_length']) < 1:
        raise ValueError('Smoke-test batch sizes and max_length must be positive.')
    poolings = poolings or (config['pooling'],)
    if not losses or not poolings:
        raise ValueError('Smoke test requires at least one loss and pooling method.')
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not allow_cpu_test:
        raise RuntimeError('GPU smoke test requires the selected Kaggle GPUs.')
    if use_cuda:
        ids = list(range(torch.cuda.device_count())) if gpu_ids is None else list(gpu_ids)
        if not ids or any(gpu < 0 or gpu >= torch.cuda.device_count() for gpu in ids):
            raise ValueError('Invalid smoke-test GPU IDs.')
        devices = [torch.device(f'cuda:{gpu}') for gpu in ids]
    else:
        devices = [torch.device('cpu')]
    previous = {'random': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if use_cuda else [],
                'threads': torch.get_num_threads(), 'benchmark': torch.backends.cudnn.benchmark,
                'deterministic': torch.backends.cudnn.deterministic}
    start, reports = time.monotonic(), []
    try:
        torch.set_num_threads(config.get('cpu_threads', 2))
        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True)
        encoded = tokenizer(texts, truncation=True, max_length=config['max_length'], padding=False)
        for device in devices:
            for loss in losses:
                for pooling in poolings:
                    variant = dict(config, loss=loss, pooling=pooling)
                    print(f'Smoke test {device}: {loss}/{pooling}', flush=True)
                    if device.type == 'cuda':
                        with torch.cuda.device(device):
                            reports.append(_smoke_variant(variant, model_dir, tokenizer, encoded, device, seed))
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        reports.append(_smoke_variant(variant, model_dir, tokenizer, encoded, device, seed))
                    print(f"Passed: train {reports[-1]['train_shape']}, eval {reports[-1]['eval_shape']}, "
                          f"peak {reports[-1]['peak_allocated_mib']} MiB", flush=True)
    finally:
        random.setstate(previous['random'])
        np.random.set_state(previous['numpy'])
        torch.set_rng_state(previous['torch'])
        if use_cuda:
            torch.cuda.set_rng_state_all(previous['cuda'])
            gc.collect()
            torch.cuda.empty_cache()
        torch.set_num_threads(previous['threads'])
        torch.backends.cudnn.benchmark = previous['benchmark']
        torch.backends.cudnn.deterministic = previous['deterministic']
    return {'passed': True, 'elapsed_seconds': time.monotonic() - start,
            'model_revision': config.get('model_revision'), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'variants': reports}


def run_job(job, model_factory=None, tokenizer_override=None):
    start = time.monotonic()
    elapsed_before = 0.0
    directory = Path(job['job_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    config = job['config']
    integrity = sweep_helpers()
    integrity.verify_job_inputs(job, worker_path=__file__)
    signature = integrity.expected_job_signature(job, torch.__version__, transformers.__version__)
    telemetry.context(fold=job['fold'], candidate_id=job['candidate_id'])
    telemetry.emit('load_fold')
    if (directory / 'complete.json').exists():
        report = integrity.validate_completed(job, torch.__version__, transformers.__version__,
                                              worker_path=__file__, verify_inputs=False)
        print(f"Cached {job['candidate_id']} seed={job['seed']} fold={job['fold']}", flush=True)
        return report
    device = torch.device('cuda:0' if torch.cuda.is_available() and not job.get('force_cpu') else 'cpu')
    if device.type == 'cpu' and not job.get('allow_cpu_test', False):
        raise RuntimeError('Vast training requires CUDA; CPU is reserved for integration tests.')
    torch.set_num_threads(config.get('cpu_threads', 2))
    set_seed(job['seed'] + config.get('model_seed_stride',1009) * job['fold'])
    frame = pd.read_parquet(job['data_path'])
    train = frame[frame['split'].eq('train') & frame.fold.ne(job['fold'])].reset_index(drop=True)
    valid = frame[frame['split'].eq('train') & frame.fold.eq(job['fold'])].reset_index(drop=True)
    test = frame[frame['split'].eq('test')].reset_index(drop=True)
    assert len(train) and len(valid) and len(test)
    assert not set(train.filename).intersection(valid.filename)
    tokenizer = (tokenizer_override or AutoTokenizer.from_pretrained(job['model_dir'], use_fast=False,
                                                                    local_files_only=True)) if audio.uses_text(config) else None
    store = audio.FeatureStore(config) if audio.uses_audio(config) else None
    spec_store=spectrogram.Store(config) if spectrogram.uses(config) else None
    collator=spectrogram.MultiCollator(tokenizer) if (spec_store or store) else (audio.AudioCollator(tokenizer) if store else DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8,return_tensors='pt'))
    def loader(data, labelled, shuffle=False):
        dataset=spectrogram.MultiDataset(data,tokenizer,config['max_length'],store,spec_store,labelled) if (spec_store or store) else (audio.AudioDataset(data,tokenizer,config['max_length'],store.rows(data),labelled) if store else TextDataset(data,tokenizer,config['max_length'],labelled))
        return DataLoader(dataset,
                          batch_size=config['batch_size'] if shuffle else config['eval_batch_size'],
                          shuffle=False, collate_fn=collator,
                          sampler=torch.utils.data.RandomSampler(dataset,generator=torch.Generator()) if shuffle else None,
                          generator=torch.Generator().manual_seed(config['seed']),
                          **loader_utils.options(config,device))
    training_loader = loader(train, True, True)
    valid_loader = loader(valid, True)
    test_loader = loader(test, False)
    model = (model_factory(config, job['model_dir'], float(train.label.mean())) if model_factory
             else GrammarModel(config, job['model_dir'], float(train.label.mean()))).to(device)
    if store:
        model.fit_audio_normalization(store.rows(train))
    if spec_store:
        model.fit_spectrogram_normalization(spec_store,train)
    atomic_json(dict(model.layer_report, initialization_seed=job['seed'] + config.get('model_seed_stride',1009) * job['fold']), directory / 'layers.json')
    optimizer = torch.optim.AdamW(parameter_groups(model, config), eps=config['adam_epsilon'], betas=tuple(config['adam_betas']))
    updates_per_epoch = math.ceil(len(training_loader) / config['gradient_accumulation'])
    total_updates = updates_per_epoch * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config['warmup_ratio'] * total_updates),
                                                total_updates)
    amp_enabled = device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled and amp_dtype == torch.float16)
    precision = str(amp_dtype) if amp_enabled else 'torch.float32'
    gpu_name = torch.cuda.get_device_name(device) if amp_enabled else None
    history, best, best_epoch, stale, first_epoch = [], math.inf, -1, 0, 0
    resume_path = directory / 'last.pt'
    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        if checkpoint['signature'] != signature:
            raise ValueError('Resume checkpoint belongs to a different experiment/environment.')
        load_delta(model, checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        history, best, best_epoch, stale = (checkpoint[k] for k in ['history', 'best', 'best_epoch', 'stale'])
        first_epoch = checkpoint['next_epoch']
        elapsed_before = float(checkpoint['elapsed_seconds'])
        if not math.isfinite(elapsed_before) or elapsed_before < 0:
            raise ValueError('Resume checkpoint has invalid accumulated runtime.')
        if history and elapsed_before < history[-1]['elapsed_seconds']:
            raise ValueError('Resume runtime is earlier than its saved training history.')
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_state'])
        torch.set_rng_state(checkpoint['torch_state'].cpu())
        if amp_enabled:
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint['cuda_states']])
        del checkpoint
        print(f'Resuming epoch {first_epoch + 1}', flush=True)
    print(f"Loaded model on {device}: {len(train)} train / {len(valid)} validation / {len(test)} test; "
          f"{len(training_loader)} batches per epoch; max epochs {config['epochs']}", flush=True)
    for epoch in range(first_epoch, config['epochs']):
        if stale >= config['patience']:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_start, last_progress = time.monotonic(), 0.0
        total_loss, samples = 0.0, 0
        gradient_norm = 0.0
        telemetry.context(epoch=epoch+1)
        telemetry.emit('training',batch=0,batches=len(training_loader))
        print(f"Starting epoch {epoch+1}/{config['epochs']}", flush=True)
        training_loader.sampler.generator.manual_seed(job['seed'] + config.get('model_seed_stride',1009)*job['fold'] + epoch*100003)
        for step, batch in enumerate(training_loader):
            inputs, labels = move_batch(batch, device)
            # Weight each microbatch by its actual size, including partial final groups.
            group_start = (step // config['gradient_accumulation']) * config['gradient_accumulation']
            group_stop = min(group_start + config['gradient_accumulation'], len(training_loader))
            group_samples = min(group_stop * config['batch_size'], len(train)) - group_start * config['batch_size']
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                prediction, logits = model(inputs)
                loss = objective(prediction, logits, labels, config)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss; inspect the worker log.')
            scaler.scale(loss * len(labels) / group_samples).backward()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
            if step + 1 == group_stop:
                scaler.unscale_(optimizer)
                gradient_norm = float(nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'], error_if_nonfinite=not scaler.is_enabled()))
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step=epoch*updates_per_epoch+(step//config['gradient_accumulation'])+1
                telemetry.scalar(f"fold_{job['fold']}/gradient_norm",gradient_norm,update_step)
                for group_index,group in enumerate(optimizer.param_groups):
                    telemetry.scalar(f"fold_{job['fold']}/learning_rate_{group_index}",group['lr'],update_step)
            telemetry.emit('training',batch=step+1,batches=len(training_loader),train_loss=total_loss/samples,
                           learning_rates=[g['lr'] for g in optimizer.param_groups],gradient_norm=gradient_norm,
                           epoch_eta_seconds=(time.monotonic()-epoch_start)/(step+1)*(len(training_loader)-step-1))
            now = time.monotonic()
            if now-last_progress >= 15 or step+1 == len(training_loader):
                elapsed = now-epoch_start
                eta = elapsed/(step+1)*(len(training_loader)-step-1)
                memory = torch.cuda.memory_allocated(device)/2**30 if amp_enabled else 0.0
                status = {'stage':'training','epoch':epoch+1,'epochs':config['epochs'],
                    'batch':step+1,'batches':len(training_loader),'train_loss':total_loss/samples,
                    'epoch_eta_seconds':eta,'updated_at':time.time()}
                atomic_json(status,directory/'live_progress.json')
                print(f"{job['candidate_id']} fold={job['fold']} epoch={epoch+1}/{config['epochs']} "
                      f"batch={step+1}/{len(training_loader)} ({100*(step+1)/len(training_loader):.1f}%) "
                      f"train_loss={total_loss/samples:.5f} lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"GPU_memory={memory:.2f}GiB epoch_train_ETA={eta/60:.1f}min", flush=True)
                last_progress = now
        predictions = predict(model, valid_loader, device, amp_dtype, amp_enabled, stage='validation')
        rmse = float(np.sqrt(np.mean((valid.label.to_numpy() - predictions) ** 2)))
        telemetry.emit('validation_result',validation_rmse=rmse,train_loss=total_loss/samples)
        telemetry.scalar(f"fold_{job['fold']}/validation_rmse",rmse,epoch+1)
        telemetry.scalar(f"fold_{job['fold']}/train_mse",total_loss/samples,epoch+1)
        if rmse < best - config['min_improvement']:
            best, best_epoch, stale = rmse, epoch + 1, 0
            atomic_save({'model': trainable_state(model), 'config': config,
                         'signature': signature, 'epoch': best_epoch}, directory / 'best.pt')
        else:
            stale += 1
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / samples,
                        'validation_rmse': rmse,
                        'learning_rates': [float(group['lr']) for group in optimizer.param_groups],
                        'elapsed_seconds': elapsed_before + time.monotonic() - start})
        atomic_json(history, directory / 'history.json')
        print(f"{job['candidate_id']} seed={job['seed']} fold={job['fold']} epoch={epoch+1} "
              f"loss={total_loss/samples:.5f} RMSE={rmse:.5f} best={best:.5f}", flush=True)
        atomic_save({'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                     'signature': signature, 'next_epoch': epoch + 1, 'history': history,
                     'best': best, 'best_epoch': best_epoch, 'stale': stale,
                     'elapsed_seconds': elapsed_before + time.monotonic() - start,
                     'precision': precision, 'gpu_name': gpu_name,
                     'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
                     'torch_state': torch.get_rng_state(),
                     'cuda_states': torch.cuda.get_rng_state_all() if amp_enabled else []}, resume_path)
        if time.time() >= job.get('deadline_epoch_time', math.inf):
            print('Session budget reached; epoch checkpoint saved.', flush=True)
            return {'paused': True, 'signature': signature}
    telemetry.emit('load_best_checkpoint',best_epoch=best_epoch,best_rmse=best)
    checkpoint = torch.load(directory / 'best.pt', map_location='cpu', weights_only=False)
    if checkpoint['signature'] != signature:
        raise ValueError('Best checkpoint belongs to a different experiment/environment.')
    load_delta(model, checkpoint['model'])
    del checkpoint
    predictions = predict(model, valid_loader, device, amp_dtype, amp_enabled,stage='oof_inference')
    test_predictions = predict(model, test_loader, device, amp_dtype, amp_enabled, stage="test inference")
    valid_output = valid[['filename', 'label', 'fold']].copy()
    valid_output['prediction'] = predictions
    test_output = test[['filename']].copy()
    test_output['prediction'] = test_predictions
    valid_output.to_csv(directory / 'validation_predictions.csv', index=False)
    test_output.to_csv(directory / 'test_predictions.csv', index=False)
    report = {'signature': signature, 'candidate_id': job['candidate_id'], 'seed': job['seed'],
              'fold': job['fold'], 'best_epoch': best_epoch, 'best_rmse': best,
              'validation_rows': len(valid), 'training_rows': len(train),
              'elapsed_seconds': elapsed_before + time.monotonic() - start, 'device': str(device),
              'precision': precision, 'gpu_name': gpu_name,
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'history': history,
              'artifact_sha256': {name: integrity.file_sha256(directory / name)
                                  for name in ['best.pt', 'validation_predictions.csv', 'test_predictions.csv']}}
    atomic_json(report, directory / 'complete.json')
    # Completion checksums cover best weights and predictions. Optimizer state
    # is needed only for an unfinished fold and is the largest redundant file.
    resume_path.unlink(missing_ok=True)
    telemetry.emit('fold_complete',best_epoch=best_epoch,best_rmse=best)
    del model, optimizer
    gc.collect()
    if amp_enabled:
        torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    run_job(json.loads(args.job.read_text()))
