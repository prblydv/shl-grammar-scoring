"""Isolated two-update worst-case CUDA probe; OOM never poisons the trainer."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
import train_worker as worker
import audio_features as audio
import spectrogram_features as spectrogram
import waveform_features as waveform
from telemetry import atomic_json, emit


def probe(config, model_dir, texts, mode, size):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This run requires a CUDA GPU with BF16 support')
    torch.set_num_threads(config.get('cpu_threads', 4))
    worker.set_seed(config['seed'])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, local_files_only=True) if audio.uses_text(config) else None
    encoded = tokenizer(texts, truncation=True, max_length=config['max_length'], padding=False) if tokenizer else {}
    order = sorted(range(len(texts)), key=lambda i:len(encoded['input_ids'][i]), reverse=True) if tokenizer else list(range(len(texts)))
    features = None
    worst_wave = None
    if audio.uses_audio(config):
        import pandas as pd
        store = audio.FeatureStore(config)
        # Training-fold probes measure the selected feature dimensions on real rows.
        frame = pd.read_parquet(Path(config['audio_data_path']).parent / 'data.parquet')
        training_audio = frame.loc[frame.split.eq('train')].reset_index(drop=True)
        if waveform.uses(config):
            longest = max(training_audio.itertuples(), key=lambda r: store.lookup[('train',str(r.filename))]['samples'])
            worst_wave = store.item('train',str(longest.filename))
        else:
            features = store.rows(training_audio)
    spec_store=None
    if spectrogram.uses(config):
        spec_store=spectrogram.Store(config)
        training=__import__('pandas').read_parquet(Path(config['spectrogram_index_path']).parent/'data.parquet')
        training=training.loc[training.split.eq('train')].reset_index(drop=True)
        longest=max(range(len(training)),key=lambda i:spec_store.lookup[('train',str(training.iloc[i].filename))]['frames'])
        worst_spec=spec_store.item('train',str(training.iloc[longest].filename))
    def batch(n, maximum=True):
        rows = [{k:v[order[i % len(order)]] for k,v in encoded.items()} for i in range(n)]
        result = tokenizer.pad(rows, padding='max_length' if maximum else True,
                               max_length=config['max_length'] if maximum else None,
                               pad_to_multiple_of=8, return_tensors='pt') if tokenizer else {}
        if features is not None:
            result['audio_features'] = torch.from_numpy(features[[order[i % len(order)] for i in range(n)]])
        if worst_wave is not None:
            for key,value in worst_wave.items(): result[key]=torch.cat([value]*n)
            result['waveform_owner']=torch.arange(n).repeat_interleave(len(worst_wave['waveform']))
        if spec_store:
            for key,value in worst_spec.items(): result[key]=torch.cat([value]*n)
            result['spectrogram_owner']=torch.arange(n).repeat_interleave(len(worst_spec['spectrogram']))
        return {k:v.cuda() for k,v in result.items()}
    torch.cuda.reset_peak_memory_stats()
    model = worker.GrammarModel(config, model_dir).cuda()
    if features is not None:
        model.fit_audio_normalization(features)
    if spec_store: model.fit_spectrogram_normalization(spec_store,training)
    optimizer = torch.optim.AdamW(worker.parameter_groups(model, config),
                                  eps=config['adam_epsilon'], betas=tuple(config['adam_betas']))
    train_size = size if mode == 'train' else config['batch_size']
    inputs = batch(train_size)
    labels = torch.linspace(0,5,train_size,device='cuda')
    before = model.head.weight.detach().clone()
    model.train()
    for update in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            prediction, logits = model(inputs)
            loss = worker.objective(prediction,logits,labels,config)
        if not torch.isfinite(loss): raise ValueError('Nonfinite probe loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        emit('batch_probe', probe_mode=mode, probe_size=size, optimizer_update=update+1, loss=float(loss.detach()), gradient_norm=float(norm))
    if torch.equal(before,model.head.weight.detach()): raise ValueError('Optimizer did not update the head')
    optimizer.zero_grad(set_to_none=True)
    # Keep training optimizer states resident for a conservative inference measurement.
    model.eval()
    eval_size = size if mode == 'eval' else 1
    evaluation = batch(eval_size)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        predicted,_ = model(evaluation)
        ordinary,_ = model(batch(train_size,maximum=False))
        padded,_ = model(inputs)
    if not torch.isfinite(predicted).all(): raise ValueError('Nonfinite inference')
    torch.testing.assert_close(ordinary,padded,rtol=2e-3,atol=2e-3)
    torch.cuda.synchronize()
    maximum = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(0).total_memory
    return {'mode':mode,'size':size,'optimizer_updates':2,'precision':'torch.bfloat16',
            'loss':float(loss.detach()),'peak_reserved_mib':maximum/2**20,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'reserved_fraction':maximum/total,'accepted':maximum/total<.9,
            'parameter_count':sum(p.numel() for p in model.parameters())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--mode',choices=['train','eval'],required=True)
    parser.add_argument('--size',type=int,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import pandas as pd
    try:
        result=probe(json.loads(args.config.read_text()),args.model,pd.read_parquet(args.data).query('split == "train"').text.tolist(),args.mode,args.size)
    except torch.cuda.OutOfMemoryError as error:
        result={'mode':args.mode,'size':args.size,'accepted':False,'oom':True,'error':str(error)}
    atomic_json(args.report,result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
