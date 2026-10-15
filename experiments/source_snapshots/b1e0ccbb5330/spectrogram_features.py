"""A full-recording spectrogram branch with fold-local scaling and timed chunks."""
import functools
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

CNN_WIDTHS = {'cnn_tiny': [16,32], 'cnn_small': [32,64,128],
              'cnn_medium': [48,96,192,256], 'cnn_large': [64,128,256,384,512]}
AST_REPOS = {'ast_audioset':'MIT/ast-finetuned-audioset-10-10-0.4593',
             'ast_audioset_v2':'MIT/ast-finetuned-audioset-10-10-0.448-v2'}


def uses(config):
    n = config.get('num_backbones',1)
    return n==3 or (n==1 and config.get('single_backbone_modality')=='spectrogram') or (
        n==2 and 'spectrogram' in config.get('backbone_pair','text_audio').split('_'))


def validate(config):
    if not uses(config): return
    if config['spectrogram_backbone'] not in {*CNN_WIDTHS,*AST_REPOS}:
        raise ValueError('Unknown SPECTROGRAM_BACKBONE; choose '+str(list(CNN_WIDTHS)+list(AST_REPOS)))
    for key in ['spectrogram_chunk_frames','spectrogram_output_dim','spectrogram_chunk_batch_size']:
        if type(config[key]) is not int or config[key]<1: raise ValueError('Invalid '+key)
    if config['spectrogram_chunk_frames']<32: raise ValueError('Spectrogram chunks need at least 32 frames')
    if not 0<=config['spectrogram_dropout']<1: raise ValueError('Invalid SPECTROGRAM_DROPOUT')
    if not np.isfinite(config['spectrogram_lr']) or config['spectrogram_lr']<=0: raise ValueError('Invalid SPECTROGRAM_LR')


def prepare(config, output):
    import pandas as pd
    from sweep_utils import file_sha256
    from telemetry import atomic_json
    validate(config);output=Path(output)
    source=Path(config['spectrogram_feature_file']).expanduser().resolve()
    with h5py.File(source,'r') as f:
        if not f.attrs.get('complete') or not f['index/completed'][:].all():
            raise ValueError('Spectrogram extraction is incomplete')
        feature_config=json.loads(f.attrs['config_json'])
        if feature_config['n_mels']!=128: raise ValueError('This branch expects 128 mel bands')
        keys=list(zip(f['index/split'].asstr()[:],f['index/filename'].asstr()[:]))
        if len(keys)!=len(set(keys)): raise ValueError('Duplicate spectrogram keys')
        lookup={key:i for i,key in enumerate(keys)}
        frame=pd.read_parquet(output/'data.parquet')
        wanted=list(zip(frame.split.astype(str),frame.filename.astype(str)))
        if set(wanted)-set(lookup): raise ValueError('Missing spectrogram records')
        indices=[lookup[key] for key in wanted]
        known=frame.split.eq('train').to_numpy()
        if not np.array_equal(f['index/label'][:][indices][known],frame.loc[known,'label'].to_numpy(float)):
            raise ValueError('Spectrogram training labels differ')
        rows=[]
        for key,i in zip(wanted,indices):
            g=f[f'records/{i:06d}'];n=len(g['frame_center_s'])
            if not g.attrs.get('complete') or g['log_mel'].shape!=(n,128): raise ValueError('Invalid saved spectrogram')
            if g.attrs['audio_sha256']!=f['index/audio_sha256'].asstr()[i]: raise ValueError('Spectrogram source hash mismatch')
            rows.append(dict(split=key[0],filename=key[1],record_index=i,frames=n,audio_sha256=g.attrs['audio_sha256']))
        fingerprint=f.attrs['fingerprint']
    result=dict(config,spectrogram_data_path=str(source),spectrogram_data_sha256=file_sha256(source),
                spectrogram_source_fingerprint=fingerprint,spectrogram_feature_config=feature_config)
    if config['spectrogram_backbone'] in AST_REPOS:
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoConfig
        repo=AST_REPOS[config['spectrogram_backbone']]
        report_path=output/'spectrogram_pretrained.json'
        prior=json.loads(report_path.read_text()) if report_path.exists() else None
        if prior and prior['model_id']!=repo: raise ValueError('Changed spectrogram model: use a new experiment name')
        revision=prior['revision'] if prior else HfApi().model_info(repo,revision=config.get('spectrogram_revision') or 'main').sha
        files=HfApi().list_repo_files(repo,revision=revision)
        weight=['*.safetensors'] if any('/' not in name and name.endswith('.safetensors') for name in files) else ['pytorch_model*.bin']
        directory=snapshot_download(repo,revision=revision,cache_dir=str(output.parents[1]/'model_cache'),
                                    allow_patterns=['*.json',*weight],ignore_patterns=['*/*'])
        architecture=AutoConfig.from_pretrained(directory,local_files_only=True)
        if architecture.model_type!='audio-spectrogram-transformer' or architecture.num_mel_bins!=128:
            raise ValueError('Unsupported pretrained spectrogram architecture')
        result.update(spectrogram_model_dir=directory,spectrogram_model_config=architecture.to_dict(),
                      spectrogram_model_id=repo,spectrogram_model_revision=revision,
                      spectrogram_chunk_frames=int(architecture.max_length))
        provenance=dict(model_id=repo,revision=revision,model_sha256=__import__('sweep_utils').model_fingerprint(directory))
        if prior and provenance!=prior: raise ValueError('Pretrained spectrogram weights changed')
        atomic_json(report_path,provenance)
        result['spectrogram_model_sha256']=provenance['model_sha256']
    manifest=dict(rows=rows,source=str(source),fingerprint=fingerprint,sha256=result['spectrogram_data_sha256'],
                  feature_config=feature_config,chunk_frames=result['spectrogram_chunk_frames'],
                  policy='All frames in non-overlapping timed chunks; last chunk masked; no random crop or silence removal.',
                  normalization='Mean/std per mel band fitted only on training-fold frames; AST uses half-standard scaling.',
                  pretrained_frontend_note='AST weights are adapted to this saved HTK log-power frontend, which differs from the original AST Kaldi frontend. This is a fine-tuning experiment, not exact native preprocessing.')
    target=output/'spectrogram_index.json'
    if target.exists() and json.loads(target.read_text())!=manifest: raise ValueError('Spectrogram inputs changed: choose a new experiment name')
    atomic_json(target,manifest);atomic_json(output/'spectrogram_report.json',manifest)
    result['spectrogram_index_path']=str(target.resolve())
    result['spectrogram_index_sha256']=file_sha256(target)
    return result


def verify(config):
    if not uses(config): return
    import sweep_utils
    for path,digest in [('spectrogram_data_path','spectrogram_data_sha256'),('spectrogram_index_path','spectrogram_index_sha256')]:
        if sweep_utils.file_sha256(config[path])!=config[digest]: raise ValueError('Changed spectrogram input: '+config[path])
    if config.get('spectrogram_model_dir') and sweep_utils.model_fingerprint(config['spectrogram_model_dir'])!=config['spectrogram_model_sha256']:
        raise ValueError('Changed pretrained spectrogram weights')


class Store:
    def __init__(self,config):
        verify(config);self.config=config
        report=json.loads(Path(config['spectrogram_index_path']).read_text())
        self.lookup={(row['split'],row['filename']):row for row in report['rows']}
        self.load=functools.lru_cache(maxsize=8)(self._load)

    def _load(self,split,filename):
        row=self.lookup[(split,filename)]
        with h5py.File(self.config['spectrogram_data_path'],'r') as f:
            g=f["records/{:06d}".format(row['record_index'])]
            mel=g['log_mel'][:];times=g['frame_center_s'][:];valid=g['valid_fraction'][:]
        if not np.isfinite(mel).all() or not np.isfinite(times).all(): raise ValueError('Nonfinite spectrogram')
        return mel,times,valid

    def item(self,split,filename):
        mel,times,valid=self.load(str(split),str(filename));k=self.config['spectrogram_chunk_frames'];n=len(mel)
        count=math.ceil(n/k);chunks=np.full((count,k,128),self.config['spectrogram_feature_config']['db_floor'],np.float32)
        lengths=np.minimum(k,n-np.arange(count)*k).astype(np.int64)
        for i,length in enumerate(lengths): chunks[i,:length]=mel[i*k:i*k+length]
        starts=times[np.arange(count)*k].astype(np.float32)
        return dict(spectrogram=torch.from_numpy(chunks),spectrogram_lengths=torch.from_numpy(lengths),
                    spectrogram_times=torch.from_numpy(starts))

    def normalization(self,frame):
        total=np.zeros(128,np.float64);squares=np.zeros(128,np.float64);weight=0.
        for row in frame.itertuples():
            mel,_,valid=self.load(str(row.split),str(row.filename));x=mel.astype(np.float64);w=valid[:,None]
            total+=(x*w).sum(0);squares+=(x*x*w).sum(0);weight+=valid.sum()
        mean=total/weight;scale=np.sqrt(np.maximum(squares/weight-mean**2,0));scale[scale<1e-6]=1
        return mean.astype(np.float32),scale.astype(np.float32)


class MultiDataset(Dataset):
    def __init__(self,frame,tokenizer,max_length,audio_store,spec_store,labelled=True):
        self.frame=frame.reset_index(drop=True);self.spec_store=spec_store
        self.audio=audio_store.rows(frame) if audio_store else None
        self.encoded=tokenizer(frame.text.tolist(),truncation=True,max_length=max_length,padding=False) if tokenizer else {}
        self.labels=frame.label.to_numpy(np.float32) if labelled else None
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row={key:values[index] for key,values in self.encoded.items()}
        if self.audio is not None: row['audio_features']=self.audio[index]
        if self.spec_store:
            item=self.frame.iloc[index];row.update(self.spec_store.item(item.split,item.filename))
        if self.labels is not None: row['labels']=float(self.labels[index])
        return row


class MultiCollator:
    def __init__(self,tokenizer):
        from transformers import DataCollatorWithPadding
        self.text=DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8,return_tensors='pt') if tokenizer else None
    def __call__(self,rows):
        special={'audio_features','spectrogram','spectrogram_lengths','spectrogram_times'}
        text=[{k:v for k,v in row.items() if k not in special} for row in rows]
        batch=self.text(text) if self.text else ({'labels':torch.tensor([r['labels'] for r in rows])} if 'labels' in rows[0] else {})
        if 'audio_features' in rows[0]: batch['audio_features']=torch.from_numpy(np.stack([r['audio_features'] for r in rows]))
        if 'spectrogram' in rows[0]:
            for key in ['spectrogram','spectrogram_lengths','spectrogram_times']: batch[key]=torch.cat([r[key] for r in rows])
            batch['spectrogram_owner']=torch.cat([torch.full((len(row['spectrogram']),),i,dtype=torch.long) for i,row in enumerate(rows)])
        return batch


class ChannelNorm(nn.Module):
    def __init__(self,width): super().__init__();self.norm=nn.LayerNorm(width)
    def forward(self,x): return self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)


class Encoder(nn.Module):
    def __init__(self,config,random_backbone=False):
        super().__init__();self.config=config;self.kind=config['spectrogram_backbone'];width=config['spectrogram_output_dim']
        self.register_buffer('mel_mean',torch.zeros(128));self.register_buffer('mel_scale',torch.ones(128))
        self.ast=None
        if self.kind in CNN_WIDTHS:
            channels=[1,*CNN_WIDTHS[self.kind]]
            self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(a,b,3,stride=2,padding=1),ChannelNorm(b),nn.GELU(),nn.Dropout(config['spectrogram_dropout'])) for a,b in zip(channels[:-1],channels[1:])])
            hidden=channels[-1]
        else:
            from transformers import ASTConfig,ASTModel
            architecture=ASTConfig.from_dict(config['spectrogram_model_config'])
            self.ast=ASTModel(architecture) if random_backbone else ASTModel.from_pretrained(config['spectrogram_model_dir'],local_files_only=True)
            if config['gradient_checkpointing']: self.ast.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            for module in self.ast.modules():
                if isinstance(module,nn.Dropout): module.p=config['spectrogram_dropout']
            hidden=architecture.hidden_size
        self.project=nn.Linear(hidden,width)
        self.position=nn.Linear(config['spectrogram_feature_config']['pe_dimensions'],width,bias=False)
        self.norm=nn.LayerNorm(width);self.frame_attention=nn.Linear(width,1);self.chunk_attention=nn.Linear(width,1)

    def positional(self,times):
        c=self.config['spectrogram_feature_config'];d=c['pe_dimensions']
        divisor=c['pe_base']**(torch.arange(0,d,2,device=times.device,dtype=torch.float32)/d)
        angles=(times.float()/c['pe_reference_step_s'])[...,None]/divisor
        return torch.stack([angles.sin(),angles.cos()],dim=-1).flatten(-2)

    def encode_chunks(self,mel,lengths,starts):
        x=(mel.float()-self.mel_mean)/self.mel_scale
        valid=torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]
        x=x.masked_fill(~valid[:,:,None],0)
        if self.ast is not None:
            embedding=self.project(self.ast(input_values=x*.5).pooler_output)
            center=starts+(lengths.float()-1)*self.config['spectrogram_feature_config']['hop_samples']/self.config['spectrogram_feature_config']['sample_rate']/2
            if self.config['spectrogram_use_time_encoding']: embedding=embedding+self.position(self.positional(center))
            return self.norm(embedding)
        x=x.transpose(1,2).unsqueeze(1);stride=1
        for block in self.blocks:
            x=block(x);stride*=2
            size=(lengths+stride-1)//stride
            valid=torch.arange(x.shape[-1],device=x.device)[None,:]<size[:,None]
            x=x.masked_fill(~valid[:,None,None,:],0)
        tokens=self.project(x.mean(2).transpose(1,2))
        hop=self.config['spectrogram_feature_config']['hop_samples']/self.config['spectrogram_feature_config']['sample_rate']
        times=starts[:,None]+torch.arange(tokens.shape[1],device=x.device)[None,:]*stride*hop
        if self.config['spectrogram_use_time_encoding']: tokens=tokens+self.position(self.positional(times))
        tokens=self.norm(tokens)
        logits=self.frame_attention(tokens).squeeze(-1).float().masked_fill(~valid,-torch.inf)
        return (tokens*logits.softmax(-1)[...,None].to(tokens.dtype)).sum(1)

    def forward(self,inputs):
        mel=inputs['spectrogram'];lengths=inputs['spectrogram_lengths'];starts=inputs['spectrogram_times'];owner=inputs['spectrogram_owner']
        pieces=[];step=self.config['spectrogram_chunk_batch_size']
        for first in range(0,len(mel),step):
            args=(mel[first:first+step],lengths[first:first+step],starts[first:first+step])
            if self.training and self.config['gradient_checkpointing']:
                from torch.utils.checkpoint import checkpoint
                pieces.append(checkpoint(self.encode_chunks,*args,use_reentrant=False))
            else: pieces.append(self.encode_chunks(*args))
        embeddings=torch.cat(pieces)
        scores=self.chunk_attention(embeddings).squeeze(-1).float()+lengths.float().log()
        results=[]
        for index in range(int(owner.max())+1):
            mask=owner==index;weights=scores[mask].softmax(0).to(embeddings.dtype)
            results.append((embeddings[mask]*weights[:,None]).sum(0))
        return torch.stack(results)


class GrammarModel(nn.Module):
    def __init__(self,config,text_model,initial_mean,random_backbone=False):
        super().__init__();import audio_features as audio
        self.config=config;self.backbone=text_model.backbone if text_model else None
        widths=[]
        if text_model: self.pooler=text_model.pooler;widths.append(self.pooler.output_dim)
        self.audio_backbone=None
        if audio.uses_audio(config):
            previous=audio.AudioGrammarModel(config,None,initial_mean)
            self.audio_backbone=previous.audio_backbone
            self.register_buffer('audio_mean',previous.audio_mean);self.register_buffer('audio_scale',previous.audio_scale)
            widths.append(config['audio_output_dim'])
        self.spectrogram_backbone=Encoder(config,random_backbone);widths.append(config['spectrogram_output_dim'])
        combined=sum(widths)
        self.fusion=nn.Sequential(nn.LayerNorm(combined),nn.Linear(combined,config['fusion_hidden_dim']),nn.GELU(),nn.Dropout(config['fusion_dropout'])) if len(widths)>1 else nn.Identity()
        width=config['fusion_hidden_dim'] if len(widths)>1 else combined
        self.norm=nn.LayerNorm(width);self.dropout=nn.Dropout(config['head_dropout']);self.head=nn.Linear(width,1)
        nn.init.normal_(self.head.weight,std=.01)
        rate=np.clip(initial_mean/config['prediction_max'],.01,.99)
        nn.init.constant_(self.head.bias,float(np.log(rate/(1-rate))) if config['output_mode']=='bounded' else initial_mean)
        self.layer_report=dict(text_model.layer_report if text_model else {},modalities=[name for name,yes in [('text',text_model is not None),('audio',self.audio_backbone is not None),('spectrogram',True)] if yes],
            spectrogram_backbone=config['spectrogram_backbone'],spectrogram_output_dim=config['spectrogram_output_dim'],
            spectrogram_parameters=sum(p.numel() for p in self.spectrogram_backbone.parameters()),
            total_parameters=sum(p.numel() for p in self.parameters()),trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))

    def fit_audio_normalization(self,values):
        from audio_features import AudioGrammarModel
        AudioGrammarModel.fit_audio_normalization(self,values)

    def fit_spectrogram_normalization(self,store,frame):
        mean,scale=store.normalization(frame)
        self.spectrogram_backbone.mel_mean.copy_(torch.as_tensor(mean,device=self.head.weight.device))
        self.spectrogram_backbone.mel_scale.copy_(torch.as_tensor(scale,device=self.head.weight.device))

    def forward(self,inputs):
        embeddings=[]
        if self.backbone is not None:
            text={k:v for k,v in inputs.items() if k in ['input_ids','attention_mask','token_type_ids']}
            output=self.backbone(**text,output_hidden_states=True) if self.pooler.needs_layers else self.backbone(**text)
            embeddings.append(self.pooler(output,text['attention_mask']))
        if self.audio_backbone is not None:
            values=inputs['audio_features'].float();values=torch.where(torch.isfinite(values),values,self.audio_mean)
            if self.config['audio_normalization']=='standard': values=values-self.audio_mean
            embeddings.append(self.audio_backbone(values/self.audio_scale))
        embeddings.append(self.spectrogram_backbone(inputs))
        logits=self.head(self.dropout(self.norm(self.fusion(torch.cat(embeddings,dim=-1))))).float().squeeze(-1)
        return (self.config['prediction_max']*logits.sigmoid() if self.config['output_mode']=='bounded' else logits),logits
