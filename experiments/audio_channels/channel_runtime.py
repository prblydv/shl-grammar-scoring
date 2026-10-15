"""Self-contained, inference-only extraction into one resumable HDF5 file."""
import gc
import hashlib
import json
import math
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from tqdm.auto import tqdm

ACCURACY = ['error_rate', 'severe_error_rate', 'error_free_clause_ratio',
 'error_free_sentence_ratio', 'tense_error_rate', 'agreement_error_rate',
 'verb_form_error_rate', 'article_error_rate', 'preposition_error_rate',
 'pronoun_error_rate', 'plural_error_rate', 'word_order_error_rate',
 'missing_word_rate', 'extra_word_rate', 'negation_error_rate', 'multi_error_clause_ratio']
SYNTAX = ['mean_clause_length', 'clauses_per_sentence', 'subordinate_clause_ratio',
 'relative_clause_rate', 'conditional_rate', 'passive_construction_rate',
 'complex_sentence_ratio', 'mean_dependency_depth', 'syntactic_depth_p90',
 'complete_clause_ratio', 'fragment_rate', 'correct_complex_clause_ratio']
REPAIRS = ['repair_attempt_rate', 'successful_repair_rate', 'failed_repair_rate',
 'repair_success_ratio', 'false_start_rate', 'abandoned_clause_rate',
 'candidate_disfluency_repetition_rate', 'restart_completion_ratio']
TIMING = ['within_clause_pause_rate', 'boundary_pause_fraction',
 'pause_before_repair_median', 'long_within_clause_pause_rate',
 'mean_words_per_fluent_run', 'local_speech_rate_cv']
MODELS = {
 'asr': 'nyralabs/CrisperWhisper2.0_large',
 'gec': 'vennify/t5-base-grammar-correction',
 'semantic': 'sentence-transformers/all-MiniLM-L6-v2',
 'surprisal': 'distilbert/distilgpt2',
 'wavlm': 'microsoft/wavlm-base-plus',
 'hubert': 'facebook/hubert-base-ls960',
 'wav2vec2': 'facebook/wav2vec2-base-960h',
 'phoneme': 'facebook/wav2vec2-lv-60-espeak-cv-ft',
 'align': 'facebook/wav2vec2-base-960h',
}
STAGES = ['asr', 'fluent', 'align', 'grammar', 'text', 'dsp',
          'wavlm', 'hubert', 'wav2vec2', 'phoneme']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def words(text): return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
def ratio(a, b): return float(a / b) if b else np.nan
def rate(a, b): return 100 * ratio(a, b)
def stats(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if not len(x): return np.full(8, np.nan, np.float32)
    return np.asarray([x.mean(), x.std(), x.min(), x.max(),
        *np.quantile(x, [.1, .5, .9]), ratio(x.std(), abs(x.mean()))], np.float32)


def resolve_audio(folder, filenames):
    """Resolve only CSV source audio; Jupyter preview checkpoints are not inputs."""
    folder = Path(folder); expected = set(filenames); lookup = {}; extras = []
    for path in folder.rglob('*.wav'):
        if '.ipynb_checkpoints' in path.relative_to(folder).parts:
            continue
        if path.name not in expected:
            extras.append(path.name)
            continue
        if path.name in lookup:
            raise ValueError(f'Duplicate source WAV {path.name}: {lookup[path.name]} and {path}')
        lookup[path.name] = path
    missing = sorted(expected - set(lookup))
    if missing:
        raise ValueError(f'{folder.name}: {len(missing)} required WAVs missing; examples: {missing[:10]}')
    if extras:
        print(f'{folder.name}: ignored {len(extras)} WAVs not listed in source CSV: {sorted(extras)[:10]}', flush=True)
    return lookup


def schema():
    definitions = []
    for model in ['wavlm', 'hubert', 'wav2vec2']:
        views = [(l, p) for l in [4, 8, 12] for p in ['mean', 'std']] if model == 'wavlm' else [(4, 'mean'), (12, 'mean'), (12, 'std')]
        for layer, pooling in views:
            definitions.append((f'{model}_layer{layer}_{pooling}', 768, None))
    definitions.extend([(name, dim, None) for name, dim in [
        ('logmel', 128), ('mfcc', 26), ('mfcc_delta', 26), ('mfcc_delta2', 26),
        ('pitch', 8), ('energy', 8), ('pauses', 8), ('word_timing', 8),
        ('fluent_semantic', 384), ('fluent_surprisal', 8)]])
    definitions.extend([(n, len(cols), cols) for n, cols in [
        ('grammar_accuracy', ACCURACY), ('syntax_complexity', SYNTAX),
        ('repair_behaviour', REPAIRS), ('grammar_timing', TIMING)]])
    offset = 0; result = {}
    for name, width, columns in definitions:
        result[name] = dict(start=offset, stop=offset + width, dimensions=width,
            columns=columns or [f'{name}_{i}' for i in range(width)])
        offset += width
    assert offset == 9888 and len(result) == 26
    return result


def cpu_capacity():
    count=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1)
    for path in ['/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu/cpu.cfs_quota_us']:
        try:
            if path.endswith('cpu.max'):
                quota,period=Path(path).read_text().split()
                if quota!='max':count=min(count,max(1,math.ceil(int(quota)/int(period))))
            else:
                quota=int(Path(path).read_text());period=int(Path(path).with_name('cpu.cfs_period_us').read_text())
                if quota>0:count=min(count,max(1,math.ceil(quota/period)))
        except (OSError,ValueError):pass
    return count


def gpu_batch_map(items,batch_size,fn):
    """Yield results in input order; split CUDA OOM batches without dropping inputs."""
    def attempt(batch):
        try:
            values=fn(batch)
            if len(values)!=len(batch):raise ValueError('Batch output count differs from inputs.')
            return values
        except torch.cuda.OutOfMemoryError:
            if len(batch)==1:raise
        # Leave the exception handler first so failed-forward tensors can be freed.
        gc.collect();torch.cuda.empty_cache()
        mid=len(batch)//2
        return attempt(batch[:mid])+attempt(batch[mid:])
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size]
        yield from zip(batch,attempt(batch))


def complete_json(text):
    """Read the first complete object, ignoring fences or trailing model commentary."""
    start=text.find('{')
    if start<0:return None
    try:value,_=json.JSONDecoder().raw_decode(text[start:])
    except ValueError:return None
    return value if isinstance(value,dict) else None


def dsp_worker(task):
    """CPU-only worker. Parent owns the HDF5 writer; workers never open its file."""
    i,path,aligned=task
    torch.set_num_threads(1)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    import librosa,webrtcvad
    x,sr=sf.read(path,dtype='float32',always_2d=True);x=x.mean(axis=1)
    if sr!=16000:
        d=math.gcd(sr,16000);x=resample_poly(x,16000//d,sr//d).astype('f4')
    sr=16000
    power=np.abs(librosa.stft(x,n_fft=512,hop_length=160,win_length=400))**2
    mel=librosa.feature.melspectrogram(S=power,sr=sr,n_mels=64)
    logmel=librosa.power_to_db(mel,ref=1.,top_db=None)
    mfcc=librosa.feature.mfcc(S=logmel,n_mfcc=13)
    pool=lambda a:np.r_[a.mean(axis=1),a.std(axis=1)]
    vectors={'logmel':pool(logmel),'mfcc':pool(mfcc)}
    width=min(9,mfcc.shape[1] if mfcc.shape[1]%2 else mfcc.shape[1]-1)
    for order,name in [(1,'mfcc_delta'),(2,'mfcc_delta2')]:
        vectors[name]=pool(librosa.feature.delta(mfcc,order=order,width=width)) if width>=3 else np.full(26,np.nan)
    rms=librosa.feature.rms(y=x,frame_length=400,hop_length=160)[0]
    vectors['energy']=stats(20*np.log10(np.maximum(rms,1e-10)))
    f0,flag,prob=librosa.pyin(x,sr=sr,fmin=65,fmax=500,frame_length=1024,hop_length=640)
    selected=f0[flag & (prob>=.7) & np.isfinite(f0)]
    vectors['pitch']=stats(12*np.log2(selected/np.median(selected))) if len(selected) else np.full(8,np.nan)
    pcm=(np.clip(x,-1,1)*32767).astype('<i2');vad=webrtcvad.Vad(2)
    voiced=np.asarray([vad.is_speech(pcm[a:a+480].tobytes(),sr) for a in range(0,len(pcm)-479,480)])
    true=np.flatnonzero(voiced);pauses=[]
    if len(true):
        interior=voiced[true[0]:true[-1]+1];edges=np.flatnonzero(np.diff(np.r_[True,interior,True]))
        pauses=[(b-a)*.03 for a,b in zip(edges[::2],edges[1::2]) if (b-a)*.03>=.25]
    vectors['pauses']=stats(pauses) if pauses else np.array([0,0,0,0,0,0,0,np.nan])
    durations=[w['end']-w['start'] for w in aligned if 'start' in w and 'end' in w and w['end']>w['start']]
    vectors['word_timing']=stats(durations)
    metadata=dict(duration_s=len(x)/sr,speech_fraction=float(voiced.mean()) if len(voiced) else None,
        pause_count=len(pauses),words_per_second=len(aligned)/(len(x)/sr))
    return i,vectors,metadata


class Extractor:
    def __init__(self, config):
        self.config = config; self.device = 'cuda:0'
        if not torch.cuda.is_available(): raise RuntimeError('Enable a Kaggle GPU accelerator.')
        capacity=cpu_capacity()
        self.cpu_threads=int(config.get('CPU_THREADS') or min(16,capacity))
        self.cpu_workers=int(config.get('CPU_WORKERS') or min(32,capacity))
        self.gpu_batch=int(config.get('GPU_BATCH_SIZE',16))
        self.text_batch=int(config.get('TEXT_BATCH_SIZE',4))
        if min(self.cpu_threads,self.cpu_workers,self.gpu_batch,self.text_batch)<1:raise ValueError('Resource settings must be positive.')
        torch.set_num_threads(self.cpu_threads); torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        print(f'Resources: {capacity} available CPUs; {self.cpu_workers} DSP workers; {self.cpu_threads} CPU threads; GPU batch {self.gpu_batch}; text batch {self.text_batch}',flush=True)
        self.channels = schema(); self.jobs = self.discover()
        if config.get('LIMIT'): self.jobs = self.jobs[:config['LIMIT']]
        self.path = Path(config['OUTPUT']); self.path.parent.mkdir(parents=True, exist_ok=True)
        resume = config.get('RESUME_FILE')
        if resume and not self.path.exists():
            import shutil
            shutil.copy2(resume, self.path)
        self.h = h5py.File(self.path, 'a')
        signature = hashlib.sha256(json.dumps(dict(channels=self.channels,
            audio=[(r['split'],r['filename'],r['sha'],r['source_row'],r['label'] if r['split']=='train' else None) for r in self.jobs]),sort_keys=True).encode()).hexdigest()
        if 'X' in self.h:
            # Migration for the original notebook: performance changes must not discard completed work.
            if self.h['X'].shape!=(len(self.jobs),9888) or json.loads(self.h.attrs['channels_json'])!=self.channels:
                raise ValueError('Existing HDF5 schema differs; use a new OUTPUT.')
            for key in ['filename','split','sha']:
                if list(self.h['metadata'][key].asstr()[:])!=[r[key] for r in self.jobs]:
                    raise ValueError('Existing HDF5 source audio differs; use a new OUTPUT.')
            for key in ['source_row','label']:
                if not np.array_equal(self.h['metadata'][key][:],np.asarray([r[key] for r in self.jobs]),equal_nan=True):
                    raise ValueError('Existing HDF5 ordering/labels differ; use a new OUTPUT.')
            if 'data_signature' in self.h.attrs and self.h.attrs['data_signature']!=signature:
                raise ValueError('Existing HDF5 input signature differs.')
            for key,model_id in MODELS.items():
                if key in self.h['provenance'] and self.h['provenance'][key].attrs.get('model_id')!=model_id:
                    raise ValueError('Existing model provenance differs.')
            print('Resuming existing HDF5; completed recording/stage checkpoints preserved.',flush=True)
        if 'X' not in self.h:
            n = len(self.jobs)
            self.h.attrs.update(signature=signature, complete=False, inference_only=True,
                schema_version=1, fixed_channels=26, sequence_channels=3,
                fixed_dimensions=9888, test_labels='unknown; random CSV placeholders ignored',
                attention_pooling='Not trained: save chunk inputs for downstream fold-local training',
                channels_json=json.dumps(self.channels), configuration_json=json.dumps(config),
                source_code=Path(__file__).read_text(), limitations_json=json.dumps([
                    'ASR, grammatical corrections, repairs, syntax and word alignment are estimates.',
                    'GEC edits include stylistic changes; no edit is asserted to be ground truth.',
                    'Severity and clause-completeness are explicitly heuristic proxies.',
                    'Qwen cleanup disabled; repair behaviour and pause-before-repair are unavailable.',
                    'Phoneme posteriors are uncalibrated acoustic model probabilities, not grammar scores.',
                    'Frozen mean/std views are correlated; predictive power is unvalidated.',
                    'Fit all learned pooling, scalers, imputers, PCA and stackers inside training folds.',
                    'Full inference can exceed one Kaggle session; resume from the same single HDF5.',
                ]))
            self.h.create_dataset('X', (n,9888), dtype='f4', fillvalue=np.nan,
                chunks=(1,9888), compression='gzip')
            self.h.create_dataset('available', (n,9888), dtype='bool', chunks=(1,9888), compression='gzip')
            meta = self.h.create_group('metadata'); string = h5py.string_dtype('utf-8')
            for key in ['filename','split','sha']:
                meta.create_dataset(key, data=np.asarray([r[key] for r in self.jobs],dtype=object), dtype=string)
            for key in ['source_row','label','duration']:
                meta.create_dataset(key, data=np.asarray([r[key] for r in self.jobs]))
            meta.create_dataset('label_known', data=[r['split']=='train' for r in self.jobs])
            meta.create_dataset('train_indices', data=[i for i,r in enumerate(self.jobs) if r['split']=='train'])
            meta.create_dataset('test_indices', data=[i for i,r in enumerate(self.jobs) if r['split']=='test'])
            source = self.h.create_group('source_tables')
            for name in ['train.csv', 'sample_submission.csv']:
                p = self.root/name
                if p.exists(): source.create_dataset(name, data=p.read_text(), dtype=string)
            source.create_dataset('test_filenames', data=np.asarray(
                pd.read_csv(self.root/'test.csv').filename.astype(str),dtype=object),dtype=string)
            self.h.create_group('records'); self.h.create_group('provenance')
        self.h.attrs['data_signature']=signature
        versions=self.h['provenance'].require_group('code_versions')
        source=Path(__file__).read_text();code_sha=sha(__file__)
        if code_sha not in versions:versions.create_dataset(code_sha,data=source,dtype=h5py.string_dtype('utf-8'))
        self.h.attrs['performance_configuration_json']=json.dumps(dict(cpu_threads=self.cpu_threads,cpu_workers=self.cpu_workers,gpu_batch=self.gpu_batch,text_batch=self.text_batch))
        self.h.attrs['complete'] = False
        self.h.flush()

    def discover(self):
        if self.config.get('DATA_ROOT'):
            candidates = [Path(self.config['DATA_ROOT'])]
        else:
            candidates = [p.parent for p in Path('/kaggle/input').rglob('train.csv')
                if (p.parent/'test.csv').exists() and (p.parent/'train').is_dir()
                and (p.parent/'test').is_dir()]
        if len(candidates)!=1: raise ValueError('Attach competition data or set DATA_ROOT explicitly.')
        self.root = candidates[0]; jobs=[]
        for split, expected in [('train',769),('test',216)]:
            table = pd.read_csv(self.root/f'{split}.csv')
            if len(table)!=expected or not table.filename.is_unique: raise ValueError('Unexpected source counts/duplicates.')
            lookup=resolve_audio(self.root/split,table.filename)
            for source_row,row in table.iterrows():
                p=lookup[row.filename]; info=sf.info(p)
                label=float(row.label) if split=='train' else np.nan
                if split=='train' and (not np.isfinite(label) or not 0<=label<=5): raise ValueError('Invalid train label.')
                jobs.append(dict(split=split,filename=str(row.filename),source_row=int(source_row),
                    label=label,path=str(p),sha=sha(p),duration=info.duration))
        print('Validated all 769 train + 216 test WAVs; test labels ignored.',flush=True)
        return jobs

    def record(self,i): return self.h['records'].require_group(f'{i:04d}')
    def done(self,i,stage): return bool(self.record(i).attrs.get(stage+'_done',False))
    def put(self,g,name,value):
        if name in g: del g[name]
        if isinstance(value,str): g.create_dataset(name,data=value,dtype=h5py.string_dtype('utf-8'))
        else:
            value=np.asarray(value)
            g.create_dataset(name,data=value,compression='lzf' if value.ndim and value.size else None)
    def getjson(self,g,name,default=None):
        if name not in g: return default
        return json.loads(g[name].asstr()[()])
    def fixed(self,i,name,value):
        v=np.asarray(value,np.float32); spec=self.channels[name]
        if v.shape!=(spec['dimensions'],) or np.isinf(v).any(): raise ValueError('Invalid channel '+name)
        self.h['X'][i,spec['start']:spec['stop']]=v
        self.h['available'][i,spec['start']:spec['stop']]=np.isfinite(v)
    def pending(self,stage): return [i for i in range(len(self.jobs)) if not self.done(i,stage)]
    def each(self,stage,indices,fn,progress=True):
        for i in (tqdm(indices,desc=stage) if progress else indices):
            g=self.record(i)
            try:
                fn(i,g); g.attrs[stage+'_done']=True
                if stage+'_error' in g.attrs: del g.attrs[stage+'_error']
            except Exception as e:
                g.attrs[stage+'_error']=f'{type(e).__name__}: {str(e)[:2000]}'
                tqdm.write(f"{stage} failed: {self.jobs[i]['split']}/{self.jobs[i]['filename']}: {e}")
                if isinstance(e,torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
            self.drain_dsp()
            self.h.flush()
    def audio(self,i):
        p=self.jobs[i]['path']
        if sha(p)!=self.jobs[i]['sha']: raise ValueError('Source WAV changed.')
        x,sr=sf.read(p,dtype='float32',always_2d=True); x=x.mean(axis=1)
        if sr!=16000:
            d=math.gcd(sr,16000); x=resample_poly(x,16000//d,sr//d).astype('f4')
        if len(x)<400 or not np.isfinite(x).all(): raise ValueError('Invalid or too-short audio.')
        return x
    def revision(self,key):
        from huggingface_hub import HfApi
        g=self.h['provenance'].require_group(key)
        if 'revision' not in g.attrs:
            revision = 'f4334f6e8193f2691212d49b20fa12d370e13896' if key=='asr' else HfApi().model_info(MODELS[key]).sha
            g.attrs.update(model_id=MODELS[key],revision=revision)
        return str(g.attrs['revision'])
    def release(self,*objects):
        # Callers also delete their own references before empty_cache.
        gc.collect(); torch.cuda.empty_cache()

    def transcribe(self):
        """Historical stage name retained for checkpoint compatibility; no ASR runs."""
        sources={};originals={}
        root=Path(self.config.get('CRISPER_TRANSCRIPT_ROOT') or '/workspace/shl/data/text_input_crisper')
        for split in ['train','test']:
            path=root/f'{split}_transcripts_crisper.csv'
            if path.exists():
                table=pd.read_csv(path,keep_default_na=False)
                if not table.filename.is_unique:raise ValueError('Duplicate Crisper transcript keys.')
                for r in table.to_dict('records'):sources[(split,r['filename'])]=(r,str(path))
        for p in Path('/kaggle/input').rglob('transcripts_crisper.jsonl'):
            for line in p.open():
                if line.strip():
                    r=json.loads(line);sources.setdefault((r.get('split'),r.get('filename')),(r,str(p)))
        original_root=Path(self.config.get('ORIGINAL_TRANSCRIPT_ROOT') or '/workspace/shl/data/text_input')
        for split in ['train','test']:
            p=original_root/f'{split}.csv'
            if p.exists():
                table=pd.read_csv(p,keep_default_na=False)
                if not table.filename.is_unique:raise ValueError('Duplicate original transcript keys.')
                for r in table.to_dict('records'):originals[(split,r['filename'])]=(r.get('text',r.get('transcript','')),str(p))
        missing=[(r['split'],r['filename']) for r in self.jobs if (r['split'],r['filename']) not in sources]
        if missing:raise ValueError(f'{len(missing)} existing Crisper records missing. Set CRISPER_TRANSCRIPT_ROOT or attach Crisper outputs; automatic transcription is disabled. Examples: {missing[:5]}')
        reused=0;unavailable=0
        for i,job in enumerate(self.jobs):
            r,path=sources[(job['split'],job['filename'])];g=self.record(i)
            if r.get('audio_sha256')!=job['sha']:raise ValueError('Crisper transcript WAV hash mismatch: '+job['filename'])
            if r.get('model_id')!=MODELS['asr'] or r.get('model_revision')!='f4334f6e8193f2691212d49b20fa12d370e13896':raise ValueError('Unexpected Crisper source model/revision.')
            good=not r.get('error') and r.get('mode')=='verbatim' and bool(r.get('transcript','').strip())
            text=r.get('transcript','') if good else ''
            if good:reused+=1
            else:unavailable+=1
            old=g['verbatim'].asstr()[()] if 'verbatim' in g else None
            if old is not None and old!=text:
                for stage in ['fluent','align','grammar','text','dsp','wavlm']:g.attrs[stage+'_done']=False
                for name in ['grammar_accuracy','syntax_complexity','repair_behaviour','grammar_timing','fluent_semantic','fluent_surprisal','word_timing']:
                    self.fixed(i,name,np.full(self.channels[name]['dimensions'],np.nan))
            self.put(g,'verbatim',text);self.put(g,'crisper_transcript',text)
            asrwords=r.get('words',[]) if 'words' in r else json.loads(r.get('words_json') or '[]')
            self.put(g,'asr_words_json',json.dumps(asrwords))
            g.attrs.update(asr_done=True,asr_source=path,asr_estimated=True,
                transcript_available=good,transcript_source_error=r.get('error',''),asr_inference_disabled=True)
            original=originals.get((job['split'],job['filename']))
            if original:
                self.put(g,'original_transcript',original[0]);g.attrs['original_transcript_source']=original[1]
                g.attrs['original_transcript_join']='split+filename from legacy export; no per-WAV hash in source'
        self.h.flush()
        print(f'Existing transcripts only: {reused} usable Crisper records; {unavailable} unavailable. Original transcripts retained when present. No ASR weights downloaded or transcription performed.',flush=True)

    def fluent(self):
        """Compatibility stage: reuse existing text verbatim, with no cleanup model."""
        method='Existing verbatim transcript unchanged; Qwen cleanup disabled'
        for i in range(len(self.jobs)):
            if not self.done(i,'asr'):continue
            g=self.record(i)
            if g.attrs.get('fluent_method')==method and self.done(i,'fluent'):continue
            text=g['verbatim'].asstr()[()]
            # Text-derived checkpoints from cleaned text must be recomputed.
            for stage in ['grammar','text']:g.attrs[stage+'_done']=False
            for name in ['grammar_accuracy','syntax_complexity','repair_behaviour',
                         'grammar_timing','fluent_semantic','fluent_surprisal']:
                self.fixed(i,name,np.full(self.channels[name]['dimensions'],np.nan))
            self.put(g,'fluent',text)
            self.put(g,'fluent_to_verbatim_token',np.arange(len(text.split()),dtype='i4'))
            self.put(g,'verbatim_tokens_json',json.dumps(text.split()))
            self.put(g,'repair_events_json','[]')
            g.attrs.update(fluent_done=True,fluent_method=method,
                           repair_events_available=False,qwen_cleanup_disabled=True)
            if 'fluent_error' in g.attrs:del g.attrs['fluent_error']
        self.h.attrs['qwen_cleanup_disabled']=True
        self.h.attrs['text_feature_source']='Existing verbatim transcript unchanged'
        self.h.flush()
        print('Qwen disabled: existing transcripts used unchanged. Repair features unavailable.',flush=True)

    def align(self):
        pending=[i for i in self.pending('align') if self.done(i,'asr')]
        if not pending:return
        import whisperx
        from difflib import SequenceMatcher
        model,metadata=whisperx.load_align_model(language_code='en',device=self.device,
            model_name=MODELS['align'])
        def one(i,g):
            text=g['verbatim'].asstr()[()]
            if not text.strip(): aligned=[]
            else:
                result=whisperx.align([dict(start=0.,end=self.jobs[i]['duration'],text=text)],
                    model,metadata,self.audio(i),self.device,return_char_alignments=False)
                aligned=result.get('word_segments',[])
            self.put(g,'aligned_words_json',json.dumps(aligned))
            # Map original whitespace-token positions to aligner's words without inventing timings.
            normalize=lambda s:re.sub(r'\W','',s.lower())
            a=[normalize(t) for t in text.split()]; b=[normalize(w['word']) for w in aligned]
            positions=np.full((len(a),2),np.nan,dtype='f4')
            for block in SequenceMatcher(None,a,b,autojunk=False).get_matching_blocks():
                for k in range(block.size):
                    w=aligned[block.b+k]
                    if 'start' in w and 'end' in w: positions[block.a+k]=[w['start'],w['end']]
            self.put(g,'verbatim_token_times',positions)
            g.attrs['alignment_method']='WhisperX character-CTC forced alignment; not phoneme posterior scores'
        self.each('align',pending,one); del model; self.release()

    def grammar(self):
        pending=[i for i in self.pending('grammar') if self.done(i,'fluent') and self.done(i,'align')]
        if not pending:return
        import spacy,errant
        from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
        nlp=spacy.load('en_core_web_sm',disable=['ner']); annotator=errant.load('en',nlp=nlp)
        rev=self.revision('gec');tok=AutoTokenizer.from_pretrained(MODELS['gec'],revision=rev)
        model=AutoModelForSeq2SeqLM.from_pretrained(MODELS['gec'],revision=rev).to(self.device).eval()
        self.h['provenance'].attrs['spacy_model']=json.dumps(nlp.meta)
        def generate_corrections(batch):
            inp=tok(['grammar: '+s for s in batch],padding=True,return_tensors='pt',truncation=False).to(self.device)
            if inp.input_ids.shape[1]>512:raise ValueError('GEC span exceeds token budget; no silent truncation.')
            with torch.inference_mode():generated=model.generate(**inp,max_new_tokens=384,num_beams=4,do_sample=False)
            return tok.batch_decode(generated,skip_special_tokens=True)
        def one(i,g):
            text=g['fluent'].asstr()[()];doc=nlp(text); n=len(words(text))
            if not n:
                for name in ['grammar_accuracy','syntax_complexity','repair_behaviour','grammar_timing']:
                    self.fixed(i,name,np.full(self.channels[name]['dimensions'],np.nan))
                self.put(g,'corrected_proposal','');self.put(g,'grammar_edits_json','[]');return
            sentlist=list(doc.sents); edits=[];proposals=[]
            inputs=[]
            for sent in sentlist:
                tokens=list(sent)
                for a in range(0,len(tokens),80):
                    chunk=tokens[a:a+80];inputs.append(doc[chunk[0].i:chunk[-1].i+1].text)
            corrections={s:c for s,c in gpu_batch_map(inputs,min(16,self.gpu_batch),generate_corrections)}
            # Split unusually long ASR sentences without silently truncating GEC input.
            for sent in sentlist:
                tokens=list(sent); chunks=[tokens[a:a+80] for a in range(0,len(tokens),80)]
                for chunk in chunks:
                    original=doc[chunk[0].i:chunk[-1].i+1].text
                    corrected=corrections[original]; proposals.append(corrected)
                    original_doc=annotator.parse(original);corrected_doc=annotator.parse(corrected)
                    # ERRANT uses spaCy tokenization; map local offsets by character spans.
                    base=chunk[0].idx
                    for e in annotator.annotate(original_doc,corrected_doc):
                        if e.o_start<len(original_doc):char=base+original_doc[e.o_start].idx
                        else:char=base+len(original)
                        global_start=next((t.i for t in doc if t.idx>=char),len(doc))
                        endchar=base+(original_doc[e.o_end-1].idx+len(original_doc[e.o_end-1]) if e.o_end>e.o_start else char-base)
                        global_end=next((t.i for t in doc if t.idx>=endchar),len(doc))
                        edits.append(dict(type=e.type,original=e.o_str,correction=e.c_str,start=global_start,end=global_end))
            self.put(g,'corrected_proposal',' '.join(proposals));self.put(g,'grammar_edits_json',json.dumps(edits))
            # Explicit clause proxy: verb heads of roots and clausal dependencies.
            clause_deps={'ROOT','advcl','ccomp','xcomp','relcl','acl','conj','parataxis'}
            heads=[t for t in doc if t.pos_ in {'VERB','AUX'} and t.dep_ in clause_deps]
            def owner(token):
                chain=[token]+list(token.ancestors)
                return next((t.i for t in chain if t in heads),None)
            owners=[owner(t) for t in doc]
            counts={h.i:sum(v==h.i for v in owners) for h in heads}
            edit_counts={h.i:0 for h in heads};sentence_errors=[0]*len(sentlist)
            for e in edits:
                k=min(e['start'],len(doc)-1); o=owners[k]
                if o is not None:edit_counts[o]+=1
                for j,s in enumerate(sentlist):
                    if s.start<=k<s.end:sentence_errors[j]+=1;break
            kinds=[e['type'] for e in edits]
            count=lambda suffix:sum(k.endswith(suffix) for k in kinds)
            severe=sum(k.endswith(('VERB:SVA','VERB:FORM','WO')) or k in {'M:VERB','M:AUX'} for k in kinds)
            negation=sum(bool(set(words(e['original'].lower())+words(e['correction'].lower())) & {'not','never','no',"n't"}) for e in edits)
            accuracy=[rate(len(edits),n),rate(severe,n),ratio(sum(v==0 for v in edit_counts.values()),len(heads)),
                ratio(sum(v==0 for v in sentence_errors),len(sentlist)),rate(count('VERB:TENSE'),n),
                rate(count('VERB:SVA'),n),rate(count('VERB:FORM'),n),rate(count('DET'),n),rate(count('PREP'),n),
                rate(count('PRON'),n),rate(count('NOUN:NUM'),n),rate(count('WO'),n),
                rate(sum(k.startswith('M:') for k in kinds),n),rate(sum(k.startswith('U:') for k in kinds),n),
                rate(negation,n),ratio(sum(v>=2 for v in edit_counts.values()),len(heads))]
            depths=np.asarray([len(list(t.ancestors)) for t in doc if not t.is_punct])
            complex_heads=[h for h in heads if h.dep_ in {'advcl','ccomp','xcomp','relcl','acl'}]
            complete=[h for h in heads if any(c.dep_ in {'nsubj','nsubjpass','csubj'} for c in h.children)
                or (h.dep_=='ROOT' and h.tag_=='VB')]
            syntax=[ratio(sum(counts.values()),len(heads)),ratio(len(heads),len(sentlist)),
                ratio(len(complex_heads),len(heads)),rate(sum(h.dep_=='relcl' for h in heads),n),
                rate(sum(t.lower_ in {'if','unless'} and t.dep_=='mark' for t in doc),n),
                rate(sum(t.dep_=='auxpass' for t in doc),n),ratio(sum(any(t.dep_ in {'advcl','ccomp','relcl'} for t in s) for s in sentlist),len(sentlist)),
                float(depths.mean()) if len(depths) else np.nan,float(np.quantile(depths,.9)) if len(depths) else np.nan,
                ratio(len(complete),len(heads)),rate(sum(not any(t in heads for t in s) for s in sentlist),n),
                ratio(sum(edit_counts[h.i]==0 for h in complex_heads),len(complex_heads))]
            events=self.getjson(g,'repair_events_json',[]); nw=len(words(g['verbatim'].asstr()[()]))
            repairs=[e for e in events if e['type']=='grammar_repair']; evaluated=[e for e in repairs if type(e.get('success'))==bool]
            restarts=[e for e in events if e['type']=='false_start']
            repairs_v=[rate(len(repairs),nw),rate(sum(e.get('success') is True for e in repairs),nw),
                rate(sum(e.get('success') is False for e in repairs),nw),
                ratio(sum(e.get('success') is True for e in evaluated),len(evaluated)),
                rate(len(restarts),nw),rate(sum(e['type']=='abandoned_clause' for e in events),nw),
                rate(sum(e['type']=='repetition' for e in events),nw),
                ratio(sum(e.get('completed') is True for e in restarts),sum(type(e.get('completed'))==bool for e in restarts))]
            if not g.attrs.get('repair_events_available',False):repairs_v=[np.nan]*8
            # Timing on existing transcript tokens; clause boundaries are parser estimates.
            token_times=g['verbatim_token_times'][:]; retained=g['fluent_to_verbatim_token'][:]
            fluent_times=token_times[retained]; self.put(g,'fluent_token_times',fluent_times)
            spans=list(re.finditer(r'\S+',text)); boundary_positions=[]
            for j,span in enumerate(spans):
                t=next((t for t in doc if t.idx>=span.start()),None)
                if t is not None and (t.is_sent_start or t.dep_=='mark' or (t in heads and t.dep_!='ROOT')):boundary_positions.append(j)
            gaps=[]; inside=[];boundaries=[]; speeds=[];runs=[];run=0
            for j in range(len(fluent_times)):
                a,b=fluent_times[j]
                if np.isfinite(a+b) and b>a:speeds.append(1/(b-a))
                run+=1
                if j and np.isfinite(fluent_times[j-1,1]+a):
                    gap=max(0.,a-fluent_times[j-1,1])
                    if gap>=.25:
                        gaps.append(gap);is_boundary=j in boundary_positions
                        boundaries.append(is_boundary)
                        if not is_boundary:inside.append(gap)
                        runs.append(run-1);run=1
            if run:runs.append(run)
            before=[]
            for e in repairs:
                ids=e.get('replacement_indices',[])
                if ids:
                    j=min(ids)
                    if j and j<len(token_times) and np.isfinite(token_times[j,0]+token_times[j-1,1]):
                        before.append(max(0.,token_times[j,0]-token_times[j-1,1]))
            timing=[rate(len(inside),n),ratio(sum(boundaries),len(boundaries)),float(np.median(before)) if before else np.nan,
                rate(sum(p>1 for p in inside),n),float(np.mean(runs)) if runs else np.nan,
                ratio(np.std(speeds),np.mean(speeds)) if speeds else np.nan]
            if not np.isfinite(fluent_times).any():timing=[np.nan]*6
            for name,value in [('grammar_accuracy',accuracy),('syntax_complexity',syntax),('repair_behaviour',repairs_v),('grammar_timing',timing)]:self.fixed(i,name,value)
            self.put(g,'clause_analysis_json',json.dumps(dict(head_indices=[h.i for h in heads],word_counts=counts,
                grammatical_edit_counts=edit_counts,complex_head_indices=[h.i for h in complex_heads])))
            g.attrs['grammar_method']='GEC+ERRANT proposals; DET includes determiners; severity/negation/clause completeness heuristic; not human grammar annotation'
        self.each('grammar',pending,one);del model,tok,nlp,annotator;self.release()

    def text(self):
        pending=[i for i in self.pending('text') if self.done(i,'fluent')]
        if not pending:return
        from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
        rev=self.revision('semantic');tok=AutoTokenizer.from_pretrained(MODELS['semantic'],revision=rev)
        model=AutoModel.from_pretrained(MODELS['semantic'],revision=rev).to(self.device).eval()
        def semantic(i,g):
            ids=tok(g['fluent'].asstr()[()],add_special_tokens=False)['input_ids']; sums=[];weights=[]
            for start in range(0,len(ids),254):
                part=ids[start:start+254];batch=tok.prepare_for_model(part,return_tensors='pt')
                batch={k:v.unsqueeze(0).to(self.device) for k,v in batch.items()}
                with torch.inference_mode():states=model(**batch).last_hidden_state[0][1:-1]
                sums.append(states.sum(0).float().cpu().numpy());weights.append(len(states))
            v=sum(sums)/sum(weights) if weights else np.full(384,np.nan)
            if weights:v=v/max(float(np.linalg.norm(v)),1e-12)
            self.fixed(i,'fluent_semantic',v)
        # Do not mark text done until both text channels are written.
        successful=[]
        for i in tqdm(pending,desc='semantic'):
            try:semantic(i,self.record(i));successful.append(i)
            except Exception as e:self.record(i).attrs['text_error']=str(e)
            self.drain_dsp()
            self.h.flush()
        del model,tok;self.release()
        rev=self.revision('surprisal');tok=AutoTokenizer.from_pretrained(MODELS['surprisal'],revision=rev)
        model=AutoModelForCausalLM.from_pretrained(MODELS['surprisal'],revision=rev).to(self.device).eval()
        def surprise(i,g):
            ids=tok(g['fluent'].asstr()[()],add_special_tokens=False)['input_ids']; losses=[]
            # Each target token scored once with up to 512 preceding tokens; no suffix truncation.
            for start in range(0,len(ids),256):
                target=ids[start:start+256];context=ids[max(0,start-512):start]
                seq=[tok.bos_token_id]+context+target; inp=torch.tensor([seq],device=self.device)
                with torch.inference_mode():logits=model(inp).logits[0]
                lp=torch.log_softmax(logits[:-1].float(),-1)
                nll=-lp.gather(1,inp[0,1:,None]).squeeze(1)
                losses.extend(nll[-len(target):].cpu().numpy())
            self.fixed(i,'fluent_surprisal',stats(losses))
        self.each('text',successful,surprise);del model,tok;self.release()

    def dsp(self,background=False):
        import multiprocessing
        pending=[i for i in self.pending('dsp') if self.done(i,'align')]
        if not pending:return
        tasks=[(i,self.jobs[i]['path'],self.getjson(self.record(i),'aligned_words_json',[])) for i in pending]
        self._dsp_pool=ProcessPoolExecutor(max_workers=min(self.cpu_workers,len(tasks)),
            mp_context=multiprocessing.get_context('spawn'))
        self._dsp_futures={self._dsp_pool.submit(dsp_worker,task):task[0] for task in tasks}
        self._dsp_bar=tqdm(total=len(tasks),desc=f'DSP ({self.cpu_workers} CPU workers; overlaps GPU)')
        if not background:self.drain_dsp(wait=True)

    def drain_dsp(self,wait=False):
        futures=getattr(self,'_dsp_futures',{})
        completed=as_completed(list(futures)) if wait else [f for f in futures if f.done()]
        for future in completed:
            i=futures.pop(future);g=self.record(i)
            try:
                _,vectors,metadata=future.result()
                for name,value in vectors.items():self.fixed(i,name,value)
                g.attrs['acoustic_rates_json']=json.dumps(metadata,allow_nan=False)
                g.attrs['dsp_done']=True
                if 'dsp_error' in g.attrs:del g.attrs['dsp_error']
            except Exception as e:
                g.attrs['dsp_error']=str(e)[:2000];tqdm.write(f'DSP row {i} failed: {e}')
            self._dsp_bar.update(1);self.h.flush()
        if hasattr(self,'_dsp_pool') and not futures:
            self._dsp_pool.shutdown(wait=True);self._dsp_bar.close();del self._dsp_pool

    def ranges(self,length):
        size=16000*10; ranges=[(a,min(a+size,length)) for a in range(0,length,size)]
        if len(ranges)>1 and ranges[-1][1]-ranges[-1][0]<1024:
            ranges[-2]=(ranges[-2][0],length);ranges.pop()
        return ranges

    @staticmethod
    def centers(model,frames,start):
        cfg=model.config;jump=1;field=1
        for kernel,stride in zip(cfg.conv_kernel,cfg.conv_stride):field+=(kernel-1)*jump;jump*=stride
        return (start+np.arange(frames)*jump+(field-1)/2)/16000

    def speech(self,kind):
        pending=self.pending(kind)
        if kind=='wavlm':pending=[i for i in pending if self.done(i,'align')]
        if not pending:return
        from transformers import AutoFeatureExtractor,WavLMModel,HubertModel,Wav2Vec2Model
        rev=self.revision(kind);processor=AutoFeatureExtractor.from_pretrained(MODELS[kind],revision=rev)
        cls={'wavlm':WavLMModel,'hubert':HubertModel,'wav2vec2':Wav2Vec2Model}[kind]
        model=cls.from_pretrained(MODELS[kind],revision=rev).to(self.device).eval()
        def one(i,g):
            x=self.audio(i);layers=[4,8,12] if kind=='wavlm' else [4,12]
            sums={l:np.zeros(768,np.float64) for l in layers};squares={l:np.zeros(768,np.float64) for l in layers};count=0
            frames8=[];times=[];chunks=[];bounds=[]
            segments=self.ranges(len(x));groups={};encoded={}
            # Equal-length groups avoid padded frames and padding-dependent encoder changes.
            for a,b in segments:groups.setdefault(b-a,[]).append((a,b))
            def encode(batch):
                inputs=processor([x[a:b] for a,b in batch],sampling_rate=16000,return_tensors='pt',padding=False).to(self.device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    output=model(**inputs,output_hidden_states=True)
                return [{l:output.hidden_states[l][j].float().cpu().numpy() for l in layers} for j in range(len(batch))]
            for group in groups.values():
                for segment,result in gpu_batch_map(group,self.gpu_batch,encode):encoded[segment[0]]=result
            for a,b in segments:
                for l in layers:
                    f=encoded[a][l]
                    sums[l]+=f.astype('f8').sum(0);squares[l]+=(f.astype('f8')**2).sum(0)
                count+=len(f)
                if kind=='wavlm':
                    f8=encoded[a][8]
                    frames8.append(f8);times.append(self.centers(model,len(f8),a))
                    chunks.append(f8.mean(0));bounds.append([a/16000,b/16000])
                del encoded[a]
            for l in layers:
                mean=sums[l]/count;std=np.sqrt(np.maximum(squares[l]/count-mean**2,0))
                self.fixed(i,f'{kind}_layer{l}_mean',mean)
                if kind=='wavlm' or l==12:self.fixed(i,f'{kind}_layer{l}_std',std)
            if kind=='wavlm':
                frames=np.concatenate(frames8); t=np.concatenate(times)
                aligned=self.getjson(g,'aligned_words_json',[]); W=np.full((len(aligned),768),np.nan,np.float32)
                word_bounds=np.full((len(aligned),2),np.nan,np.float32)
                for j,w in enumerate(aligned):
                    if 'start' not in w or 'end' not in w:continue
                    a,b=float(w['start']),float(w['end']);word_bounds[j]=[a,b]
                    mask=(t>=a)&(t<b)
                    if mask.any():W[j]=frames[mask].mean(0)
                self.put(g,'word_wavlm',W);self.put(g,'word_wavlm_available',np.isfinite(W).all(1))
                self.put(g,'word_times',word_bounds);self.put(g,'chunk_wavlm',np.asarray(chunks,'f4'))
                self.put(g,'chunk_times',np.asarray(bounds,'f4'))
                g.attrs['word_embedding_layer']=8;g.attrs['chunk_embedding_layer']=8
                g.attrs['pooling']='Global frame-weighted population moments across unpadded 10-second contexts'
            g.attrs[kind+'_precision']='bfloat16 CUDA autocast' if torch.cuda.is_bf16_supported() else 'float16 CUDA autocast'
        self.each(kind,pending,one);del model,processor;self.release()

    def phoneme(self):
        pending=self.pending('phoneme')
        if not pending:return
        from transformers import AutoFeatureExtractor,Wav2Vec2ForCTC
        from huggingface_hub import hf_hub_download
        rev=self.revision('phoneme');processor=AutoFeatureExtractor.from_pretrained(MODELS['phoneme'],revision=rev)
        model=Wav2Vec2ForCTC.from_pretrained(MODELS['phoneme'],revision=rev).to(self.device).eval()
        vocab=json.loads(Path(hf_hub_download(MODELS['phoneme'],'vocab.json',revision=rev)).read_text())
        self.h['provenance']['phoneme'].attrs['vocabulary_json']=json.dumps(vocab)
        if model.config.vocab_size!=392:raise ValueError('Phoneme vocabulary width changed.')
        def one(i,g):
            x=self.audio(i);rows=[];times=[]
            segments=self.ranges(len(x));groups={};encoded={}
            for a,b in segments:groups.setdefault(b-a,[]).append((a,b))
            def encode(batch):
                inputs=processor([x[a:b] for a,b in batch],sampling_rate=16000,return_tensors='pt',padding=False).to(self.device)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                    logits=model(**inputs).logits
                probabilities=logits.float().softmax(-1).cpu().numpy().astype('f2')
                return [probabilities[j] for j in range(len(batch))]
            for group in groups.values():
                for segment,result in gpu_batch_map(group,self.gpu_batch,encode):encoded[segment[0]]=result
            for a,b in segments:
                probabilities=encoded.pop(a);rows.append(probabilities)
                times.append(self.centers(model,len(probabilities),a))
            self.put(g,'phoneme_probabilities',np.concatenate(rows));self.put(g,'phoneme_frame_times',np.concatenate(times).astype('f4'))
            g.attrs['phoneme_note']='392 CTC symbols including blanks/special tokens; frame posteriors, not calibrated pronunciation or grammar scores'
        self.each('phoneme',pending,one);del model,processor;self.release()

    def finish(self):
        failures={s:[i for i in range(len(self.jobs)) if not self.done(i,s)] for s in STAGES}
        complete=all(not v for v in failures.values()) and len(self.jobs)==985
        self.h.attrs.update(complete=complete,processing_status_json=json.dumps(failures),
            completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            missing_fixed_values=int((~self.h['available'][:]).sum()))
        self.h.flush();self.h.close()
        print('COMPLETE' if complete else 'PARTIAL: resume to retry missing stages',self.path)
        print('Only one output artifact. Missing/uncertain features stay NaN with availability masks.')
        if not complete:print('Unfinished row indices by stage:',{k:v for k,v in failures.items() if v})

    def run(self):
        try:
            self.transcribe();self.fluent();self.align();self.dsp(background=True);self.grammar();self.text()
            for kind in ['wavlm','hubert','wav2vec2']:self.speech(kind)
            self.phoneme();self.drain_dsp(wait=True);self.finish()
        finally:
            if hasattr(self,'_dsp_pool'):
                for process in (getattr(self._dsp_pool,'_processes',{}) or {}).values():
                    if process.is_alive():process.terminate()
                self._dsp_pool.shutdown(wait=False,cancel_futures=True)
                self._dsp_bar.close()
            if self.h.id.valid:self.h.flush();self.h.close()
