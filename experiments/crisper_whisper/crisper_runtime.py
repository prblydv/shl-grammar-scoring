"""Kaggle CrisperWhisper transcription, resumable exports and private publication."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import soundfile as sf


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def make_jobs(data_root,expected_counts=None):
    """Join audio by split AND filename; preserve source CSV row order and labels."""
    root=Path(data_root);rows=[];sources={}
    for split in ['train','test']:
        csv=root/f'{split}.csv';table=pd.read_csv(csv)
        if 'filename' not in table or table.filename.isna().any() or not table.filename.is_unique:
            raise ValueError(f'{split}: source CSV needs unique nonempty filenames')
        if split=='train' and 'label' not in table:raise ValueError('Missing training labels')
        files=list((root/split).rglob('*.wav'))
        by_name={p.name:p for p in files}
        if len(by_name)!=len(files) or set(by_name)!=set(table.filename):
            raise ValueError(f'{split}: WAV filenames do not match the source CSV exactly')
        sources[split]={'csv_sha256':digest(csv),'rows':len(table)}
        for source_index,row in table.iterrows():
            filename=str(row.filename)
            if Path(filename).name!=filename or not filename.endswith('.wav'):
                raise ValueError('Unexpected audio filename')
            label=float(row.label) if split=='train' else None
            if label is not None and (not math.isfinite(label) or not 0<=label<=5):
                raise ValueError('Invalid original training label')
            path=by_name[filename];info=sf.info(path)
            if info.frames<=0:raise ValueError(f'Empty WAV: {split}/{filename}')
            entry=dict(job_index=len(rows),source_row_index=int(source_index),filename=filename,
                audio_id=filename,split=split,label=label,label_known=split=='train',
                label_source='original_train_csv' if split=='train' else 'unlabelled_test',
                audio_path=str(path),audio_relative_path=path.relative_to(root).as_posix(),
                duration_s=float(info.duration),sample_rate_hz=int(info.samplerate),
                channels=int(info.channels),audio_sha256=digest(path))
            entry['input_fingerprint']=fingerprint({k:entry[k] for k in ['split','filename','label',
                'label_known','audio_sha256','duration_s','sample_rate_hz','channels']})
            rows.append(entry)
    counts=dict(Counter(r['split'] for r in rows))
    if expected_counts is not None and counts!=expected_counts:
        raise ValueError(f'Expected {expected_counts}, found {counts}')
    return rows,sources


def checkpoint_path(root,row):
    return Path(root)/row['split']/(row['filename']+'.json')


def reusable(record,row,config_hash):
    return (record.get('config_fingerprint')==config_hash and
        record.get('input_fingerprint')==row['input_fingerprint'] and
        record.get('split')==row['split'] and record.get('filename')==row['filename'] and
        not record.get('error') and isinstance(record.get('transcript'),str) and
        bool(record['transcript'].strip()))


def read_checkpoint(root,row,config_hash):
    path=checkpoint_path(root,row)
    try:record=json.loads(path.read_text())
    except (OSError,ValueError):return None
    return record if reusable(record,row,config_hash) else None


def restore_previous(input_root,checkpoint_root,jobs,config_hash):
    """An attached completed or partial export can resume a new Kaggle session."""
    by_key={(r['split'],r['filename']):r for r in jobs};restored=0
    for path in sorted(Path(input_root).rglob('transcripts_crisper.jsonl')):
        with path.open() as handle:
            for line in handle:
                if not line.strip():continue
                try:record=json.loads(line)
                except ValueError:continue
                row=by_key.get((record.get('split'),record.get('filename')))
                if row and reusable(record,row,config_hash) and not read_checkpoint(checkpoint_root,row,config_hash):
                    atomic_json(checkpoint_path(checkpoint_root,row),record);restored+=1
    return restored


def result_record(row,result,config,attempts):
    text=result.text
    if not isinstance(text,str) or not text.strip():raise ValueError('Model returned empty transcript')
    if result.mode!='verbatim':raise ValueError('Unexpected model output mode')
    words=[asdict(w) for w in (result.words or [])]
    warnings=0
    for word in words:
        start,end=word.get('start'),word.get('end')
        if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
            raise ValueError('Nonfinite word timestamp')
        if end<start or start<-.25 or end>row['duration_s']+.5:warnings+=1
    record={**row,'transcript':text,'error':None,'model_id':config['model_id'],
        'model_revision':config['model_revision'],'crisperwhisper_version':config['package_version'],
        'mode':'verbatim','language':result.language,'backend':config['backend'],
        'compute_type':config['compute_type'],'config_fingerprint':fingerprint(config),
        'processing_time_s':float(result.processing_time),'attempts':attempts,
        'transcribed_at_utc':datetime.now(timezone.utc).isoformat(),
        'words':words,'timing_warning_count':warnings,
        'chunks':[asdict(c) for c in (result.chunks or [])]}
    # Keep original model text; no spelling, grammar, filler or punctuation cleanup.
    json.dumps(record,allow_nan=False)
    return record


def worker(job_file,config_file,model_dir,checkpoint_root):
    import torch
    from crisperwhisper import CrisperWhisperModel
    from importlib.metadata import version
    jobs=json.loads(Path(job_file).read_text());config=json.loads(Path(config_file).read_text())
    assert version('crisperwhisper')==config['package_version']
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    signature=fingerprint(config)
    remaining=[r for r in jobs if not read_checkpoint(checkpoint_root,r,signature)]
    print(f'{os.environ.get("WORKER_NAME","GPU")}: {len(jobs)-len(remaining)} cached, {len(remaining)} remaining',flush=True)
    if not remaining:return
    model=CrisperWhisperModel(model_dir,backend=config['backend'],device='cuda',
        device_index=0,compute_type=config['compute_type'])
    started=time.monotonic();processed_audio=0.;total_audio=sum(r['duration_s'] for r in remaining)
    for index,row in enumerate(remaining,1):
        error=None;record=None
        for attempt in range(1,config['max_attempts']+1):
            try:
                result=model.transcribe(row['audio_path'],**config['transcribe'])
                record=result_record(row,result,config,attempt);break
            except Exception as exc:
                error=f'{type(exc).__name__}: {str(exc)[:1500]}'
                gc.collect();torch.cuda.empty_cache()
                print(f'{row["split"]}/{row["filename"]}: retry {attempt}/{config["max_attempts"]}',flush=True)
        if record is None:
            record={**row,'transcript':None,'error':error,'words':[],'chunks':[],
                'model_id':config['model_id'],'model_revision':config['model_revision'],
                'config_fingerprint':signature,'attempts':config['max_attempts']}
        atomic_json(checkpoint_path(checkpoint_root,row),record)
        processed_audio+=row['duration_s'];elapsed=time.monotonic()-started
        speed=processed_audio/max(elapsed,.001);eta=(total_audio-processed_audio)/max(speed,.001)
        print(f'{os.environ.get("WORKER_NAME","GPU")}: {index}/{len(remaining)} | '
            f'{speed:.2f}x audio | ETA {eta/60:.1f} min | {row["split"]}/{row["filename"]} | '
            f'{"OK" if record.get("error") is None else "FAILED"}',flush=True)


def run_workers(jobs,config,model_dir,checkpoint_root,runtime_file,work_root,gpu_count):
    root=Path(work_root);root.mkdir(parents=True,exist_ok=True)
    partitions=[[] for _ in range(gpu_count)];loads=[0.]*gpu_count
    signature=fingerprint(config)
    pending=[r for r in jobs if not read_checkpoint(checkpoint_root,r,signature)]
    for row in sorted(pending,key=lambda r:r['duration_s'],reverse=True):
        gpu=min(range(gpu_count),key=lambda i:loads[i]);partitions[gpu].append(row);loads[gpu]+=row['duration_s']
    config_file=root/'config.json';atomic_json(config_file,config)
    processes=[]
    try:
        for gpu,partition in enumerate(partitions):
            if not partition:continue
            job_file=root/f'jobs_gpu_{gpu}.json';atomic_json(job_file,partition)
            environment={k:v for k,v in os.environ.items() if k not in ['KAGGLE_KEY','KAGGLE_API_TOKEN']}
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu),WORKER_NAME=f'GPU {gpu}',
                OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
            command=[sys.executable,'-u',str(runtime_file),'worker',str(job_file),str(config_file),
                str(model_dir),str(checkpoint_root)]
            processes.append(subprocess.Popen(command,env=environment))
        while any(p.poll() is None for p in processes):
            failures=[p.returncode for p in processes if p.poll() is not None and p.returncode!=0]
            if failures:raise RuntimeError(f'GPU worker failed with exit code {failures}; checkpoints preserved')
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('GPU worker failed; checkpoints preserved')
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()


def export_bundle(jobs,config,checkpoint_root,destination,data_root,sources,model_license=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    records=[];word_rows=[];signature=fingerprint(config)
    for row in jobs:
        record=read_checkpoint(checkpoint_root,row,signature)
        if record is None:
            path=checkpoint_path(checkpoint_root,row)
            try:previous=json.loads(path.read_text())
            except (OSError,ValueError):previous={}
            record={**row,'transcript':None,'error':previous.get('error') or 'Missing/invalid checkpoint',
                'words':[],'chunks':[],'model_id':config['model_id'],
                'model_revision':config['model_revision'],'config_fingerprint':signature}
        else:
            # Refresh ordering and current mounted path on a cross-session resume.
            record={**record,**row}
        records.append(record)
        for index,word in enumerate(record.get('words',[])):
            word_rows.append(dict(split=row['split'],filename=row['filename'],audio_id=row['audio_id'],
                word_index=index,word=word['word'],start_s=word['start'],end_s=word['end']))
    with (destination/'transcripts_crisper.jsonl').open('w') as handle:
        for record in records:handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
    flat=[{**{k:v for k,v in r.items() if k not in ['words','chunks']},
        'words_json':json.dumps(r.get('words',[]),ensure_ascii=False),
        'chunks_json':json.dumps(r.get('chunks',[]),ensure_ascii=False)} for r in records]
    frame=pd.DataFrame(flat);frame['label']=pd.array(frame.label,dtype='Float64')
    frame.to_csv(destination/'transcripts_crisper.csv',index=False)
    frame.to_parquet(destination/'transcripts_crisper.parquet',index=False)
    for split in ['train','test']:
        frame[frame.split.eq(split)].to_csv(destination/f'{split}_transcripts_crisper.csv',index=False)
        shutil.copy2(Path(data_root)/f'{split}.csv',destination/f'source_{split}.csv')
    words=pd.DataFrame(word_rows,columns=['split','filename','audio_id','word_index','word','start_s','end_s'])
    words.to_parquet(destination/'word_timestamps.parquet',index=False)
    words.to_csv(destination/'word_timestamps.csv',index=False)
    failures=frame[frame.error.fillna('').ne('')|frame.transcript.fillna('').str.strip().eq('')]
    failures[['split','filename','error']].to_csv(destination/'transcription_errors.csv',index=False)
    schema={'primary_key':['split','filename'],'row_order':'train then test, preserving source CSV order',
        'label':'Original 0–5 training score; null for test. Test CSV labels are placeholders.',
        'label_known':'True only for original training labels.',
        'transcript':'Unmodified CrisperWhisper verbatim text, not human ground truth.',
        'audio_relative_path':'Path relative to competition Dataset_Final directory.',
        'words_json':'JSON list of {word,start,end}; seconds in original recording.',
        'chunks_json':'Raw model chunk metadata; model may use overlapping windows.',
        'error':'Null on success; incomplete exports must not be used as completed datasets.',
        'timing_warning_count':'Number of unusual timestamp intervals; text is preserved.',
        'all_columns':{c:str(frame[c].dtype) for c in frame.columns}}
    atomic_json(destination/'schema.json',schema)
    from importlib.metadata import version,PackageNotFoundError
    packages={}
    for package in ['crisperwhisper','torch','transformers','numpy','pandas','soundfile','pyarrow','kaggle']:
        try:packages[package]=version(package)
        except PackageNotFoundError:pass
    manifest=dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),
        competition='shl-hiring-assessment-2026',config=config,config_fingerprint=signature,
        total_rows=len(frame),split_counts=frame.split.value_counts().to_dict(),errors=len(failures),
        complete=len(failures)==0,source_csvs=sources,packages=packages,
        label_policy='Train labels preserved. Test labels are unknown/null; source_test.csv retains placeholders.',
        transcript_policy='Verbatim model output; no grammar correction or filler removal.',
        model_license_url=f'https://huggingface.co/{config["model_id"]}/blob/{config["model_revision"]}/LICENSE.md')
    atomic_json(destination/'run_manifest.json',manifest)
    if model_license and Path(model_license).exists():shutil.copy2(model_license,destination/'MODEL_LICENSE.md')
    (destination/'README.md').write_text(
        '1. SHL 2026 CrisperWhisper verbatim transcripts: all train/test recordings.\n\n'
        '2. Join on split + filename. Filename values overlap across train/test. '
        'Original training labels are unchanged; combined/test transcript labels are blank. '
        'source_test.csv contains original placeholders, not ground-truth test scores.\n\n'
        '3. Preserve transcript text, filler tags, repetitions and original grammar. '
        'These are ASR predictions and can contain omissions or hallucinations.\n\n'
        '4. Word/chunk times are in seconds; review timing_warning_count. '
        'See schema.json, run_manifest.json and file_checksums.json for provenance.\n\n'
        '5. Attach this private dataset in future notebooks and read transcripts_crisper.parquet '
        'or train_transcripts_crisper.csv/test_transcripts_crisper.csv. '
        'Keep source-related recordings together in scoring validation.\n\n'
        '6. Competition data terms apply. CrisperWhisper 2.0 weights and generated outputs '
        'carry the upstream non-commercial research licence; see MODEL_LICENSE.md. '
        'This metadata does not grant additional rights.\n\n'
        '7. To resume in a new session, attach this export containing transcripts_crisper.jsonl; '
        'only matching successful model/config/audio checkpoints are reused.\n')
    atomic_json(destination/'file_checksums.json',{p.name:digest(p) for p in sorted(destination.iterdir())
        if p.is_file() and p.name not in ['file_checksums.json','dataset-metadata.json']})
    validate_bundle(destination,jobs,config)
    return frame,manifest


def validate_bundle(destination,jobs,config,require_complete=False):
    destination=Path(destination);frame=pd.read_parquet(destination/'transcripts_crisper.parquet')
    if len(frame)!=len(jobs) or frame.duplicated(['split','filename']).any():raise ValueError('Invalid export keys/counts')
    expected=pd.DataFrame(jobs)
    if list(zip(frame.split,frame.filename))!=list(zip(expected.split,expected.filename)):
        raise ValueError('Source CSV ordering changed')
    train=frame[frame.split.eq('train')]
    expected_train=expected[expected.split.eq('train')]
    if not np.array_equal(train.label.to_numpy(dtype=float),expected_train.label.to_numpy(dtype=float)):
        raise ValueError('Training labels changed')
    test=frame[frame.split.eq('test')]
    if not test.label.isna().all() or test.label_known.any():raise ValueError('Test labels must be unknown')
    if not train.label_known.all():raise ValueError('Missing label-known flag')
    if not frame.model_id.eq(config['model_id']).all() or not frame.model_revision.eq(config['model_revision']).all():
        raise ValueError('Export contains another model')
    manifest=json.loads((destination/'run_manifest.json').read_text())
    if manifest['config_fingerprint']!=fingerprint(config):raise ValueError('Configuration mismatch')
    for name,sha in json.loads((destination/'file_checksums.json').read_text()).items():
        if Path(name).name!=name or digest(destination/name)!=sha:raise ValueError('Export checksum mismatch')
    failed=frame.error.fillna('').ne('')|frame.transcript.fillna('').str.strip().eq('')
    if manifest['complete']!=bool(not failed.any()) or manifest['errors']!=int(failed.sum()):
        raise ValueError('Manifest completion mismatch')
    if require_complete and failed.any():raise ValueError('Incomplete transcripts; rerun workers before publication')
    return frame


def kaggle_from_secrets(owner,secret_name='KAGGLE_KEY',auth_method='auto'):
    """Read only attached Kaggle secrets. Never write credentials into output files."""
    try:
        from kaggle_secrets import UserSecretsClient
        client=UserSecretsClient()
        key=None
        labels=list(dict.fromkeys([secret_name,'KAGGLE_KEY','KAGGLE_API_TOKEN','KAGGLE_TOKEN','KAGGLE_API_KEY']))
        for label in labels:
            try:key=client.get_secret(label).strip()
            except Exception:continue
            if key:break
        if not key:raise ValueError('No enabled Kaggle credential')
    except Exception:
        raise RuntimeError('Enable your existing Kaggle API secret in Add-ons > Secrets and set KAGGLE_SECRET_NAME to its label.') from None
    # Accept quotation marks accidentally copied with the JSON key value.
    if len(key)>=2 and key[0]==key[-1] and key[0] in ['"',"'"]:
        key=key[1:-1].strip()
    if key.startswith('{'):
        try:
            credentials=json.loads(key)
            if credentials.get('username',owner)!=owner:raise ValueError('Wrong account')
            key=credentials['key'];auth_method='legacy'
        except (ValueError,KeyError,TypeError):
            raise ValueError('Kaggle JSON secret must contain a key for DATASET_OWNER') from None
    if auth_method=='auto':
        auth_method='legacy' if re.fullmatch(r'[0-9a-fA-F]{32,64}',key) else 'token'
    if auth_method not in ['legacy','token']:raise ValueError('KAGGLE_AUTH_METHOD must be auto, legacy or token')
    if auth_method=='legacy' and not re.fullmatch(r'[0-9a-fA-F]{32,64}',key):
        raise ValueError('Legacy mode needs only the hex key value from kaggle.json, without the key: label or surrounding JSON.')
    client=IsolatedKaggleClient(owner,key,auth_method)
    client._request('probe')
    return client


class IsolatedKaggleClient:
    """Run API calls in fresh Python processes after pip installs/updates the SDK.

    Kaggle notebooks may already have older kaggle/kagglesdk modules loaded.
    Credentials pass only over stdin, never via command arguments or disk.
    """
    def __init__(self,owner,credential,auth_method):
        self.owner=owner;self._credential=credential;self.auth_method=auth_method

    def _request(self,operation,**kwargs):
        payload=dict(owner=self.owner,credential=self._credential,auth_method=self.auth_method,
            operation=operation,kwargs=kwargs)
        try:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),'kaggle_api'],
                input=json.dumps(payload),capture_output=True,text=True,timeout=1200)
            reply=json.loads(process.stdout)
        except (ValueError,subprocess.SubprocessError):
            raise RuntimeError('Kaggle API helper could not start. Restart the Kaggle session and Run All to load the installed SDK.') from None
        if not reply.get('ok'):raise RuntimeError(reply.get('message','Kaggle API request failed'))
        return reply.get('value')

    def dataset_list(self,**kwargs):
        return [SimpleNamespace(**r) for r in self._request('dataset_list',**kwargs)]

    def dataset_metadata(self,dataset,path):
        return self._request('dataset_metadata',dataset=dataset,path=path)

    def dataset_create_new(self,folder,**kwargs):
        return SimpleNamespace(**self._request('dataset_create_new',folder=folder,**kwargs))

    def dataset_create_version(self,folder,**kwargs):
        return SimpleNamespace(**self._request('dataset_create_version',folder=folder,**kwargs))

    def dataset_status(self,dataset):return self._request('dataset_status',dataset=dataset)


def kaggle_request(payload,api_factory=None):
    """Fresh-process SDK adapter. Return safe diagnostics without raw exceptions."""
    import contextlib
    import io
    owner=payload['owner'];key=payload['credential'];method=payload['auth_method']
    previous={k:os.environ.get(k) for k in ['KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN']}
    try:
        for name in previous:os.environ.pop(name,None)
        if method=='legacy':os.environ.update(KAGGLE_USERNAME=owner,KAGGLE_KEY=key)
        else:os.environ['KAGGLE_API_TOKEN']=key
        # Discard SDK authentication/upload messages; only emit our JSON result.
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            if api_factory is None:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api_factory=KaggleApi
            api=api_factory();api.authenticate()
            if api.config_values.get('username')!=owner:
                return {'ok':False,'message':'Authenticated account differs from DATASET_OWNER. Set DATASET_OWNER to the exact username in your kaggle.json.'}
            operation=payload['operation'];kwargs=payload.get('kwargs',{})
            if operation=='probe':
                api.dataset_list(mine=True,page=1)
                return {'ok':True,'value':{'owner':owner,'auth_method':method}}
            if operation=='dataset_list':
                value=[{'ref':getattr(r,'ref','')} for r in api.dataset_list(**kwargs)]
            elif operation in ['dataset_metadata','dataset_status']:
                value=getattr(api,operation)(**kwargs)
            elif operation in ['dataset_create_new','dataset_create_version']:
                response=getattr(api,operation)(**kwargs)
                error=getattr(response,'error',None) or getattr(response,'error_message',None)
                if error:return {'ok':False,'message':'Kaggle rejected the dataset upload. Files remain saved in /kaggle/working/crisper_dataset.'}
                value={'error':None}
            else:return {'ok':False,'message':'Unknown Kaggle API operation'}
        return {'ok':True,'value':value}
    except (ImportError,AttributeError):
        return {'ok':False,'message':'Kaggle SDK dependencies could not load. Restart the Kaggle session and Run All; the installed Kaggle SDK is 2.2.4.'}
    except (Exception,SystemExit) as exc:
        status=http_status(exc)
        if status in [400,401,403]:
            message=(f'Kaggle rejected this request (HTTP {status}). For a legacy key, DATASET_OWNER must be '
                'the exact username in kaggle.json and the enabled secret must contain its key value. '
                'Check whether the key was expired/revoked. Set CREATE_DATASET=False to transcribe and save output files without API upload.')
        elif isinstance(exc,SystemExit):
            message='Kaggle could not authenticate this credential. Check the key and DATASET_OWNER; use KAGGLE_AUTH_METHOD="legacy" for the key field in kaggle.json.'
        else:
            detail=f'HTTP {status}' if status else type(exc).__name__
            message=f'Kaggle API request failed ({detail}). Check Internet access and retry. Transcript outputs remain available.'
        return {'ok':False,'message':message}
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value


def http_status(exc):
    for attribute in ['status','status_code']:
        value=getattr(exc,attribute,None)
        if value is not None:return value
    return getattr(getattr(exc,'response',None),'status_code',None)


def publish_private(api,destination,jobs,config,dataset_id):
    validate_bundle(destination,jobs,config,require_complete=True)
    owner,slug=dataset_id.split('/')
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',owner) or not re.fullmatch(r'[a-z0-9-]{6,50}',slug):
        raise ValueError('Invalid dataset ID')
    metadata={'id':dataset_id,'title':'SHL 2026 CrisperWhisper Verbatim Transcripts',
        'licenses':[{'name':'other'}],
        'description':Path(destination,'README.md').read_text(),
        'subtitle':'Verbatim train/test transcripts, original labels and word timestamps'}
    atomic_json(Path(destination)/'dataset-metadata.json',metadata)
    # Kaggle metadata can return 403 for a nonexistent private destination.
    # Search the authenticated account's own datasets instead of guessing from 403.
    refs=set()
    for page in range(1,101):
        try:owned=api.dataset_list(mine=True,page=page)
        except Exception:
            raise RuntimeError('Could not list your datasets. No dataset was changed.') from None
        if not owned:break
        refs.update(getattr(d,'ref','') for d in owned)
    else:raise RuntimeError('Dataset listing exceeded 100 pages; choose a dedicated account/destination')
    exists=dataset_id in refs
    if exists:
        with tempfile.TemporaryDirectory(prefix='crisper-dataset-metadata-') as temp:
            try:
                path=api.dataset_metadata(dataset_id,temp)
                existing=json.loads(Path(path).read_text())
            except Exception:
                raise RuntimeError('Could not check existing dataset privacy. No dataset was changed.') from None
    if exists:
        private=existing.get('isPrivate',existing.get('is_private'))
        if private is not True:raise ValueError('Existing destination is not confirmed private; choose a new slug')
        if 'CrisperWhisper' not in existing.get('title',''):
            raise ValueError('Existing destination is unrelated; choose a new dataset slug')
        response=api.dataset_create_version(str(destination),version_notes='Complete CrisperWhisper verbatim export',
            quiet=True,convert_to_csv=False,delete_old_versions=False)
    else:
        response=api.dataset_create_new(str(destination),public=False,quiet=True,convert_to_csv=False)
    error=getattr(response,'error',None) or getattr(response,'error_message',None)
    if error:raise RuntimeError('Kaggle rejected the dataset publication; local outputs remain available')
    url=f'https://www.kaggle.com/datasets/{dataset_id}'
    print('Dataset upload accepted:',url,flush=True)
    for _ in range(40):
        status=api.dataset_status(dataset_id)
        if status in ['ready','complete']:print('Private dataset ready:',url,flush=True);return url
        if status in ['error','failed']:raise RuntimeError('Kaggle dataset processing failed; local outputs preserved')
        print('Dataset processing:',status,flush=True);time.sleep(15)
    raise RuntimeError('Upload accepted but still processing. Rerun the publication cell after checking Kaggle.')


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='worker':worker(*sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='kaggle_api':
        print(json.dumps(kaggle_request(json.loads(sys.stdin.read()))))
