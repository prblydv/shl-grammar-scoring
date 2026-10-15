"""Full-WAV log-mel and separate sinusoidal time arrays; no model training."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from scipy import fft, signal
import soundfile as sf
from tqdm.auto import tqdm

VERSION = 'spectrogram-time-v1'
DEFAULTS = dict(sample_rate=16000, window_samples=400, hop_samples=160,
                n_fft=1024, n_mels=128, f_min=20.0, f_max=8000.0,
                db_floor=-100.0, pe_dimensions=64, pe_base=10000.0,
                pe_reference_step_s=0.01, compression='gzip', compression_level=4)
ARRAYS = ['log_mel', 'time_encoding', 'frame_start_s', 'frame_center_s', 'valid_fraction']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_config(config):
    c = dict(DEFAULTS, **config)
    for key in ['sample_rate','window_samples','hop_samples','n_fft','n_mels','pe_dimensions']:
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not 0 < c['hop_samples'] <= c['window_samples'] <= c['n_fft']:
        raise ValueError('Require hop <= window <= FFT size')
    if not 0 <= c['f_min'] < c['f_max'] <= c['sample_rate']/2:
        raise ValueError('Invalid frequency range')
    if c['pe_dimensions']%2 or c['pe_base']<=1 or c['pe_reference_step_s']<=0:
        raise ValueError('Time encoding requires even dimensions, base>1 and positive time step')
    if c['compression'] not in ['gzip','lzf'] or not 0<=c['compression_level']<=9:
        raise ValueError('Unsupported compression settings')
    if not np.isfinite(c['db_floor']) or c['db_floor']>=0:
        raise ValueError('db_floor must be finite and negative')
    return c


def filterbank(config):
    c = config
    to_mel = lambda hz: 2595*np.log10(1+np.asarray(hz)/700)
    to_hz = lambda mel: 700*(10**(np.asarray(mel)/2595)-1)
    edges = to_hz(np.linspace(to_mel(c['f_min']),to_mel(c['f_max']),c['n_mels']+2))
    frequencies = fft.rfftfreq(c['n_fft'],1/c['sample_rate'])
    bank = np.maximum(0,np.minimum(
        (frequencies[None,:]-edges[:-2,None])/(edges[1:-1]-edges[:-2])[:,None],
        (edges[2:,None]-frequencies[None,:])/(edges[2:]-edges[1:-1])[:,None]))
    if np.any(bank.sum(axis=1)==0):
        raise ValueError('Empty mel filter: increase n_fft or reduce n_mels')
    return bank.astype(np.float32), edges[1:-1].astype(np.float32)


def time_encoding(times, config):
    positions = np.asarray(times,dtype=np.float64)/config['pe_reference_step_s']
    angles = positions[:,None]/config['pe_base']**(
        np.arange(0,config['pe_dimensions'],2,dtype=np.float64)/config['pe_dimensions'])[None,:]
    out = np.empty((len(times),config['pe_dimensions']),dtype=np.float32)
    out[:,0::2], out[:,1::2] = np.sin(angles), np.cos(angles)
    return out


def extract_array(waveform, config):
    c = validate_config(config)
    x = np.asarray(waveform,dtype=np.float32)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Require nonempty finite mono waveform')
    win,hop,sr = c['window_samples'],c['hop_samples'],c['sample_rate']
    count = max(1,1+math.ceil((len(x)-win)/hop))
    padded_samples = (count-1)*hop+win-len(x)
    padded = np.pad(x,(0,padded_samples))
    frames = np.lib.stride_tricks.sliding_window_view(padded,win)[::hop]
    window = signal.get_window('hann',win,fftbins=True).astype(np.float32)
    # Fixed reference, normalized FFT amplitude; no per-clip gain normalization.
    spectrum = fft.rfft(frames*window,n=c['n_fft'],axis=1)/window.sum()
    power = (spectrum.real**2+spectrum.imag**2).astype(np.float32)
    bank,_ = filterbank(c)
    mel_power = power@bank.T
    log_mel = (10*np.log10(np.maximum(mel_power,10**(c['db_floor']/10)))).astype(np.float32)
    starts = np.arange(count,dtype=np.float64)*hop/sr
    centers = starts+win/(2*sr)
    valid = np.minimum(win,np.maximum(0,len(x)-np.arange(count)*hop))/win
    return dict(log_mel=log_mel,time_encoding=time_encoding(centers,c),
                frame_start_s=starts,frame_center_s=centers,
                valid_fraction=valid.astype(np.float32)), padded_samples


def discover(data_root, label_root=None):
    root = Path(data_root).resolve()
    label_root = Path(label_root).resolve() if label_root else root
    rows, csv_hashes = [], {}
    for split in ['train','test']:
        csv_path = label_root/f'{split}.csv'
        table = pd.read_csv(csv_path,keep_default_na=False)
        if 'filename' not in table or table.filename.duplicated().any():
            raise ValueError(f'Invalid filename column in {csv_path}')
        wanted = set(table.filename)
        if any(Path(name).name!=name or not name.lower().endswith('.wav') for name in wanted):
            raise ValueError('Expected plain WAV filenames in CSV')
        lookup = {}
        for path in (root/split).rglob('*'):
            if path.suffix.lower()!='.wav' or '.ipynb_checkpoints' in path.parts or path.name not in wanted:
                continue
            if path.name in lookup:
                raise ValueError(f'Duplicate source: {split}/{path.name}')
            lookup[path.name] = path
        missing = sorted(wanted-set(lookup))
        if missing:
            raise FileNotFoundError(f'{split}: {len(missing)} WAVs missing: {missing[:8]}')
        csv_hashes[f'{split}.csv'] = sha(csv_path)
        for source_row, row in enumerate(table.to_dict('records')):
            filename = row['filename']
            label = float(row['label']) if split=='train' else None
            if label is not None and (not np.isfinite(label) or not 0<=label<=5):
                raise ValueError(f'Invalid original training label: {filename}')
            path = lookup[filename]
            info = sf.info(path)
            if info.frames<=0 or info.samplerate<=0:
                raise ValueError(f'Empty/invalid WAV: {path}')
            rows.append(dict(split=split,filename=filename,source_row=source_row,label=label,
                             path=str(path),audio_sha256=sha(path),
                             native_sample_rate=info.samplerate,native_channels=info.channels,
                             native_samples=info.frames,duration_s=info.duration))
    return rows,csv_hashes


def load_waveform(row, config):
    if sha(row['path']) != row['audio_sha256']:
        raise ValueError('WAV changed since preflight: '+row['path'])
    x, rate = sf.read(row['path'],dtype='float32',always_2d=True)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite audio: '+row['path'])
    x = x.mean(axis=1)
    if rate != config['sample_rate']:
        gcd = math.gcd(rate,config['sample_rate'])
        x = signal.resample_poly(x,config['sample_rate']//gcd,rate//gcd).astype(np.float32)
    return x


def array_digest(values):
    h = hashlib.sha256()
    for key in ARRAYS:
        a = np.ascontiguousarray(values[key])
        h.update(key.encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()


def check_record(group, row, config):
    if not group.attrs.get('complete',False) or group.attrs['audio_sha256']!=row['audio_sha256']:
        raise ValueError('Incomplete or mismatched saved record')
    arrays = {key:group[key][:] for key in ARRAYS}
    n = len(arrays['frame_center_s'])
    if n==0 or arrays['log_mel'].shape!=(n,config['n_mels']) or arrays['time_encoding'].shape!=(n,config['pe_dimensions']):
        raise ValueError('Invalid saved shapes')
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite saved features')
    if any(arrays[key].shape!=(n,) for key in ['frame_start_s','valid_fraction']):
        raise ValueError('Invalid frame metadata shape')
    if array_digest(arrays)!=group.attrs['array_sha256']:
        raise ValueError('Saved feature checksum mismatch')


@contextlib.contextmanager
def writer_lock(output):
    # Keep lock inode on disk so another process cannot race through an unlinked lock.
    import fcntl
    with open(str(output)+'.lock','a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another extractor is writing this output') from exc
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def source_code():
    return Path(__file__).read_text()


def initialize(output, rows, csv_hashes, config, fingerprint):
    handle,name = tempfile.mkstemp(prefix=output.name+'.init-',dir=output.parent)
    os.close(handle)
    temporary = Path(name)
    try:
        with h5py.File(temporary,'w') as f:
            f.attrs.update(schema_version=1,version=VERSION,complete=False,
                           fingerprint=fingerprint,config_json=canonical(config),
                           created_at_utc=datetime.now(timezone.utc).isoformat(),
                           expected_records=len(rows),completed_records=0,
                           axis_order='time,mel_frequency',time_encoding_axis_order='time,encoding_dimension',
                           normalization='No per-record peak/loudness normalization; later learned scaling must fit training folds only.',
                           log_mel_definition='10*log10(max(mel(abs(rfft(Hann*frame)/sum(Hann))**2),10**(db_floor/10))); HTK triangular filters, no area normalization, reference=1',
                           frame_policy='No centering or left padding; final partial window right-zero-padded; valid_fraction saved.',
                           pe_definition='position=original frame center seconds/pe_reference_step_s; alternating sin and cos; denominator=base**(2i/dimension). No PE is painted into acoustic values.',
                           test_label_policy='Test CSV labels are placeholders and ignored; stored NaN with label_known=False.')
            strings = h5py.string_dtype('utf-8')
            idx = f.create_group('index')
            for key in ['filename','split','audio_sha256']:
                idx.create_dataset(key,data=[r[key] for r in rows],dtype=strings)
            idx.create_dataset('source_row',data=[r['source_row'] for r in rows])
            idx.create_dataset('label',data=[np.nan if r['label'] is None else r['label'] for r in rows])
            idx.create_dataset('label_known',data=[r['label'] is not None for r in rows])
            idx.create_dataset('completed',data=np.zeros(len(rows),dtype=bool))
            _,centers = filterbank(config)
            f.create_dataset('mel_centers_hz',data=centers)
            f.create_dataset('mel_filterbank',data=filterbank(config)[0])
            f.create_dataset('source_manifest_json',data=canonical(dict(rows=rows,csv_sha256=csv_hashes)),dtype=strings)
            f.create_dataset('extractor_source',data=source_code(),dtype=strings)
            versions = {package:importlib.metadata.version(package) for package in ['numpy','scipy','pandas','soundfile','h5py','tqdm']}
            f.create_dataset('environment_json',data=canonical(dict(python=platform.python_version(),packages=versions)),dtype=strings)
            f.create_group('records');f.create_group('pending');f.flush()
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def run(data_root, output, label_root=None, config=None, limit=None, min_free_gb=2):
    c = validate_config(config or {})
    output = Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    if limit is not None and (type(limit) is not int or limit<=0):
        raise ValueError('limit must be None or a positive integer')
    print('Preflight: matching source CSVs, reading WAV headers and hashing all sources...',flush=True)
    rows,csv_hashes = discover(data_root,label_root)
    identity = [{k:r[k] for k in ['split','filename','source_row','label','audio_sha256']} for r in rows]
    fingerprint = hashlib.sha256(canonical(dict(version=VERSION,config=c,rows=identity,source_sha256=hashlib.sha256(source_code().encode()).hexdigest())).encode()).hexdigest()
    print(f'Found {len(rows)} WAVs; train={sum(r["split"]=="train" for r in rows)}, test={sum(r["split"]=="test" for r in rows)}. Output: {output}',flush=True)
    with writer_lock(output):
        if not output.exists():
            initialize(output,rows,csv_hashes,c,fingerprint)
        with h5py.File(output,'r+') as f:
            if f.attrs.get('fingerprint')!=fingerprint:
                raise ValueError('Existing output belongs to different inputs/settings/code. Select a new OUTPUT filename; original file was preserved.')
            for key in list(f['pending']):
                del f['pending'][key]
            completed = np.zeros(len(rows),dtype=bool)
            for i,row in enumerate(rows):
                key = f'{i:06d}'
                if key in f['records']:
                    check_record(f['records'][key],row,c)
                    completed[i] = True
            f['index/completed'][:] = completed
            f.attrs['completed_records'] = int(completed.sum())
            f.attrs['complete'] = bool(completed.all());f.flush()
            targets = list(range(len(rows)))[:limit] if limit else list(range(len(rows)))
            for i in tqdm(targets,desc='Full-WAV spectrograms',unit='wav'):
                if completed[i]:
                    continue
                row = rows[i]
                # Upper bound for one uncompressed record plus user's reserve.
                samples = math.ceil(row['duration_s']*c['sample_rate'])
                frames = max(1,1+math.ceil((samples-c['window_samples'])/c['hop_samples']))
                budget = frames*(4*(c['n_mels']+c['pe_dimensions']+1)+16)
                if shutil.disk_usage(output.parent).free < min_free_gb*1024**3+budget:
                    raise OSError('Disk reserve reached; output retains completed recordings. Free space and rerun.')
                x = load_waveform(row,c);values,padding = extract_array(x,c)
                key = f'{i:06d}';g = f['pending'].create_group(key)
                g.attrs.update(filename=row['filename'],split=row['split'],audio_sha256=row['audio_sha256'],
                               source_row=row['source_row'],native_sample_rate=row['native_sample_rate'],
                               native_channels=row['native_channels'],resampled_samples=len(x),
                               duration_s=len(x)/c['sample_rate'],right_padding_samples=padding,
                               array_sha256=array_digest(values),complete=False)
                options = dict(compression=c['compression'],shuffle=True)
                if c['compression']=='gzip':
                    options['compression_opts'] = c['compression_level']
                for name,array in values.items():
                    g.create_dataset(name,data=array,**options)
                g.attrs['complete'] = True;f.flush()
                f.move(f'pending/{key}',f'records/{key}');f.flush()
                completed[i] = True;f['index/completed'][i] = True
                f.attrs['completed_records'] = int(completed.sum());f.flush()
            f.attrs['complete'] = bool(completed.all())
            f.attrs['last_run_utc'] = datetime.now(timezone.utc).isoformat();f.flush()
            print(f'Saved {int(completed.sum())}/{len(rows)} recordings; complete={bool(completed.all())}; size={output.stat().st_size/1024**3:.3f} GiB',flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default='/workspace/shl/data/audio_full')
    parser.add_argument('--label-root')
    parser.add_argument('--output',default='/workspace/shl/audio_embeddings/shl_spectrogram_time_v1.h5')
    parser.add_argument('--limit',type=int,default=None)
    args = parser.parse_args()
    run(args.data_root,args.output,args.label_root,limit=args.limit)


if __name__ == '__main__':
    main()
