"""Synthetic-only checks: timing, amplitude, source joins, resume and preservation."""
import importlib.util
from pathlib import Path
import tempfile

import h5py
import numpy as np
import pandas as pd
import soundfile as sf

spec = importlib.util.spec_from_file_location('spectrogram_fixture_runtime',Path(__file__).with_name('extract_spectrograms.py'))
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
c = runtime.validate_config({})
sr = c['sample_rate']
wave = .1*np.sin(2*np.pi*440*np.arange(sr+83)/sr).astype(np.float32)
a,pad = runtime.extract_array(wave,c)
b,_ = runtime.extract_array(wave*2,c)
assert a['log_mel'].shape==(len(a['frame_center_s']),128)
active = a['log_mel']>-85
np.testing.assert_allclose((b['log_mel']-a['log_mel'])[active],20*np.log10(2),atol=2e-4)
np.testing.assert_array_equal(a['time_encoding'],b['time_encoding'])
np.testing.assert_allclose(a['time_encoding'][:,0],np.sin(a['frame_center_s']/.01),atol=1e-7)
np.testing.assert_allclose(a['time_encoding'][:,1],np.cos(a['frame_center_s']/.01),atol=1e-7)
assert a['frame_start_s'][-1]*sr+c['window_samples']>=len(wave)
assert pad>0 and 0<a['valid_fraction'][-1]<1
delayed = np.concatenate([np.zeros(sr*30,dtype=np.float32),wave])
d,_ = runtime.extract_array(delayed,c)
assert np.max(d['log_mel'][:2998])<=-99.99
assert np.max(d['log_mel'][3000:])>-60
short,_ = runtime.extract_array(np.ones(20,dtype=np.float32)*.1,c)
assert short['log_mel'].shape==(1,128) and np.isclose(short['valid_fraction'][0],.05)

with tempfile.TemporaryDirectory(prefix='shl-spectrogram-fixture-') as folder:
    root = Path(folder)
    for split in ['train','test']:
        (root/split).mkdir()
        sf.write(root/split/'audio_0.wav',wave,sr,subtype='PCM_16')
        pd.DataFrame({'filename':['audio_0.wav'],'label':[0 if split=='train' else 4.5]}).to_csv(root/f'{split}.csv',index=False)
    checkpoints = root/'train/.ipynb_checkpoints'
    checkpoints.mkdir();sf.write(checkpoints/'audio_0.wav',wave,sr)
    output = root/'out.h5'
    runtime.run(root,output,limit=1,min_free_gb=0)
    with h5py.File(output) as f:
        assert not f.attrs['complete'] and f.attrs['completed_records']==1
        first = f['records/000000'].attrs['array_sha256']
        assert f['index/label'][0]==0 and np.isnan(f['index/label'][1])
        assert list(f['index/split'].asstr()[:])==['train','test']
    runtime.run(root,output,min_free_gb=0)
    with h5py.File(output) as f:
        assert f.attrs['complete'] and f.attrs['completed_records']==2
        assert f['records/000000'].attrs['array_sha256']==first
    # A source/settings mismatch must not change the existing output bytes.
    original = runtime.sha(output)
    try:
        runtime.run(root,output,config={'n_mels':64},min_free_gb=0)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed settings accepted')
    assert runtime.sha(output)==original
    sf.write(root/'test/audio_0.wav',wave*.5,sr,subtype='PCM_16')
    try:
        runtime.run(root,output,min_free_gb=0)
    except ValueError:
        pass
    else:
        raise AssertionError('Changed source accepted')
    assert runtime.sha(output)==original
    # A moved-but-not-indexed committed record is recovered; a partial pending record is discarded.
    sf.write(root/'test/audio_0.wav',wave,sr,subtype='PCM_16')
    with h5py.File(output,'r+') as f:
        f['index/completed'][1] = False
        f['pending'].create_group('unfinished')
    runtime.run(root,output,min_free_gb=0)
    with h5py.File(output) as f:
        assert f['index/completed'][:].all() and not len(f['pending'])
    with runtime.writer_lock(output):
        try:
            with runtime.writer_lock(output):
                raise AssertionError('Concurrent writer accepted')
        except RuntimeError:
            pass
    with h5py.File(output,'r+') as f:
        f['records/000000/log_mel'][0,0] += 1
    try:
        runtime.run(root,output,min_free_gb=0)
    except ValueError as exc:
        assert 'checksum' in str(exc)
    else:
        raise AssertionError('Corruption accepted')
print('Synthetic checks passed: amplitude, separate PE, 30s silence, short/tail coverage, split joins, ignored test labels, resume, mismatches, pending recovery, lock and corruption.')
