"""Focused checks only: no notebook execution, model download, or GPU inference."""
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import channel_runtime as runtime


def main():
    def fake_batch(items):
        if len(items)>2:raise torch.cuda.OutOfMemoryError('test batch split')
        return [v*10 for v in items]
    assert list(runtime.gpu_batch_map(list(range(9)),8,fake_batch))==[(i,i*10) for i in range(9)]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        capacities=list(pool.map(runtime.cpu_capacity,[]))
        assert pool.submit(runtime.cpu_capacity).result()>0
    root=Path('/workspace/shl/data/text_input_crisper')
    tables={s:pd.read_csv(root/f'{s}_transcripts_crisper.csv',keep_default_na=False) for s in ['train','test']}
    index=pd.read_csv('/workspace/shl/data/audio_full/audio_index.csv',keep_default_na=False)
    for split,t in tables.items():
        expected=index[index.split.eq(split)].set_index('filename').audio_sha256
        assert len(t)==len(expected)
        assert all(expected[r.filename]==r.audio_sha256 for r in t.itertuples())
    selected=[tables['train'][tables['train'].error.eq('')].iloc[0],tables['test'].iloc[0],
              tables['train'][tables['train'].error.ne('')].iloc[0]]
    jobs=[dict(split=r['split'],filename=r.filename,source_row=int(r.source_row_index),
               sha=r.audio_sha256,label=float(r.label) if r['split']=='train' else np.nan,
               path=f'/workspace/shl/data/audio_full/{r["split"]}/{r.filename}',duration=float(r.duration_s)) for r in selected]
    class Fixture(runtime.Extractor):
        def discover(self):
            self.root=Path('/workspace/shl/data/audio_full')
            return jobs
    with TemporaryDirectory() as temporary:
        config=dict(OUTPUT=str(Path(temporary)/'fixture.h5'),CPU_WORKERS=2,CPU_THREADS=2,
                    GPU_BATCH_SIZE=8,TEXT_BATCH_SIZE=2)
        e=Fixture(config);e.transcribe()
        assert e.record(0)['verbatim'].asstr()[()]==selected[0].transcript
        assert e.record(0)['original_transcript'].asstr()[()]
        assert e.record(2)['verbatim'].asstr()[()]==''
        assert not e.record(2).attrs['transcript_available']
        assert all(e.done(i,'asr') for i in range(3))
        e.record(0).attrs['hubert_done']=True
        del e.h.attrs['data_signature']  # Original notebook compatibility.
        e.h.attrs['signature']='legacy code-specific fingerprint'
        e.h.close()
        updated=dict(config,GPU_BATCH_SIZE=16,CPU_WORKERS=4)
        e=Fixture(updated)
        assert e.done(0,'hubert')
        assert e.h['X'].shape==(3,9888)
        assert np.isnan(e.h['metadata/label'][1])
        e.h.close()
    print('PASS: OOM splitting preserves order; spawned workers import; all 985 transcript hashes match; original/Crisper reuse; missing text remains missing; legacy checkpoints survive performance-setting changes. No model inference run.')


if __name__=='__main__':main()
