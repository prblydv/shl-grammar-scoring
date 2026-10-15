"""CPU integration using a tiny local backbone; never trains the competition model."""
import copy
import json
from pathlib import Path
import tempfile
import sys
import os

import telemetry
import runner

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2Model

import pipeline
import train_worker as worker
import sweep_utils as integrity


def main():
    torch.set_num_threads(1)
    config = pipeline.read_json(Path(__file__).with_name('first_run.json'))
    config.update(design_confirmed=True, output_mode='bounded', max_length=24, batch_size=3,
                  eval_batch_size=3, gradient_accumulation=2, max_runtime_hours=1,
                  epochs=2, cpu_threads=1, model_revision='tiny-test-model')
    with tempfile.TemporaryDirectory() as temporary:
        out = Path(temporary)
        model_dir = out / 'backbone'
        model_dir.mkdir()
        vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'i', 'go', 'went', 'to', 'school', '.', 'he', 'is', 'happy']
        (model_dir / 'vocab.txt').write_text('\n'.join(vocab)+'\n')
        tokenizer = BertTokenizerFast(vocab_file=str(model_dir / 'vocab.txt'))
        tokenizer.save_pretrained(model_dir)
        model = DebertaV2Model(DebertaV2Config(vocab_size=len(vocab), hidden_size=16,
            num_hidden_layers=2, num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=32, relative_attention=True, pos_att_type=['p2c','c2p'],
            hidden_dropout_prob=.2, attention_probs_dropout_prob=.2))
        model.save_pretrained(model_dir)
        data = out / 'input'
        data.mkdir()
        train = pd.DataFrame([{'filename': f'audio_{i}.wav', 'label': float(i % 6),
            'text': ['i go school .', 'i went to school .', 'he is happy .'][i % 3]} for i in range(18)])
        test = pd.DataFrame([{'filename': f'test_{i}.wav', 'label': -1., 'text': 'he is happy .'} for i in range(3)])
        train.to_csv(data / 'train.csv', index=False)
        test.to_csv(data / 'test.csv', index=False)
        pd.DataFrame({'zero_filename':['audio_0.wav','audio_6.wav'],
                      'positive_filename':['audio_5.wav','audio_11.wav']}).to_csv(data / 'source_pairs.csv', index=False)
        test[['filename','label']].iloc[::-1].to_csv(data / 'submission_template.csv', index=False)
        pipeline.write_json(data / 'manifest.json', {'files': {p.name: integrity.file_sha256(p) for p in data.glob('*.csv')}})
        pipeline.write_json(out / 'model_provenance.json', {'model_id': 'tiny', 'revision': 'tiny-test-model'})
        pipeline.analyze(data, out)
        report = pipeline.read_json(out / 'length_report.json')
        assert report['recommended_limit'] == 32 and report['native_covers_95_percent']
        unlimited=dict(config,max_runtime_hours=None)
        assert pipeline.resolved_config(unlimited,out)['max_runtime_hours'] is None
        with telemetry.run_lock(out):
            try:
                with telemetry.run_lock(out): pass
            except RuntimeError: pass
            else: raise AssertionError('Concurrent writer accepted')
        telemetry.atomic_json(out/'progress.json',{'stage':'training','updated_at':__import__('time').time()-301})
        assert runner.snapshot(out,os.getpid())['stalled_warning']
        monitored=out/'monitor-test'
        assert runner.monitor([sys.executable,'-c','import time; print("started",flush=True); time.sleep(.1)'],monitored,interval=.03)==0
        assert pipeline.read_json(monitored/'terminal_status.json')['process_state']=='complete'
        assert len((monitored/'heartbeats.jsonl').read_text().splitlines())>=2
        failed=out/'failure-test'
        assert runner.monitor([sys.executable,'-c','raise RuntimeError("intentional-test-failure")'],failed,interval=.03)!=0
        assert pipeline.read_json(failed/'terminal_status.json')['process_state']=='failed'
        assert (failed/'error.json').exists()
        assert 'intentional-test-failure' in (failed/'train.log').read_text()
        pipeline.assign_folds(data, out, config)
        before = (out / 'folds.csv').read_bytes()
        pipeline.assign_folds(data, out, config)
        assert (out / 'folds.csv').read_bytes() == before
        folds = pd.read_csv(out / 'folds.csv').set_index('filename')
        assert folds.loc['audio_0.wav', 'fold'] == folds.loc['audio_5.wav', 'fold']
        frame = pd.read_parquet(out / 'data.parquet')
        for frozen, reset in [(0, 0), (1, 0), (0, 1), (2, 0)]:
            scorer = worker.GrammarModel(dict(config, freeze_top_n_layers=frozen, re_init_top_n_layers=reset), model_dir)
            assert all(m.p == 0 for m in scorer.modules() if isinstance(m, torch.nn.Dropout))
            assert all(p.requires_grad for p in scorer.backbone.embeddings.parameters())
            for i, block in enumerate(scorer.backbone.encoder.layer):
                assert all(p.requires_grad == (i < 2-frozen) for p in block.parameters())
        for frozen, reset in [(-1, 0), (3, 0), (True, 0), (1, 1)]:
            try: worker.GrammarModel(dict(config, freeze_top_n_layers=frozen, re_init_top_n_layers=reset), model_dir)
            except ValueError: pass
            else: raise AssertionError('Invalid layer settings accepted')
        scorer = worker.GrammarModel(config, model_dir).eval()
        batch = tokenizer(['i go school .'], return_tensors='pt')
        padded = {k: torch.nn.functional.pad(v,(0,4)) for k,v in batch.items()}
        torch.testing.assert_close(scorer(batch)[0], scorer(padded)[0])
        smoke = worker.smoke_test(config, model_dir, frame.text, allow_cpu_test=True)
        assert smoke['passed'] and len(smoke['variants']) == 1
        jobs = pipeline.jobs_for(config, out)
        for job in jobs: job.update(allow_cpu_test=True, force_cpu=True)
        assert worker.run_job(dict(jobs[0], deadline_epoch_time=0))['paused']
        assert (Path(jobs[0]['job_dir']) / 'last.pt').exists()
        worker.run_job(jobs[0])
        reference = dict(jobs[0], job_dir=str(out / 'reference'))
        worker.run_job(reference)
        a = pd.read_csv(Path(jobs[0]['job_dir']) / 'validation_predictions.csv')
        b = pd.read_csv(Path(reference['job_dir']) / 'validation_predictions.csv')
        np.testing.assert_allclose(a.prediction,b.prediction,atol=1e-7)
        assert not (Path(jobs[0]['job_dir']) / 'last.pt').exists()
        for job in jobs[1:]: worker.run_job(job)
        pipeline.export(config, out)
        result = pd.read_csv(out / 'submission.csv')
        assert result.filename.tolist() == test.filename.iloc[::-1].tolist()
        assert result.label.between(0,5).all()
        assert len(pd.read_csv(out / 'oof_predictions.csv')) == len(train)
        # Fail closed for tampered completed predictions.
        bad = Path(jobs[0]['job_dir']) / 'test_predictions.csv'
        original = bad.read_bytes()
        bad.write_bytes(original+b'\n')
        try: pipeline.export(config, out)
        except ValueError: pass
        else: raise AssertionError('Tampered predictions accepted')
        bad.write_bytes(original)
        pipeline.backup(out, out.parent / (out.name+'-backup'))
        import shutil
        shutil.rmtree(out.parent / (out.name+'-backup'))
    print('PASS: unlimited runtime, writer lock, independent heartbeat, failure reporting, token reports, grouped reproducible folds, layer controls, disabled dropout, padding, optimizer smoke, exact epoch resume, aligned submission, artifact integrity and backup.')


if __name__ == '__main__': main()
