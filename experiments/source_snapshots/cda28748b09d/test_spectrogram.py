"""Synthetic-only integration of timed spectra, modalities, CV/resume and delivery."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2Model

import pipeline
import spectrogram_features as spectrogram
import audio_features as audio
import train_worker as worker
import sweep_utils as integrity
import full_data_train as full
import publish_kaggle
import publish_full_model


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='shl-spectrogram-cv-fixture-') as temporary:
        root=Path(temporary);cv=root/'cv';backbone=cv/'backbone';backbone.mkdir(parents=True)
        config=json.loads(Path(__file__).with_name('first_run.json').read_text())
        config.update(num_backbones=3,backbone_pair='text_audio',single_backbone_modality='spectrogram',
            run_id='spectrogram_fixture',model_id='tiny',model_revision='fixture',max_length=16,
            batch_size=3,eval_batch_size=16,gradient_accumulation=2,epochs=2,patience=3,cpu_threads=1,
            n_folds=3,max_runtime_hours=None,freeze_top_n_layers=0,re_init_top_n_layers=0,
            spectrogram_backbone='cnn_tiny',spectrogram_chunk_frames=32,spectrogram_chunk_batch_size=3,
            spectrogram_output_dim=16,spectrogram_dropout=0.,spectrogram_lr=1e-3,
            spectrogram_use_time_encoding=True,spectrogram_revision=None,
            audio_backbone='gated_mlp',audio_hidden_dim=16,audio_num_layers=2,audio_output_dim=8,
            audio_dropout=0.,audio_lr=1e-3,audio_normalization='standard',audio_encodings=['fixture'],
            fusion='concat',fusion_hidden_dim=16,fusion_dropout=0.,gradient_checkpointing=True)
        vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','this','is','a','test','.']
        (backbone/'vocab.txt').write_text('\n'.join(vocab)+'\n')
        tokenizer=BertTokenizerFast(vocab_file=str(backbone/'vocab.txt'))
        tokenizer.save_pretrained(backbone);tokenizer.save_pretrained(cv/'tokenizer')
        worker.set_seed(42)
        DebertaV2Model(DebertaV2Config(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=2,
            num_attention_heads=2,intermediate_size=32,max_position_embeddings=32,
            relative_attention=True,pos_att_type=['p2c','c2p'])).save_pretrained(backbone)
        inputs=cv/'input';inputs.mkdir()
        training=pd.DataFrame([dict(filename=f'audio_{i}.wav',label=float(i%6),text='this is a test .') for i in range(18)])
        # Train/test intentionally share filenames: joins must include split.
        test=pd.DataFrame([dict(filename=f'audio_{i}.wav',label=-1.,text='this is a test .') for i in range(216)])
        training.to_csv(inputs/'train.csv',index=False);test.to_csv(inputs/'test.csv',index=False)
        pd.DataFrame(columns=['zero_filename','positive_filename']).to_csv(inputs/'source_pairs.csv',index=False)
        test[['filename','label']].iloc[::-1].to_csv(inputs/'submission_template.csv',index=False)
        pipeline.write_json(inputs/'manifest.json',dict(files={p.name:integrity.file_sha256(p) for p in inputs.glob('*.csv')}))
        pipeline.write_json(cv/'model_provenance.json',dict(model_id='tiny',revision='fixture'))
        pipeline.analyze(inputs,cv);pipeline.assign_folds(inputs,cv,config)
        frame=pd.read_parquet(cv/'data.parquet');rng=np.random.default_rng(42)
        feature_config=dict(sample_rate=16000,hop_samples=160,pe_dimensions=64,pe_base=10000.,pe_reference_step_s=.01,n_mels=128,db_floor=-100.)
        source=root/'fixture_spectrograms.h5'
        with h5py.File(source,'w') as f:
            f.attrs.update(complete=True,fingerprint='fixture',config_json=json.dumps(feature_config))
            idx=f.create_group('index');strings=h5py.string_dtype()
            for key in ['split','filename']: idx.create_dataset(key,data=frame[key].tolist(),dtype=strings)
            idx.create_dataset('label',data=frame.label.where(frame.split.eq('train'),np.nan))
            idx.create_dataset('completed',data=np.ones(len(frame),bool))
            idx.create_dataset('audio_sha256',data=['fixture-'+str(i) for i in range(len(frame))],dtype=strings)
            for i,row in enumerate(frame.itertuples()):
                n=37+i%5;g=f.create_group(f'records/{i:06d}')
                g.attrs.update(complete=True,audio_sha256='fixture-'+str(i))
                g.create_dataset('log_mel',data=rng.normal(-50+i*.1,5,(n,128)).astype(np.float32))
                g.create_dataset('frame_center_s',data=.0125+np.arange(n)*.01)
                g.create_dataset('valid_fraction',data=np.ones(n,np.float32))
        config['spectrogram_feature_file']=str(source)
        config=spectrogram.prepare(config,cv)
        vectors=cv/'audio_features.npz';np.savez_compressed(vectors,values=rng.normal(size=(len(frame),4)).astype(np.float32),keys=frame[['split','filename']].to_numpy(str))
        config.update(audio_data_path=str(vectors),audio_data_sha256=integrity.file_sha256(vectors),audio_input_dim=4)
        audio.validate(config);store=audio.FeatureStore(config);spec_store=spectrogram.Store(config)
        a=spec_store.item('train','audio_0.wav');b=spec_store.item('test','audio_0.wav')
        assert not torch.equal(a['spectrogram'],b['spectrogram'])
        assert int(a['spectrogram_lengths'].sum())==spec_store.lookup[('train','audio_0.wav')]['frames']
        batch=spectrogram.MultiCollator(tokenizer)([spectrogram.MultiDataset(frame.head(2),tokenizer,16,store,spec_store)[i] for i in range(2)])
        inputs_batch={k:v for k,v in batch.items() if k!='labels'}
        counts=[]
        for name in spectrogram.CNN_WIDTHS:
            candidate=dict(config,spectrogram_backbone=name)
            model=worker.GrammarModel(candidate,backbone).eval()
            model.fit_spectrogram_normalization(spec_store,frame.query('split == "train" and fold != 0'))
            expected=spec_store.normalization(frame.query('split == "train" and fold != 0'))
            np.testing.assert_allclose(model.spectrogram_backbone.mel_mean.numpy(),expected[0])
            output,_=model(inputs_batch);assert output.shape==(2,) and torch.isfinite(output).all()
            output.sum().backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.spectrogram_backbone.parameters())
            counts.append(model.layer_report['spectrogram_parameters'])
            # Record prediction must not depend on which other recording shares the batch.
            dataset=spectrogram.MultiDataset(frame.head(1),tokenizer,16,store,spec_store)
            solo=spectrogram.MultiCollator(tokenizer)([dataset[0]]);solo.pop('labels')
            torch.testing.assert_close(model(solo)[0],output[:1],rtol=1e-5,atol=1e-6)
            state=copy.deepcopy(model.state_dict());restored=worker.GrammarModel(candidate,backbone,random_backbone=True).eval()
            restored.load_state_dict(state);torch.testing.assert_close(restored(inputs_batch)[0],output)
        assert counts==sorted(counts) and len(set(counts))==4
        for n,pair,single in [(1,'text_audio','spectrogram'),(2,'audio_spectrogram','spectrogram'),(2,'text_spectrogram','spectrogram'),(3,'text_audio','spectrogram')]:
            candidate=dict(config,num_backbones=n,backbone_pair=pair,single_backbone_modality=single)
            audio.validate(candidate);model=worker.GrammarModel(candidate,backbone)
            assert torch.isfinite(model(inputs_batch)[0]).all()
        # Exercise actual AST implementation without downloading pretrained weights.
        ast_config=dict(hidden_size=24,num_hidden_layers=1,num_attention_heads=3,intermediate_size=48,max_length=32,
                        num_mel_bins=128,patch_size=16,frequency_stride=16,time_stride=16)
        for name in spectrogram.AST_REPOS:
            candidate=dict(config,spectrogram_backbone=name,spectrogram_model_config=ast_config)
            model=worker.GrammarModel(candidate,backbone,random_backbone=True)
            output,_=model(inputs_batch);output.sum().backward();assert torch.isfinite(output).all()
        pipeline.write_json(cv/'resolved_config.json',config)
        jobs=pipeline.jobs_for(config,cv)
        for job in jobs: job.update(allow_cpu_test=True,force_cpu=True)
        assert worker.run_job(dict(jobs[0],deadline_epoch_time=0))['paused']
        worker.run_job(jobs[0]);reference=dict(jobs[0],job_dir=str(root/'reference_fold'));worker.run_job(reference)
        np.testing.assert_allclose(pd.read_csv(Path(jobs[0]['job_dir'])/'validation_predictions.csv').prediction,
                                   pd.read_csv(root/'reference_fold/validation_predictions.csv').prediction,atol=1e-7)
        for job in jobs[1:]: worker.run_job(job)
        pipeline.export(config,cv)
        assert pd.read_csv(cv/'submission.csv').filename.tolist()==test.filename.iloc[::-1].tolist()
        # Metadata must reload best checkpoints' OOF arrays and keep the two-file download contract.
        sys.path.insert(0,str(Path(__file__).parent.parent))
        import run_metadata
        submission,metadata=run_metadata.export(Path(__file__).parent,cv,cv/'submission.csv')
        document=json.loads(metadata.read_text());assert len(document['oof_predictions'])==18
        assert 'spectrogram_features.py' in document['source_code']
        assert document['run_reports']['spectrogram_report.json']['sha256']==config['spectrogram_data_sha256']
        assert sorted(p.name for p in submission.parent.iterdir())==sorted([submission.name,metadata.name])
        # Resume, inference and optional model packaging after full-data refit.
        shutil.copytree(Path(__file__).parent,cv/'source',ignore=shutil.ignore_patterns('__pycache__'))
        job=full.prepare(cv,epochs=2,allow_cpu_test=True);out=root/'full'
        assert full.train(job,out,allow_cpu_test=True,pause_after_epoch=1)['paused']
        full.train(job,out,allow_cpu_test=True)
        full.train(job,root/'full_reference',allow_cpu_test=True)
        np.testing.assert_allclose(pd.read_csv(out/'submission_full_data.csv').label,
                                   pd.read_csv(root/'full_reference/submission_full_data.csv').label,atol=1e-7)
        full_submission,full_metadata=run_metadata.export(Path(__file__).parent,cv,out/'submission_full_data.csv',action='full_data',full_run=out)
        assert json.loads(full_metadata.read_text())['approaches']['submission']['action']=='full_data'
        publish_kaggle.document(cv,root/'fold_package','fixture/spectrogram')
        publish_full_model.document(out,root/'full_package','fixture/spectrogram-full',allow_cpu_test=True)
        assert (root/'fold_package/source/spectrogram_features.py').is_file()
        assert (root/'full_package/source/spectrogram_features.py').is_file()
        assert len(list((root/'fold_package').rglob('*.pt')))==1
        assert len(list((root/'full_package').rglob('*.pt')))==1
    print('PASS: four CNN capacities, AST implementations, all spectrogram modality combinations, split-safe full-frame chunks, train-only normalization, gradients, batch invariance, checkpoint reload, exact CV/full-data resume, aligned inference, best-epoch metadata/two files and single-model packaging. Synthetic data only.')


if __name__=='__main__': main()
