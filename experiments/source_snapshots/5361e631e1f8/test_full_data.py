"""Tiny CPU checks only: fixed-epoch full-data resume, alignment, single-weight packaging."""
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, DebertaV2Config, DebertaV2Model

import full_data_train as full
import pipeline
import publish_full_model
import sweep_utils as integrity
import train_worker as worker


def main():
    with tempfile.TemporaryDirectory() as temporary:
        cv=Path(temporary)/'cv';model_dir=cv/'backbone';model_dir.mkdir(parents=True)
        config=json.loads(Path(__file__).with_name('first_run.json').read_text())
        config.update(max_length=16,batch_size=3,eval_batch_size=3,gradient_accumulation=2,
                      epochs=2,cpu_threads=1,model_revision='tiny-test',max_runtime_hours=None,
                      n_folds=4,pooling='weighted_layer',weighted_layer_start=0,freeze_top_n_layers=1)
        vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','this','is','a','test','.']
        (model_dir/'vocab.txt').write_text('\n'.join(vocab)+'\n')
        tokenizer=BertTokenizerFast(vocab_file=str(model_dir/'vocab.txt'))
        tokenizer.save_pretrained(model_dir);tokenizer.save_pretrained(cv/'tokenizer')
        worker.set_seed(42)
        DebertaV2Model(DebertaV2Config(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=2,
            num_attention_heads=2,intermediate_size=32,max_position_embeddings=32,relative_attention=True,
            pos_att_type=['p2c','c2p'])).save_pretrained(model_dir)
        inputs=cv/'input';inputs.mkdir()
        training=pd.DataFrame([{'filename':f'train_{i}','label':float(i%6),'text':'this is a test .'} for i in range(18)])
        test=pd.DataFrame([{'filename':f'test_{i}','label':-1.,'text':'this is a test .'} for i in range(3)])
        training.to_csv(inputs/'train.csv',index=False);test.to_csv(inputs/'test.csv',index=False)
        pd.DataFrame(columns=['zero_filename','positive_filename']).to_csv(inputs/'source_pairs.csv',index=False)
        test[['filename','label']].iloc[::-1].to_csv(inputs/'submission_template.csv',index=False)
        pipeline.write_json(inputs/'manifest.json',{'files':{p.name:integrity.file_sha256(p) for p in inputs.glob('*.csv')}})
        pipeline.write_json(cv/'model_provenance.json',{'model_id':'tiny','revision':'tiny-test'})
        pipeline.analyze(inputs,cv);pipeline.assign_folds(inputs,cv,config)
        pipeline.write_json(cv/'resolved_config.json',config)
        for job in pipeline.jobs_for(config,cv):
            worker.run_job(dict(job,allow_cpu_test=True,force_cpu=True))
        pipeline.export(config,cv)
        shutil.copytree(Path(__file__).parent,cv/'source',ignore=shutil.ignore_patterns('__pycache__'))
        job=full.prepare(cv,epochs=2,allow_cpu_test=True)
        automatic=full.prepare(cv,allow_cpu_test=True)
        assert automatic['plan']['epochs']==int(np.ceil(np.median(automatic['plan']['cv_best_epochs'])))
        assert job['plan']['training_rows']==18 and job['plan']['validation_rows']==0
        before=(cv/'submission.csv').read_bytes()
        out=Path(temporary)/'full'
        assert full.train(job,out,allow_cpu_test=True,pause_after_epoch=1)['paused']
        report=full.train(job,out,allow_cpu_test=True)
        assert report['epochs']==2 and report['validation_rmse'] is None
        assert all(row['training_rows']==18 for row in report['history'])
        reference=Path(temporary)/'reference'
        full.train(job,reference,allow_cpu_test=True)
        a=pd.read_csv(out/'submission_full_data.csv');b=pd.read_csv(reference/'submission_full_data.csv')
        assert a.filename.tolist()==test.filename.iloc[::-1].tolist()
        np.testing.assert_allclose(a.label,b.label,atol=1e-7)
        assert (cv/'submission.csv').read_bytes()==before
        # A changed refit plan cannot reuse the same run or checkpoint.
        try: full.train(full.prepare(cv,epochs=1,allow_cpu_test=True),out,allow_cpu_test=True)
        except ValueError: pass
        else: raise AssertionError('Changed refit config accepted')
        import publish_kaggle
        fold_stage=Path(temporary)/'fold-stage'
        fold_summary=publish_kaggle.document(cv,fold_stage,'test/best-fold')
        assert fold_summary['selected_model']['best_validation_rmse']==min(r['best_validation_rmse'] for r in fold_summary['folds'])
        assert len(fold_summary['folds'])==4
        assert [p.name for p in fold_stage.rglob('*.pt')]==['best.pt']
        stage=Path(temporary)/'stage'
        publish_full_model.document(out,stage,'test/private-model',allow_cpu_test=True)
        assert [p.relative_to(stage).as_posix() for p in stage.rglob('*.pt')]==['model.pt']
        assert not list(stage.rglob('*.safetensors'))
        assert json.loads((stage/'training_summary.json').read_text())['final_model_validation_rmse'] is None
        assert (stage/'submission_full_data.csv').read_bytes()==(out/'submission_full_data.csv').read_bytes()
        for name,digest in json.loads((stage/'upload_checksums.json').read_text())['files'].items():
            assert integrity.file_sha256(stage/name)==digest
        # Changed saved prediction must fail cached-completion validation.
        (out/'test_predictions.csv').write_text('tampered')
        try: full.train(job,out,allow_cpu_test=True)
        except ValueError: pass
        else: raise AssertionError('Changed artifact accepted')
    print('PASS: full-data fixed epochs, all labels per epoch, exact resume, alignment, independent CV files, changed-config/artifact rejection, single-model payload and checksums.')


if __name__=='__main__':main()
