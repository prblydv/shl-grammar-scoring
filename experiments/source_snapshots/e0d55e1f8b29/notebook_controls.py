"""Notebook setup and delivery; all model work runs in isolated subprocesses."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys


def setup(root,name,settings,use_existing=False):
    root=Path(root).resolve();base=root/'experiments/vast_text';run=root/'runs'/name
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name): raise ValueError('Use a simple unique experiment name')
    if use_existing:
        if not (run/'resolved_config.json').exists(): raise ValueError('Existing experiment has no resolved configuration')
        requested=root/'configs'/f'{name}.json'
        if requested.exists():
            config=json.loads(requested.read_text())
            code=base/('configurable_v3' if 'num_backbones' in config else 'configurable_v2' if 'transcript_dataset' in config else 'configurable_v1')
        else:
            config=json.loads((base/'first_run.json').read_text());requested=base/'first_run.json';code=base
            if config['run_id']!=json.loads((run/'resolved_config.json').read_text())['run_id']:
                raise ValueError('Legacy run configuration is not this first-run config')
        return code,run,requested,config
    config=dict(settings,run_id=name,design_confirmed=True)
    if 'num_backbones' in config:
        spec=importlib.util.spec_from_file_location('shl_audio_settings',base/'configurable_v3/audio_features.py')
        audio=importlib.util.module_from_spec(spec);spec.loader.exec_module(audio)
        audio.validate(config)
    dataset=config.get('transcript_dataset','whisper')
    config['transcript_dataset']=dataset
    if dataset not in {'whisper','crisper'}:
        raise ValueError('TRANSCRIPT_DATASET must be whisper or crisper')
    if config['pooling'] not in ['mean','cls','attention','concat','weighted_layer','gem','lstm']:
        raise ValueError('Choose a supported pooling')
    if config.get('preprocessing','unchanged')!='unchanged': raise ValueError('Only unchanged transcripts are implemented')
    if config['loss']!='mse' or config['precision']!='bf16': raise ValueError('MSE loss and BF16 are supported; validation uses RMSE')
    for key in ['n_folds','epochs','patience','effective_batch_size','cpu_threads']:
        if type(config[key]) is not int or config[key]<1: raise ValueError('Invalid '+key)
    for key in ['batch_size','eval_batch_size','gradient_accumulation','max_length']:
        if config.get(key) is not None and (type(config[key]) is not int or config[key]<1): raise ValueError('Invalid '+key)
    if not 0<config['heartbeat_seconds']<=60: raise ValueError('Heartbeat must be in (0,60] seconds')
    if config['n_folds']<2: raise ValueError('At least two folds required')
    for key in ['freeze_top_n_layers','re_init_top_n_layers']:
        if type(config[key]) is not int or config[key]<0: raise ValueError('Invalid '+key)
    if config['freeze_top_n_layers'] and config['re_init_top_n_layers']:
        raise ValueError('Top frozen and reinitialised layers overlap; use separate trials')
    if config['model_id']!='microsoft/deberta-v3-large' and config.get('revision')=='64a8c8eab3e352a784c658aef62be1662607476f':
        raise ValueError('Set MODEL_REVISION=None or the matching revision when changing MODEL_NAME')
    requested=root/'configs'/f'{name}.json'
    if requested.exists() and json.loads(requested.read_text())!=config:
        raise ValueError('Settings changed: choose a new EXPERIMENT_NAME')
    if run.exists() and not requested.exists(): raise ValueError('Existing run directory: use a new name or USE_EXISTING_RUN=True')
    shared=root/'data'/('text_input_crisper' if dataset=='crisper' else 'text_input')
    if (run/'input').exists() and (run/'input').resolve()!=shared.resolve():
        raise ValueError('Transcript dataset changed: choose a new EXPERIMENT_NAME')
    if not (run/'input').exists():
        # Input data outlives any individual experiment and is shared without copies.
        if not shared.exists():
            if dataset=='crisper':
                raise FileNotFoundError('Upload the Crisper text bundle to '+str(shared))
            candidates=[root/'runs/first_model/input',root/'artifacts/vast_first_model/input']
            candidates.extend(sorted((root/'runs').glob('*/input')))
            source=next((p for p in candidates if (p/'manifest.json').is_file()),None)
            if source is None:
                raise FileNotFoundError('Restore the text bundle to '+str(shared)+'; no training input bundle remains')
            shared.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,shared)
        run.mkdir(parents=True,exist_ok=True)
        (run/'input').symlink_to(shared,target_is_directory=True)
    requested.parent.mkdir(parents=True,exist_ok=True)
    requested.write_text(json.dumps(config,indent=2)+'\n')
    return base/('configurable_v3' if 'num_backbones' in config else 'configurable_v2'),run,requested,config


def verify_cv(code,run):
    sys.path.insert(0,str(code))
    import pipeline,train_worker,sweep_utils
    config=json.loads((run/'resolved_config.json').read_text())
    jobs=pipeline.jobs_for(config,run)
    reports=[sweep_utils.validate_completed(job,worker_path=train_worker.__file__) for job in jobs]
    import pandas as pd
    import numpy as np
    sample=pd.read_csv(run/'input/submission_template.csv');submission=pd.read_csv(run/'submission.csv')
    if (list(submission.columns)!=['filename','label'] or len(submission)!=216 or
        submission.filename.tolist()!=sample.filename.tolist() or not submission.filename.is_unique or
        not np.isfinite(submission.label).all() or not submission.label.between(0,5).all()):
        raise ValueError('Invalid CV submission')
    return config,jobs,reports


def best_fold_submission(code,run):
    config,jobs,reports=verify_cv(code,run)
    best=min(zip(jobs,reports),key=lambda pair:(pair[1]['best_rmse'],pair[0]['fold']))
    job,report=best
    import pandas as pd,numpy as np
    prediction=pd.read_csv(Path(job['job_dir'])/'test_predictions.csv')
    sample=pd.read_csv(run/'input/submission_template.csv')
    submission=sample[['filename']].merge(prediction,on='filename',how='left',validate='one_to_one').rename(columns={'prediction':'label'})
    if len(submission)!=216 or not np.isfinite(submission.label).all() or not submission.label.between(0,5).all():
        raise ValueError('Invalid best-fold submission')
    directory=run/'delivery/best_fold';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'submission_best_fold.csv';submission.to_csv(path,index=False)
    summary={'display_fold':job['fold']+1,'best_epoch':report['best_epoch'],'validation_rmse':report['best_rmse'],
        'local_model':str(Path(job['job_dir'])/'best.pt'),'submission':str(path)}
    (directory/'selection.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--choice',choices=['cv_submission','best_fold'],required=True)
    parser.add_argument('--publish',action='store_true');parser.add_argument('--destination',type=Path)
    parser.add_argument('--dataset-id');parser.add_argument('--update',action='store_true')
    args=parser.parse_args();args.run=args.run.resolve()
    sys.path.insert(0,str(args.code))
    import telemetry
    with telemetry.run_lock(args.run):
        if args.choice=='cv_submission':
            if args.publish: raise ValueError('Choose best_fold or full_data to publish one model')
            verify_cv(args.code,args.run);print('Verified CV submission:',args.run/'submission.csv');return
        summary=best_fold_submission(args.code,args.run);print(json.dumps(summary,indent=2),flush=True)
        if args.publish:
            if not args.destination or not args.dataset_id: raise ValueError('Missing publication destination/id')
            # Use dynamic RMSE documentation, retaining the actual run's scoring source.
            publisher_path=(Path(args.code)/'publish_kaggle.py' if Path(args.code).name=='configurable_v3'
                            else Path(__file__).parent/'configurable_v1/publish_kaggle.py')
            spec=importlib.util.spec_from_file_location('selected_fold_publisher',publisher_path)
            publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
            publisher.document(args.run,args.destination,args.dataset_id)
            shutil.copy2(summary['submission'],args.destination/'submission_best_fold.csv')
            import sweep_utils
            telemetry.atomic_json(args.destination/'upload_checksums.json',{'files':{str(p.relative_to(args.destination)):sweep_utils.file_sha256(p)
                for p in args.destination.rglob('*') if p.is_file() and p.name!='upload_checksums.json'}})
        else: return
    auth=json.loads(sys.stdin.readline())
    import os
    os.environ['KAGGLE_USERNAME']=auth['username'];os.environ['KAGGLE_KEY']=auth['key'];del auth
    publisher.upload(args.destination,args.dataset_id,args.update)


if __name__=='__main__':main()
