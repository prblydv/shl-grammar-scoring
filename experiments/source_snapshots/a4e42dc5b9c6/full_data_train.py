"""Fresh full-data refit with a CV-selected fixed epoch budget, never holdout tuning."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

import pipeline
import runner
import sweep_utils as integrity
import telemetry
import train_worker as worker
import audio_features as audio
import spectrogram_features as spectrogram


def prepare(cv_run, epochs=None, allow_cpu_test=False):
    cv_run=Path(cv_run).resolve()
    config=json.loads((cv_run/'resolved_config.json').read_text())
    jobs=pipeline.jobs_for(config,cv_run)
    reports=[integrity.validate_completed(j,worker_path=worker.__file__) for j in jobs]
    metrics=json.loads((cv_run/'metrics.json').read_text())
    frame=pd.read_parquet(cv_run/'data.parquet')
    # Also validate the exported CSV and its actual ID order.
    template=pd.read_csv(cv_run/'input/submission_template.csv')
    submission=pd.read_csv(cv_run/'submission.csv')
    if (submission.filename.tolist()!=template.filename.tolist() or
            not np.isfinite(submission.label).all() or not submission.label.between(0,5).all()):
        raise ValueError('CV submission is invalid')
    if not allow_cpu_test and (frame.split.eq('train').sum()!=769 or frame.split.eq('test').sum()!=216):
        raise ValueError('Expected 769 training and 216 test rows')
    best_epochs=[r['best_epoch'] for r in reports]
    chosen=int(math.ceil(float(np.median(best_epochs)))) if epochs is None else epochs
    if type(chosen) is not int or not 1<=chosen<=config['epochs']:
        raise ValueError('Full-data epochs must be an integer within the CV epoch cap')
    plan={'policy':'ceil(median(best_epoch across completed CV folds))' if epochs is None else 'explicit user override',
          'cv_best_epochs':best_epochs,'epochs':chosen,'seed':config['seed'],
          'initialization':'fresh pinned pretrained backbone; fresh scoring head; no fold checkpoint weights',
          'early_stopping':False,'validation_rows':0,'training_rows':int(frame.split.eq('train').sum()),
          'test_rows':int(frame.split.eq('test').sum()),'final_model_validation_rmse':None,
          'cv_metrics':metrics,'cv_folds':[{'fold':r['fold']+1,'best_epoch':r['best_epoch'],
                 'best_validation_rmse':r['best_rmse'],'validation_metric':'RMSE'} for r in reports]}
    config=dict(config,epochs=chosen)
    inputs={name:integrity.file_sha256(cv_run/name) for name in
            ['data.parquet','folds.csv','metrics.json','resolved_config.json','input/submission_template.csv']}
    implementation=integrity.implementation_fingerprint(worker.__file__)
    implementation['full_data_train.py']=integrity.file_sha256(Path(__file__))
    payload={'config':config,'plan':plan,'inputs':inputs,'model':integrity.model_fingerprint(cv_run/'backbone'),
             'tokenizer':{str(p.relative_to(cv_run/'tokenizer')):integrity.file_sha256(p)
                          for p in (cv_run/'tokenizer').rglob('*') if p.is_file()},
             'implementation':implementation,'torch':torch.__version__,'transformers':transformers.__version__}
    signature=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return {'cv_run':str(cv_run),'config':config,'plan':plan,'signature':signature,'provenance':payload}


def train(job, output, allow_cpu_test=False, pause_after_epoch=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cv_run=Path(job['cv_run']); config=job['config']; signature=job['signature']
    # Fail closed if preparation is stale, even when accepting a completed cache.
    override=None if job['plan']['policy'].startswith('ceil(') else config['epochs']
    if prepare(cv_run,override,allow_cpu_test)['signature']!=signature:
        raise ValueError('Full-data input, configuration, CV results or implementation changed')
    if (output/'job.json').exists():
        previous=json.loads((output/'job.json').read_text())
        if previous['signature']!=signature: raise ValueError('Use a new output directory for a changed refit')
    telemetry.atomic_json(output/'job.json',job)
    complete=output/'complete.json'
    if complete.exists():
        report=json.loads(complete.read_text())
        if report['signature']!=signature: raise ValueError('Completed full-data run does not match this plan')
        for name,digest in report['artifact_sha256'].items():
            if integrity.file_sha256(output/name)!=digest: raise ValueError('Changed full-data artifact: '+name)
        print('Verified completed full-data model and submission; no retraining.',flush=True)
        return report
    device=torch.device('cpu' if allow_cpu_test else 'cuda:0')
    if device.type=='cuda' and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('This refit requires the existing CUDA/BF16 environment')
    torch.set_num_threads(config['cpu_threads']); worker.set_seed(job['plan']['seed'])
    telemetry.context(fold='full_data',epochs=config['epochs']);telemetry.emit('full_data_preparation')
    frame=pd.read_parquet(cv_run/'data.parquet')
    training=frame[frame.split.eq('train')].reset_index(drop=True)
    test=frame[frame.split.eq('test')].reset_index(drop=True)
    tokenizer=AutoTokenizer.from_pretrained(cv_run/'tokenizer',local_files_only=True,use_fast=False) if audio.uses_text(config) else None
    store=audio.FeatureStore(config) if audio.uses_audio(config) else None
    spec_store=spectrogram.Store(config) if spectrogram.uses(config) else None
    collator=spectrogram.MultiCollator(tokenizer) if spec_store else (audio.AudioCollator(tokenizer) if store else DataCollatorWithPadding(tokenizer,pad_to_multiple_of=8,return_tensors='pt'))
    training_dataset=spectrogram.MultiDataset(training,tokenizer,config['max_length'],store,spec_store) if spec_store else (audio.AudioDataset(training,tokenizer,config['max_length'],store.rows(training)) if store else worker.TextDataset(training,tokenizer,config['max_length']))
    test_dataset=spectrogram.MultiDataset(test,tokenizer,config['max_length'],store,spec_store,False) if spec_store else (audio.AudioDataset(test,tokenizer,config['max_length'],store.rows(test),False) if store else worker.TextDataset(test,tokenizer,config['max_length'],False))
    train_loader=DataLoader(training_dataset,batch_size=config['batch_size'],
        shuffle=True,num_workers=0,collate_fn=collator,pin_memory=device.type=='cuda')
    test_loader=DataLoader(test_dataset,batch_size=config['eval_batch_size'],
        shuffle=False,num_workers=0,collate_fn=collator,pin_memory=device.type=='cuda')
    telemetry.emit('load_fresh_pretrained')
    model=worker.GrammarModel(config,cv_run/'backbone',float(training.label.mean())).to(device)
    if store:
        model.fit_audio_normalization(store.rows(training))
    if spec_store:
        model.fit_spectrogram_normalization(spec_store,training)
    telemetry.atomic_json(output/'resolved_config.json',config)
    telemetry.atomic_json(output/'training_plan.json',job['plan'])
    telemetry.atomic_json(output/'provenance.json',job['provenance'])
    telemetry.atomic_json(output/'layers.json',model.layer_report)
    # Keep a usable model bundle on Vast independently of publication.
    (output/'backbone').mkdir(exist_ok=True)
    shutil.copy2(cv_run/'backbone/config.json',output/'backbone/config.json')
    shutil.copytree(cv_run/'tokenizer',output/'tokenizer',dirs_exist_ok=True)
    (output/'source').mkdir(exist_ok=True)
    for name in ['train_worker.py','sweep_utils.py','pipeline.py','telemetry.py','runner.py','batch_probe.py',
                 'full_data_train.py','publish_full_model.py','publish_kaggle.py','pooling.py','audio_features.py','spectrogram_features.py']:
        source=Path(__file__).with_name(name)
        if source.exists(): shutil.copy2(source,output/'source'/name)
    print(json.dumps({'plan':job['plan'],'resolved_config':config,'parameters':model.layer_report},indent=2),flush=True)
    optimizer=torch.optim.AdamW(worker.parameter_groups(model,config),eps=config['adam_epsilon'],betas=tuple(config['adam_betas']))
    updates=math.ceil(len(train_loader)/config['gradient_accumulation'])
    scheduler=get_linear_schedule_with_warmup(optimizer,int(updates*config['epochs']*config['warmup_ratio']),updates*config['epochs'])
    amp_enabled=device.type=='cuda'; dtype=torch.bfloat16
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    history=[];first_epoch=0
    if (output/'last.pt').exists():
        saved=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if saved['signature']!=signature: raise ValueError('Resume configuration/input/implementation mismatch')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];first_epoch=saved['next_epoch']
        random.setstate(saved['random_state']);np.random.set_state(saved['numpy_state'])
        torch.set_rng_state(saved['torch_state'].cpu())
        if amp_enabled: torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_states']])
        del saved
        print(f'Resuming full-data epoch {first_epoch+1}',flush=True)
    for epoch in range(first_epoch,config['epochs']):
        model.train();optimizer.zero_grad(set_to_none=True);loss_sum=0.;samples=0;norm=0.
        started=time.monotonic();last_log=0.
        telemetry.context(epoch=epoch+1)
        for step,batch in enumerate(train_loader):
            inputs,labels=worker.move_batch(batch,device)
            group_start=(step//config['gradient_accumulation'])*config['gradient_accumulation']
            group_stop=min(group_start+config['gradient_accumulation'],len(train_loader))
            group_samples=min(group_stop*config['batch_size'],len(training))-group_start*config['batch_size']
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp_enabled):
                predictions,logits=model(inputs);loss=worker.objective(predictions,logits,labels,config)
            if not torch.isfinite(loss): raise ValueError('Nonfinite full-data loss')
            (loss*len(labels)/group_samples).backward()
            loss_sum+=float(loss.detach())*len(labels);samples+=len(labels)
            if step+1==group_stop:
                norm=float(nn.utils.clip_grad_norm_(model.parameters(),config['max_grad_norm'],error_if_nonfinite=True))
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True)
                update=epoch*updates+step//config['gradient_accumulation']+1
                telemetry.scalar('full_data/gradient_norm',norm,update)
                for i,g in enumerate(optimizer.param_groups): telemetry.scalar(f'full_data/lr_{i}',g['lr'],update)
            elapsed=time.monotonic()-started;eta=elapsed/(step+1)*(len(train_loader)-step-1)
            telemetry.emit('full_data_training',batch=step+1,batches=len(train_loader),train_mse=loss_sum/samples,
                gradient_norm=norm,learning_rates=[g['lr'] for g in optimizer.param_groups],epoch_eta_seconds=eta)
            if time.monotonic()-last_log>=15 or step+1==len(train_loader):
                print(f'FULL DATA epoch={epoch+1}/{config["epochs"]} batch={step+1}/{len(train_loader)} '
                      f'train_MSE={loss_sum/samples:.6f} gradient_norm={norm:.4f} '
                      f'LR={[g["lr"] for g in optimizer.param_groups]} epoch_ETA={eta:.1f}s',flush=True)
                last_log=time.monotonic()
        history.append({'epoch':epoch+1,'training_mse':loss_sum/samples,'training_rows':samples})
        telemetry.scalar('full_data/train_mse',loss_sum/samples,epoch+1)
        worker.atomic_json(history,output/'history.json')
        worker.atomic_save({'model':worker.trainable_state(model),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'signature':signature,'next_epoch':epoch+1,
            'history':history,'random_state':random.getstate(),'numpy_state':np.random.get_state(),
            'torch_state':torch.get_rng_state(),'cuda_states':torch.cuda.get_rng_state_all() if amp_enabled else []},output/'last.pt')
        if pause_after_epoch is not None and epoch+1>=pause_after_epoch: return {'paused':True}
    # Fixed final epoch; never select an epoch by training loss or use test labels.
    worker.atomic_save({'model':worker.trainable_state(model),'config':config,'signature':signature,
                        'epoch':config['epochs'],'training_rows':len(training),'validation_rmse':None},output/'final.pt')
    predictions=worker.predict(model,test_loader,device,dtype,amp_enabled,stage='full_data_test_inference')
    predicted=test[['filename']].assign(label=predictions)
    template=pd.read_csv(cv_run/'input/submission_template.csv')
    submission=template[['filename']].merge(predicted,on='filename',how='left',validate='one_to_one')
    if (len(submission)!=len(test) or set(submission.filename)!=set(test.filename) or
            submission.filename.tolist()!=template.filename.tolist() or
            not np.isfinite(submission.label).all() or not submission.label.between(0,5).all()):
        raise ValueError('Invalid full-data submission alignment or scores')
    submission.to_csv(output/'submission_full_data.csv.tmp',index=False)
    (output/'submission_full_data.csv.tmp').replace(output/'submission_full_data.csv')
    predicted.rename(columns={'label':'prediction'}).to_csv(output/'test_predictions.csv',index=False)
    retained=['final.pt','test_predictions.csv','submission_full_data.csv','resolved_config.json',
              'training_plan.json','provenance.json','layers.json','job.json','history.json']
    retained += [str(p.relative_to(output)) for folder in ['backbone','tokenizer','source']
                 for p in (output/folder).rglob('*') if p.is_file()]
    report={'signature':signature,'training_rows':len(training),'epochs':config['epochs'],'seed':job['plan']['seed'],
            'validation_rmse':None,'cv_metrics':job['plan']['cv_metrics'],'history':history,
            'artifact_sha256':{name:integrity.file_sha256(output/name) for name in
                              retained}}
    telemetry.atomic_json(complete,report);telemetry.emit('full_data_submission_ready',submission_rows=len(submission))
    (output/'last.pt').unlink(missing_ok=True)
    telemetry.close()
    print('Full-data model and submission ready:',output/'submission_full_data.csv',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cv-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int);p.add_argument('--worker',action='store_true');p.add_argument('--plan-only',action='store_true')
    args=p.parse_args()
    if args.cv_run.resolve()==args.output.resolve(): raise ValueError('Full-data output must be separate from CV run')
    if args.worker:
        # The parent owns the full-data lock. Hold the CV lock against concurrent modifications.
        with telemetry.run_lock(args.cv_run): train(prepare(args.cv_run,args.epochs),args.output)
    elif args.plan_only:
        with telemetry.run_lock(args.cv_run): print(json.dumps(prepare(args.cv_run,args.epochs)['plan'],indent=2))
    else:
        command=[sys.executable,'-u',str(Path(__file__).resolve()),'--cv-run',str(args.cv_run.resolve()),
                 '--output',str(args.output.resolve()),'--worker']
        if args.epochs is not None: command+=['--epochs',str(args.epochs)]
        sys.exit(runner.monitor(command,args.output,interval=json.loads((args.cv_run/'resolved_config.json').read_text()).get('heartbeat_seconds',15)))


if __name__=='__main__': main()
