"""Document and publish a private Kaggle checkpoint dataset directly from Vast."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

from sweep_utils import file_sha256
from telemetry import atomic_json, run_lock


def document(run, destination, dataset_id):
    run,destination=Path(run).resolve(),Path(destination).resolve()
    config=json.loads((run/'resolved_config.json').read_text())
    assignment=__import__('pandas').read_csv(run/'folds.csv')
    reports=[]
    for folder in sorted((run/'models').glob('*/*/fold_*')):
        fold=int(folder.name.split('_')[-1])
        history=json.loads((folder/'history.json').read_text())
        best=min(history,key=lambda x:x['validation_rmse'])
        complete=(folder/'complete.json').exists()
        reports.append({'fold_index':fold,'display_fold':fold+1,'model_seed':config['seed']+config.get('model_seed_stride',1009)*fold,
            'training_rows':int(len(assignment)-assignment.fold.eq(fold).sum()),
            'validation_rows':int(assignment.fold.eq(fold).sum()),'epochs_completed':len(history),
            'best_epoch':best['epoch'],'best_validation_rmse':best['validation_rmse'],
            'validation_metric':'RMSE',
            'status':'complete' if complete else 'paused','checkpoint':str((folder/'best.pt').relative_to(run))})
    if len(reports)!=config.get('n_folds',3): raise ValueError('Missing fold checkpoint sets')
    selected=min(reports,key=lambda r:(r['best_validation_rmse'],r['fold_index']))
    total=sum(r['validation_rows'] for r in reports)
    mean=sum(r['best_validation_rmse'] for r in reports)/len(reports)
    pooled=math.sqrt(sum(r['validation_rows']*r['best_validation_rmse']**2 for r in reports)/total)
    metrics=json.loads((run/'metrics.json').read_text()) if (run/'metrics.json').exists() else None
    state='complete' if all(r['status']=='complete' for r in reports) and (run/'submission.csv').exists() else 'paused'
    summary={'dataset_id':dataset_id,'private':True,'training_status':state,'folds':reports,
             'selection_metric':'lowest validation RMSE across all folds',
             'selected_model':selected,'uploaded_checkpoint':'best.pt','model_count':1,
             'mean_best_fold_rmse':mean,
             'aggregate_best_validation_rmse':pooled,
             'aggregate_definition':'sqrt(sum(validation_rows * best_fold_RMSE^2) / total_validation_rows)',
             'pooled_oof_rmse':metrics['cv_rmse'] if metrics else None,
             'oof_export_status':'complete' if metrics else 'pending final fold inference after manual resume',
             'public_leaderboard_score':None,'configuration':config}
    rows=['| Fold | Status | Train / validation | Epochs trained | Best epoch | Best validation RMSE | Seed |',
          '|---|---|---|---|---|---|---|']
    rows.extend(f"| {r['display_fold']} | {r['status']} | {r['training_rows']} / {r['validation_rows']} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_validation_rmse']:.8f} | {r['model_seed']} |" for r in reports)
    hyper=['| Hyperparameter | Value |','|---|---|']
    hyper.extend(f'| `{k}` | `{json.dumps(v)}` |' for k,v in sorted(config.items()))
    description=f'''# {config['model_id']}: single lowest-RMSE fold model

Training status: **{state}**. Private model dataset uploaded directly from Vast.ai.

Exactly one model: fold **{selected["display_fold"]}**, epoch **{selected["best_epoch"]}**,
validation RMSE **{selected["best_validation_rmse"]:.8f}**.
Selection uses the lowest saved validation RMSE across folds; ties use the lowest fold index.
The submission averages all configured folds; this published model contains only the selected fold.

## Validation results

{chr(10).join(rows)}

Unweighted mean of best fold RMSE: **{mean:.8f}**.
Aggregate best validation RMSE, weighted by validation rows before taking the square root: **{pooled:.8f}**.
Pooled row-aligned OOF RMSE: **{summary['pooled_oof_rmse'] if metrics else 'pending completion of final fold inference'}**.
The aggregate above is computed from saved epoch validation metrics. When paused,
it describes the retained best checkpoints so far; it does not claim the full
pipeline or submission export is complete. Public leaderboard metric/score are
separate and unverified. Original labels, including zero, are unchanged.

## Architecture and data

`{config['model_id']}` revision `{config['model_revision']}`. Freeze top layers: `{config.get('freeze_top_n_layers',0)}`; reinitialise top layers: `{config.get('re_init_top_n_layers',0)}`. Pooling: `{config['pooling']}` with all variant settings in the configuration. LayerNorm, scalar linear head and sigmoid times five.
Original saved Whisper Large-v3 transcripts: 769 training rows and 216 test rows.
Sequence length {config['max_length']} for train/validation/test, dynamic padding to multiples of eight.
{config.get('n_folds',3)} fixed source-grouped MultilabelStratifiedKFold splits use zero and positive
bins (0,2], (2,3], (3,4], (4,5]. The 31 verified noisy/clean source pairs remain
together. Other source/speaker overlap remains unverified. Hyperparameter and
checkpoint selection use validation, so these scores are not independent holdout results.

## Full hyperparameters

{chr(10).join(hyper)}

Fold initialization seed is {config['seed']} + {config.get('model_seed_stride',1009)} times the zero-based fold index. AdamW excludes
biases and one-dimensional parameters from weight decay. LR warmup ratio {config['warmup_ratio']} followed by linear decay; layer-wise LR decay {config['layer_lr_decay']}. Early stopping patience {config['patience']} and strict
validation-RMSE improvement. Gradient clipping follows accumulated gradients;
BF16 uses no enabled loss scaler. Train microbatch {config['batch_size']}, accumulation {config['gradient_accumulation']}, effective batch {config['effective_batch_size']}; evaluation batch {config['eval_batch_size']}. These sizes passed two optimizer updates
at worst-case length, finite-gradient and padding/inference checks below 90% VRAM.
The inspected GPU was an RTX4090 with 24GB VRAM; exact software versions and measured
peaks are saved in environment.json, installed-requirements.txt and batch_measurements.json.

## Files and recovery

`best.pt` is the single selected full scoring-model state: backbone, LayerNorm and head.
No other fold weights, optimizer/resume states or original pretrained weights are uploaded.
`backbone/config.json`, `tokenizer/`, `source/`, complete hyperparameters, all-fold
histories/metrics, available predictions and logs support later inference and ensembles.
`upload_checksums.json` covers the actual published payload. Original training data
and transcripts remain on Vast. The full three-fold run and resume checkpoints remain
on the existing Vast instance; this single-model dataset is not an exact training-resume backup.

Load the scoring model from this dataset (after extracting source/tokenizer archives):

```python
import sys, json, torch
from pathlib import Path
root = Path("/path/to/dataset")
sys.path.insert(0, str(root / "source"))
from train_worker import GrammarModel
config = json.loads((root / "resolved_config.json").read_text())
model = GrammarModel(config, root / "backbone", random_backbone=True)
checkpoint = torch.load(root / "best.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

Use the included tokenizer, truncation at {config['max_length']} and padding rounded to eight. The
model returns `(predictions, logits)`; predictions are bounded from zero to five. Retain filename alignment when
combining its predictions with other ensemble members.

## Run from a terminal: no notebook

On the existing Vast instance:

```bash
cd /workspace/shl
/venv/main/bin/python -u experiments/vast_text/runner.py --output /workspace/shl/runs/first_model --config /workspace/shl/experiments/vast_text/first_run.json
```

This prints timestamped logs and independent 15-second heartbeats in the terminal.
It resumes the saved epoch, verifies completed folds, finishes inference and exports
`runs/first_model/submission.csv` once all configured folds complete. Keep the same
implementation and environment for exact resume. Do not launch a second writer.
For a job that survives SSH disconnects, use `supervisorctl start shl_training`,
then `tail -f /workspace/shl/runs/first_model/train.log`. Supervisor automatic restart
is disabled. To inspect state: `/venv/main/bin/python experiments/vast_text/pipeline.py status --output /workspace/shl/runs/first_model`.

The final submission uses actual 216 test IDs/order with filename,label columns:
the provided official sample has 204 different IDs, as documented in the project.
This paused version may have no final submission yet. Publish a new private dataset
version after resuming, retaining this earlier version for checkpoint recovery.

Original competition data terms and pretrained-model/package licenses apply.
'''
    destination.mkdir(parents=True,exist_ok=False)
    # Publish one full checkpoint only; never copy optimizer/resume or base weights.
    selected_path=run/selected['checkpoint']
    os.link(selected_path,destination/'best.pt')
    for source in run.rglob('*'):
        relative=source.relative_to(run)
        if not source.is_file() or '.cache' in relative.parts: continue
        if relative.parts[0]=='input': continue
        if relative.parts[0]=='backbone' and source.name!='config.json': continue
        if relative.parts[0]=='models' and source.suffix not in {'.json','.csv'}: continue
        if source.name in {'.run.lock','backup_manifest.json','data.parquet'}: continue
        target=destination/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for name in ['publish_kaggle.py','train_and_submit.ipynb']:
        helper=Path(__file__).with_name(name)
        if helper.exists():
            (destination/'source').mkdir(exist_ok=True)
            shutil.copy2(helper,destination/'source'/name)
    if config.get('num_backbones') == 2 or config.get('single_backbone_modality') == 'audio':
        description += '''\n## Selected audio branch\n\nThis checkpoint includes the configured audio-feature backbone and its training-fold\nimputation/scaling buffers. Supply `inputs['audio_features']` as a float32 tensor\nwith shape `(batch, audio_input_dim)`, concatenating the selected `audio_encodings`\nin their saved order. Use NaN for unavailable entries; the model applies its saved\ntransforms internally. Audio-only requires no tokenizer or text inputs. With two\nbackbones, include tokenizer outputs in the same input dictionary. Selected\nchannel dimensions, modalities, fusion, and backbone settings are recorded in\n`resolved_config.json`. The extractor's pretrained audio weights are not required\nfor scoring already extracted vectors.\n'''
    (destination/'README.md').write_text(description)
    atomic_json(destination/'training_summary.json',summary)
    import pandas as pd
    pd.DataFrame(reports).to_csv(destination/'fold_metrics.csv',index=False)
    pd.DataFrame([{'name':k,'value':json.dumps(v)} for k,v in sorted(config.items())]).to_csv(destination/'hyperparameters.csv',index=False)
    metadata={'id':dataset_id,'title':('SHL '+config['run_id']+' Best RMSE Fold Model')[:100],
              'licenses':[{'name':'other'}],'description':description}
    atomic_json(destination/'dataset-metadata.json',metadata)
    atomic_json(destination/'upload_checksums.json',{'files':{str(p.relative_to(destination)):file_sha256(p)
        for p in destination.rglob('*') if p.is_file() and p.name!='upload_checksums.json'}})
    return summary


def upload(destination,dataset_id,update):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi()
    api.authenticate()
    print(f'Uploading PRIVATE dataset {dataset_id} directly from Vast',flush=True)
    if update:
        response=api.dataset_create_version(str(destination),version_notes='Updated checkpoint and validation documentation after manual resume',
            quiet=False,convert_to_csv=False,delete_old_versions=False,dir_mode='zip')
    else:
        response=api.dataset_create_new(str(destination),public=False,quiet=False,convert_to_csv=False,dir_mode='zip')
    print('Upload request completed',flush=True)
    receipt={'dataset_id':dataset_id,'url':'https://www.kaggle.com/datasets/'+dataset_id,'requested_private':True,
             'response_error':str(getattr(response,'error','') or ''),'created_at':time.time()}
    if receipt['response_error']: raise RuntimeError(receipt['response_error'])
    for attempt in range(120):
        status=api.dataset_status(dataset_id)
        print(f'Kaggle processing status: {status}',flush=True)
        if str(status).lower()=='ready': break
        if str(status).lower() in ['error','failed']: raise RuntimeError(f'Kaggle processing failed: {status}')
        time.sleep(15)
    else: raise RuntimeError('Kaggle dataset is still processing; verify before relying on this backup')
    receipt['status']='ready'
    matches=[d for d in api.dataset_list(mine=True,search=dataset_id.split('/')[1])
             if getattr(d,'ref',None)==dataset_id]
    if not matches:
        raise RuntimeError('Dataset is ready but cannot yet verify its owner/privacy in the account listing')
    private=getattr(matches[0],'isPrivate',getattr(matches[0],'is_private',None))
    if private is not True:
        raise RuntimeError(f'Private dataset verification did not return true: {private}')
    receipt['verified_private']=True
    api.dataset_metadata(dataset_id,str(destination/'remote_metadata'))
    listed=api.dataset_list_files(dataset_id,page_size=200)
    receipt['listed_files']=[{'name':getattr(f,'name',None),'bytes':getattr(f,'total_bytes',getattr(f,'size',None))} for f in listed.files]
    atomic_json(destination.parent/'kaggle_upload_receipt.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--dataset-id',required=True)
    parser.add_argument('--auth-stdin',action='store_true')
    parser.add_argument('--update',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.auth_stdin:
        auth=json.loads(sys.stdin.readline())
        os.environ['KAGGLE_USERNAME']=auth['username']
        os.environ['KAGGLE_KEY']=auth['key']
        del auth
    with run_lock(args.run):
        summary=document(args.run,args.destination,args.dataset_id)
    print(json.dumps({k:v for k,v in summary.items() if k!='configuration'},indent=2),flush=True)
    if not args.prepare_only: upload(args.destination,args.dataset_id,args.update)


if __name__=='__main__': main()
