"""Publish only the full-data refit after its final submission has been generated."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import full_data_train
from publish_kaggle import upload
import sweep_utils as integrity
import telemetry


def document(run,destination,dataset_id,allow_cpu_test=False):
    run=Path(run).resolve();destination=Path(destination).resolve()
    report=json.loads((run/'complete.json').read_text())
    plan=json.loads((run/'training_plan.json').read_text())
    config=json.loads((run/'resolved_config.json').read_text())
    for name,digest in report['artifact_sha256'].items():
        if integrity.file_sha256(run/name)!=digest: raise ValueError('Changed completed full-data artifact: '+name)
    provenance=json.loads((run/'provenance.json').read_text())
    cv_run=Path(json.loads((run/'job.json').read_text())['cv_run'])
    override=None if plan['policy'].startswith('ceil(') else config['epochs']
    fresh=full_data_train.prepare(cv_run,override,allow_cpu_test=allow_cpu_test)
    if fresh['signature']!=report['signature']: raise ValueError('Full-data provenance no longer matches inputs/implementation')
    if fresh['provenance']!=provenance or fresh['config']!=config or fresh['plan']!=plan:
        raise ValueError('Changed saved provenance/configuration/plan')
    destination.mkdir(parents=True,exist_ok=False)
    for source in run.rglob('*'):
        rel=source.relative_to(run)
        if not source.is_file() or source.name in {'.run.lock','last.pt','final.pt'}: continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    os.link(run/'final.pt',destination/'model.pt')
    for name in ['full_data_train.py','publish_full_model.py','publish_kaggle.py','train_and_submit.ipynb']:
        source=Path(__file__).with_name(name)
        if source.exists(): shutil.copy2(source,destination/'source'/name)
    for name in ['folds.csv','fold_report.json','metrics.json','batch_measurements.json','environment.json']:
        if (cv_run/name).exists(): shutil.copy2(cv_run/name,destination/('cv_'+name))
    import pandas as pd
    pd.DataFrame(plan['cv_folds']).to_csv(destination/'cv_fold_metrics.csv',index=False)
    pd.DataFrame([{'parameter':k,'value':json.dumps(v)} for k,v in sorted(config.items())]).to_csv(destination/'hyperparameters.csv',index=False)
    table='\n'.join(f'| {r["fold"]} | {r["best_epoch"]} | {r["best_validation_rmse"]:.8f} |' for r in plan['cv_folds'])
    hyper='\n'.join(f'| `{k}` | `{json.dumps(v)}` |' for k,v in sorted(config.items()))
    description=f'''# {config['model_id']}: single full-data model

Exactly **one full scoring model**, trained fresh from the pinned pretrained backbone
on all **{plan['training_rows']}** original labelled rows for **{plan['epochs']} epochs**,
seed **{plan['seed']}**. This is a full-data refit, not any of the CV fold models.
Initialization: {plan['initialization']}.
Epoch policy: `{plan['policy']}`; CV best epochs: `{plan['cv_best_epochs']}`.
No validation subset, early stopping, training-loss-based checkpoint selection, or
test-label tuning is used for this refit. Fixed final epoch is retained.
Choosing the training length from cross-validation reduces ad hoc tuning but does
not guarantee against overfitting. The refit has **no independent validation RMSE**;
all scores below belong to the earlier grouped cross-validation experiment.

## Cross-validation evidence

| Fold | Best epoch | Best validation RMSE |
|---|---|---|
{table}

Mean best fold RMSE: **{sum(r['best_validation_rmse'] for r in plan['cv_folds'])/len(plan['cv_folds']):.8f}**.
Pooled aligned OOF RMSE: **{plan['cv_metrics']['cv_rmse']:.8f}**.
Full pooled/zero-label/positive-label and same-fold training-mean baseline metrics
are in `cv_metrics.json`. Original labels, including zeros, are unchanged. {len(plan['cv_folds'])}
source-grouped stratified folds keep 31 verified audio pairs together; other shared
speakers remain unverified. Public leaderboard score/metric remain separate.

## Architecture and full hyperparameters

Backbone `{config['model_id']}`, pinned revision `{config['model_revision']}`.
Pooling: `{config['pooling']}`; all pooling/freeze/reinitialisation/dropout settings are recorded below. LayerNorm, scalar linear head and sigmoid times five. Configured backbone blocks follow the saved layer/dropout report. MSE in original score units. Unchanged Whisper Large-v3 text.
Train/validation/inference context {config['max_length']}; dynamic padding rounded to eight. BF16,
gradient checkpointing, global norm clipping {config['max_grad_norm']}. AdamW excludes bias and 1D tensors
from weight decay. Linear scheduler uses {config['warmup_ratio']} of the refit's total optimizer updates
for warmup, then decays to zero. Layer-wise LR decay {config['layer_lr_decay']}. Train microbatch {config['batch_size']}, accumulation {config['gradient_accumulation']}, effective batch {config['effective_batch_size']}; evaluation {config['eval_batch_size']}. Final partial groups
are weighted by actual sample counts. No time cap. GPU: RTX4090 24GB.

| Parameter | Value |
|---|---|
{hyper}

`patience` and `min_improvement` in inherited settings describe CV only. They are
unused in this fixed-epoch full-data refit. Logs, 15-second independent heartbeats,
training MSE/learning rates/gradient norms and TensorBoard scalars are included.

## Artifacts, inference and future ensembles

`model.pt` is the sole trained checkpoint, containing the entire backbone, LayerNorm
and head. Other fold weights, pretrained base weights and optimizer/resume states
are excluded. `submission_full_data.csv` contains 216 actual test IDs in original
order and bounded finite scores. It comes from this single full-data model; the
previous CV-fold ensemble CSV remains separate on Vast. The 204-row official
sample has different IDs; the project exports actual test order instead.
`test_predictions.csv` supports future filename-aligned ensembles.
`upload_checksums.json` records SHA256 for this published payload.

```python
import sys, json, torch
from pathlib import Path
from transformers import AutoTokenizer
root = Path('/path/to/dataset')
sys.path.insert(0, str(root / 'source'))
from train_worker import GrammarModel
config = json.loads((root / 'resolved_config.json').read_text())
model = GrammarModel(config, root / 'backbone', random_backbone=True)
checkpoint = torch.load(root / 'model.pt', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', use_fast=False)
inputs = tokenizer(['Your transcript'], return_tensors='pt', truncation=True,
                   max_length=config['max_length'], padding=True, pad_to_multiple_of=8)
with torch.inference_mode():
    predictions, logits = model(inputs)
```

Directory uploads may be zipped; extract them preserving source/tokenizer/backbone
folder names. Full-data resume states stay on Vast. Competition submission and
instance shutdown are user-controlled. Original data and model/package licenses apply.
'''
    if __import__('audio_features').uses_audio(config):
        description += '''\n## Selected audio branch\n\nThis model includes the configured audio-feature backbone and imputation/scaling\nbuffers fitted on all labelled rows. Supply `inputs['audio_features']` as a\nfloat32 tensor of shape `(batch, audio_input_dim)` with channels concatenated\nin the saved `audio_encodings` order. Use NaN for unavailable entries. The model\napplies its saved transforms internally. In audio-only mode skip the tokenizer\nand pass only audio features; with two backbones include the text tokenizer\noutputs in the same dictionary. All channel dimensions and fusion settings are\nsaved in `resolved_config.json`.\n'''
    if __import__('spectrogram_features').uses(config):
        description += "\n## Spectrogram branch\n\nSupply numerical `spectrogram` chunks shaped `(chunks, chunk_frames, 128)`, `spectrogram_lengths` (valid frames), `spectrogram_times` (original first-frame centers in seconds), and `spectrogram_owner` (zero-based recording index for every chunk). Preserve every recording's chunks in both training and inference. Chunks use the saved frontend/configuration; time encoding is added inside the model. Fold-fitted mel means/scales are checkpoint buffers. AST checkpoints are initialized from the included serialized AST architecture with `random_backbone=True`, then the full scoring state is loaded strictly; pretrained weights are not needed for inference. The pretrained AST frontend is adapted to our saved HTK representation.\n"
    (destination/'README.md').write_text(description)
    telemetry.atomic_json(destination/'training_summary.json',{'model_count':1,'type':'full_data_refit',
        'training_rows':plan['training_rows'],'epochs':plan['epochs'],'epoch_selection':plan,
        'final_model_validation_rmse':None,'private':True,'configuration':config})
    telemetry.atomic_json(destination/'dataset-metadata.json',{'id':dataset_id,
        'title':('SHL '+config['run_id']+' Full Data Model')[:100],'licenses':[{'name':'other'}],'description':description})
    telemetry.atomic_json(destination/'upload_checksums.json',{'files':{str(p.relative_to(destination)):integrity.file_sha256(p)
        for p in destination.rglob('*') if p.is_file() and p.name!='upload_checksums.json'}})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--dataset-id',required=True);p.add_argument('--auth-stdin',action='store_true')
    p.add_argument('--update',action='store_true');p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    if args.auth_stdin:
        auth=json.loads(sys.stdin.readline());os.environ['KAGGLE_USERNAME']=auth['username'];os.environ['KAGGLE_KEY']=auth['key'];del auth
    with telemetry.run_lock(args.run):
        job=json.loads((args.run/'job.json').read_text())
        with telemetry.run_lock(job['cv_run']): document(args.run,args.destination,args.dataset_id)
    print('Prepared exactly one full-data model plus final submission and documentation.',flush=True)
    if not args.prepare_only: upload(args.destination,args.dataset_id,args.update)


if __name__=='__main__': main()
