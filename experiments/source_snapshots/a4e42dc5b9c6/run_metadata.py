"""Export exactly two downloads, with best-epoch OOF and submission inside JSON TXT."""
from datetime import datetime, timezone
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd


def build_plots(oof,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    images={}
    def encode(name,figure):
        buffer=io.BytesIO();figure.tight_layout();figure.savefig(buffer,format='png',dpi=140)
        images[name]={'mime_type':'image/png','encoding':'base64','data':base64.b64encode(buffer.getvalue()).decode('ascii')}
        plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(15,4))
    for report in reports:
        history=report['history'];epochs=[r['epoch'] for r in history];label=f'Fold {report["fold"]+1}'
        axes[0].plot(epochs,[r['train_loss'] for r in history],marker='.',label=label)
        axes[1].plot(epochs,[r['validation_rmse'] for r in history],marker='.',label=label)
        axes[1].scatter([report['best_epoch']],[report['best_rmse']],marker='*',s=100)
        if history and all('learning_rates' in r for r in history):
            for group in range(len(history[0]['learning_rates'])):
                axes[2].plot(epochs,[r['learning_rates'][group] for r in history],label=f'{label}, optimizer group {group}')
    for axis,title in zip(axes,['Training MSE (during updates)','Validation RMSE; stars mark selected epochs','Learning rate at epoch end']):
        axis.set_title(title);axis.set_xlabel('Epoch');axis.grid(alpha=.2)
        if axis.lines:axis.legend(fontsize=7)
    encode('epoch_learning_curves',figure)
    figure,axes=plt.subplots(1,2,figsize=(10,4))
    for zero,label in [(False,'Positive labels'),(True,'Zero labels')]:
        p=oof[oof.label.eq(0)==zero]
        axes[0].scatter(p.label,p.prediction,s=13,alpha=.6,label=label)
    axes[0].plot([0,5],[0,5],color='black',linestyle='--');axes[0].legend()
    axes[0].set(xlabel='True score',ylabel='OOF predicted score',title='Best-epoch out-of-fold predictions')
    axes[1].hist(oof.prediction-oof.label,bins=35)
    axes[1].axvline(0,color='black',linestyle='--')
    axes[1].set(xlabel='Prediction minus true score',ylabel='Clips',title='OOF error distribution')
    encode('oof_error_diagnostics',figure)
    return images


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def assemble_oof(run,jobs,reports):
    """Accept only the checksum-verified final predictions for each selected best epoch."""
    run=Path(run)
    train=pd.read_csv(run/'input/train.csv',keep_default_na=False)
    folds=pd.read_csv(run/'folds.csv')
    if train.filename.duplicated().any() or folds.filename.duplicated().any():
        raise ValueError('Duplicate training/fold IDs')
    expected=train[['filename','label']].merge(folds[['filename','fold','source_group']],on='filename',validate='one_to_one')
    if len(expected)!=len(train):raise ValueError('Incomplete folds')
    rows=[]; selected=[]
    if len(jobs)!=len(reports) or len(jobs)!=expected.fold.nunique():
        raise ValueError('Need all completed folds')
    for job,report in zip(jobs,reports):
        directory=Path(job['job_dir'])
        for name in ['best.pt','validation_predictions.csv','test_predictions.csv']:
            if sha256(directory/name)!=report['artifact_sha256'][name]:
                raise ValueError('Best-model artifact changed: '+name)
        history=report['history']
        best_epoch=report['best_epoch']
        selected_history=[r for r in history if r['epoch']==best_epoch]
        if len(selected_history)!=1 or not np.isclose(selected_history[0]['validation_rmse'],report['best_rmse'],rtol=0,atol=1e-8):
            raise ValueError('Selected best epoch does not match fold history')
        valid=pd.read_csv(directory/'validation_predictions.csv')
        wanted=expected[expected.fold.eq(job['fold'])]
        if valid.filename.duplicated().any() or set(valid.filename)!=set(wanted.filename) or not valid.fold.eq(job['fold']).all():
            raise ValueError('OOF fold membership differs')
        joined=wanted.merge(valid[['filename','label','prediction']],on='filename',validate='one_to_one',suffixes=('','_pred'))
        if not joined.label.eq(joined.label_pred).all() or not np.isfinite(joined.prediction).all():
            raise ValueError('OOF labels or predictions are invalid')
        rmse=float(np.sqrt(np.mean((joined.label-joined.prediction)**2)))
        if not np.isclose(rmse,report['best_rmse'],rtol=1e-5,atol=1e-5):
            raise ValueError('OOF predictions do not reproduce selected best-epoch RMSE')
        joined['best_epoch']=best_epoch
        rows.append(joined.drop(columns='label_pred'))
        selected.append(dict(fold=int(job['fold']),best_epoch=int(best_epoch),selection_metric='validation_rmse',
            rmse=rmse,recorded_best_rmse=report['best_rmse'],seed=report['seed'],
            validation_rows=len(valid),checkpoint_sha256=report['artifact_sha256']['best.pt']))
    combined=pd.concat(rows,ignore_index=True)
    if combined.filename.duplicated().any():raise ValueError('An OOF row was predicted more than once')
    ordered=train[['filename']].merge(combined,on='filename',how='left',validate='one_to_one')
    if not np.isfinite(ordered.prediction).all():raise ValueError('Missing OOF rows')
    selected.sort(key=lambda r:r['fold'])
    metrics=dict(mean_fold_rmse=float(np.mean([r['rmse'] for r in selected])),
        pooled_oof_rmse=float(np.sqrt(np.mean((ordered.label-ordered.prediction)**2))))
    return ordered,selected,metrics


def export(code,run,submission_path,action='cv_submission',full_run=None):
    """Called after training; no model inference, training or publication is performed."""
    import notebook_controls
    code,run=Path(code),Path(run)
    # Re-running this notebook with a different engine must not reuse cached imports.
    for name in ['pipeline','train_worker','sweep_utils','telemetry','pooling','audio_features','spectrogram_features']:
        module=sys.modules.get(name)
        if module and Path(module.__file__).resolve().parent!=code.resolve():
            del sys.modules[name]
    config,jobs,reports=notebook_controls.verify_cv(code,run)
    oof,selected,scores=assemble_oof(run,jobs,reports)
    submission_path=Path(submission_path)
    submission=pd.read_csv(submission_path)
    template=pd.read_csv(run/'input/submission_template.csv')
    if (list(submission.columns)!=['filename','label'] or submission.filename.tolist()!=template.filename.tolist()
            or not submission.filename.is_unique or not np.isfinite(submission.label).all()
            or not submission.label.between(0,5).all()):
        raise ValueError('Invalid selected submission')
    # Provenance distinguishes CV-ensemble, selected-fold and fresh full-data outputs.
    submission_evidence={'action':action,'source_path':str(submission_path),'source_sha256':sha256(submission_path)}
    if action=='best_fold':
        chosen=min(selected,key=lambda r:(r['recorded_best_rmse'],r['fold']))
        prediction=pd.read_csv(Path(next(j for j in jobs if j['fold']==chosen['fold'])['job_dir'])/'test_predictions.csv')
        submission_evidence['selected_fold']=chosen
    elif action=='cv_submission':
        predictions=[pd.read_csv(Path(j['job_dir'])/'test_predictions.csv').set_index('filename').prediction.reindex(template.filename).to_numpy() for j in jobs]
        prediction=template[['filename']].assign(prediction=np.mean(predictions,axis=0))
    elif action=='full_data':
        prediction=None
        if full_run is None:raise ValueError('Full-data export needs its run directory')
        submission_evidence['validation_note']='OOF describes CV models; the full-data refit has no held-out validation score.'
    else:raise ValueError('Unknown submission action')
    if prediction is not None:
        values=prediction.set_index('filename').prediction.reindex(submission.filename).to_numpy()
        if not np.allclose(values,submission.label.to_numpy(),rtol=1e-10,atol=1e-10):
            raise ValueError('Selected submission differs from best-epoch test predictions')
    json_names=['resolved_config.json','measured_config.json','model_provenance.json','environment.json',
        'length_report.json','fold_report.json','batch_measurements.json','metrics.json','terminal_status.json','spectrogram_report.json','spectrogram_pretrained.json','audio_report.json']
    evidence={p.name:json.loads(p.read_text()) for p in [run/name for name in json_names] if p.is_file()}
    source_code={p.name:p.read_text() for p in code.glob('*.py') if not p.name.startswith('.')}
    controls_dir=Path(__file__).parent
    for name in ['notebook_controls.py','run_metadata.py','train_and_submit.ipynb']:
        p=controls_dir/name
        if p.is_file():source_code[name]=p.read_text()
    run_settings=controls_dir.parents[1]/'configs'/f'{run.name}.json'
    document=dict(schema_version=1,created_at_utc=datetime.now(timezone.utc).isoformat(),run_id=config['run_id'],
        score_in_filename='mean_fold_rmse',scores=scores,
        requested_config=json.loads(run_settings.read_text()) if run_settings.is_file() else config,
        run_reports=evidence,input_manifest=json.loads((run/'input/manifest.json').read_text()),
        installed_requirements=(run/'installed-requirements.txt').read_text() if (run/'installed-requirements.txt').exists() else None,
        approaches={'text_preprocessing':config.get('preprocessing'),'loss':config['loss'],
            'validation_policy':config['validation_policy'],'best_epoch_selection':'lowest validation RMSE; per-fold best checkpoint reloaded before OOF/test inference',
            'group_note':'source_group identifies verified clean/noisy audio families; it is not a verified speaker ID',
            'fold_numbering':'zero-based','oof_order':'original training CSV order','submission':submission_evidence},
        folds=selected,fold_training_reports=reports,
        plots=build_plots(oof,reports),
        plot_data={'epochs':[{'fold':r['fold'],'history':r['history']} for r in reports],
            'oof_errors':oof.assign(error=oof.prediction-oof.label)[['filename','label','prediction','error','fold']].to_dict('records'),
            'learning_rate_definition':'Actual optimizer learning rates recorded after the last update of each epoch; not reconstructed.'},
        oof_predictions=oof.rename(columns={'label':'true_score','prediction':'predicted_score'}).to_dict('records'),
        submission_predictions=submission.rename(columns={'label':'predicted_score'}).to_dict('records'),
        submission_csv_text=submission_path.read_text(),source_code=source_code)
    if full_run:
        full_run=Path(full_run)
        document['full_data_run']={p.name:json.loads(p.read_text()) for p in full_run.glob('*.json')}
    # Only these two files are offered for download; training/resume artifacts stay internal.
    destination=controls_dir.parents[1]/'outputs'/run.name/action
    destination.mkdir(parents=True,exist_ok=True)
    name=f'run_metadata_{scores["mean_fold_rmse"]:.6f}.txt'
    serialized=json.dumps(document,indent=2,allow_nan=False)+'\n'
    temporary=destination/(name+'.tmp');temporary.write_text(serialized);temporary.replace(destination/name)
    for old in destination.glob('run_metadata_*.txt'):
        if old.name!=name:old.unlink()
    shutil.copy2(submission_path,destination/'submission.csv')
    return destination/'submission.csv',destination/name
