"""Audit saved held-out predictions and build an audio/transcript listening artifact."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/architecture_error_analysis'
NAMES = {
    '0.504259': 'fusion_deep', '0.515180': 'fusion_shallow',
    '0.517052': 'fusion_hubert_grammar', '0.597296': 'audio_hubert_grammar',
    '0.600088': 'audio_wavlm', '0.815625': 'text_whisper_lstm',
    '0.872455': 'text_crisper_lstm', '0.877899': 'text_crisper_mean',
}


def table(frame, index=False, floatfmt='.3f'):
    frame = frame.reset_index() if index else frame
    def cell(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v).replace('|', '\\|')
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |'] + ['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT/'data/labels/train.csv').set_index('filename')
    assert labels.index.is_unique
    data = labels.copy()
    runs, provenance = [], []
    reference = None
    for score, name in NAMES.items():
        path = Path.home()/'Downloads'/f'run_metadata_{score}.txt'
        raw = path.read_bytes()
        (OUT/'metadata_sources').mkdir(exist_ok=True)
        shutil.copy2(path, OUT/'metadata_sources'/path.name)
        document = json.loads(raw)
        rows = pd.DataFrame(document['oof_predictions']).set_index('filename').reindex(data.index)
        assert rows.index.is_unique and len(document['oof_predictions']) == len(data)
        assert set(r['filename'] for r in document['oof_predictions']) == set(data.index)
        assert np.array_equal(rows.true_score, data.label)
        assert np.isfinite(rows.predicted_score).all()
        membership = rows[['fold', 'source_group']]
        if reference is None:
            reference = membership
            data = data.join(membership)
        else:
            assert membership.equals(reference), 'Runs use different validation partitions'
        prediction = rows.predicted_score.to_numpy()
        error = prediction-data.label.to_numpy()
        reproduced = float(np.sqrt(np.mean(error**2)))
        assert np.isclose(reproduced, document['scores']['pooled_oof_rmse'], atol=1e-8)
        fold_rmse = [float(np.sqrt(np.mean(error[rows.fold.to_numpy() == f]**2))) for f in sorted(rows.fold.unique())]
        assert np.isclose(np.mean(fold_rmse), document['scores']['mean_fold_rmse'], atol=1e-8)
        data[name] = prediction
        config = document['run_reports']['resolved_config.json']
        provenance.append(dict(name=name, source_path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                               run_id=document['run_id'], resolved_config=config,
                               input_manifest=document['input_manifest']))
        runs.append(dict(model=name, pooled_rmse=reproduced, mean_fold_rmse=np.mean(fold_rmse),
                         positive_rmse=float(np.sqrt(np.mean(error[data.label.gt(0)]**2))),
                         zero_rmse=float(np.sqrt(np.mean(error[data.label.eq(0)]**2))),
                         bias=float(error.mean()), above_075=int((abs(error)>=.75).sum()),
                         above_100=int((abs(error)>=1).sum())))
    names = list(NAMES.values())
    fusion, audio, text = names[:3], names[3:5], names[5:]
    errors = data[names].subtract(data.label, axis=0)
    errors.corr().to_csv(OUT/'residual_correlations.csv')
    for group, columns in [('fusion',fusion), ('audio',audio), ('text',text), ('all',names)]:
        e = errors[columns]
        data[f'{group}_mean_prediction'] = data[columns].mean(axis=1)
        data[f'{group}_mean_abs_error'] = e.abs().mean(axis=1)
        data[f'{group}_failures_075'] = e.abs().ge(.75).sum(axis=1)
        data[f'{group}_all_over_075'] = e.ge(.75).all(axis=1)
        data[f'{group}_all_under_075'] = e.le(-.75).all(axis=1)
    data['prediction_spread'] = data[names].max(axis=1)-data[names].min(axis=1)
    whisper = pd.read_csv(ROOT/'artifacts/transcripts-large-v3-dataset/transcripts_large_v3.csv').query('split == "train"').set_index('audio_id')
    crisper = pd.read_csv(ROOT/'artifacts/transcription_crisper/train_transcripts_crisper.csv').set_index('filename')
    data['whisper_transcript'] = whisper.transcript.reindex(data.index).fillna('')
    data['crisper_transcript'] = crisper.transcript.reindex(data.index).fillna('')
    data['crisper_error'] = crisper.error.reindex(data.index).fillna('')
    data['duration_s'] = whisper.duration_s.reindex(data.index)
    data['whisper_words'] = data.whisper_transcript.str.split().str.len()
    data['crisper_words'] = data.crisper_transcript.str.split().str.len()
    data['words_per_second'] = data.whisper_words/data.duration_s
    data['asr_agreement'] = [difflib.SequenceMatcher(None,a.lower().split(),b.lower().split(),autojunk=False).ratio() for a,b in zip(data.whisper_transcript,data.crisper_transcript)]
    data['crisper_word_ratio'] = data.crisper_words/data.whisper_words.clip(lower=1)
    audio_rows = []
    for filename in data.index:
        path = ROOT/'artifacts/competition_audio_full/train'/filename
        assert path.is_file(), filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == crisper.loc[filename,'audio_sha256']
        waveform, rate = sf.read(path, dtype='float32')
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        block = int(rate*.02)
        frames = waveform[:len(waveform)//block*block].reshape(-1,block)
        rms = np.sqrt(np.mean(frames.astype(np.float64)**2, axis=1))
        p10,p90 = np.quantile(rms,[.1,.9])
        audio_rows.append(dict(filename=filename, frame_dynamic_db=20*np.log10(max(p90,1e-12)/max(p10,1e-12)),
                               relative_quiet_fraction=float(np.mean(rms<max(p90,1e-12)*.1)),
                               rms_dbfs=20*np.log10(max(float(np.sqrt(np.mean(waveform.astype(np.float64)**2))),1e-12)),
                               audio_path='../competition_audio_full/train/'+filename))
    data = data.join(pd.DataFrame(audio_rows).set_index('filename'))
    by_label = []
    for label, subset in data.groupby('label'):
        for name in names:
            e = subset[name]-label
            by_label.append(dict(label=label,n=len(subset),model=name,rmse=np.sqrt(np.mean(e**2)),
                                 bias=e.mean(), failures_075=int(e.abs().ge(.75).sum())))
    pd.DataFrame(by_label).to_csv(OUT/'label_metrics.csv',index=False)
    pd.DataFrame(runs).to_csv(OUT/'model_metrics.csv',index=False)
    for name in names:
        data[name+'_error'] = errors[name]
    data['review_category'] = 'other'
    data.loc[data.text_failures_075.eq(3)&data.fusion_failures_075.eq(0),'review_category'] = 'text failure rescued by fusion'
    data.loc[data.prediction_spread.ge(1.5),'review_category'] = 'architecture disagreement'
    data.loc[data.fusion_failures_075.eq(3),'review_category'] = 'all fusion models struggle'
    data.loc[data.all_failures_075.eq(8),'review_category'] = 'all eight models struggle'
    data.to_csv(OUT/'all_recordings.csv')
    priority = data[data.label.gt(0)].sort_values('fusion_mean_abs_error',ascending=False).head(20).copy()
    priority['selection_reason'] = 'largest average fusion error among positive labels'
    disagreement = data[data.label.gt(0)].sort_values('prediction_spread',ascending=False).head(12).copy()
    disagreement['selection_reason'] = 'largest prediction spread among positive labels'
    rescued = data[data.review_category.eq('text failure rescued by fusion')].sort_values('text_mean_abs_error',ascending=False).head(8).copy()
    rescued['selection_reason'] = 'text misses; all fusion errors below 0.75'
    controls = []
    for label in sorted(priority.label.unique()):
        row = data[data.label.eq(label)].sort_values('fusion_mean_abs_error').head(2).copy()
        row['selection_reason'] = 'same-label easy comparison'
        controls.append(row)
    selected = pd.concat([priority,disagreement,rescued,*controls]).loc[lambda d:~d.index.duplicated()].copy()
    selected.to_csv(OUT/'listening_queue.csv')
    for field in ['heard_grammar','transcript_mismatch','noise_or_clipping','pace_pauses_repair','voice_similarity_group','label_consistency_notes','next_experiment_hypothesis']:
        selected[field] = ''
    selected.to_csv(OUT/'listening_notes_template.csv')
    normalized = data.whisper_transcript.str.lower().str.replace(r'\W+',' ',regex=True).str.strip()
    duplicates = data.assign(normalized=normalized).groupby('normalized').filter(lambda d:len(d)>1 and d.label.nunique()>1)
    duplicates.to_csv(OUT/'conflicting_transcript_labels.csv')
    associations = []
    positive = data[data.label.gt(0)]
    for feature in ['words_per_second','whisper_words','crisper_word_ratio','asr_agreement','frame_dynamic_db','relative_quiet_fraction','rms_dbfs']:
        for group in ['fusion','audio','text']:
            raw = positive[feature].corr(positive[f'{group}_mean_abs_error'],method='spearman')
            adjusted_x = positive[feature]-positive.groupby('label')[feature].transform('mean')
            target = positive[f'{group}_mean_abs_error']
            adjusted_y = target-target.groupby(positive.label).transform('mean')
            associations.append(dict(feature=feature,model_group=group,positive_n=len(positive),
                                     spearman=raw,within_label_centered_spearman=adjusted_x.corr(adjusted_y,method='spearman')))
    pd.DataFrame(associations).to_csv(OUT/'feature_error_associations.csv',index=False)
    metrics = pd.DataFrame(runs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for group, columns in [('Fusion',fusion), ('Audio feature',audio), ('Text',text)]:
        grouped = errors[columns].mean(axis=1).groupby(data.label)
        axes[0].plot(grouped.mean().index, grouped.mean(), marker='o', label=group)
    axes[0].axhline(0,color='black',linestyle='--')
    axes[0].set(xlabel='Original label',ylabel='Mean prediction minus label',title='Low scores overpredicted; high scores underpredicted')
    axes[0].legend()
    axes[1].scatter(data.label, data.fusion_deep, alpha=.35, s=14)
    axes[1].plot([0,5],[0,5],color='black',linestyle='--')
    axes[1].set(xlabel='Original label',ylabel='Best fusion OOF prediction',title='All 769 held-out recordings')
    figure.tight_layout(); figure.savefig(OUT/'error_patterns.png',dpi=160); plt.close(figure)
    report = ['1. Audited eight metadata files: all 769 training labels match, all OOF predictions are finite, all pooled/mean-fold RMSE values reproduce, and fold/source-group assignments match across all runs. Original WAV hashes match the saved Crisper input hashes for all 769 recordings.',
              '2. Naming correction: the three runs beginning firstaudrior... with num_backbones=2 are audio + DeBERTa-v3-large fusion, despite their audio-only-looking run names. The HuBERT feature-only run also includes transcript-derived grammar channels. The WavLM feature-only run has no text branch. Configuration and embedded source, rather than run name, determine modality.',
              '3. Model metrics (pooled RMSE differs from the mean-fold score in each filename):\n\n'+table(metrics,floatfmt='.4f'),
              '4. Per-label performance for the best fusion model:\n\n'+table(pd.DataFrame(by_label).query('model == "fusion_deep"'),floatfmt='.4f'),
              f'5. At the descriptive threshold |error| >= 0.75: {int(data.all_failures_075.eq(8).sum())} clips are missed by all eight models; {int(data.fusion_failures_075.eq(3).sum())} by all three fusion models. Of the latter, {int(positive.fusion_failures_075.eq(3).sum())} have positive labels. Text misses on all three runs but fusion misses on none: {int((data.text_failures_075.eq(3)&data.fusion_failures_075.eq(0)).sum())}. These are correlated runs, not eight independent votes.',
              '6. First listening candidates (positive labels, ranked by average absolute fusion error):\n\n'+table(priority[['label',*fusion,'fusion_mean_abs_error','text_mean_abs_error','asr_agreement']].head(15),index=True),
              '7. Associations among positive labels are exploratory. Within-label centering checks whether score distribution explains part of an association; it does not establish causality.\n\n'+table(pd.DataFrame(associations)),
              '8. No verified speaker IDs are available. Source groups identify supported clean/noisy source families, not speakers. Accent, pitch, voice identity, human grammar and annotation consistency require listening; no claim about a particular voice population is established by these numbers. Original labels stay unchanged. Adjacent scores and disagreement alone do not prove label errors.',
              '9. Review review.html for all audio players, Whisper/Crisper transcripts, predictions and saved personal notes. Start with listening_queue.csv and use same-label easy comparisons. Notes remain local to this browser and can be exported. No training, extraction or competition submission was started.',
              '10. This is model-development evidence: each OOF checkpoint was selected using its validation fold, and these runs already informed experiment choices. Any next strategy evaluated on the same folds is exploratory; reserve a fresh untouched evaluation before claiming an improvement.']
    low = data[data.label.between(1,2)]
    high = data[data.label.between(4,5)]
    report.extend([
        f'11. Strongest recurring pattern: score compression. Best-fusion bias is {low.fusion_deep_error.mean():+.3f} for labels 1–2 (n={len(low)}) and {high.fusion_deep_error.mean():+.3f} for labels 4–5 (n={len(high)}). All three fusion models overpredict 20 low-score recordings by at least 0.75 and underpredict 22 high-score recordings by at least 0.75. The score-1 and score-1.5 groups contain only one and three recordings, respectively. Bounded output, sample imbalance and genuinely ambiguous inputs are possible contributors, not proven causes. [Error chart](error_patterns.png).',
        '12. Shared limitations matter more than adding similar models: fusion residual correlations are 0.910–0.922. The 20 recordings with largest average fusion errors contribute about 18.0% of the best-fusion squared error. Architecture depth comparisons also change dropout and learning rate, so the small aggregate improvement cannot be attributed to depth alone.',
        '13. Audio has already addressed much of the earlier zero-label problem: text-Whisper zero RMSE is 2.485 versus best-fusion 0.408. Positive-label RMSE also improves from 0.622 to 0.510. The next review should emphasize positive recordings, while retaining zeros as quality controls.',
        '14. Listen first to audio_577 (label 4; fusion predictions 1.61–2.91): Whisper has a long repeated nature passage; Crisper is substantially different. Text-Whisper predicts 1.77, but text-Crisper predicts 4.29/4.49. This supports an ASR-input failure hypothesis to confirm against the WAV. Conversely, audio_555 (label 5; fusion 3.27–3.72) has a long repeated passage in Crisper. Neither transcript source is uniformly reliable.',
        '15. Label/input consistency review: audio_15 and audio_55 have label 2 but relatively coherent saved text; all eight predictions exceed 2.75. audio_238 has label 4, a short transcript and agreement errors, while every model predicts below 3.25. audio_91 has label 5 and a complex marketplace passage but all eight predictions are below 4.25. These are listening questions, not established label mistakes; actual speech, rubric interpretation, ASR corrections and genuine model errors must be distinguished.',
        '16. Next experiment A — transcript reliability: preserve the strongest fusion baseline and compare the original Whisper input against a second transcript view or a targeted fallback for confirmed ASR failure. Record ASR disagreement, repetition and missingness explicitly. Do not remove spoken repetitions based only on text. Evaluate the reviewed failure set and all 769 rows, including whether improvements damage previously easy cases.',
        '17. Next experiment B — score compression: hold input features, folds, seeds and architecture fixed, and compare the current bounded MSE head with a linear regression head (with a predeclared final clipping rule), then separately test modest score-balanced sampling/loss weighting. Report per-score signed bias plus pooled, positive and zero RMSE. Fit any calibration only within each training fold using inner validation; fitting it to all these OOF labels and scoring the same rows would overstate improvement.',
        '18. Next experiment C — time structure: if listening shows missed clause errors, repairs or long pauses, compare whole-recording pooled features with a trainable chunk/word sequence branch, preserving grammar mistakes in text. This tests a specific missing signal. It remains a proposal; no new model or extraction has been run.',
        '19. Listening protocol: first hear each hard clip and its same-label easy controls, then inspect both transcripts and reveal model errors. Note specific audible grammar, ASR changes, pauses/repairs and quality. Assign a tentative voice group only when recordings sound like the same speaker; mark uncertain matches and verify before designing speaker-grouped validation. Reviewing errors must not automatically change original labels. Prioritize A when ASR is wrong, B for repeated tail bias with faithful inputs, and C when faithful text omits a useful temporal signal.'
    ])
    (OUT/'report.md').write_text('\n\n'.join(report)+'\n')
    manifest = dict(runs=provenance,validation_checks='complete row alignment, labels, finite predictions, both RMSE definitions, equal folds/source groups, all audio SHA256s',
                    error_threshold=.75,speaker_ids_available=False)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records = data.reset_index().replace({np.nan:None}).to_dict('records')
    payload = json.dumps(dict(rows=records,models=names,queue=selected.index.tolist()),allow_nan=False).replace('</','<\\/')
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Architecture error listening review</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:1300px;padding:0 20px;background:#f4f5f7;color:#18222c}article{background:white;padding:22px;margin:18px 0;border-radius:10px}select,input,button{padding:8px;margin:4px}table{border-collapse:collapse;width:100%}td,th{padding:6px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;font:15px system-ui;line-height:1.55}textarea{width:98%;height:100px}audio{width:100%}.transcripts{display:grid;grid-template-columns:1fr 1fr;gap:22px}.error{color:#b62727}@media(max-width:700px){.transcripts{grid-template-columns:1fr}}</style>
<h1>1. Architecture error listening review</h1><p>Eight runs, 769 held-out recordings. Error = prediction − original label. Start with the listening queue, compare easy examples with the same label, and record what you actually hear. Voice identities and accents have not been established.</p>
<h2>2. Choose recordings</h2><select id="category"><option value="queue">Listening queue</option><option value="all">All recordings</option><option>all eight models struggle</option><option>all fusion models struggle</option><option>architecture disagreement</option><option>text failure rescued by fusion</option><option value="easy">Easy comparison (fusion error &lt; 0.25)</option></select><select id="label"><option value="all">All labels</option></select><input id="search" placeholder="Filename or transcript"><select id="sort"><option value="fusion_mean_abs_error">Fusion error</option><option value="prediction_spread">Model disagreement</option><option value="all_mean_abs_error">All-model error</option></select><button id="export">Export listening notes</button><div id="summary"></div><button id="prev">Previous 15</button><button id="next">Next 15</button><main id="results"></main>
<script>const D=PAYLOAD;let page=0;const $=id=>document.getElementById(id);const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const fmt=x=>Number(x).toFixed(3);let notes={};try{notes=JSON.parse(localStorage.getItem('shl-architecture-review-v1')||'{}')}catch(e){};
[...new Set(D.rows.map(r=>r.label))].sort((a,b)=>a-b).forEach(v=>$('label').add(new Option('Label '+v,v)));
function render(){let rows=D.rows.filter(r=>($('label').value==='all'||r.label===Number($('label').value))&&($('category').value==='all'||($('category').value==='queue'?D.queue.includes(r.filename):$('category').value==='easy'?r.fusion_mean_abs_error<.25:r.review_category===$('category').value))&&(! $('search').value||(r.filename+' '+r.whisper_transcript+' '+r.crisper_transcript).toLowerCase().includes($('search').value.toLowerCase()))).sort((a,b)=>b[$('sort').value]-a[$('sort').value]);page=Math.min(page,Math.max(0,Math.ceil(rows.length/15)-1));$('summary').textContent=rows.length+' recordings · page '+(page+1)+' · notes saved locally in this browser';$('results').innerHTML=rows.slice(page*15,page*15+15).map((r,i)=>`<article><h2>${3+page*15+i}. ${esc(r.filename)} · label ${r.label}</h2><p>${esc(r.review_category)} · fusion mean absolute error ${fmt(r.fusion_mean_abs_error)} · Whisper/Crisper word agreement ${fmt(r.asr_agreement)} · level variation ${fmt(r.frame_dynamic_db)} dB</p><audio controls preload="none" src="${esc(r.audio_path)}"></audio><details><summary>Show model predictions and errors</summary><table><tr><th>Run</th><th>Prediction</th><th>Prediction − label</th></tr>${D.models.map(m=>`<tr><td>${esc(m)}</td><td>${fmt(r[m])}</td><td class="${Math.abs(r[m]-r.label)>=.75?'error':''}">${fmt(r[m]-r.label)}</td></tr>`).join('')}</table></details><div class="transcripts"><section><h3>1. Whisper Large-v3</h3><pre>${esc(r.whisper_transcript)}</pre></section><section><h3>2. Crisper verbatim</h3><pre>${esc(r.crisper_transcript||r.crisper_error||'Unavailable')}</pre></section></div><label>3. Listening notes: actual grammar; ASR omissions/corrections; noise; pace/pauses/repairs; suspected repeated voice; label consistency; proposed experiment.</label><textarea data-file="${esc(r.filename)}">${esc(notes[r.filename]||'')}</textarea></article>`).join('');document.querySelectorAll('textarea').forEach(t=>t.oninput=()=>{notes[t.dataset.file]=t.value;try{localStorage.setItem('shl-architecture-review-v1',JSON.stringify(notes))}catch(e){$('summary').textContent='Browser storage unavailable: export notes before closing.'}})}
['category','label','sort'].forEach(id=>$(id).onchange=()=>{page=0;render()});$('search').oninput=()=>{page=0;render()};$('prev').onclick=()=>{page=Math.max(0,page-1);render()};$('next').onclick=()=>{page++;render()};$('export').onclick=()=>{let a=document.createElement('a'),blob=new Blob([JSON.stringify(notes,null,2)],{type:'application/json'});a.href=URL.createObjectURL(blob);a.download='listening_notes.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();</script></html>'''
    (OUT/'review.html').write_text(template.replace('PAYLOAD',payload))
    print(metrics.to_string(index=False))
    print('Fusion failures',data.fusion_failures_075.eq(3).sum(),'all failures',data.all_failures_075.eq(8).sum())
    print(priority[['label',*fusion,'fusion_mean_abs_error']].head(15).to_string())


if __name__ == '__main__':
    main()
