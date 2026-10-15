"""Descriptive transcript/audio audit; preserves labels and does not train models."""
from collections import Counter
from pathlib import Path
import difflib
import hashlib
import html
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from analyze_zero_audio import load
from analyze_zero_labels import features, markdown_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/score_pattern_analysis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(ROOT/'data/labels/train.csv')
    large = pd.read_csv(ROOT/'artifacts/transcripts-large-v3-dataset/transcripts_large_v3.csv')
    small = pd.read_csv(ROOT/'artifacts/transcription/transcripts.csv')
    data = labels.merge(large[large.split.eq('train')], left_on='filename',
                        right_on='audio_id', validate='one_to_one')
    assert len(data) == len(labels) == 769
    data = data.merge(small[small.split.eq('train')][['audio_id','transcript']].rename(
        columns={'transcript':'small_transcript'}), on='audio_id', validate='one_to_one')
    assert data.transcript.notna().all() and data.small_transcript.notna().all()
    derived = pd.DataFrame([features(t) for t in data.transcript])
    data = pd.concat([data.reset_index(drop=True), derived], axis=1)
    data['words_per_second'] = data.words / data.duration_s
    data['asr_agreement'] = [difflib.SequenceMatcher(None, a.lower().split(),
        b.lower().split(), autojunk=False).ratio() for a,b in zip(data.transcript,data.small_transcript)]
    data['repetition_flag'] = data.repeated_4gram_fraction.gt(.30)
    # Descriptive lexical marker only, not a grammar complexity score.
    data['clause_marker_per_100_words'] = [100*len(re.findall(
        r'\b(?:because|although|though|while|unless|if|whether|which|whose)\b',
        t.lower()))/max(w,1) for t,w in zip(data.transcript,data.words)]
    summary = data.groupby('label').agg(n=('filename','size'),
        duration_median=('duration_s','median'), words_median=('words','median'),
        words_per_second_median=('words_per_second','median'),
        asr_agreement_median=('asr_agreement','median'),
        repetition_flags=('repetition_flag','sum'),
        clause_marker_median=('clause_marker_per_100_words','median'))
    summary.to_csv(OUT/'transcript_group_summary.csv')

    audio_paths = {}
    # Known competition training directories only; exclude unrelated/test WAVs.
    for folder in ['zero_label_analysis/audio','audio_spectrogram_preview/audio']:
        for path in sorted((ROOT/'artifacts'/folder).glob('*.wav')):
            if path.name in set(labels.filename):
                if path.name in audio_paths:
                    assert hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(
                        audio_paths[path.name].read_bytes()).digest()
                else:
                    audio_paths[path.name] = path
    audio_rows = []
    search_waveforms = {}
    for name,path in sorted(audio_paths.items()):
        rate,x,raw = load(path)
        assert rate == 16000
        freq,power = welch(x,fs=rate,nperseg=min(2048,len(x)))
        block = int(rate*.02)
        frames = x[:len(x)//block*block].reshape(-1,block)
        frame_rms = np.sqrt(np.mean(frames**2,axis=1))
        p10,p90 = np.quantile(frame_rms,[.10,.90])
        audio_rows.append(dict(filename=name, rate=rate, duration_audio=len(x)/rate,
            rms=float(np.sqrt(np.mean(x*x))),
            rms_dbfs=float(20*np.log10(max(np.sqrt(np.mean(x*x)),1e-12))),
            clipping_fraction=float(np.mean(np.abs(x)>=.999)),
            near_zero_fraction=float(np.mean(np.abs(x)<.001)),
            high_frequency_power_fraction=float(power[freq>4000].sum()/max(power.sum(),1e-20)),
            frame_dynamic_db=float(20*np.log10(max(p90,1e-12)/max(p10,1e-12))),
            zcr=float(np.mean(np.signbit(x[1:])!=np.signbit(x[:-1]))),
            audio_path_local=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        # Cover the full recording so opening silence cannot hide a source match.
        search_waveforms[name] = x[::4].astype(np.float32)
    aud = pd.DataFrame(audio_rows).merge(labels,on='filename',validate='one_to_one')
    aud.to_csv(OUT/'audio_features.csv',index=False)
    audio_summary = aud.groupby('label').agg(n_local=('filename','size'),
        rms_dbfs=('rms_dbfs','median'), high_frequency_power_fraction=('high_frequency_power_fraction','median'),
        frame_dynamic_db=('frame_dynamic_db','median'),near_zero_fraction=('near_zero_fraction','median'),
        clipping_fraction=('clipping_fraction','median'))
    audio_summary.to_csv(OUT/'audio_group_summary.csv')
    data = data.merge(aud.drop(columns=['label','rate','sha256']),on='filename',how='left',validate='one_to_one')
    data.to_csv(OUT/'all_train_features.csv',index=False)

    correlations=[]
    for population,part in [('all_train_text',data),('positive_train_text',data[data.label.gt(0)]),
                            ('positive_local_audio',data[data.label.gt(0)&data.rms.notna()])]:
        cols = (['rms_dbfs','high_frequency_power_fraction','frame_dynamic_db','near_zero_fraction','zcr']
            if population.endswith('audio') else ['duration_s','words','words_per_second',
                'asr_agreement','repeated_4gram_fraction','clause_marker_per_100_words'])
        for col in cols:
            rho,_ = spearmanr(part.label,part[col])
            correlations.append(dict(population=population,n=len(part),feature=col,spearman_rho=float(rho)))
    pd.DataFrame(correlations).to_csv(OUT/'descriptive_correlations.csv',index=False)

    # Search every local positive source, independently of transcript wording.
    zeros = sorted(aud[aud.label.eq(0)].filename)
    positives = sorted(aud[aud.label.gt(0)].filename)
    max_samples=max(len(x) for x in search_waveforms.values())
    zm = np.stack([np.pad(search_waveforms[n],(0,max_samples-len(search_waveforms[n]))) for n in zeros])
    pm = np.stack([np.pad(search_waveforms[n],(0,max_samples-len(search_waveforms[n]))) for n in positives])
    zm /= np.maximum(np.linalg.norm(zm,axis=1,keepdims=True),1e-20)
    pm /= np.maximum(np.linalg.norm(pm,axis=1,keepdims=True),1e-20)
    cos = zm@pm.T
    pairs=[]
    for i,name in enumerate(zeros):
        j=int(np.argmax(cos[i])); source=positives[j]
        rate,x,_=load(audio_paths[name]); _,y,_=load(audio_paths[source])
        n=min(len(x),len(y));x=x[:n];y=y[:n]
        gain=float(x@y/max(y@y,1e-30)); component=gain*y; residual=x-component
        fullcos=float(x@y/np.sqrt(max((x@x)*(y@y),1e-30)))
        sr=float(10*np.log10(max(np.mean(component**2),1e-30)/max(np.mean(residual**2),1e-30)))
        xf,p=welch(residual,fs=rate,nperseg=2048)
        q=p[(xf>0)&(xf<rate/2)]
        flatness=float(np.exp(np.log(np.maximum(q,1e-30)).mean())/max(q.mean(),1e-30))
        pairs.append(dict(zero_filename=name,positive_filename=source,
            positive_label=float(labels.set_index('filename').loc[source,'label']),
            waveform_search_cosine=float(cos[i,j]),full_waveform_cosine=fullcos,
            supported_source_match=fullcos>.45, fitted_gain=gain,
            signal_to_residual_db=sr,residual_spectral_flatness=flatness,
            residual_lag1_correlation=float(np.corrcoef(residual[:-1],residual[1:])[0,1]),
            compared_seconds=n/rate))
    pairs=pd.DataFrame(pairs)
    pairs.to_csv(OUT/'local_source_search.csv',index=False)

    # Identical normalized ASR text with conflicting labels.
    data['normalized_text']=data.transcript.str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
    conflicts=data.groupby('normalized_text').filter(lambda g:g.label.nunique()>1)
    conflicts[['filename','label','transcript']].to_csv(OUT/'identical_text_different_labels.csv',index=False)

    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    groups=list(summary.index)
    for ax,col,frame,title in [(axes[0],'asr_agreement',data,'ASR agreement: all 769 transcripts'),
        (axes[1],'high_frequency_power_fraction',aud,'Energy above 4 kHz: local audio'),
        (axes[2],'frame_dynamic_db',aud,'20 ms RMS dynamic range: local audio')]:
        ax.boxplot([frame[frame.label.eq(g)][col].dropna().values for g in groups],
                   tick_labels=[str(g) for g in groups],showfliers=False)
        ax.set(title=title,xlabel='Provided score',ylabel=col)
    fig.tight_layout();fig.savefig(OUT/'score_patterns.png',dpi=160);plt.close(fig)

    blocks=['<!doctype html><meta charset="utf-8"><title>Score pattern review</title>',
        '<style>body{font:16px system-ui;max-width:1000px;margin:32px auto}section{border-top:1px solid #bbb;padding:14px 0}p{white-space:pre-wrap}audio{width:100%}summary{cursor:pointer}</style>',
        '<h1>Training score review</h1><p>All 769 training transcripts; original audio players for available local WAVs. Whisper text is model-generated. Local positive audio is a selected sample. No labels changed.</p>']
    for score,part in data.sort_values('filename').groupby('label'):
        blocks.append(f'<details><summary>Score {score}: {len(part)} recordings</summary>')
        for _,r in part.iterrows():
            blocks.append(f'<section><h3>{html.escape(r.filename)} — {r.label}</h3>')
            if r.filename in audio_paths:
                src='../../'+str(audio_paths[r.filename].relative_to(ROOT))
                blocks.append(f'<audio controls preload="none" src="{html.escape(src)}"></audio>')
            else:blocks.append('<p>Original WAV not downloaded locally.</p>')
            blocks.append('<p>'+html.escape(r.transcript)+'</p><details><summary>Whisper Small comparison</summary><p>'+html.escape(r.small_transcript)+'</p></details></section>')
        blocks.append('</details>')
    (OUT/'review.html').write_text('\n'.join(blocks))
    manifest={'date':'2026-10-12','train_transcripts':len(data),'distinct_local_training_wavs':len(aud),
        'local_zero_wavs':len(zeros),'local_positive_wavs':len(positives),
        'supported_source_pairs':int(pairs.supported_source_match.sum()),
        'audio_assessment':'Numerical waveform/spectral analysis, not direct listening.',
        'positive_audio_sample_representative':False,
        'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'data/labels/train.csv',ROOT/'artifacts/transcripts-large-v3-dataset/transcripts_large_v3.csv',ROOT/'artifacts/transcription/transcripts.csv']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'summary_tables.md').write_text('## Transcript groups\n\n'+markdown_table(summary)+
        '\n\n## Selected local audio groups\n\n'+markdown_table(audio_summary)+
        '\n\n## Descriptive correlations\n\n'+markdown_table(pd.DataFrame(correlations),index=False)+'\n')
    print(json.dumps(manifest,indent=2))
    print(summary.round(3).to_string())
    print(audio_summary.round(4).to_string())
    print(pd.DataFrame(correlations).round(4).to_string(index=False))
    print('Unresolved local source searches:')
    print(pairs[~pairs.supported_source_match].to_string(index=False))
    print('Identical transcript / different labels:')
    print(conflicts[['filename','label','transcript']].to_string(index=False))


if __name__=='__main__':
    main()
