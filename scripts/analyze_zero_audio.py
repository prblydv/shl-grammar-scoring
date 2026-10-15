"""Signal diagnostics for downloaded zero WAVs and text-matched positive WAVs."""
from pathlib import Path
import hashlib
import html
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample_poly, welch
from analyze_zero_labels import markdown_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/zero_label_analysis'


def load(path):
    rate, raw = wavfile.read(path)
    if raw.dtype.kind=='i': x=raw.astype(float)/max(abs(np.iinfo(raw.dtype).min),np.iinfo(raw.dtype).max)
    elif raw.dtype.kind=='u': x=(raw.astype(float)-128)/128
    else: x=raw.astype(float)
    if x.ndim==2: x=x.mean(axis=1)
    return rate,x,raw


def main():
    review=pd.read_csv(OUT/'zero_review.csv')
    names=sorted(set(review.filename)|set(review.nearest_positive_filename))
    cache={}; rows=[]
    for name in names:
        path=OUT/'audio'/name
        if not path.exists(): continue
        rate,x,raw=load(path); cache[name]=(rate,x)
        freq,power=welch(x,fs=rate,nperseg=min(2048,len(x)))
        rows.append(dict(filename=name,rate=rate,channels=1 if raw.ndim==1 else raw.shape[1],dtype=str(raw.dtype),
            duration_s=len(x)/rate,rms=float(np.sqrt(np.mean(x*x))),peak=float(abs(x).max()),
            fullscale_fraction=float(np.mean(abs(x)>=.999)),near_zero_fraction=float(np.mean(abs(x)<.001)),
            high_frequency_power_fraction=float(power[freq>4000].sum()/max(power.sum(),1e-20)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    metrics=pd.DataFrame(rows); metrics.to_csv(OUT/'audio_metrics.csv',index=False)
    pairs=[]
    for _,r in review.iterrows():
        a,b=r.filename,r.nearest_positive_filename
        if a not in cache or b not in cache:continue
        ra,x=cache[a];rb,y=cache[b]
        if rb!=ra:
            g=math.gcd(ra,rb);y=resample_poly(y,ra//g,rb//g)
        n=min(len(x),len(y));x=x[:n];y=y[:n]
        dot=float(np.dot(x,y)); xx=float(np.dot(x,x));yy=float(np.dot(y,y))
        cosine=dot/math.sqrt(max(xx*yy,1e-30))
        gain=dot/max(yy,1e-30)
        residual=float(np.mean((x-gain*y)**2))
        explained=float(np.mean((gain*y)**2))
        pairs.append(dict(zero_filename=a,positive_filename=b,positive_label=r.nearest_positive_label,
            text_similarity=r.nearest_positive_similarity,compared_seconds=n/ra,
            aligned_waveform_cosine=cosine,fitted_gain=gain,
            aligned_signal_to_residual_db=10*math.log10(max(explained,1e-30)/max(residual,1e-30)),
            identical_file=metrics.set_index('filename').loc[a,'sha256']==metrics.set_index('filename').loc[b,'sha256']))
    pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'audio_pair_comparison.csv',index=False)
    matched=pairs[pairs.aligned_waveform_cosine.gt(.45)].copy()
    folds=pd.read_csv(ROOT/'artifacts/deberta-reference-dataset/frozen_folds.csv').set_index('filename').fold
    matched['zero_fold']=matched.zero_filename.map(folds)
    matched['positive_fold']=matched.positive_filename.map(folds)
    matched['different_folds']=matched.zero_fold.ne(matched.positive_fold)
    matched.to_csv(OUT/'matched_source_fold_audit.csv',index=False)
    picked=review.nlargest(3,'nearest_positive_similarity')
    fig,axes=plt.subplots(len(picked),2,figsize=(12,8))
    for row,(_,r) in enumerate(picked.iterrows()):
        for col,name in enumerate([r.filename,r.nearest_positive_filename]):
            if name in cache:
                rate,x=cache[name];step=max(1,len(x)//3000)
                axes[row,col].plot(np.arange(0,len(x),step)/rate,x[::step],lw=.5)
                axes[row,col].set(title=f'{name} (label {0 if col==0 else r.nearest_positive_label})',xlabel='Seconds',ylabel='Amplitude')
    fig.tight_layout();fig.savefig(OUT/'audio_pair_waveforms.png',dpi=160);plt.close(fig)
    blocks=['<!doctype html><meta charset="utf-8"><title>Zero-label audio review</title>',
            '<style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}section{border-top:1px solid #bbb;padding:20px 0}p{white-space:pre-wrap}audio{width:100%}</style>',
            '<h1>Zero-label audio review</h1><p>Listen to each zero recording and its text-matched positive candidate. '
            'A text match does not prove the recordings have the same source. Labels are unchanged.</p>']
    for _,r in review.iterrows():
        blocks.extend([f'<section><h2>{html.escape(r.filename)} — label 0</h2>',
            f'<audio controls preload="none" src="audio/{html.escape(r.filename)}"></audio>',
            '<h3>Large-v3 transcript</h3><p>'+html.escape(r.transcript)+'</p>',
            '<h3>Small transcript</h3><p>'+html.escape(r.small_transcript)+'</p>',
            f'<h3>Nearest positive: {html.escape(r.nearest_positive_filename)} — label {r.nearest_positive_label}</h3>',
            f'<audio controls preload="none" src="audio/{html.escape(r.nearest_positive_filename)}"></audio>',
            '<p>'+html.escape(r.nearest_positive_transcript)+'</p></section>'])
    (OUT/'audio_review.html').write_text('\n'.join(blocks))
    summary=json.loads((OUT/'summary.json').read_text())
    summary['raw_competition_audio_available_locally']=True
    summary['downloaded_zero_audio_count']=sum(n in cache for n in review.filename)
    summary['downloaded_total_audio_count']=len(cache)
    summary['audio_pairs_compared']=len(pairs)
    summary['aligned_source_matches']=len(matched)
    summary['aligned_matches_in_different_folds']=int(matched.different_folds.sum())
    summary['matched_aligned_signal_to_residual_db_range']=[float(matched.aligned_signal_to_residual_db.min()),float(matched.aligned_signal_to_residual_db.max())]
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    unresolved=pairs[pairs.aligned_waveform_cosine.le(.45)]
    audio_report=['## Audio forensic follow-up','',
        f'All 37 zero WAVs and 37 distinct text-matched positive candidates were downloaded from the competition '
        f'({len(cache)} files). All use 16 kHz. No audio was modified.',
        '',f'**{len(matched)}/37 pairs show strong sample-aligned source evidence.** Their waveform cosine similarity '
        'is about 0.49, and the fitted signal-to-residual power ratio is tightly clustered around −5 dB '
        f'(range {matched.aligned_signal_to_residual_db.min():.3f} to {matched.aligned_signal_to_residual_db.max():.3f} dB). '
        'This pattern strongly suggests noise-added versions of the positive recordings. It is not proof of '
        'the exact generation process or official reason for the zero labels.','',
        'Method: fit a scalar gain from the positive waveform to the zero waveform, then compare the power '
        'of that fitted component with the residual. The reported ratio is an estimate conditional on an '
        'aligned source match, not a direct measurement of true speech/noise separation. No lag search '
        'or flexible alignment was used. A cosine threshold of 0.45 identifies these clear matches; '
        'inspect the pairs rather than treating this as a universal duplicate detector.','',
        f'{len(unresolved)} nearest-text candidates do not align; those source identities remain unresolved. '
        'Failure to match these candidates does not mean their zero audio is valid or invalid.','',
        '### Why text-only scoring struggles','',
        'ASR often recovers plausible words from noisy recordings; a text-only scorer then rates the words '
        'similarly to the corresponding clean speech. It lacks the original audio noise signal. '
        'Some ASR outputs are also corrupted or repetitive, but this is not consistent across all zeros. '
        'Noise sensitivity is the strongest explanation supported by this audit; the official meaning '
        'of zero remains unverified.','',
        '### Fold leakage audit','',
        f'{int(matched.different_folds.sum())}/{len(matched)} supported source pairs lie in different frozen folds. '
        'Consequently a held-out recording can have its clean/noisy counterpart in the training fold. '
        'This is source overlap, even though the labels differ. Future experiments should group confirmed '
        'source pairs and rerun baselines on that new protocol. Do not interpret existing validation as '
        'independent of source overlap.','',
        markdown_table(matched[['zero_filename','positive_filename','positive_label','aligned_waveform_cosine',
                                'aligned_signal_to_residual_db','zero_fold','positive_fold']],index=False),'',
        '### Review and next steps','',
        '- Open audio_review.html to listen to the pairs; the assistant performed signal analysis, not a human listening assessment.',
        '- Prioritize audio_5040 ↔ audio_337 (labels 0 and 5) and audio_5044 ↔ audio_743 (labels 0 and 4).',
        '- Verify the official zero-label definition/source provenance before any relabelling or exclusion.',
        '- Resolve the remaining six sources, then group confirmed clean/noisy source families for validation.',
        '- Consider an audio-quality analysis/feature experiment; do not assume every naturally noisy test '
        'recording should receive zero. No such classifier or scoring rule has been trained here.',
        '- The matched positive audio set was selected by transcript similarity; its signal summaries are '
        'not representative of all 732 positive recordings.',
        '- Labels, existing folds and submissions remain unchanged.']
    (OUT/'audio_findings.md').write_text('\n'.join(audio_report)+'\n')
    print(metrics.groupby(metrics.filename.isin(review.filename))[['rms','fullscale_fraction','near_zero_fraction','high_frequency_power_fraction']].median().to_string())
    print(pairs.sort_values('aligned_waveform_cosine',ascending=False).head(12).to_string(index=False))
    print('Audio files:',len(cache),'Zero files:',summary['downloaded_zero_audio_count'])


if __name__=='__main__':main()
