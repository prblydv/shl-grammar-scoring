"""Audit existing labels/transcripts without modifying data or training a scorer."""
from collections import Counter
from pathlib import Path
import difflib
import hashlib
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/zero_label_analysis'


def markdown_table(df, index=True):
    df = df.reset_index() if index else df
    rows = [[str(v).replace('|','\\|') for v in row] for row in df.itertuples(index=False,name=None)]
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |',
                      '| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows])


def features(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    counts = Counter(grams)
    repeated = sum(n-1 for n in counts.values()) / max(1, len(grams))
    return {'words': len(words), 'repeated_4gram_fraction': repeated,
            'most_common_4gram_count': max(counts.values(), default=0),
            'punctuation_fraction': sum(c in '.,!?;:' for c in str(text))/max(1,len(str(text)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/'data/labels/train.csv', ROOT/'data/labels/test.csv',
             ROOT/'artifacts/transcripts-large-v3-dataset/transcripts_large_v3.csv',
             ROOT/'artifacts/transcription/transcripts.csv',
             ROOT/'artifacts/deberta-reference-dataset/frozen_folds.csv',
             ROOT/'experiments/deberta_no_inputs/deberta_no_inputs.ipynb']
    train, test, large, small, folds = [pd.read_csv(p) for p in paths[:5]]
    expected = {(split, name) for split, df in [('train',train),('test',test)] for name in df.filename}
    integrity = {}
    for name, df in [('large_v3',large),('small',small)]:
        keys = list(zip(df.split, df.audio_id))
        assert len(keys)==len(set(keys)), 'Duplicate split/audio IDs'
        assert set(keys)==expected, 'Missing/extra transcripts'
        integrity[name] = {'rows':len(df), 'missing_text':int(df.transcript.fillna('').str.strip().eq('').sum()),
                           'recorded_errors':int(df.error.fillna('').ne('').sum()),
                           'path_filename_mismatches':int(sum(Path(p).name!=n for p,n in zip(df.audio_path,df.audio_id))),
                           'path_split_mismatches':int(sum(Path(p).parent.name!=s for p,s in zip(df.audio_path,df.split)))}
    assert train.filename.is_unique and folds.filename.is_unique
    assert len(folds)==len(train)
    joined = train.merge(large[large.split.eq('train')],left_on='filename',right_on='audio_id',validate='one_to_one')
    joined = joined.merge(folds.rename(columns={'label':'fold_label'}),on='filename',validate='one_to_one')
    assert joined.label.eq(joined.fold_label).all(), 'Frozen labels changed'
    joined['id_number'] = joined.filename.str.extract(r'(\d+)')[0].astype(int)
    joined['group'] = np.where(joined.label.eq(0),'zero','positive')
    f = pd.DataFrame([features(t) for t in joined.transcript])
    joined = pd.concat([joined.reset_index(drop=True),f],axis=1)
    joined['words_per_second'] = joined.words/joined.duration_s
    small_train = small[small.split.eq('train')][['audio_id','transcript']].rename(columns={'transcript':'small_transcript'})
    joined = joined.merge(small_train,on='audio_id',validate='one_to_one')
    joined['asr_word_sequence_similarity'] = [difflib.SequenceMatcher(None,a.lower().split(),b.lower().split(),autojunk=False).ratio()
                                            for a,b in zip(joined.transcript,joined.small_transcript)]
    joined['possible_repetition'] = joined.repeated_4gram_fraction.gt(.30)
    joined['normalized_text'] = joined.transcript.str.lower().str.replace(r'\s+',' ',regex=True).str.strip()
    joined['exact_duplicate_text'] = joined.normalized_text.duplicated(keep=False)
    zeros = joined[joined.label.eq(0)].copy()
    positives = joined[joined.label.gt(0)].copy()
    # Descriptive text matching only: no predictive model or label fitting.
    vec = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2)
    matrix = vec.fit_transform(joined.transcript)
    similarities = cosine_similarity(matrix[zeros.index],matrix[positives.index])
    best = similarities.argmax(axis=1)
    nearest = positives.iloc[best]
    zeros['nearest_positive_filename'] = nearest.filename.to_numpy()
    zeros['nearest_positive_label'] = nearest.label.to_numpy()
    zeros['nearest_positive_similarity'] = similarities[np.arange(len(zeros)),best]
    zeros['nearest_positive_transcript'] = nearest.transcript.to_numpy()
    # Extract only displayed top errors, never pretend this is the full OOF file.
    notebook = json.loads(paths[5].read_text())
    errors = {}
    metrics = []
    for cell in notebook['cells']:
        for output in cell.get('outputs',[]):
            plain = output.get('data',{}).get('text/plain',[])
            plain = ''.join(plain) if isinstance(plain,list) else str(plain)
            for line in plain.splitlines():
                m = re.match(r'\s*\d+\s+(audio_\d+\.wav)\s+0\.0\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',line)
                if m: errors[m[1]] = float(m[2])
                m = re.match(r'\s*\d+\s+(ridge|neural|ridge_blend)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$',line)
                if m and 'pearson_correlation' in plain:
                    metrics.append(dict(model=m[1],rmse=float(m[2]),correlation=float(m[3]),zero_rmse=float(m[4]),positive_rmse=float(m[5])))
    zeros['displayed_oof_prediction'] = zeros.filename.map(errors)
    zeros['review_priority'] = np.where(zeros.displayed_oof_prediction.notna(),'displayed_large_error',
                                       np.where(zeros.possible_repetition,'repetition','other_zero'))
    zeros.sort_values(['displayed_oof_prediction','nearest_positive_similarity'],ascending=False).to_csv(OUT/'zero_review.csv',index=False)
    joined.drop(columns='normalized_text').to_csv(OUT/'all_train_audit.csv',index=False)
    fold_counts = pd.crosstab(joined.fold,joined.group)
    fold_counts.to_csv(OUT/'fold_counts.csv')
    summary = joined.groupby('group').agg(rows=('filename','size'),duration_median=('duration_s','median'),
        words_median=('words','median'),repetition_flags=('possible_repetition','sum'),
        repeated_fraction_median=('repeated_4gram_fraction','median'),asr_similarity_median=('asr_word_sequence_similarity','median'))
    summary.to_csv(OUT/'group_summary.csv')
    neural = next((m for m in metrics if m['model']=='neural'),None)
    error_share = len(zeros)*neural['zero_rmse']**2/(len(joined)*neural['rmse']**2) if neural else None
    zero_ids = sorted(zeros.id_number.tolist())
    test_ids = test.filename.str.extract(r'(\d+)')[0].astype(int)
    audit = {'inputs_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'integrity':integrity,'train_rows':len(train),'zero_rows':len(zeros),'positive_rows':len(positives),
        'zero_contiguous':zero_ids==list(range(min(zero_ids),max(zero_ids)+1)),
        'zero_id_range':[min(zero_ids),max(zero_ids)],'positive_id_range':[int(positives.id_number.min()),int(positives.id_number.max())],
        'test_id_range':[int(test_ids.min()),int(test_ids.max())],
        'zero_exact_60_0746875_seconds':int(np.isclose(zeros.duration_s,60.0746875,rtol=0,atol=1e-8).sum()),
        'positive_exact_60_0746875_seconds':int(np.isclose(positives.duration_s,60.0746875,rtol=0,atol=1e-8).sum()),
        'zero_nearest_similarity_ge_0_65':int(zeros.nearest_positive_similarity.ge(.65).sum()),
        'displayed_zero_predictions_available':len(errors),'metrics':metrics,'zero_fraction_of_squared_error':error_share,
        'raw_competition_audio_available_locally':False,'official_zero_definition_verified':False}
    (OUT/'summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for group,color in [('positive','steelblue'),('zero','darkorange')]:
        d=joined[joined.group.eq(group)]
        axes[0].hist(d.duration_s,bins=np.linspace(0,100,31),density=True,alpha=.5,label=group,color=color)
        axes[1].hist(d.repeated_4gram_fraction,bins=np.linspace(0,1,21),density=True,alpha=.5,label=group,color=color)
    axes[0].set(xlabel='Recorded duration (seconds)',ylabel='Density',title='Duration distribution (0–100s view)')
    axes[1].set(xlabel='Repeated 4-gram fraction',ylabel='Density',title='Text repetition diagnostic')
    axes[2].scatter(joined.id_number,joined.label,s=10,alpha=.5)
    axes[2].set(xlabel='Filename numeric ID (audit only)',ylabel='Label',title='Label/source block')
    for ax in axes[:2]: ax.legend()
    fig.tight_layout();fig.savefig(OUT/'zero_label_diagnostics.png',dpi=160);plt.close(fig)
    fraction=f'{100*error_share:.1f}%' if error_share is not None else 'not available'
    lines=['# Zero-label audit — 11 October 2026','',
        'See [audio forensic findings](audio_findings.md) for the subsequent original-WAV analysis and '
        '[audio review](audio_review.html) to listen to zero/positive candidates.','',
        '## Main finding','',
        'Zero labels are perfectly confounded with a separate contiguous filename block. '
        'The transcript and duration evidence warrants an audio/source audit before changing training or labels. '
        'Neither corruption nor incorrect labels is proven.','',
        f'- {len(zeros)} of {len(train)} train recordings have label zero ({100*len(zeros)/len(train):.2f}%).',
        f'- All zeros occupy IDs {min(zero_ids)}–{max(zero_ids)}; all positives occupy IDs {audit["positive_id_range"]}.',
        f'- Test IDs span {audit["test_id_range"]}; filename numbering alone must not be used to infer quality.',
        f'- Zero recordings account for approximately {fraction} of neural OOF squared error, using rounded saved metrics.',
        f'- {audit["zero_exact_60_0746875_seconds"]}/{len(zeros)} zero recordings have exactly 60.0746875s recorded duration; '
        f'{audit["positive_exact_60_0746875_seconds"]}/{len(positives)} positives do.',
        '- Transcript keys match labels exactly by (split, audio_id); no missing/extra or duplicate transcript keys. '
        'Paths agree with filenames and splits. This rules out obvious join errors, not incorrect source transcription.',
        '', '## Group comparison','',markdown_table(summary),'',
        'A repetition flag means more than 30% of word 4-gram occurrences repeat an earlier occurrence. '
        'This is an exploratory diagnostic, not proof of ASR failure or poor grammar.','',
        '## Frozen-fold distribution','',markdown_table(fold_counts),'',
        '## Nearest positive transcript matches','',
        markdown_table(zeros.nlargest(10,'nearest_positive_similarity')[['filename','nearest_positive_filename','nearest_positive_label','nearest_positive_similarity']],index=False),'',
        'Similarity is character 3–5-gram TF-IDF cosine, fitted without targets. Shared topic and wording can yield high '
        'similarity; it does not prove shared audio or speakers. Full paired text is in zero_review.csv.','',
        '## Limits and next steps','',
        '- Full neural OOF predictions are not downloaded. Only the 15 displayed worst predictions were extracted; '
        'empty prediction fields are unavailable, not zero.',
        '- Original audio must be inspected to distinguish silence/noise, altered recordings, ASR hallucination and genuine speech.',
        '- Listen first to audio_5040, audio_5045 and audio_5048 (large errors), and audio_5043, audio_5049, '
        'audio_5050, audio_5066, audio_5069 and audio_5070 (visible repetition).',
        '- Confirm the official meaning of label zero and the source of the 5037–5073 block.',
        '- If source recordings match positive examples, inspect audio similarity before deciding grouping or exclusion.',
        '- Keep original labels unchanged. Excluding zeros would change the evaluation population and is not a solution by itself.',
        '- No new scorer was trained, no labels modified, and no submission created by this analysis.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(summary.to_string())
    print(json.dumps({k:v for k,v in audit.items() if k not in ['inputs_sha256','integrity','metrics']},indent=2))


if __name__=='__main__':
    main()
