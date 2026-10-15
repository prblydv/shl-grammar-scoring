"""Move user-downloaded exports and audit text, without training or audio inference."""
from pathlib import Path
from collections import Counter
from difflib import SequenceMatcher
import hashlib
import json
import re
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'artifacts/transcription_crisper'
OUT = DEST / 'analysis'


def tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())


def features(text):
    w = tokens(text)
    grams = Counter(tuple(w[i:i+4]) for i in range(max(0, len(w)-3)))
    return dict(word_count=len(w), repeated_4gram_fraction=sum(n-1 for n in grams.values())/max(1,len(w)-3),
                filler_count=sum(x in {'um','uh','erm','hmm'} for x in w),
                immediate_repeat_count=sum(a == b for a,b in zip(w,w[1:])))


def similarity(a,b):
    return SequenceMatcher(None,tokens(a),tokens(b),autojunk=False).ratio()


def main():
    DEST.mkdir(parents=True,exist_ok=True); OUT.mkdir(exist_ok=True)
    moved=[]
    for name in ['train_transcripts_crisper.csv','test_transcripts_crisper.csv','transcription_errors.csv']:
        src=(Path.home()/'Downloads')/name; dst=DEST/name
        if src.exists():
            if dst.exists():
                raise FileExistsError(f'Will not overwrite {dst}')
            shutil.move(str(src),str(dst)); moved.append(name)
    frames=[pd.read_csv(DEST/f'{split}_transcripts_crisper.csv') for split in ['train','test']]
    crisp=pd.concat(frames,ignore_index=True)
    labels=pd.read_csv(ROOT/'data/labels/train.csv')
    expected=set(zip(['train']*len(labels),labels.filename))
    test_labels=pd.read_csv(ROOT/'data/labels/test.csv')
    expected |= set(zip(['test']*len(test_labels),test_labels.filename))
    assert len(crisp)==985 and not crisp.duplicated(['split','filename']).any()
    assert set(zip(crisp.split,crisp.filename))==expected
    train=crisp[crisp.split.eq('train')].merge(labels.rename(columns={'label':'original_score'}),on='filename',validate='one_to_one')
    assert train.label.eq(train.original_score).all()
    assert crisp.loc[crisp.split.eq('test'),'label'].isna().all()
    crisp['transcript']=crisp.transcript.fillna('')
    crisp['failed']=crisp.error.fillna('').ne('') | crisp.transcript.str.strip().eq('')
    crisp['group']=['test' if s=='test' else ('zero' if y==0 else 'positive') for s,y in zip(crisp.split,crisp.label)]
    refs={'small':ROOT/'artifacts/transcription/transcripts.csv',
          'large_v3':ROOT/'artifacts/transcripts-large-v3-dataset/transcripts_large_v3.csv'}
    paired=crisp[['split','filename','label','group','duration_s','transcript','failed','processing_time_s','attempts','timing_warning_count']].copy()
    paired=paired.rename(columns={'transcript':'crisper_text'})
    summaries=[]; changes={}
    for name,path in refs.items():
        ref=pd.read_csv(path)
        ref['filename']=ref.audio_id
        assert set(zip(ref.split,ref.filename))==expected
        paired=paired.merge(ref[['split','filename','transcript']].rename(columns={'transcript':name+'_text'}),on=['split','filename'],validate='one_to_one')
        paired[name+'_similarity']=[similarity(a,b) for a,b in zip(paired.crisper_text,paired[name+'_text'])]
        changes[name]={}
        for g,p in paired.groupby('group'):
            changes[name][g]={'rows':len(p),'raw_changed':int(p.crisper_text.ne(p[name+'_text']).sum()),
                'median_word_sequence_similarity':float(p[name+'_similarity'].median()),
                'similarity_below_0_6':int(p[name+'_similarity'].lt(.6).sum())}
    for name,col in [('crisper','crisper_text'),('small','small_text'),('large_v3','large_v3_text')]:
        fs=pd.DataFrame([features(t) for t in paired[col]])
        for f in fs:
            paired[name+'_'+f]=fs[f].values
        for g,p in paired.groupby('group'):
            summaries.append(dict(model=name,group=g,rows=len(p),
                empty_texts=int(p[col].fillna('').str.strip().eq('').sum()),
                median_words=float(p[name+'_word_count'].median()),
                total_words=int(p[name+'_word_count'].sum()),
                repetition_above_30pct=int(p[name+'_repeated_4gram_fraction'].gt(.3).sum()),
                total_fillers=int(p[name+'_filler_count'].sum()),
                clips_with_fillers=int(p[name+'_filler_count'].gt(0).sum()),
                total_immediate_repeats=int(p[name+'_immediate_repeat_count'].sum())))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'group_summary.csv',index=False)
    paired.to_csv(OUT/'paired_transcripts.csv',index=False)
    paired[paired.group.eq('zero')].to_csv(OUT/'zero_comparison.csv',index=False)
    pairs=pd.read_csv(ROOT/'artifacts/score_pattern_analysis/deeper/expanded_source_search.csv')
    pairs=pairs[pairs.supported_source_match].copy()
    source_summary={}
    for name,col in [('crisper','crisper_text'),('small','small_text'),('large_v3','large_v3_text')]:
        text=paired[paired.split.eq('train')].set_index('filename')[col]
        pairs[name+'_source_similarity']=[similarity(text[z],text[p]) for z,p in zip(pairs.zero_filename,pairs.positive_filename)]
        source_summary[name]={'supported_audio_pairs':len(pairs),
            'median_text_similarity':float(pairs[name+'_source_similarity'].median()),
            'pairs_above_0_8':int(pairs[name+'_source_similarity'].ge(.8).sum())}
    pairs.to_csv(OUT/'known_noisy_source_pairs.csv',index=False)
    report={'files_in_project':[p.name for p in DEST.glob('*.csv')], 'moved_this_invocation':moved,'rows':len(crisp),'train_rows':len(train),'test_rows':len(test_labels),
        'labels_match_source':True,'test_labels_null':True,
        'model_ids':list(crisp.model_id.dropna().unique()),'modes':list(crisp['mode'].dropna().unique()),
        'backends':list(crisp.backend.dropna().unique()),
        'failures':crisp.loc[crisp.failed,['split','filename','label','error']].to_dict('records'),
        'zero_successes':int((crisp.group.eq('zero') & ~crisp.failed).sum()),
        'positive_successes':int((crisp.group.eq('positive') & ~crisp.failed).sum()),
        'test_successes':int((crisp.group.eq('test') & ~crisp.failed).sum()),
        'text_changes':changes,'known_noisy_source_pairs':source_summary,
        'processing_time_s_total':float(crisp.processing_time_s.sum()),
        'timing_warnings_total':int(crisp.timing_warning_count.sum()),
        'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.glob('*.csv')},
        'limits':['No human transcript reference: disagreement is not word error rate.',
                  'No scoring model trained or OOF comparison: grading improvement remains untested.',
                  'Empty transcripts are recorded failures, not confirmed absence of speech.',
                  'Model processing-time sum excludes failed attempts and is not elapsed wall time.',
                  'Downloaded run_manifest.json and JSONL were not available.']}
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    examples=[]
    for g in ['positive','zero']:
        examples.extend(paired[paired.group.eq(g)].nsmallest(3,'large_v3_similarity').to_dict('records'))
    lines=['# CrisperWhisper comparison','',
           'Comparison uses stable split/filename keys. Original transcripts and labels are unchanged.',
           '',f'- {len(crisp)} rows; {len(report["failures"])} recorded failures, both zero-labelled training clips.',
           '- No inference, training, Kaggle edits, or submission was performed.',
           '- Text disagreement measures model differences, not transcription accuracy.',
           '- Positive clips containing um/uh/erm/hmm: Crisper 569/732, Large-v3 126/732, Small 15/732. This is a retention proxy, not proof every filler is correct.',
           '- Repeated four-word phrase fraction above 30% flags: Crisper 126/732 positives versus Large-v3 5/732 and Small 7/732. Flags need audio review; genuine speech repetitions can also be flagged.',
           '- Example audio_185.wav: Crisper has 448 words versus Large-v3 71, including extensive repeated playground sentences. Audio_505.wav has 478 versus 104 words, with repeated music sentences.',
           '- Zero-labelled clips still yield text in 35/37 cases; two empty-text failures do not prove absent speech or solve zero identification.',
           '- Median Crisper word count is 127 for zeros and 117 for positives; zero clips do not stand out simply as short or empty transcripts.',
           '- Better transcription can recover speech from noisy copies of positively labelled recordings. Recovering those words does not explain why their source labels are zero.',
           '- Do not replace original transcripts or change labels on this evidence. Compare downstream grading with identical valid folds before claiming a scoring gain.',
           '', '## Group summaries','',summary.to_string(index=False),'',
           '## Noisy zero/source pairs','',json.dumps(source_summary,indent=2),'',
           '## Selected disagreements (audio review required)','']
    for r in examples:
        lines += [f'### {r["filename"]}, label {r["label"]}',
                  '',f'Small: {r["small_text"]}', '', f'Large-v3: {r["large_v3_text"]}',
                  '',f'Crisper: {r["crisper_text"]}', '']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(report,indent=2));print(summary.to_string(index=False))


if __name__=='__main__':
    main()
