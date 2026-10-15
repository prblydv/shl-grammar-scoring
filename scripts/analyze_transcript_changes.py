#!/usr/bin/env python3
"""Compare Small and Large-v3 outputs; measure disagreement, not true ASR error."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import unicodedata

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

from finish_transcription import validate

ROOT = Path(__file__).resolve().parents[1]


def words(text):
    text = unicodedata.normalize('NFKC', text).replace('’', "'").casefold()
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)


def repetition_fraction(tokens, n=5):
    grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return 1-len(set(grams))/len(grams) if grams else 0.0


def longest_word_run(tokens):
    longest = current = 0
    previous = None
    for token in tokens:
        current = current+1 if token == previous else 1
        longest = max(longest, current)
        previous = token
    return longest


def analyze(small_path, large_path, output):
    small, large = validate(small_path), validate(large_path)
    assert small.model_id.eq('openai/whisper-small').all()
    assert large.model_id.eq('openai/whisper-large-v3').all()
    paired = small.merge(large, on=['split', 'audio_id'], suffixes=('_small', '_large'),
                         how='outer', indicator=True, validate='one_to_one')
    assert paired['_merge'].eq('both').all()
    np.testing.assert_allclose(paired.duration_s_small, paired.duration_s_large, rtol=0, atol=1e-6)
    rows, substitutions = [], Counter()
    for row in paired.to_dict('records'):
        a, b = words(row['transcript_small']), words(row['transcript_large'])
        operations = Levenshtein.editops(a, b)
        counts = Counter(operation.tag for operation in operations)
        for operation in operations:
            if operation.tag == 'replace':
                substitutions[(a[operation.src_pos], b[operation.dest_pos])] += 1
        seconds = row['duration_s_small']
        result = {
            'split': row['split'], 'audio_id': row['audio_id'], 'duration_s': seconds,
            'raw_text_changed': row['transcript_small'] != row['transcript_large'],
            'normalized_words_changed': a != b,
            'small_words': len(a), 'large_words': len(b), 'word_count_delta': len(b)-len(a),
            'word_substitutions': counts['replace'], 'word_insertions': counts['insert'],
            'word_deletions': counts['delete'], 'word_edits': len(operations),
            'word_disagreement_rate': len(operations)/max(len(a), 1),
            'small_repeated_5gram_fraction': repetition_fraction(a),
            'large_repeated_5gram_fraction': repetition_fraction(b),
            'small_longest_word_run': longest_word_run(a),
            'large_longest_word_run': longest_word_run(b),
            'small_words_per_minute': len(a)*60/seconds,
            'large_words_per_minute': len(b)*60/seconds,
            'small_alphabetic_uppercase_fraction': sum(c.isupper() for c in row['transcript_small'])/max(sum(c.isalpha() for c in row['transcript_small']), 1),
            'large_alphabetic_uppercase_fraction': sum(c.isupper() for c in row['transcript_large'])/max(sum(c.isalpha() for c in row['transcript_large']), 1),
            'small_sentence_punctuation': sum(row['transcript_small'].count(c) for c in '.!?'),
            'large_sentence_punctuation': sum(row['transcript_large'].count(c) for c in '.!?'),
            'transcript_small': row['transcript_small'], 'transcript_large': row['transcript_large'],
        }
        for model in ['small', 'large']:
            # Review heuristics, not validated hallucination/accuracy measurements.
            result[f'{model}_repetition_flag'] = (result[f'{model}_repeated_5gram_fraction'] >= 0.5
                                                  and result[f'{model}_words'] >= 80)
            result[f'{model}_speed_flag'] = result[f'{model}_words_per_minute'] > 300
        rows.append(result)
    changes = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    changes.to_csv(output/'word_changes.csv', index=False)
    pd.DataFrame([{'small_word': a, 'large_word': b, 'count': count}
                  for (a, b), count in substitutions.most_common()]).to_csv(output/'word_substitutions.csv', index=False)
    report = {
        'recordings': len(changes), 'split_counts': changes['split'].value_counts().to_dict(),
        'raw_text_changed': int(changes.raw_text_changed.sum()),
        'normalized_word_sequences_changed': int(changes.normalized_words_changed.sum()),
        'formatting_only_changes': int((changes.raw_text_changed & ~changes.normalized_words_changed).sum()),
        'small_total_words': int(changes.small_words.sum()), 'large_total_words': int(changes.large_words.sum()),
        'word_count_percent_change': float(100*(changes.large_words.sum()/changes.small_words.sum()-1)),
        'word_edits': int(changes.word_edits.sum()),
        'word_substitutions': int(changes.word_substitutions.sum()),
        'word_insertions': int(changes.word_insertions.sum()), 'word_deletions': int(changes.word_deletions.sum()),
        'corpus_word_disagreement_percent_using_small_as_reference': float(100*changes.word_edits.sum()/changes.small_words.sum()),
        'median_recording_word_disagreement_percent': float(100*changes.word_disagreement_rate.median()),
        'small_max_words': int(changes.small_words.max()), 'large_max_words': int(changes.large_words.max()),
        'small_repetition_flags': int(changes.small_repetition_flag.sum()),
        'large_repetition_flags': int(changes.large_repetition_flag.sum()),
        'repetition_flags_resolved': int((changes.small_repetition_flag & ~changes.large_repetition_flag).sum()),
        'repetition_flags_introduced': int((~changes.small_repetition_flag & changes.large_repetition_flag).sum()),
        'small_above_300_words_per_minute': int(changes.small_speed_flag.sum()),
        'large_above_300_words_per_minute': int(changes.large_speed_flag.sum()),
        'small_total_sentence_punctuation': int(changes.small_sentence_punctuation.sum()),
        'large_total_sentence_punctuation': int(changes.large_sentence_punctuation.sum()),
        'zero_transcription_errors_both': True,
        'caution': 'Word disagreement uses Small as a comparison reference, not ground truth. '
                   'It is not word error rate or a percentage accuracy improvement. '
                   'Repetition and speed thresholds identify recordings for audio review; they do not prove hallucinations.',
        'normalization': 'Unicode NFKC, apostrophe normalization, casefolding, punctuation-free word tokenization.',
        'repetition_flag_definition': 'At least 80 words and at least 50% duplicate five-gram occurrences.',
    }
    (output/'transcript_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    summary = []
    for split, group in changes.groupby('split'):
        summary.append({'split': split, 'rows': len(group), 'word_sequences_changed': int(group.normalized_words_changed.sum()),
                        'small_words': int(group.small_words.sum()), 'large_words': int(group.large_words.sum()),
                        'word_disagreement_percent': 100*group.word_edits.sum()/group.small_words.sum(),
                        'small_repetition_flags': int(group.small_repetition_flag.sum()),
                        'large_repetition_flags': int(group.large_repetition_flag.sum())})
    pd.DataFrame(summary).to_csv(output/'split_summary.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(changes.small_words, changes.large_words, alpha=0.4, s=12)
    upper = max(changes.small_words.max(), changes.large_words.max())
    axes[0].plot([0, upper], [0, upper], 'k--', lw=1)
    axes[0].set(xlabel='Small: normalized word count', ylabel='Large-v3: normalized word count', title='Length changes')
    axes[1].hist(100*changes.word_disagreement_rate, bins=40)
    axes[1].set(xlabel='Word disagreement (%)', ylabel='Recordings', title='Disagreement, not accuracy')
    axes[2].bar(['Small', 'Large-v3'], [report['small_repetition_flags'], report['large_repetition_flags']])
    axes[2].set(ylabel='Flagged recordings', title='Repetition review heuristic')
    fig.tight_layout(); fig.savefig(output/'transcript_comparison.png', dpi=160); plt.close(fig)
    examples = changes.nlargest(8, 'word_edits')
    example_path = output/'transcript_examples.md'
    with example_path.open('w') as handle:
        handle.write('# Largest transcript disagreements\n\nThese excerpts require audio review to establish which words are correct.\n')
        for row in examples.to_dict('records'):
            handle.write(f"\n## {row['split']}/{row['audio_id']}\n\nSmall: {row['small_words']} words; Large-v3: {row['large_words']} words; word edits: {row['word_edits']}.\n\n")
            for name in ['small', 'large']:
                text = row[f'transcript_{name}']
                handle.write(f"**{name}:** {text[:900]}{' … [excerpt shortened]' if len(text)>900 else ''}\n\n")
    print(json.dumps(report, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    return changes, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--small', type=Path, default=ROOT/'artifacts/whisper_comparison/reference/small_transcripts.parquet')
    parser.add_argument('--large', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/whisper_comparison/analysis')
    args = parser.parse_args()
    analyze(args.small, args.large, args.output)
