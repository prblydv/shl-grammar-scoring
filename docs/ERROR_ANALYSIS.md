# 1. What the errors taught me

1. **Average error hides concentrated failures.** The audited AST fusion run has pooled OOF RMSE 0.49524 and MAE 0.38544. Of 769 recordings, 103 have absolute error at least 0.75 and 42 at least 1. The worst 50 account for 37.60% of total squared error. Reviewing hard examples alongside same-label easy controls is therefore more informative than only watching aggregate RMSE.

2. **Score slices show shrinkage, not a proven calibration defect.** Low positive labels tend to be overpredicted and high labels underpredicted. However, prediction-decile mean label-minus-prediction differences range only from about −0.101 to +0.093. Conditioning on the true label can create apparent compression even for reasonable conditional-mean predictions. Blanket expansion of every score was not justified by this audit.

3. **Several models share the same misses.** Strong models have residual correlations around 0.90–0.93, with 57 difficult recordings shared across four models. Additional highly correlated predictions have limited ability to correct those examples. This motivated checking residual diversity rather than collecting models solely for their standalone scores.

4. **ASR disagreement needs case-level review.** Only 13 training transcripts exceed the audited 256-token limit, and only one of those is among the large-error cases. Truncation is therefore not a general explanation. Large disagreement, repetition loops, and long initial silence can identify review candidates, but model disagreement is not transcription accuracy and does not establish that a label is wrong.

5. **Acoustic shortcuts were not supported as general rules.** Duration, quietness, and transcript disagreement show weak associations with error after controlling descriptively for label. Fillers do not automatically imply poor grammar. Listener difficulty does not establish an accent-based scoring rule. Original labels and raw evidence are preserved.

6. **Training and frontend limits remain.** Best AST fold epochs are 9, 11, and 16. Training losses fall much lower than validation error, consistent with a generalization gap; optimizer-update training MSE is not directly equivalent to evaluation-mode RMSE. The HTK frontend differs from AST's native frontend. Zeroing padded tails is not an AST attention mask. These are plausible ablation targets, not proven causes of the observed misses.

7. **Zero-case gains need separate accounting.** In an earlier Elastic-to-quadratic comparison, zero-case development RMSE fell from approximately 0.27190 to 0.13871, while positive-case RMSE changed from 0.49396 to 0.49231. That gain largely concerns the unusual-zero subset. This observation motivated the hurdle experiment and also explains why I am cautious about treating its benchmark gain as general grammar modelling progress.

8. **Next controlled tests.** Lower only AST LR first; then isolate audio width, compare transcript inputs on identical folds, test the native frontend, and consider temporal sequence features. Rebuild all known source groups and compare positive-only results before relying on the zero-case correction. These are proposed checks, not completed results. The numerical analysis code is [research_ast_errors.py](../scripts/research_ast_errors.py); its record-level inputs remain private.
