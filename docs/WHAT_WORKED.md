# 1. What worked, what did not, and what remains untested

1. **Useful results, with the evidence attached.**

   | Decision | Recorded outcome | Lesson |
   |---|---|---|
   | Start with TF-IDF + Ridge | Small-ASR pooled OOF 1.11795 versus mean reference 1.24253 | Establish a cheap, explainable floor before GPU work |
   | Inspect original audio and source relationships | 31 initially supported clean/noisy pairs; later evidence supported 35 | Validation quality depends on source structure, not just stratification |
   | Preserve grammar errors and disfluencies | Prevented deliberately rewriting the signal the score should assess | Cleaner language is not automatically a better input for grammar rating |
   | Combine text, speech summaries, and spectrograms | Audited AST fusion pooled OOF 0.49524, public full-data result 0.3318 | Complementary modalities were productive; individual causal contributions need controls |
   | Regularize prediction-level combinations | Elastic Net public 0.3204; quadratic 0.3169; 11-model quadratic/RBF 0.3166 | Small residual corrections can be useful on correlated base predictions |
   | Save full experiment metadata | Labels, folds, test order, source hashes, settings, and reported RMSE could be checked | Reproducibility and alignment are part of modelling quality |

2. **Mixed results and approaches that did not win.**

   | Experiment | Recorded outcome | What I would not conclude |
   |---|---|---|
   | Whisper Small → Large-v3 with scoring fixed | Local RMSE 1.11795 → 1.12624; public 0.6607 → 0.6522 | Bigger ASR consistently improves the downstream model |
   | Bagged Elastic Net | Development 0.485635 versus original 0.485603; public 0.3206 versus 0.3204 | Bagging was an established improvement |
   | Pick the lowest-development kernel blend | Kernel-average development 0.47823, public 0.3188; all-11 RBF blend development 0.47893, public 0.3166 | Tiny development differences reliably rank public candidates |
   | Genetic MSE/Pearson mixture in an earlier screen | Public 0.3198, worse than the simpler all-11 result 0.3166 | Evolutionary search was the strongest method; final work subsequently used RMSE only |
   | Add the latest two runs to the ensemble | Three 14-model candidates scored 0.3174 / 0.3176 / 0.3170 publicly | More model exports necessarily add useful diversity |
   | Broader RMSE-only meta-model screen | Standalone Ridge/polynomial 0.47986, spline 0.49005, SVR 0.48735, ExtraTrees 0.49400, histogram boosting 0.50116, ordered CatBoost 0.49597 | More flexible meta-learners established a substantial improvement |
   | Look for a simple global audio-quality explanation | Quietness, duration, and ASR-disagreement associations were weak after label-centering | One loudness/pause rule explains grammar scores |
   | Diagnose compression by true-label slices alone | Prediction-bin calibration was comparatively close | Global score expansion is justified |

3. **The hurdle result: benchmark progress with a generalization concern.** The 60% hurdle / 40% retained joint blend reached public RMSE 0.3141. It was the lowest recorded public result, but the zero subset has unusual characteristics, the recipe was public-informed, and its development score was worse than the 35% version. I retain it as a documented experiment rather than presenting it as the best-generalizing solution. A learned gate can still capture an accidental dataset distinction. Machine learning should transfer beyond that distinction.

4. **Suggestions that were not completed improvements in this record.** Lower AST LR with all else fixed; compare AST and CNN at equal audio width; use native AST preprocessing; compare transcript sources on identical grouped folds; rebuild all 35 supported source families; and test temporal sequence representations. FLAN-T5 Small was a proposal, not a completed training result. Additional selectable waveform/AST backbones in later code are implementation options, not evidence they all ran or worked.

5. **Evidence provenance.** These conclusions were reconstructed from `EXPERIMENT_MEMORY.md` and the saved source/zero-label audits, regression follow-up, optimized-ensemble, latest-ensemble, RMSE-research, AST-error, and winning-handoff memory records. Those private working notes remain in the original workspace; public model-level summaries and selected fit records are under `reports/`. Historical observations are labelled as such, and no private-test performance is claimed.
