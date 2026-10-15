# SHL Grammar Scoring

![Recorded rank, public RMSE, and training-data development RMSE](docs/assets/results_card.svg)

**Training-data evaluation RMSE: 0.477589*** · **3 grouped folds** · **769 labelled recordings**  
**Recorded public leaderboard RMSE: 0.3141 · Recorded public rank: 5th**

**Write-up updated: 15 October 2026 at 23:19 BST (22:19 UTC).** The rank refers to the saved 15 October leaderboard snapshot; its exact capture time was not retained. This timestamp dates the write-up, not a fresh leaderboard check.

> [!NOTE]
> **About the training-data score.** The training-data RMSE combines the final ensemble's evaluation predictions across all three folds. It is one error measure over all 769 labelled recordings. The average of the three fold RMSEs is slightly different: **0.477121**.
>
> The base models' saved out-of-fold predictions were reused, so the complete system was not retrained inside every ensemble-validation split. Public leaderboard feedback also informed the final blend weights. Treat this as a development score. An end-to-end in-sample training RMSE was not recorded.

**Final submission notebook:** [final_submission.ipynb](final_submission.ipynb) explains the method, reports each fold score, and includes code to verify the RMSE and rebuild the submission using the private saved predictions.

**Learning to assess spoken grammar from text, speech representations, and spectrograms.**

1. **Research approach.**

   I first reviewed related Kaggle grammar and language-scoring competitions to learn what worked and where the approaches struggled. I then read arXiv papers on audio processing and transformer representations. From this research, I built an experiment plan: start with a simple baseline, compare transcription methods, test text and audio features, and combine models with different strengths. I recorded the results and used validation scores and error analysis to guide each next step.

   **Related Kaggle problems → audio research → documented hypotheses → experiments → error analysis**

   The target is one grammar score on a **0–5 scale**. Earlier essay-scoring work informed the text experiments; audio research informed the speech features. Because those tasks use different inputs and targets, I tested the ideas on SHL before drawing conclusions.

   | Research question | How it shaped an experiment | What the evidence showed |
   |---|---|---|
   | What transfers from earlier language-scoring competitions? | Test DeBERTa text features and different ways to summarize its output, inspired by the saved Feedback Prize example | Text models gave a starting point; their errors motivated adding audio |
   | Is the final speech-transformer layer the only useful representation? | Combine middle- and final-layer features from WavLM, HuBERT, and wav2vec2 | Used in the audio branch; a controlled comparison with final-layer-only features is still needed |
   | Does better transcription automatically improve grammar scoring? | Change Whisper Small to Large-v3 while keeping the scoring model and folds fixed | Public RMSE improved, but local RMSE worsened |
   | Can combining models resolve their remaining errors? | Test regularized combinations and examine shared errors | Some combinations helped; adding models did not reliably help |

   Follow the [research journey](docs/RESEARCH.md), [annotated references](docs/REFERENCES.md), and [what worked / what did not](docs/WHAT_WORKED.md) for the reasoning and recorded results.

2. **Results and how they were measured.** These are saved experiment results. Public scores refer to the leaderboard at the time of submission.

   | Approach | Local evaluation | Recorded public RMSE |
   |---|---:|---:|
   | Whisper Small + word/character TF-IDF + Ridge | 1.11795, six-fold pooled OOF | 0.6607 |
   | Whisper Large-v3 + the same Ridge pipeline | 1.12624, six-fold pooled OOF | 0.6522 |
   | DeBERTa + pooled speech features + AST | 0.49524, three-fold grouped pooled OOF | 0.3318, full-data refit |
   | Elastic Net prediction ensemble | 0.48560, ensemble development | 0.3204 |
   | Smooth quadratic prediction ensemble | 0.48128, ensemble development | 0.3169 |
   | 11-model quadratic/kernel ensemble | 0.47893, ensemble development | 0.3166 |
   | Experimental hurdle/joint blend | 0.47759, ensemble development | **0.3141** |

   > [!IMPORTANT]
   > **Recorded public result: 0.3141 RMSE · 5th place** at the saved 15 October 2026 snapshot. This is the lowest observed public RMSE in my records; private performance is unknown.

   **How to read the local scores:** out-of-fold (OOF) predictions come from base models that held out the recording being scored. Pooled RMSE combines errors across all folds. Ensemble-development scores evaluate the model-combination stage using saved base OOF predictions, with the limitations noted above. The early baselines use six folds; later neural experiments use three grouped folds. Local and public scores come from different evaluation sets.

   **Why the best public score needs caution:** the hurdle model separates zero from positive scores, and the zero-labelled recordings have unusual noise/source patterns. It may learn a pattern specific to this dataset. I therefore retain the earlier non-hurdle ensemble as a comparison and do not claim that the 0.3141 result will generalize best to new speakers or recording conditions.

3. **The pipeline at a glance.**

   ![Multimodal architecture and experimental ensemble](docs/assets/architecture.svg)

   The audited AST model combines three views of each recording:

   | Branch | Input and role |
   |---|---|
   | Text | Whisper Large-v3 transcribes the speech; DeBERTa-v3-large encodes the words in context |
   | Speech features | WavLM, HuBERT, and wav2vec2 supply 11 layer summaries; a gated neural network combines them |
   | Spectrogram | An Audio Spectrogram Transformer (AST), pretrained on AudioSet, processes a time–frequency view of the full recording |

   The three outputs feed a regression head that predicts a score between 0 and 5. The audio network's feature gates and the later ensemble's zero/nonzero gate serve different purposes. The [architecture guide](docs/ARCHITECTURE.md) gives the dimensions and training settings.

4. **The decisions that mattered.**

   1. Start with a cheap baseline. Larger speech-recognition models did not consistently improve scoring.
   2. Preserve grammar errors, repetitions, and hesitations. Correcting the transcript could remove useful scoring information.
   3. Check related recordings before choosing validation splits. Keep known clean/noisy pairs together and document pairs discovered later.
   4. Combine different views of speech. Text captures the words; audio features retain information the transcript misses.
   5. Review errors before adding complexity. A small set of recordings caused much of the error, and strong models shared many misses.
   6. Test whether models add useful differences. More models and wider searches did not reliably improve the result.
   7. Report leaderboard gains and generalization limits together. A dataset-specific correction may improve this benchmark without transferring to new data.

5. **Explore the research.**

   | Document | What it explains |
   |---|---|
   | [What worked and what did not](docs/WHAT_WORKED.md) | Results, mixed outcomes, unsuccessful directions, and untested ideas |
   | [Research references](docs/REFERENCES.md) | Papers, prior competition code, and the decisions they inform |
   | [Research and experiment journey](docs/RESEARCH.md) | Starting assumptions, ASR comparison, feature design, unsuccessful directions |
   | [Architecture and encoding](docs/ARCHITECTURE.md) | Backbones, feature dimensions, pooling, temporal encoding, fusion, training settings |
   | [Validation and generalization](docs/GENERALIZATION.md) | Source groups, leakage controls, regularization, zero-case concerns, remaining risks |
   | [Ensemble mechanics](docs/ENSEMBLES.md) | OOF alignment, quadratic/kernel corrections, learned hurdle experiment, exact blend |
   | [Error analysis](docs/ERROR_ANALYSIS.md) | Error concentration, score slices, ASR failures, calibration, controlled next steps |
   | [Reproduction guide](docs/REPRODUCING.md) | Environments, private inputs, training entry point, final prediction reconstruction |

6. **Results in pictures.**

   ![Local and public scores](docs/assets/score_comparison.svg)

   ![AST training and validation curves](docs/assets/learning_curves.svg)

   More diagnostic figures and their interpretation are in the [visual results gallery](docs/RESULTS.md).

7. **Find the code.**

   | Location | Purpose |
   |---|---|
   | `experiments/vast_text/train_and_submit.ipynb` | Commented settings and training/export workflow recovered from the audited AST run |
   | `experiments/vast_text/configurable_v4/` | Actual training implementation, pooling, feature stores, AST/CNN branches, and existing integration checks |
   | `experiments/audio_channels/` | Reusable speech-encoder feature extraction |
   | `experiments/spectrogram_time/` | Full-recording mel extraction and temporal coordinates |
   | `experiments/whisper_small/`, `experiments/whisper_large_v3/`, `experiments/crisper_whisper*/` | Transcription experiments |
   | `experiments/tfidf_ridge*/` | Baseline notebooks |
   | `scripts/` | Historical analysis and ensemble implementations, plus a portable final-prediction entry point |
   | `experiments/source_snapshots/` | Recovered per-run implementations, mapped by `reports/run_sources.json` |
   | `configs/`, `reports/` | Audited settings, model-level summaries, selected meta-model fits, and source hashes |

8. **Reproducibility and scope.** The public repository contains code, settings, aggregate results, and source hashes. Competition data, transcripts, individual predictions, model weights, and credentials remain private. Authorized reviewers need the original inputs and saved prediction exports to reproduce the submission. Public notebook outputs are cleared. The [source manifest](reports/source_manifest.json) records the code versions and publication edits. Some older research scripts expect the original folder layout; the reproduction guide explains those dependencies.

9. **AI assistance.** AI tools helped with implementation, analysis, and documentation. The repository records the modelling choices, evidence, and limitations so that reviewers can inspect the work and discuss the reasoning.

10. **Training hardware.**

    | Stage | Hardware used | Evidence / scope |
    |---|---|---|
    | Original Whisper transcription workflow | Kaggle NVIDIA Tesla T4 ×2 | Two-worker transcription notebook and experiment records |
    | Most later scorer runs | NVIDIA GeForce RTX 4090, on Vast.ai | Per-fold hardware recorded in run exports |
    | Audited AST fusion and CNN-large runs | NVIDIA GeForce RTX 5090, on Vast.ai | Per-fold hardware recorded in run exports |
    | TF-IDF baselines and final prediction-level ensembles | CPU | No GPU required once transcripts/base predictions are available |

    The audited AST run used BF16 precision and gradient checkpointing to manage memory. Training/evaluation batches were 8/16; accumulating two training batches gave an effective batch size of 16. GPU probes checked which batch sizes fit. See the [recorded GPU and software versions](reports/training_hardware.json). Complete timing, cost, and machine specifications were not retained.

11. **Re-create the pipeline.**

    1. Clone this repository and obtain the competition inputs through your authorized Kaggle account. Private data and checkpoints are not bundled here.
    2. Install a suitable CUDA PyTorch build, then `requirements.txt` and `requirements-training.txt`; use the separate transcription requirements when regenerating transcripts.
    3. Rebuild or restore the text bundle and the selected audio/spectrogram HDF5 features. Preserve original labels and join by split plus filename.
    4. Review `configs/ast_original_settings.py` and `configs/ast_resolved.json`. Open `experiments/vast_text/train_and_submit.ipynb`, set the data paths and a new experiment name, and run grouped CV before a fresh full-data refit.
    5. Retain aligned OOF/test exports from each base run. The original per-run implementations/configurations are mapped in `reports/run_sources.json`; the 12- and 14-model column inventories are in `reports/ensemble_inventory.json`.
    6. To rebuild the documented experimental final blend from private ordered prediction arrays, run:

       ```bash
       python scripts/predict_final.py \
         --hurdle-inputs /private/path/hurdle_inputs.npz \
         --joint-inputs /private/path/joint_inputs.npz \
         --output outputs/submission.csv
       ```

    7. Follow the [full reproduction guide](docs/REPRODUCING.md) for input schemas, paths, dependency groups, numeric reference verification, and the difference between historical research scripts and the portable reconstruction command. Reproducing the score requires private inputs; a fresh GPU retrain is not guaranteed to yield byte-identical predictions.

12. **References and research motivation.**

    The [annotated references](docs/REFERENCES.md) explain how layer-wise speech research, DeBERTa, the speech backbones, AST, and validation literature informed the experiments. They also describe the earlier Feedback Prize example. Each reference is connected to a design choice, with completed experiments separated from ideas still awaiting a test.
