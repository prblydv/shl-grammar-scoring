# 1. Architecture and encoding

This page follows one recording through the audited model: speech-to-text, three feature branches, and a score prediction. A **backbone** is a pretrained model used to extract features; **pooling** summarizes a sequence of features into one vector. Exact settings are retained below.

1. **Inputs and target.** The working dataset has 769 training and 216 test recordings. The target is a continuous grammar rating from 0 to 5. Every modality joins by `(split, filename)` because train/test filenames can overlap. Original labels, including zeros, are retained.

2. **Transcription.** Whisper Small established the initial baseline. Whisper Large-v3 provided the original text input for the audited AST fusion model. CrisperWhisper was investigated as a verbatim alternative, with empty failed transcriptions retained rather than silently dropping labelled rows. Transcripts are model-generated observations, not human reference text. The scoring pipeline preserves their grammar and disfluencies.

3. **Text branch.** `microsoft/deberta-v3-large` tokenizes the transcript into subword IDs and attention masks. The audited run used a resolved maximum length of 256 tokens and dynamic padding. All 24 encoder blocks were trainable. Masked last-layer mean pooling includes attended special tokens and excludes padding. The checkpoint revision was `64a8c8eab3e352a784c658aef62be1662607476f`. Alternative implemented poolers include CLS, learned token attention, concatenated layer means, learned layer weighting, signed GeM, and bidirectional token LSTM; availability is not proof each option improved the final model.

4. **Speech feature branch.** Pretrained WavLM Base Plus, HuBERT Base, and wav2vec2 Base produce frame representations. The selected recording-level channels are:

   | Encoder | Layer summaries used | Width |
   |---|---|---:|
   | WavLM | layer 4 mean/std, layer 8 mean/std, layer 12 mean | 5 × 768 |
   | HuBERT | layer 4 mean, layer 12 mean/std | 3 × 768 |
   | wav2vec2 | layer 4 mean, layer 12 mean/std | 3 × 768 |

   Concatenation gives **8,448 features**. Imputation and standardization fit on each training fold. A **10-layer gated MLP**, hidden width **2,048**, projects these into **256 dimensions**, with dropout **0.005**. Its learned feature gates are not zero-label classifiers. The encoder summaries are extracted features; this branch does not jointly fine-tune the original waveform encoders during fusion training.

5. **Spectrogram representation.** Audio is mono 16 kHz. Extraction uses 25 ms Hann windows, 10 ms hops, FFT size 1,024, 128 HTK mel filters over 20–8,000 Hz, and fixed-reference log power with a −100 dB floor. It preserves the full recording, including silence. Each recording stores variable-length log-mel frames, absolute frame-center timestamps, valid fractions, and separate 64-dimensional sine/cosine time coordinates. These coordinates describe time; they are not pretrained speech embeddings.

6. **Spectrogram branch.** The audited model fine-tunes `MIT/ast-finetuned-audioset-10-10-0.4593`, revision `f826b80d28226b62986cc218e5cec390b1096902`. The resolved chunk length is **1,024 frames**, about 10.24 seconds, even though the requested generic CNN setting was 256 frames. All chunks are retained. AST has native patch positions, with optional absolute chunk-time encoding added after feature projection; chunk aggregation accounts for valid duration. The branch produces **256 dimensions**. Fold-local per-band normalization adapts the saved HTK frontend. This frontend differs from AST's native Kaldi preprocessing. Tail zeros do not constitute a patch attention mask inside AST.

7. **Fusion and output.** Text, audio-vector, and spectrogram representations concatenate. The fusion block uses LayerNorm, a 256-unit projection, GELU, and dropout 0.2. The scalar regression output uses `5 × sigmoid(logit)`. Training minimizes MSE in original score units; validation and checkpoint selection use RMSE. There is no separate zero/nonzero head in this neural architecture.

8. **Audited training settings.**

   | Setting | Value |
   |---|---|
   | Validation | 3 grouped folds, score-bin stratification |
   | Fold/model seeds | 42; model seed adds 1,009 per fold |
   | Resolved train/eval batch | 8 / 16 |
   | Gradient accumulation | 2, effective batch 16 |
   | Text / audio MLP / AST / head LR | 2e−5 / 5e−5 / 1e−4 / 1e−4 |
   | Optimizer | AdamW, weight decay 0.01 |
   | Schedule | 10% linear warmup, then linear decay |
   | Precision and memory | BF16, gradient checkpointing |
   | Epoch cap / patience | 20 / 3 |
   | Gradient norm cap | 10 |
   | Best fold epochs | 9 / 11 / 16 |
   | Full-data refit | 11 epochs, fresh initialization |

9. **Inspect the implementation.** See [original settings](../configs/ast_original_settings.py), [resolved settings](../configs/ast_resolved.json), [training model](../experiments/vast_text/configurable_v4/train_worker.py), [audio branch](../experiments/vast_text/configurable_v4/audio_features.py), [spectrogram branch](../experiments/vast_text/configurable_v4/spectrogram_features.py), and [pooling implementations](../experiments/vast_text/configurable_v4/pooling.py).
