# ===== Experiment / backbone =====r
EXPERIMENT_NAME = 'spectrogram_fcnn_smdalyl_run01'  # NEW unique name when changing settings
TRANSCRIPT_DATASET = 'whisper'     # crisper | whisper (original Whisper Large-v3)
USE_EXISTING_RUN = False          # True loads that run's saved settings; ignores tweaks below

# ===== Modality selection =====
NUM_BACKBONES = 3                # 1 = single; 2 = selected pair; 3 = text + audio + spectrogram
BACKBONE_PAIR = 'text_audio'       # text_audio | text_spectrogram | audio_spectrogram; used only with 2
SINGLE_BACKBONE_MODALITY = 'spectrogram' # text | audio | spectrogram; used only with 1
# Missing pretrained text weights download automatically; cached weights are reused.
# Audio backbones below train on saved vectors; they need no pretrained download.

# ===== Backbone 1: text (ignored in audio-only mode) =====
# MODEL_NAME = 'microsoft/deberta-v3-large'
# MODEL_NAME = "microsoft/deberta-v3-base"
MODEL_NAME = "microsoft/deberta-v3-large"
# MODEL_NAME = "microsoft/deberta-v2-xlarge"
# MODEL_NAME = "FacebookAI/roberta-large"
# MODEL_NAME = "distilbert/distilbert-base-uncased"
# MODEL_NAME = "distilbert/distilbert-base-uncased"  ---- very bad gave cv of 0.4664
MODEL_REVISION = None             # Resolve and pin exact revision on first download; or specify SHA
# Encoder candidates from memory: microsoft/deberta-v3-base, microsoft/deberta-v3-large,
# microsoft/deberta-v2-xlarge, FacebookAI/roberta-large, distilbert/distilbert-base-uncased
PREPROCESSING = 'unchanged'       # Preserve grammar errors/disfluencies and all labels, including zeros

# ===== Audio inputs: actual fixed-channel names from the extraction schema =====
AUDIO_FEATURE_FILE = '/workspace/shl/audio_embeddings/shl_all_channels.h5'
# Change this path if your extracted HDF5 file is stored elsewhere on Vast.
# Add/remove entries below. Selected vectors concatenate; input width is inferred.
# To concatenate middle and final layers, select both named layer channels.
AUDIO_ENCODINGS = [
    'wavlm_layer8_mean',
    'wavlm_layer4_mean',
    'wavlm_layer4_std',
    'wavlm_layer8_std',
    'wavlm_layer12_mean',
    # 'wavlm_layer12_std',  # Also complete; enable if wanted

    'hubert_layer4_mean',
    'hubert_layer12_mean',
    'hubert_layer12_std',

    'wav2vec2_layer4_mean',
    'wav2vec2_layer12_mean',
    'wav2vec2_layer12_std',
]
# Sequence channels (word_wavlm/chunk_wavlm/phoneme_probabilities) are not fixed vectors.
AUDIO_NORMALIZATION = 'standard' # standard | none; imputation is training-fold-only

# ===== Backbone 2: extracted audio features =====
# AUDIO_BACKBONE = 'residual_mlp'
# AUDIO_BACKBONE = 'mlp'          # Feed-forward network
AUDIO_BACKBONE = 'gated_mlp'    # Learned gates blend transformed features
AUDIO_HIDDEN_DIM = 2048
AUDIO_NUM_LAYERS = 10
AUDIO_OUTPUT_DIM = 256
AUDIO_DROPOUT = 0.005
AUDIO_LR = 5e-5

# ===== Backbone 3: full-recording mel spectrograms + time encoding =====
SPECTROGRAM_FEATURE_FILE = '/workspace/shl/audio_embeddings/shl_spectrogram_time_v1.h5'
# Uncomment exactly ONE SPECTROGRAM_BACKBONE line. Choose a new experiment name for changes.
# CNN choices below train from scratch; capacities increase from tiny to large.
# SPECTROGRAM_BACKBONE = 'cnn_tiny'      # 2 convolution stages: 16, 32 channels
# SPECTROGRAM_BACKBONE = 'cnn_small'       # 3 convolution stages: 32, 64, 128 channels
# SPECTROGRAM_BACKBONE = 'cnn_medium'    # 4 stages: 48, 96, 192, 256 channels
# SPECTROGRAM_BACKBONE = 'cnn_large'     # 5 stages: 64, 128, 256, 384, 512 channels
SPECTROGRAM_BACKBONE = 'ast_audioset'  # Pretrained Audio Spectrogram Transformer; fine-tune all layers
# SPECTROGRAM_BACKBONE = 'ast_audioset_v2' # Alternative pretrained AudioSet AST checkpoint
SPECTROGRAM_REVISION = None             # Pin exact HF revision automatically for pretrained choices
SPECTROGRAM_OUTPUT_DIM = 256
SPECTROGRAM_DROPOUT = 0.0
SPECTROGRAM_LR = 1e-4                   # CNN starting point; consider 1e-5 for pretrained AST
SPECTROGRAM_CHUNK_FRAMES = 256          # CNN: 2.56s at 10ms hop; AST uses native 1024 frames automatically
SPECTROGRAM_CHUNK_BATCH_SIZE = 8        # Internal chunk batch; optimizer batch is still TRAIN_MICROBATCH
SPECTROGRAM_USE_TIME_ENCODING = True    # Separate sine/cosine coordinates added after feature projection
# Every chunk is used, including silence; last chunk is padded/masked. No random crops.
# CNN attention pools timed frames, then timed chunks. AST uses learned native patch positions
# plus optional absolute chunk time. Fold-local mel scaling is stored in the checkpoint.
# AST weights are adapted to the saved HTK frontend; original AST Kaldi preprocessing differs.

# ===== Fusion of the selected backbones =====
FUSION = 'concat'
FUSION_HIDDEN_DIM = 256
FUSION_DROPOUT = 0.2

# ===== Architecture =====
POOLING = 'mean'                  # mean | cls | attention | concat | weighted_layer | gem | lstm
CONCAT_LAST_N_LAYERS = 15         # Concatenate masked means of last N transformer blocks
WEIGHTED_LAYER_START = 0          # Zero-based first transformer layer; learned softmax weights
GEM_INITIAL_P = 3.0               # Signed masked power mean, p > 1
GEM_LEARNABLE_P = True
GEM_EPSILON = 1e-6
LSTM_HIDDEN_SIZE = 128            # One or more bidirectional token LSTMs, then masked mean
LSTM_NUM_LAYERS = 1
FREEZE_TOP_N_LAYERS = 0           # Final N output-side blocks; embeddings/pool/head remain trainable
RE_INIT_TOP_N_LAYERS = 0          # Final N blocks reset using backbone initialisation rules
# Nonzero freeze and reinitialisation overlap: use separate trials, never silently freeze reset layers.
BACKBONE_DROPOUT = 0.0
HEAD_DROPOUT = 0.0
OUTPUT_MODE = 'bounded'           # bounded = sigmoid × 5; linear trial must still pass CSV bounds
PREDICTION_MAX = 5.0
LOSS = 'mse'                     # Training objective; validation/selection ALWAYS uses RMSE

# ===== Grouped validation and seeds =====
N_FOLDS = 3
FOLD_SEED = 42
MODEL_SEED = 42
MODEL_SEED_STRIDE = 1009          # Fold model seed = MODEL_SEED + stride × zero-based fold index
VALIDATION_POLICY = 'grouped_multilabel_score_bins'  # zero, (0,2], (2,3], (3,4], (4,5]

# ===== Sequence length / measured batches =====
MAX_LENGTH = None                # None: tokenizer analysis selects supported >=95%-coverage candidate
# Explicit 512 reproduces the first run. Same length for training, validation and test.
TRAIN_MICROBATCH = None           # None probes 8,4,2,1; explicit size is also GPU-tested
EVAL_BATCH = None                 # None probes 16,8,4,2,1
GRADIENT_ACCUMULATION = None      # None computes accumulation for effective batch below
EFFECTIVE_BATCH_SIZE = 16
GRADIENT_CHECKPOINTING = True
PRECISION = 'bf16'               # Measured GPU pipeline currently supports BF16

# ===== Optimizer / schedule / stopping =====
BACKBONE_LR = 2e-5
HEAD_LR = 1e-4
LAYER_LR_DECAY = 1.0              # 1 disables layer-wise LR decay
WEIGHT_DECAY = 0.01              # Biases and one-dimensional parameters excluded
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WARMUP_RATIO = 0.10              # Linear warmup then linear decay
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 3
MIN_RMSE_IMPROVEMENT = 0.0        # 0: strictly lowest validation RMSE selects best checkpoint
MAX_GRAD_NORM = 10.0
MAX_RUNTIME_HOURS = None         # None = unlimited CV time; optional cap pauses at saved epochs
CPU_THREADS = 4
HEARTBEAT_SECONDS = 15           # Independent process heartbeat; five-minute progress warning

# ===== Optional full-data refit, used only if chosen in LAST cell =====
FULL_DATA_EPOCHS = None           # None = ceil(median(best CV epochs)); optional integer override
FULL_DATA_RUN_NAME = EXPERIMENT_NAME + '_full_data'
# Refit is fresh pretrained initialisation, fixed epochs, no held-out validation score.

# ===== Optional private Kaggle publication =====
KAGGLE_USERNAME = 'prabalsinghx'
UPDATE_EXISTING_DATASET = False
DATASET_SLUG = 'shl-' + EXPERIMENT_NAME.replace('_','-') + '-model'
# Hidden API-key prompt appears only when upload is enabled in the LAST cell.
