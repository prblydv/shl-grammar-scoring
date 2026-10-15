# 1. Reproducing the work

1. **Access and limitations.** Obtain the competition data through your authorized Kaggle account. This public repository contains source, configuration, aggregate charts, and model-level summaries. It does not distribute private data, prediction rows, or trained weights. Exact reruns need the recorded private feature stores and ordered base-model prediction exports. This publication was checked locally; it does not represent a newly executed full GPU rerun.

2. **Recorded training hardware.** Most later base-model exports record NVIDIA GeForce RTX 4090; the audited AST fusion and CNN-large exports record RTX 5090, rented through Vast.ai. The original transcription workflow used Kaggle T4 ×2. CPU suffices for final prediction-level ensemble reconstruction. See [per-run hardware/software records](../reports/training_hardware.json).

3. **Create an environment.** Python 3.12 and a CUDA-compatible PyTorch build are appropriate for the recorded GPU pipeline. Install PyTorch for the actual machine separately, then install the relevant dependency groups. The lightweight final ensemble check was also executed in the existing local Python environment.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt -r requirements-training.txt
   python -m pip install jupyterlab ipykernel
   # For transcription only:
   python -m pip install -r requirements-transcription.txt
   # For the optional broader CatBoost research screen only:
   python -m pip install -r requirements-research.txt
   ```

4. **Prepare private inputs.** The original training workflow expects the repository at `/workspace/shl`. Its paths can be edited in the settings/notebook when running elsewhere. It expects:

   ```text
   /workspace/shl/
     data/text_input/
       train.csv                 # filename,label,text
       test.csv                  # filename,label,text; test labels are placeholders
       source_pairs.csv          # zero_filename,positive_filename
       submission_template.csv   # filename,label; actual 216-row test order
       manifest.json             # {"files": {"train.csv": "SHA256", ...}}
     audio_embeddings/
       shl_all_channels.h5
       shl_spectrogram_time_v1.h5
   ```

   Generate source pairs from verified waveform/source analysis; do not invent speaker identities. `source_pairs.csv` may be empty with its header for an ungrouped-source fixture, but that would not reproduce the recorded grouped experiment. HDF5 extraction entry points live in `experiments/audio_channels/` and `experiments/spectrogram_time/`. The historical extractor notebooks include their dependency/setup cells; upstream model access and licences apply.

5. **Run the neural workflow deliberately.** Open `experiments/vast_text/train_and_submit.ipynb`, review the fully commented settings, choose a new experiment name, set `USE_EXISTING_RUN=False`, and keep upload disabled. The recovered code lives in its adjacent `configurable_v4/` directory. The resolved original settings are in `configs/ast_resolved.json`; use these to distinguish measured settings from automatic requested values. The last cell offers CV-ensemble, best-fold, or fresh full-data inference/export. Running GPU training is a separate compute operation; publication preparation does not launch it.

6. **Reconstruct the final experimental blend without repeating model search.** Build two private NPZ inputs with `x`, `test`, `y`, `names`, and `test_names`. Shapes are `(769,14)`/`(216,14)` for the hurdle and `(769,12)`/`(216,12)` for the retained joint. The numeric training labels must match, and rows and model columns must follow the archived inventory order. Names are string arrays readable with `allow_pickle=False`. The first prediction column is the residual anchor, so column permutation changes the result. [Model order](../reports/ensemble_inventory.json) records the required identities.

   ```bash
   python scripts/predict_final.py \
     --hurdle-inputs /private/path/hurdle_inputs.npz \
     --joint-inputs /private/path/joint_inputs.npz \
     --output outputs/submission.csv \
     --reference /private/path/03_hurdle60_joint40.csv
   ```

   The optional reference verifies all predictions within 1e−10. CSV bytes can vary with numerical libraries; the original immutable file hash is in `reports/winner_manifest.json`. This command fits the saved full-fit parameters, not a new parameter search, and does not compute an independent OOF score or submit to Kaggle.

7. **Historical research scripts.** Scripts with date suffixes preserve the actual experiment implementations, including their original `artifacts/`, Downloads, and frozen-metadata assumptions. They are research provenance, not a portable one-command replacement for missing private artifacts. In particular, the dated fixed-weight script also checks the earlier 25% hurdle submission, and the broader screen needs optional CatBoost. Use `predict_final.py` for the bounded reconstruction path above.

8. **Checks.** Existing synthetic integration suites are in `experiments/vast_text/configurable_v4/test_*.py`; they exercise model plumbing, fold-local normalization, resume, and output integrity with tiny generated data. They do not establish competition-model quality. `scripts/validate_publication.py` checks notebook schemas, Python syntax, documentation image/link targets, and publication hashes. `scripts/render_figures.py` regenerates the charts from the published aggregate data without requiring competition records.

9. **Submission details.** Use the Kaggle username `prabalsinghx` recorded in the project only after confirming it matches the account used in the actual form submission. Upload the exact intended output file from the private archive. The public repository documents both the non-hurdle comparisons and the experimental 0.3141 result; publishing source does not choose or send the hiring form on your behalf.
