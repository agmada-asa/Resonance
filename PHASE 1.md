# Phase 1

Updated: 2026-06-26

## Purpose

Phase 1 is the synthetic-audio proof of concept for Resonance:

```text
current audio state + action -> next audio state
```

The current implementation trains a conditional spectrogram transition model on generated waveforms and deterministic DSP-style transformations. The model predicts a delta in spectrogram space, then reconstructs the next state as:

```text
predicted_target = input_spectrogram + predicted_delta
```

This document records the current implementation before the planned `/src` refactor. After the refactor and test pass, use the later sections to capture the final architecture and test results.

## Current Status

Phase 1 is implemented as an experiment pipeline under `experiments/`.

Completed:

- Synthetic waveform generation for sine, square, and sawtooth signals.
- CQT spectrogram conversion for model inputs and targets.
- Action encoding for five action types.
- Synthetic train/validation/test dataset generation.
- PyTorch dataset loading and normalization.
- Baseline CNN transition model.
- Current U-Net-style spectrogram transition model.
- Training scripts for full-action and pitch-only runs.
- Evaluation helpers for plotting, per-action losses, and exported audio comparisons.
- Saved datasets and checkpoints under `data/synthetic/v001/` and `build/`.
- Full-action 75-epoch training and evaluation run recorded below.
- Experiment components refactored into a formal class system under `src/resonance/`.

Still in progress:

- Formal automated tests.
- Clean command-line entry points.
- Rollout evaluation across multiple predicted steps.
- Machine-readable test-result logs or reports.
- README update to reflect that implementation now exists.

## Implementation Snapshot

### Configuration

Core constants live in `experiments/config.py`.

| Setting | Current value |
| --- | --- |
| Sample rate | 44100 Hz |
| Duration | 2 seconds |
| FFT window | 2048 |
| Hop length | 512 |
| Mel bands | 128 |
| CQT bins per octave | 12 |
| CQT bins | 113 |
| CQT minimum frequency | 32.70 Hz |
| CQT dB floor | -80.0 dB |

The implemented training path currently uses CQT spectrograms rather than mel spectrograms.

### Data Generation

Primary generator:

```text
experiments/generate_synthetic_audio.py
```

Pitch-only generator:

```text
experiments/generate_synthetic_audio_pitch_only.py
```

Generated waveform families:

- `sine`
- `square`
- `sawtooth`

Implemented action families:

- `no_action`
- `gain`
- `pitch_change`
- `low_pass`
- `high_pass`

The full synthetic dataset generates more examples for `pitch_change` and `high_pass`, because those actions are called out in the code as harder cases with weaker performance.

### Action Encoding

Actions are encoded in `experiments/actions.py` as 8-value vectors:

```text
[no_action, gain, pitch_change, low_pass, high_pass, gain_db, semitones, cutoff]
```

Parameter normalization:

- Gain is clamped to `[-12 dB, 12 dB]` and normalized to `[-1, 1]`.
- Pitch shift is clamped to `[-12, 12]` semitones and normalized to `[-1, 1]`.
- Filter cutoff is log-normalized between `20 Hz` and `20000 Hz`.

DSP/action implementations:

- Gain uses amplitude scaling.
- Pitch change uses `librosa.effects.pitch_shift`.
- Low-pass and high-pass filters use 5th-order Butterworth filters via SciPy `sosfilt`.

### Spectrogram Representation

Spectrogram conversion lives in:

```text
experiments/audio_to_spectogram.py
```

The active path uses `audio_to_cqt`, which:

- Computes a librosa CQT.
- Converts amplitude to dB with `ref=1.0`.
- Does not use relative `top_db` clipping.
- Floors values at `CQT_DB_FLOOR`.

Current stored spectrogram shape:

```text
113 frequency bins x 173 time frames
```

### Dataset Loader

Dataset loading lives in:

```text
experiments/dataset.py
```

`SyntheticSpectrogramDataset`:

- Loads `.npz` files containing `input_spectrograms`, `target_spectrograms`, and `action_vectors`.
- Applies train-set mean/std normalization supplied by the training script.
- Stores model inputs as `[N, 1, frequency, time]`.
- Computes `target_delta = target_spectrogram - input_spectrogram`.
- Loads matching metadata from `metadata_<split>.jsonl` when present.

### Models

Baseline model:

```text
experiments/basic_model.py
```

`SpectrogramTransitionModel` is a compact encoder/action-embedding/decoder CNN that predicts a spectrogram delta.

Current model:

```text
experiments/unet_model.py
```

`SpectrogramUNetModel` is the current training model. It has:

- Two convolutional encoder blocks.
- Two max-pooling stages.
- A small MLP action encoder.
- A bottleneck that concatenates spectrogram features with the action embedding.
- Two decoder/up-sampling blocks with skip connections.
- A final `1x1` convolution for the predicted refinement.

Pitch-change handling includes an explicit frequency-axis alignment step:

- The model reads the pitch-action flag and normalized semitone value from the action vector.
- It builds an affine sampling grid.
- It shifts the input CQT along the frequency axis before encoding.
- It predicts the final delta as the alignment shift plus a learned U-Net refinement.

This is the key model-specific idea in the current implementation.

### Training

Full-action training:

```text
python3 experiments/train.py
```

Pitch-only training:

```text
python3 experiments/train_pitch_only.py
```

Current shared training setup:

- Batch size: `32`
- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: MSE on predicted delta vs target delta
- Device selection: CUDA, then MPS, then CPU
- Checkpoint selection: lowest validation delta loss

Epochs:

- `experiments/train.py`: fixed at `75`
- `experiments/train_pitch_only.py`: defaults to `50`, override with `NUM_EPOCHS`

Training also reports:

- Train delta loss
- Validation delta loss
- Validation target loss
- Validation identity baseline
- Final test delta loss
- Final test target loss
- Final test identity baseline

### Evaluation

Evaluation helpers live in:

```text
experiments/test.py
experiments/test_pitch_only.py
```

Current evaluation capabilities:

- Plot input, predicted target, true target, predicted delta, true delta, and delta error.
- Report per-sample MSE and L1 loss.
- Report per-action test losses.
- Export before/target-after/predicted-after audio comparison files.
- For pitch-only evaluation, estimate dominant CQT frequency and export a resynthesized pitch prediction.

The current scripts are useful for research inspection, but they are not yet formal automated tests.

## Current Artifacts

Dataset directory:

```text
data/synthetic/v001/
```

Full-action dataset:

| Split | Examples | Input shape | Action vector shape |
| --- | ---: | --- | --- |
| train | 8400 | `(8400, 113, 173)` | `(8400, 8)` |
| val | 1050 | `(1050, 113, 173)` | `(1050, 8)` |
| test | 1050 | `(1050, 113, 173)` | `(1050, 8)` |

Full-action test split by action:

| Action | Test examples |
| --- | ---: |
| gain | 152 |
| high_pass | 289 |
| low_pass | 149 |
| no_action | 138 |
| pitch_change | 322 |

Pitch-only dataset:

| Split | Examples | Input shape | Action vector shape |
| --- | ---: | --- | --- |
| train | 2400 | `(2400, 113, 173)` | `(2400, 8)` |
| val | 300 | `(300, 113, 173)` | `(300, 8)` |
| test | 300 | `(300, 113, 173)` | `(300, 8)` |

Checkpoint directory:

```text
build/
```

Current checkpoints:

- `build/best_spectrogram_transition_model.pth`
- `build/spectrogram_transition_model_1.pth`
- `build/best_spectrogram_transition_model_pitch_only.pth`
- `build/spectrogram_transition_model_pitch_only_1.pth`

Recent evaluation exports:

- `build/audio_samples/` contains 27 exported full-action audio comparison files from the latest evaluation run.

## `/src` Architecture

The experiment implementation has been moved into a class-based package under `src/resonance/`. The old `experiments/` files are now compatibility wrappers around the new classes, so existing commands such as `experiments/train.py` still work.

Current structure:

```text
src/resonance/
  __init__.py
  actions.py
  config.py
  data/
    __init__.py
    dataset.py
    synthetic.py
  features/
    __init__.py
    spectrogram.py
  models/
    __init__.py
    basic.py
    unet.py
  training/
    __init__.py
    trainer.py
  evaluation/
    __init__.py
    evaluator.py
    spectrogram_viewer.py
```

Class owners:

- `AudioConfig` and `ProjectPaths` hold runtime constants and repository paths.
- `AudioActionProcessor` owns action encoding and DSP transforms.
- `SpectrogramTransformer` owns mel and CQT feature extraction.
- `WaveformSynthesizer` owns waveform generation.
- `SyntheticTrainingDataGenerator` owns full-action and pitch-only dataset generation.
- `SyntheticSpectrogramDataset` owns `.npz` loading, normalization, delta targets, and metadata.
- `SpectrogramTransitionModel` remains the compact baseline CNN.
- `SpectrogramUNetModel` remains the current U-Net-style transition model.
- `SpectrogramTransitionTrainer` owns normalization stats, dataloaders, training, evaluation, and checkpointing.
- `SpectrogramEvaluator` owns full-action plotting, per-action metrics, and audio comparison export.
- `PitchOnlySpectrogramEvaluator` owns pitch-only evaluation and predicted-frequency export.
- `SpectrogramViewer` owns the manual mel-spectrogram inspection workflow.

Still to add later:

- Stable command-line entry points outside the compatibility wrappers.
- Rollout evaluation under `src/resonance/evaluation/rollout.py`.
- Formal package/dependency metadata.

## Test Results

These are current experiment results from the `experiments/` implementation. They should be replaced or extended after the `/src` refactor with formal automated test output.

### Full-Action Training Run

Command:

```text
/Users/agmad/Documents/Resonance/.venv/bin/python /Users/agmad/Documents/Resonance/experiments/train.py
```

Environment:

- Device: `mps`
- Epochs: `75`
- Batch size: `32`
- Learning rate: `1e-3`
- Loss: MSE on predicted delta
- Approximate training time: `58.7 minutes`

Training summary:

| Metric | Value |
| --- | ---: |
| Best validation delta loss | 0.0260 |
| Best validation epoch | 56 |
| Best validation target loss | 0.0260 |
| Validation identity baseline | 0.6534 |
| Final train delta loss | 0.0129 |
| Final validation delta loss | 0.0280 |
| Final validation target loss | 0.0280 |
| Test delta loss | 0.0290 |
| Test target loss | 0.0290 |
| Test identity baseline | 0.6650 |

Interpretation:

- The model is substantially better than the identity baseline on the full-action test split.
- Validation performance bottoms out around epoch 56, then fluctuates slightly through epoch 75.
- Final test target loss matches test delta loss because the target is reconstructed as `input + predicted_delta`.

### Full-Action Evaluation Run

Command:

```text
/Users/agmad/Documents/Resonance/.venv/bin/python /Users/agmad/Documents/Resonance/experiments/test.py
```

Environment:

- Device: `mps`
- Checkpoint loaded by script: `build/spectrogram_transition_model_1.pth`
- Audio exports: 27 files written under `build/audio_samples/`

Per-action test losses:

| Action | Count | Delta MSE | Delta L1 | Identity MSE | Identity L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gain | 152 | 0.000677 | 0.014228 | 0.050330 | 0.165055 |
| high_pass | 289 | 0.041855 | 0.069105 | 1.730832 | 0.933648 |
| low_pass | 149 | 0.024403 | 0.046477 | 0.345539 | 0.194024 |
| no_action | 138 | 0.000007 | 0.002019 | 0.000000 | 0.000000 |
| pitch_change | 322 | 0.045381 | 0.122087 | 0.431423 | 0.384009 |

Representative plotted sample losses:

| Action | Sample index | Waveform | Parameter | MSE | L1 |
| --- | ---: | --- | --- | ---: | ---: |
| gain | 6 | sine | 0.10908575289449196 | 0.000045 | 0.002883 |
| high_pass | 1 | sawtooth | 3276.1890815923593 | 0.326493 | 0.320811 |
| low_pass | 3 | sawtooth | 18276.52633182281 | 0.001353 | 0.010463 |
| no_action | 4 | sawtooth | None | 0.000007 | 0.001843 |
| pitch_change | 0 | square | -5 | 0.053765 | 0.157326 |

Notes:

- `gain` and `no_action` are currently the easiest actions.
- `high_pass` and `pitch_change` are still the hardest actions by delta MSE.
- The high-pass identity baseline is very high, so even the comparatively larger high-pass model error is still a large improvement over doing nothing.
- Exported audio samples cover sine, square, and sawtooth examples across several action types.

### Automated Tests

No formal automated test suite has been run yet.

Smoke checks run after the `/src` refactor:

| Date | Command | Result | Notes |
| --- | --- | --- | --- |
| 2026-06-26 | `python -m compileall -q src experiments` | Pass | Byte-compiled source and wrappers |
| 2026-06-26 | import `resonance.*` classes | Pass | Checked config/actions/data/features/models/training/evaluation imports |
| 2026-06-26 | import `experiments.*` wrappers | Pass | Checked old action/train/test imports still resolve |
| 2026-06-26 | U-Net dummy forward pass | Pass | Output shape `(2, 1, 113, 173)` |
| 2026-06-26 | dataset loader smoke check | Pass | Loaded 1050 test examples with shape `(1, 113, 173)` |

## Next Checkpoint

Before calling Phase 1 complete:

- Add dependency and environment documentation.
- Add tests around action encoding, waveform generation, spectrogram shape, dataset loading, and model forward-pass shape.
- Add non-interactive evaluation commands suitable for CI.
- Record fresh test output in this document.
