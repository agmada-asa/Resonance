# Phase 1

Updated: 2026-06-29

## Purpose

Phase 1 is the synthetic-audio proof of concept for Resonance:

```text
current audio state + action -> next audio state
```

The current implementation trains a conditional spectrogram transition model on generated waveforms and deterministic DSP-style transformations. The model predicts a delta in spectrogram space, then reconstructs the next state as:

```text
predicted_target = input_spectrogram + predicted_delta
```

This document records the current refactored implementation, the latest 30-epoch full-action model run, and the current transition-evaluation results.

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
- Full-action 30-epoch training and evaluation run recorded below.
- Experiment components refactored into a formal class system under `src/resonance/`.
- Multi-action transition probes for model-input chaining and summed-vector one-shot inference.

Still in progress:

- Clean command-line entry points.
- Expanded automated test coverage beyond the current smoke/unit tests.
- Native multi-action training data and model conditioning.
- Machine-readable test-result reports beyond captured console logs.

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

- `experiments/train.py`: fixed at `30`
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
| train | 16800 | `(16800, 113, 173)` | `(16800, 8)` |
| val | 2100 | `(2100, 113, 173)` | `(2100, 8)` |
| test | 2100 | `(2100, 113, 173)` | `(2100, 8)` |

Full-action test split by action:

| Action | Test examples |
| --- | ---: |
| gain | 290 |
| high_pass | 581 |
| low_pass | 284 |
| no_action | 302 |
| pitch_change | 643 |

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
- `build/spectrogram_transition_model_old.pth`
- `build/best_spectrogram_transition_model_pitch_only.pth`
- `build/spectrogram_transition_model_pitch_only_1.pth`

Recent evaluation exports:

- `build/audio_samples/` contains 54 exported full-action audio comparison files from the latest evaluation run.
- `build/latent_space_transition_samples/` contains 10 chained-rollout comparison plots.
- `build/latent_space_vector_chaining_samples/` contains 10 summed-vector one-shot comparison plots.
- `build/logs/` contains the 2026-06-29 training and evaluation console logs.

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
    transition_evaluator.py
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
- `LatentTransitionEvaluator` owns the current multi-action transition probes.

Still to add later:

- Stable command-line entry points outside the compatibility wrappers.
- Native multi-action dataset generation and model conditioning.
- Formal package/dependency metadata.

## Test Results

These are current experiment results from the refactored `/src` implementation, run through the compatibility scripts in `experiments/`.

### Full-Action Training Run

Command:

```text
/Users/agmad/Documents/Resonance/.venv/bin/python /Users/agmad/Documents/Resonance/experiments/train.py
```

Environment:

- Device: `mps`
- Epochs: `30`
- Batch size: `32`
- Learning rate: `1e-3`
- Loss: MSE on predicted delta
- Approximate training time: `45.5 minutes`
- Log: `build/logs/train_30_epoch_2026-06-29.log`

Training summary:

| Metric | Value |
| --- | ---: |
| Best validation delta loss | 0.0123 |
| Best validation epoch | 28 |
| Best validation target loss | 0.0123 |
| Validation identity baseline | 0.4041 |
| Final train delta loss | 0.0139 |
| Final validation delta loss | 0.0128 |
| Final validation target loss | 0.0128 |
| Test delta loss | 0.0146 |
| Test target loss | 0.0146 |
| Test identity baseline | 0.3942 |

Interpretation:

- The model is substantially better than the identity baseline on the full-action test split.
- Validation performance kept improving past epoch 10 and reached its best point at epoch 28.
- Validation loss is noisy, with occasional upward spikes, but the later-epoch plateau is clearly lower than the early run.
- Final test target loss matches test delta loss because the target is reconstructed as `input + predicted_delta`.

### Full-Action Evaluation Run

Command:

```text
/Users/agmad/Documents/Resonance/.venv/bin/python /Users/agmad/Documents/Resonance/experiments/test.py
```

Environment:

- Device: `mps`
- Checkpoint loaded by script: `build/spectrogram_transition_model_1.pth`
- Audio exports: 54 files written under `build/audio_samples/`
- Log: `build/logs/eval_regular_30_epoch_2026-06-29.log`

Per-action test losses:

| Action | Count | Delta MSE | Delta L1 | Identity MSE | Identity L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gain | 290 | 0.000781 | 0.011498 | 0.025544 | 0.073116 |
| high_pass | 581 | 0.016676 | 0.034228 | 0.373939 | 0.190453 |
| low_pass | 284 | 0.008876 | 0.027884 | 0.305020 | 0.151296 |
| no_action | 302 | 0.000010 | 0.001772 | 0.000000 | 0.000000 |
| pitch_change | 643 | 0.028205 | 0.063047 | 0.803283 | 0.364099 |

Representative plotted sample losses:

| Action | Sample index | Waveform | Parameter | MSE | L1 |
| --- | ---: | --- | --- | ---: | ---: |
| gain | 44 | sine | -6.676240747807732 | 0.002283 | 0.033379 |
| high_pass | 16 | sine | 4275.492961159641 | 0.001049 | 0.007119 |
| low_pass | 15 | sine | 5215.065327464878 | 0.000743 | 0.009491 |
| no_action | 8 | sine | None | 0.000005 | 0.001076 |
| pitch_change | 0 | sine | 12 | 0.033033 | 0.103153 |

Notes:

- `gain` and `no_action` are currently the easiest actions.
- `pitch_change` remains the hardest action by delta MSE, but it improved substantially from the previous 10-epoch run.
- `high_pass` is still harder than gain/no-op/low-pass, but the 30-epoch model reduced its average test delta MSE below 0.02.
- Exported audio samples cover waveform and chord examples for sine, square, and sawtooth inputs.

### Multi-Action Transition Probes

The model is still trained on single-action examples. These probes evaluate whether repeated model-input chaining or summed-vector one-shot inference is more useful for multi-action sequences.

Chained rollout command:

```text
PYTHONPATH=src .venv/bin/python experiments/test_model_latent_space.py
```

Vector-once command:

```text
PYTHONPATH=src .venv/bin/python experiments/test_model_vector_chaining.py
```

Logs:

- `build/logs/eval_latent_rollout_30_epoch_2026-06-29.log`
- `build/logs/eval_vector_chaining_30_epoch_2026-06-29.log`

Chained rollout losses:

| Case | Loss |
| --- | ---: |
| sine_880_gain_up_then_pitch_down | 0.057101 |
| square_chord_low_pass_then_gain_down | 0.060026 |
| sine_660_pitch_round_trip_with_gain | 0.081249 |
| square_220_gain_up_then_low_pass | 0.108455 |
| sawtooth_chord_pitch_up_then_high_pass | 0.109507 |
| sine_440_pitch_up_then_gain_down | 0.117047 |
| sine_chord_gain_up_then_pitch_up | 0.172205 |
| sawtooth_330_high_pass_then_gain_up | 0.199412 |
| square_330_low_pass_then_pitch_up | 0.240475 |
| sawtooth_180_pitch_down_then_low_pass | 0.623570 |

Vector-once comparison:

| Metric | Value |
| --- | ---: |
| Average chained-input MSE | 0.176905 |
| Average vector-once MSE | 0.344634 |
| Average vector/chained ratio | 1.95x |

Per-case vector-once comparison:

| Case | Chained MSE | Vector-once MSE | Ratio |
| --- | ---: | ---: | ---: |
| sine_880_gain_up_then_pitch_down | 0.057101 | 0.096639 | 1.69x |
| square_chord_low_pass_then_gain_down | 0.060026 | 0.142641 | 2.38x |
| sine_660_pitch_round_trip_with_gain | 0.081249 | 0.173900 | 2.14x |
| sawtooth_chord_pitch_up_then_high_pass | 0.109507 | 0.201053 | 1.84x |
| sine_440_pitch_up_then_gain_down | 0.117047 | 0.318946 | 2.72x |
| sine_chord_gain_up_then_pitch_up | 0.172205 | 0.390146 | 2.27x |
| square_330_low_pass_then_pitch_up | 0.240475 | 0.407244 | 1.69x |
| sawtooth_330_high_pass_then_gain_up | 0.199412 | 0.427768 | 2.15x |
| square_220_gain_up_then_low_pass | 0.108455 | 0.551294 | 5.08x |
| sawtooth_180_pitch_down_then_low_pass | 0.623570 | 0.736707 | 1.18x |

Interpretation:

- Chaining model inputs is more accurate by MSE on every current probe case.
- Summed-vector one-shot inference can visually preserve sharper spectrogram detail because it avoids repeated model-output smoothing, but it is out of distribution for the current single-action one-hot training setup.
- Proper multi-action training should preserve ordered action slots or encode an action sequence into a fixed conditioning embedding, rather than summing single-action one-hot vectors and hoping the existing model extrapolates.

### Comparison To Earlier Runs

Direct 10-epoch predecessor run:

| Metric | 10 epochs | 30 epochs |
| --- | ---: | ---: |
| Final test delta loss | 0.0204 | 0.0146 |
| gain delta MSE | 0.002002 | 0.000781 |
| high_pass delta MSE | 0.019183 | 0.016676 |
| low_pass delta MSE | 0.011560 | 0.008876 |
| no_action delta MSE | 0.000039 | 0.000010 |
| pitch_change delta MSE | 0.043291 | 0.028205 |

Multi-action comparison against the pre-30-epoch checkpoint used for earlier probes:

| Metric | Earlier checkpoint | 30-epoch checkpoint |
| --- | ---: | ---: |
| Average chained rollout MSE | 0.358226 | 0.176905 |
| Average vector-once MSE | 0.654557 | 0.344634 |
| Vector/chained ratio | 1.83x | 1.95x |

Older saved artifact comparison, using `build/spectrogram_transition_model_old.pth`:

| Metric | Old artifact | 30-epoch checkpoint |
| --- | ---: | ---: |
| Total test delta loss | 0.291647 | 0.0146 |
| gain delta MSE | 0.050254 | 0.000781 |
| high_pass delta MSE | 0.736598 | 0.016676 |
| low_pass delta MSE | 0.173683 | 0.008876 |
| no_action delta MSE | 0.000017 | 0.000010 |
| pitch_change delta MSE | 0.187543 | 0.028205 |
| Average chained rollout MSE | 0.505189 | 0.176905 |
| Average vector-once MSE | 0.954446 | 0.344634 |

This comparison is less direct than the 10-epoch comparison because the old artifact predates the current training state, but it is useful as a saved-checkpoint regression reference.

### Automated Tests

The current formal test suite is `unittest`-based.

Smoke checks run after the `/src` refactor:

| Date | Command | Result | Notes |
| --- | --- | --- | --- |
| 2026-06-26 | `python -m compileall -q src experiments` | Pass | Byte-compiled source and wrappers |
| 2026-06-26 | import `resonance.*` classes | Pass | Checked config/actions/data/features/models/training/evaluation imports |
| 2026-06-26 | import `experiments.*` wrappers | Pass | Checked old action/train/test imports still resolve |
| 2026-06-26 | U-Net dummy forward pass | Pass | Output shape `(2, 1, 113, 173)` |
| 2026-06-26 | dataset loader smoke check | Pass | Loaded 1050 test examples with shape `(1, 113, 173)` |
| 2026-06-29 | `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests` | Pass | Ran 8 tests in 0.489s |
| 2026-06-29 | `PYTHONPATH=src .venv/bin/python -m pytest` | Blocked | `pytest` is not installed in the venv |

## Next Checkpoint

Before calling Phase 1 complete:

- Add dependency and environment documentation.
- Add `pytest` or standardize explicitly on `unittest` in the development setup.
- Add tests around action encoding, spectrogram shape, dataset loading, model forward-pass shape, and transition-evaluator output shape.
- Add non-interactive evaluation commands that write structured JSON/CSV summaries for CI and comparison tracking.
- Train a native multi-action model with fixed ordered action slots, for example four 8-value action slots padded with `no_action`.
- Compare fixed-slot multi-action training against a sequence encoder that maps variable-length action lists into a fixed conditioning embedding.
- Add side-by-side audio and spectrogram review for chained rollout, vector-once, and native multi-action inference.
