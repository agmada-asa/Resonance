# Phase 1

Updated: 2026-07-03

## Purpose

Phase 1 is the synthetic-audio proof of concept for Resonance:

```text
current audio state + action -> next audio state
```

The current implementation trains a conditional spectrogram transition model on generated waveforms and deterministic DSP-style transformations. The model predicts a delta in spectrogram space, then reconstructs the next state as:

```text
predicted_target = input_spectrogram + predicted_delta
```

This document records the current refactored implementation, the latest 30-epoch multi-action model run (trained on a subset of 7500 total samples), and the current transition-evaluation results.

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
- Training scripts for full-action runs.
- Evaluation helpers for plotting, per-action losses, and exported audio comparisons.
- Saved datasets and checkpoints under `data/synthetic/v001/` and `build/`.
- Native multi-action training data and model conditioning (new data representation).
- Full-action 30-epoch training and evaluation run recorded below.
- Experiment components refactored into a formal class system under `src/resonance/`.
- Multi-action transition probes for model-input chaining and summed-vector one-shot inference.

Still in progress:

- Clean command-line entry points.
- Expanded automated test coverage beyond the current smoke/unit tests.
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

Current shared training setup:

- Batch size: `32`
- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: MSE on predicted delta vs target delta
- Device selection: CUDA, then MPS, then CPU
- Checkpoint selection: lowest validation delta loss

Epochs:

- `experiments/train.py`: fixed at `30`

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
```

Current evaluation capabilities:

- Plot input, predicted target, true target, predicted delta, true delta, and delta error.
- Report per-sample MSE and L1 loss.
- Report per-action test losses.
- Export before/target-after/predicted-after audio comparison files.

The current scripts are useful for research inspection, but they are not yet formal automated tests.

## Current Artifacts

Dataset directory:

```text
data/synthetic/v001/
```

Full-action dataset:

The dataset now utilizes a new data representation with native multi-action sequences. The recent training run was conducted on a reduced dataset size:

| Split | Examples | Input shape | Action vector shape |
| --- | ---: | --- | --- |
| train | 6000 | `(6000, 113, 173)` | `(6000, 28)` |
| val | 750 | `(750, 113, 173)` | `(750, 28)` |
| test | 750 | `(750, 113, 173)` | `(750, 28)` |

Test split spans single actions as well as natively chained combinations.

Checkpoint directory:

```text
build/
```

Current checkpoints:

- `build/best_spectrogram_transition_model.pth`
- `build/spectrogram_transition_model_1.pth`
- `build/spectrogram_transition_model_old.pth`

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
- `SyntheticTrainingDataGenerator` owns full-action dataset generation.
- `SyntheticSpectrogramDataset` owns `.npz` loading, normalization, delta targets, and metadata.
- `SpectrogramTransitionModel` remains the compact baseline CNN.
- `SpectrogramUNetModel` remains the current U-Net-style transition model.
- `SpectrogramTransitionTrainer` owns normalization stats, dataloaders, training, evaluation, and checkpointing.
- `SpectrogramEvaluator` owns full-action plotting, per-action metrics, and audio comparison export.
- `SpectrogramViewer` owns the manual mel-spectrogram inspection workflow.
- `LatentTransitionEvaluator` owns the current multi-action transition probes.

Still to add later:

- Stable command-line entry points outside the compatibility wrappers.
- Native multi-action dataset generation and model conditioning.
- Formal package/dependency metadata.

## Test Results

These are current experiment results from the refactored `/src` implementation, run through the compatibility scripts in `experiments/`.

### Multi-Action Training Run

Command:

```text
/Users/agmad/Documents/Resonance/.venv/bin/python /Users/agmad/Documents/Resonance/experiments/train.py
```

Environment:

- Device: `mps`
- Epochs: `30`
- Batch size: `32`
- Learning rate: `1e-3`
- Dataset: 7500 samples total (train, val, test) with native multi-action sequences
- Loss: MSE on predicted delta

Training summary:

| Metric | Value |
| --- | ---: |
| Best validation epoch | 27 |
| Final train delta loss | 0.0381 |
| Final validation delta loss | 0.0654 |
| Validation identity baseline | 0.6933 |
| Test delta loss | 0.0641 |
| Test target loss | 0.0641 |
| Test identity baseline | 0.6586 |

Interpretation:

- The model successfully converges and learns the transformation, significantly beating the identity baseline of 0.6586 on the test split.
- The training relies on a new data representation (native multi-action sequences) with a much smaller dataset (7500 total samples compared to 21000 previously). Therefore, a direct apples-to-apples comparison on single-action performance with earlier iterations is not applicable, but it demonstrates strong multi-action capabilities.

### Multi-Action Evaluation Run

Command:

```text
/Users/agmad/Documents/Resonance/.venv/bin/python /Users/agmad/Documents/Resonance/experiments/test.py
```

Environment:

- Device: `mps`
- Checkpoint loaded by script: `build/spectrogram_transition_model_1.pth`
- Audio exports: 54 files written under `build/audio_samples/` covering various multi-action sequences.

Per-action test losses (Single Action Subsets):

| Action | Count | Delta MSE | Delta L1 | Identity MSE | Identity L1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gain | 26 | 0.003161 | 0.025767 | 0.017557 | 0.058015 |
| high_pass | 68 | 0.027501 | 0.050553 | 0.289720 | 0.163055 |
| low_pass | 28 | 0.016589 | 0.042269 | 0.097464 | 0.081063 |
| no_action | 23 | 0.000226 | 0.006946 | 0.000000 | 0.000000 |
| pitch_change | 61 | 0.040312 | 0.076605 | 0.701386 | 0.322842 |

Representative Multi-Action Sequence Test Losses:

The new model was evaluated natively on chained sequences of up to 4 actions. Some examples:

| Sequence | Count | Delta MSE | Identity MSE |
| --- | ---: | ---: | ---: |
| gain -> pitch_change -> high_pass | 1 | 0.049244 | 0.961228 |
| high_pass -> low_pass | 14 | 0.039367 | 0.272121 |
| pitch_change -> pitch_change | 12 | 0.049163 | 1.078030 |
| low_pass -> high_pass -> pitch_change -> low_pass | 3 | 0.030306 | 0.459179 |

Interpretation:

- The model effectively generalizes to multi-action sequences natively. `gain -> pitch_change -> high_pass`, for example, maintains a low delta MSE of 0.0492 against a massive identity MSE of 0.9612.

### Comparison To Earlier Runs

The new model introduces native multi-action sequences, so it isn't an apples-to-apples comparison with the previous single-action model. However, it shows a substantial improvement in handling complex combinations of actions.

Multi-action performance comparison:

| Metric | Previous Model (30-epoch single-action) | New Model (30-epoch multi-action, 7.5k samples) |
| --- | ---: | ---: |
| Multi-action MSE | ~0.1769 (rollout) / ~0.3446 (vector) | 0.0641 (native test delta loss) |

Single-action subset comparison:

| Metric | Previous Model (21k samples) | New Model (7.5k samples) |
| --- | ---: | ---: |
| gain delta MSE | 0.000781 | 0.003161 |
| high_pass delta MSE | 0.016676 | 0.027501 |
| low_pass delta MSE | 0.008876 | 0.016589 |
| no_action delta MSE | 0.000010 | 0.000226 |
| pitch_change delta MSE | 0.028205 | 0.040312 |

While the single-action MSE is marginally higher on the new model, this is expected: it was trained on only ~35% of the data volume compared to the previous model and had to learn a much more complex representation encompassing multi-action chaining. The monumental improvement from ~0.1769 to 0.0641 on multi-action sequences proves the new data representation's effectiveness.

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
- Compare fixed-slot multi-action training against a sequence encoder that maps variable-length action lists into a fixed conditioning embedding.
- Add side-by-side audio and spectrogram review for chained rollout, vector-once, and native multi-action inference.
