# Resonance

Resonance is an early-stage research project about small **audio world models**: models that learn how audio changes when an action is applied.

```text
current audio state + action -> next audio state
```

The first version starts with generated tones and predictable DSP effects. The aim is not to generate full songs. The aim is to learn controllable audio transformations and test whether those transformations remain stable over multiple steps.

## Core Idea

A normal one-step predictor learns:

```text
sine wave + low-pass filter -> filtered sine wave
```

The more interesting test is rollout:

```text
state_0 + action_1 -> predicted_state_1
predicted_state_1 + action_2 -> predicted_state_2
predicted_state_2 + action_3 -> predicted_state_3
```

If the model can keep producing reasonable future audio states from its own previous predictions, it starts to behave like a small world model rather than just an audio effect predictor.

## Current Phase

Phase 1 is implemented as a synthetic-audio experiment pipeline.

Implemented:

- Generate sine, square, and sawtooth waveforms.
- Apply deterministic actions: `no_action`, `gain`, `pitch_change`, `low_pass`, and `high_pass`.
- Convert audio into CQT spectrograms.
- Train a conditional PyTorch model to predict spectrogram deltas.
- Evaluate one-step predictions against deterministic DSP targets.
- Export audio comparison samples for inspection.
- Keep old `experiments/` commands working through wrappers around the `/src` package.

Current model:

- `SpectrogramUNetModel`, a U-Net-style transition model with action conditioning.
- Pitch-change actions include an explicit frequency-axis alignment step before the U-Net refinement.

Current result snapshot:

- Full-action training ran for 75 epochs on MPS.
- Best validation delta loss: `0.0260` at epoch 56.
- Final test delta loss: `0.0290`.
- Final test identity baseline: `0.6650`.
- Per-action evaluation shows `gain` and `no_action` are easiest; `high_pass` and `pitch_change` remain the hardest actions.

See [PHASE 1.md](PHASE%201.md) for the implementation notes, architecture, artifacts, and test-result details.

## Repository Layout

```text
.
|-- README.md
|-- PLAN.md
|-- PHASE 1.md
|-- data/
|   `-- synthetic/v001/
|-- build/
|-- experiments/
`-- src/
    `-- resonance/
        |-- actions.py
        |-- config.py
        |-- data/
        |-- features/
        |-- models/
        |-- training/
        `-- evaluation/
```

`src/resonance/` contains the class-based implementation. `experiments/` contains compatibility wrappers for the earlier script names.

## Main Components

- `AudioActionProcessor`: action encoding and DSP transforms.
- `SpectrogramTransformer`: mel and CQT feature extraction.
- `WaveformSynthesizer`: synthetic waveform generation.
- `SyntheticTrainingDataGenerator`: full-action and pitch-only dataset generation.
- `SyntheticSpectrogramDataset`: `.npz` loading, normalization, targets, and metadata.
- `SpectrogramUNetModel`: current transition model.
- `SpectrogramTransitionTrainer`: dataloaders, training, evaluation, and checkpointing.
- `SpectrogramEvaluator`: plots, per-action metrics, and audio sample export.

## Common Commands

Generate full synthetic data:

```bash
.venv/bin/python experiments/generate_synthetic_audio.py
```

Train the full-action model:

```bash
.venv/bin/python experiments/train.py
```

Evaluate the full-action model:

```bash
.venv/bin/python experiments/test.py
```

Pitch-only variants:

```bash
.venv/bin/python experiments/generate_synthetic_audio_pitch_only.py
.venv/bin/python experiments/train_pitch_only.py
.venv/bin/python experiments/test_pitch_only.py
```

## Next Directions

- Add formal automated tests.
- Add dependency and environment documentation.
- Add non-interactive evaluation commands suitable for CI.
- Add rollout evaluation across multiple predicted steps.
- Add text-conditioned actions later, such as `"make it muffled"` or `"make it brighter"`.
- Move from synthetic audio to licensed open-source audio datasets after the synthetic setup is stable.
