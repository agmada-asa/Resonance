# Resonance

Resonance is an early-stage research project about small **audio world models**: models that learn how audio changes when an action is applied.

```text
current audio state + action -> next audio state
```

The first version will start small: generated tones, simple melodies, and predictable DSP effects such as gain, filters, distortion, delay, and reverb. The aim is not to generate full songs. The aim is to learn controllable audio transformations and test whether those transformations remain stable over multiple steps.

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

## First Milestone

Build the smallest useful demo:

1. Generate simple synthetic audio.
2. Apply known effects to create before/after pairs.
3. Train a conditional model on spectrograms.
4. Compare one-step predictions against the real DSP target.
5. Run a short multi-step rollout and measure how quickly errors grow.

## Later Directions

- Add text-conditioned actions with CLAP, such as `"make it muffled"` or `"make it brighter"`.
- Move from synthetic audio to licensed open-source audio datasets.
- Try better conditioning, latent audio representations, or sequence models if rollout fails.

## Repository Layout

```text
.
|-- README.md
|-- PLAN.md
|-- data/
|-- experiments/
|-- notebooks/
`-- src/
```

This repository is intentionally minimal for now.

## Status

Scaffold only. No implementation code has been added yet.
