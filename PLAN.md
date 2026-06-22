# Plan

## Goal

Build a small audio world model that learns:

```text
current audio state + action -> next audio state
```

The project should begin with simple generated audio, then later explore text-conditioned actions and real audio datasets.

## Phase 1: Synthetic Audio

Start with generated signals:

- sine waves
- square waves
- saw waves
- noise
- simple chords or melodies

Apply simple effects:

- gain
- low-pass filter
- high-pass filter
- distortion
- delay
- reverb

Represent each example as:

```text
input audio
action
target audio
```

The first model can work on mel spectrograms rather than raw waveform.

## Phase 2: Rollout

Evaluate whether the model can predict multiple future steps:

```text
x_0 -> x_1 -> x_2 -> x_3
```

Compare the predicted chain with the true DSP chain.

This is the main research hook.

## Phase 3: Text Actions

Add language control later:

```text
"make it darker" -> low-pass style action
"turn it up" -> gain action
"make it distorted" -> distortion action
```

CLAP can be used as a text feature extractor, but the project should learn its own action embedding space.

## Phase 4: Real Audio

Once the synthetic setup works, try licensed datasets such as NSynth, Slakh, MedleyDB, ESC-50, FSD50K, or similar sources.

Use real clips as inputs, then apply known effects to create targets.

## Notes

Keep the repo flexible. Add folders, docs, configs, and experiment tracking only when they become useful.

