# Compact CQ audio world model

This experiment keeps the project as a learned world model. Deterministic gain,
pitch, low-pass, and high-pass implementations generate supervised targets only.
Inference encodes the current audio, predicts one next compact state per action,
and decodes the final predicted state. It never calls those effect
implementations.

## Model

The state packs the ragged complex CQ-NSGT representation into a phase-aware
`[2, 122, 96]` grid. It stores magnitude, initial phase, and local phase
increments. This is 7.74 times smaller than the native CQ vectors. The fixed
codec has a 23.93 dB median reconstruction ceiling on the held-out test set.

The transition is a 143,625-parameter action-conditioned residual U-Net with
four learned response heads. Pitch uses a frequency displacement whose scale is
fitted from training pairs; the fitted normalized scale is `-0.1959`. Applying
several actions means feeding each predicted state back into this same learned
transition.

The training cache contains every intermediate state:

| Split | Trajectories | One-step transitions |
|---|---:|---:|
| Train | 6,000 | 14,593 |
| Validation | 750 | 1,835 |
| Test | 750 | 1,872 |

The loss measures normalized state error, active spectral magnitude, magnitude
level, and complex phase. Rollout batches train the model on its own earlier
predictions. A compact-state range guard prevents numerical excursions during
longer rollouts.

## Held-out results

The selected checkpoint is epoch 7. On all 750 test trajectories:

| Metric, median | Learned rollout | Copy source |
|---|---:|---:|
| Spectral magnitude cosine | **0.854** | 0.264 |
| Waveform SNR | **-1.94 dB** | -6.73 dB |
| Output/target RMS ratio | **0.969** | — |

Results remain useful as action sequences get longer:

| Actions | Learned cosine | Copy-source cosine | Learned SNR | Copy-source SNR |
|---:|---:|---:|---:|---:|
| 1 | 0.974 | 0.886 | -0.91 dB | -3.41 dB |
| 2 | 0.918 | 0.416 | -2.22 dB | -6.35 dB |
| 3 | 0.713 | 0.105 | -3.17 dB | -9.88 dB |
| 4 | 0.462 | 0.004 | -3.19 dB | -23.13 dB |

One-step compact-state error improves over identity by 52.3% overall. The
action breakdown is 66.0% for pitch, 53.7% for high-pass, 13.0% for low-pass,
and 2.3% for gain.

The checkpoint is
`build/compact_cq_world_complete_energy/best_compact_world_model.pth`. Full
metrics are in `build/compact_cq_world_complete_energy/results.json`, and
representative source, target, predicted, and codec-ceiling WAVs are under
`build/compact_cq_world_complete_energy/test_audio_samples/`.

## Run it

Build the resumable intermediate-state cache:

```bash
PYTHONPATH=src .venv/bin/python experiments/build_compact_cq_trajectories.py
```

Train with the recommended seven-epoch schedule:

```bash
PYTHONPATH=src .venv/bin/python experiments/train_compact_cq_trajectories.py \
  --output-dir build/compact_cq_world_complete_energy
```

Apply the learned model to a two-second mono clip. Repeat `--action` in rollout
order, up to four actions:

```bash
PYTHONPATH=src .venv/bin/python experiments/apply_compact_cq_world_model.py \
  input.wav output.wav \
  --action pitch_change=4 \
  --action low_pass=6000
```

## Limits

This result covers synthetic two-second tones and three-note chords at 44.1 kHz.
It does not establish performance on recorded audio. Waveform SNR is still
negative because phase is less accurate than spectral magnitude. Four-step
rollouts lose detail, and nearly silent filter targets still produce large RMS
ratio outliers even though the overall median is calibrated. Gain and low-pass
state improvements are modest. Those are the next modeling targets before
moving to real audio.
