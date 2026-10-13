# PhyGround eight-case attention study

This directory contains the completed GOOSE outputs for the frozen eight-case
Wan2.1-I2V baseline/PhaseLock comparison at seed 42. The study was executed as
two Slurm shards and records attention without changing inference behavior.

## Execution record

| Job | Shard | Slurm state | Exit | Elapsed | MaxRSS |
| --- | ---: | --- | --- | --- | --- |
| 4629 | 0 | `COMPLETED` | `0:0` | 01:22:50 | 49,344,776 K |
| 4630 | 1 | `COMPLETED` | `0:0` | 01:23:19 | 50,843,120 K |

Code baseline: `ba057d143b32a5daa60a9b0f73a0a028019db427`.

## Frozen cases

The four low/high physical-reasoning contrasts are:

1. `collision_145` / `mechanics_411`
2. `0032_perspective-center_trimmed-balls-collide` / `0029_perspective-center_trimmed-ball-train`
3. `0035_perspective-center_trimmed-block-domino` / `0050_perspective-center_trimmed-dominos-with-space`
4. `0173_perspective-center_trimmed-stable-blocks` / `0182_perspective-center_trimmed-unstable-block-stack`

Each case contains `baseline` and `phaselock` arms under `seed_42/`, for 16
arms total.

## Integrity audit

All 16/16 arms passed the same audit:

- `run.json` exists.
- `attention_metadata.json` contains 60 events: 30 cross-attention and 30
  self-attention events.
- `attention_maps.npz` contains 150 finite arrays: 90 arrays with shape
  `(21, 30, 52)` and 60 arrays with shape `(21, 21)`.
- `few_step.mp4` and `final.mp4` are non-empty, decodable, and contain 81
  frames each.
- `latent_snapshots.pt` is intentionally excluded from the public repository.

Each `analysis/` directory contains the descriptive JSON summary and four
human-readable figures. All eight cross-attention panels and all eight
primary-to-secondary connectivity panels were visually reviewed after
generation; labels, arm ordering, denoising-step ordering, color bars, and
heatmaps are readable.

## Descriptive aggregate signal

Across the eight cases, the two arms are identical at denoising step 0, as
expected before PhaseLock can alter later evolution. Relative to baseline,
PhaseLock changes the mean primary-to-secondary same-frame self-attention by
`+0.00471` at step 1 and `+0.00473` at step 4. At step 4 it reduces mean
primary next-frame self-attention by `-0.00163`. By step 19, the mean event
cross-attention centroid displacement is larger by `+0.819` spatial tokens,
while primary next-frame self-attention remains lower by `-0.00166`.

These are descriptive associations from one seed and selected layers, not
causal evidence and not yet a success/failure classifier. The intended next
analysis is to align these traces with blinded video-level physical outcomes
and test the pre-registered formation-failure versus preservation-failure
split.
