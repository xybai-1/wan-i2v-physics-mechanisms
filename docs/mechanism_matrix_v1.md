# Formation-versus-preservation mechanism matrix v1

This is a diagnostic study, not a leaderboard submission. Its purpose is to
locate whether a physics failure is already present at the earliest denoising
steps (formation failure) or appears only after a useful early prior is formed
(preservation failure).

## Frozen design

- Backbones: Wan2.1-I2V-14B, CogVideoX-5B-I2V, and LTX-Video-0.9.5-I2V.
- Benchmarks: PhyGround, PhyGenBench, and Physics-IQ Verified.
- Strata within every benchmark: single-entity motion, two-entity contact, and
  a multi-entity or multi-stage causal chain.
- Seeds: 42, 123, and 777.
- Arms: baseline and PhaseLock.
- Full sampling: 50 steps.
- Decoded trajectory checkpoints: 1, 2, 4, 8, 25, and 50.
- Total independent final-video cells: `3 * 9 * 3 * 2 = 162`.

The six checkpoint videos are observations of a cell's one sampling trajectory;
they are not counted as six additional experiments.

## Input policy

Each case has one conditioning image shared by every backbone, seed, and arm.
PhyGround and Physics-IQ Verified use their official public first/switch frames.
PhyGenBench is T2V-only, so three pre-action frames are generated once with a
pinned FLUX.1-schnell revision and frozen seeds. The actual generation prompt is
the original benchmark prompt. The pre-action image prompt is never passed to a
video model. `input_lock.json` records every conditioning image's SHA-256.

This design tests matched I2V mechanisms; it is not an official cross-model
ranking because the backbones have different native frame counts, frame rates,
and resolutions.

## Instrumentation

Every transformer attention module is scanned at every full-trajectory call.
To bound storage and avoid changing the attention computation, hooks retain
only deterministic sampled scalar reductions of module inputs and outputs:
mean, standard deviation, RMS, residual RMS, and input/output cosine. Raw
attention matrices and raw latent snapshots are not saved.

The decoded checkpoints and the two-step PhaseLock prior are videos. The
per-step latent-delta trace records cosine alignment with that prior. Attention
activation summaries are correlational diagnostics; representative cases must
still be tested later with activation patching before making a causal claim.

## Deployment gates

1. CPU setup downloads pinned snapshots and public inputs.
2. One H100 creates and locks the three PhyGenBench pre-action frames.
3. A compatibility gate runs both arms for one matched collision case on all
   three backbones. It validates pipeline-specific latent layouts, PhaseLock,
   six checkpoint decodes, all-layer statistics, and video decoding.
4. Only if all six gate cells pass are two full shards submitted. The gate cells
   are part of the 162-cell matrix and are skipped rather than regenerated.

The full shards are grouped by backbone so a pipeline is loaded once and reused
across its cells. Shard 0 runs Wan; shard 1 runs CogVideoX followed by LTX-Video.

## Acceptance criteria

For every cell:

- `run.json`, `few_step.mp4`, `final.mp4`, and all six checkpoint videos exist;
- every MP4 is non-empty and decodable;
- `latent_delta_trace.json` has 50 finite records;
- attention metadata reports all discovered attention modules and nonzero records;
- every array in `attention_activation_stats.npz` is finite;
- no raw attention matrix or latent tensor is present.

After completion, compare baseline and PhaseLock by complexity stratum and model.
Only 3--6 representative cases should proceed to expensive activation patching
and solver/trajectory intervention.
