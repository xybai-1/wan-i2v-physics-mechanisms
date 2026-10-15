# Wan I2V Physics Mechanism Study

Mechanistic study of physical failures in image-to-video diffusion on a fixed
Wan2.1-I2V backbone and a selected PhyGround subset.

The frozen three-backbone follow-up matrix is documented in
[`docs/mechanism_matrix_v1.md`](docs/mechanism_matrix_v1.md), with its registered
cases and revisions in
[`configs/mechanism_matrix_v1.json`](configs/mechanism_matrix_v1.json).
The completed 162-run audit, mechanism analysis, compact activation summaries,
and review videos are published under
[`results/goose/mechanism_matrix_v1/`](results/goose/mechanism_matrix_v1/ANALYSIS.md).

## Scientific question

For a failed physical video, was the motion/event structure already incorrect
in the few-step prior (**formation failure**), or was a reasonable early prior
destroyed by later visual refinement (**preservation failure**)?

## First-round conditions

Only two generation conditions are valid in the first round:

1. `Wan2.1-I2V baseline`
2. `Wan2.1-I2V + PhaseLock`

Both conditions must share the same model revision, first frame, prompt, seed,
resolution, frame count, scheduler, and inference-step count.

The following are **measurements**, not four independent models:

- image/text attention balance;
- event/frame attention localization;
- early trajectory attention;
- latent-delta evolution and similarity to the 2-step prior.

AlignVid, SwitchCraft, and the physics-attention repository are upstream
references for instrumentation. Their interventions are not enabled in the
first-round causal comparison. SwitchCraft and the released physics toolkit
are T2V implementations and cannot be treated as matched I2V baselines without
an explicit, tested port.

## Repository policy

- Git stores code, manifests, small plots, summarized metrics, environment
  locks, provenance, and explicitly reviewed compact result videos.
- Model weights, caches, raw attention matrices, latent dumps, and large
  checkpoint-video collections stay on GOOSE and are backed up separately.
- Every run writes a machine-readable record containing the Git commit,
  upstream revisions, model revision, sample ID, seed, and Slurm job ID.
- No full benchmark generation is submitted before one matched smoke case
  passes artifact and reproducibility checks.

## GOOSE layout

The deployment defaults to:

```text
/home/baix2/wan-i2v-physics-mechanisms/   # this repository
/home/baix2/wan-i2v-state/
  envs/
  upstream/
  models/
  data/
  cache/
  runs/
```

See [`docs/experiment_protocol.md`](docs/experiment_protocol.md) and
[`slurm/README.md`](slurm/README.md).

