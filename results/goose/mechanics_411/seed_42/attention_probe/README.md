# mechanics_411 attention gate

This directory contains the reviewed single-case attention-instrumentation
gate for Wan2.1-I2V baseline and `+PhaseLock`, seed 42.

## Execution

- baseline: GOOSE Slurm job `4627`, completed in `00:10:22`;
- PhaseLock: GOOSE Slurm job `4628`, completed in `00:10:23`;
- each arm recorded 150 arrays and 60 probe events;
- all arrays have the expected shapes and finite values;
- the large latent snapshot tensors remain on GOOSE and are not committed.

## Non-invasive check

The instrumented final-video SHA-256 equals the corresponding uninstrumented
video SHA-256 for both arms:

- baseline: `7b48ae0d22ea19522496bb0a3c0ca0242dd6aa9e1057a92ca93b2d5f01f7e0e6`;
- PhaseLock: `300ba3c98d8518768294e378e5b38715dd99f5929808b99e606064aba8eeb24a`.

This establishes that the probe records attention without changing the
generated video for the gated sample.

## Preliminary reading

The two arms have identical earliest recorded attention, then diverge during
denoising. PhaseLock produces a tighter entity-centroid path in later sampled
steps, consistent with its role as a preservation constraint. The sparse
primary-to-secondary self-attention becomes mostly frame-diagonal in both
arms; PhaseLock does not create a uniquely clear collision-transfer pattern in
this one sample. Together with the reviewed two-step video, this supports
testing the **formation-versus-preservation** split on a larger paired set.

These observations are descriptive and single-case only. Attention values are
not causal evidence, and no model-level conclusion should be drawn until the
eight-case contrast run is reviewed.
