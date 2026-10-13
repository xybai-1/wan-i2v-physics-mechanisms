# PhyGround mechanics_411 smoke result

Date: 2026-10-13

This report records the first matched Wan2.1-I2V comparison on GOOSE. It is a
single-case mechanism smoke test, not a benchmark score or a general claim
about PhaseLock.

## Matched inputs

- Benchmark/case: PhyGround `mechanics_411`
- Prompt: `A bowling ball rolls down a polished wooden lane, hitting the pins at the end.`
- Conditioning image SHA-256: `8c12c8f4b6d18ea5460f8d0b711b8581895dda843009f468fed33cbda8070f82`
- Seed: `42`
- Backbone: `Wan-AI/Wan2.1-I2V-14B-720P-Diffusers`
- Model revision: `eb849f76dfa246545b65774a9e25943ee69b3fa3`
- PhaseLock revision: `4ebb65ba1348e754723f08a190723cf82294db19`
- Generation: 480x832, 81 frames, 16 fps, 40 denoising steps,
  guidance scale 5.0
- PhaseLock: strength 0.05, active on steps 0--19
- Study revision: `cb9c1ae9e0c98b5d8140c03cf61a21ffe4c8e2d3`

## GOOSE jobs

| Job | Arm | State | Elapsed | Outputs |
| --- | --- | --- | --- | --- |
| 4624 | Wan2.1-I2V baseline | COMPLETED | 00:10:16 | 2-step video, final video, 40-step latent trace, run metadata |
| 4625 | Wan2.1-I2V + PhaseLock | COMPLETED | 00:10:17 | 2-step video, final video, 40-step latent trace, run metadata |
| 4626 | CPU visualization | COMPLETED | 00:00:07 | contact sheet, latent plot, JSON summary, review checklist |

Canonical compute path:

`/home/baix2/wan-i2v-state/runs/smoke/mechanics_411/seed_42`

## Visual review

The shared two-step prior does not form a coherent rolling-ball trajectory.
After the early frames, the ball is not independently identifiable and the
scene collapses into a blurred lane/pin representation. There is no visible
ball--pin contact chain in the two-step diagnostic.

The 40-step baseline introduces a bowling ball, but its size and depth change
abruptly. It expands into extreme close-up frames, then becomes small again;
the pins remain mostly standing and the required impact/knock-down sequence is
not visibly completed.

The PhaseLock result keeps the lane and pin identities somewhat more stable
than the baseline, but turns into a sustained camera/object push-in. The ball
occludes the pins in late frames, and no complete, physically credible
ball--pin collision outcome is visible.

## Mechanism interpretation

Classification: **formation failure (primary), with preservation/geometry
failure in the final generations**.

The evidence does not support the hypothesis that the two-step prior already
contains the correct event and PhaseLock merely needs to preserve it. The
required event/trajectory is absent at two steps. PhaseLock therefore has no
correct early physical structure to lock. Its output is visually more stable
in some respects, but it does not recover the missing collision chain.

The latent diagnostic is consistent with different denoising behavior but is
not causal proof. Cosine similarity to the two-step latent rises from
`-0.0161` to `0.2529` for the baseline and to `0.1501` for PhaseLock. The lower
final PhaseLock similarity should not be interpreted as worse preservation by
itself because the two-step reference is already semantically inadequate.

## Boundary and next experiment

This smoke run exports latent evolution and temporal video comparisons. It
does **not** yet export attention maps. Attention instrumentation should be
added only after verifying Wan's spatial/temporal token layout, then run on
this exact frozen sample to measure:

1. image-versus-text attention across denoising steps;
2. event/frame attention around the intended impact time;
3. early trajectory attention over the ball region;
4. whether incorrect 2-step structure is already present before PhaseLock.

The next research decision should focus on early event/trajectory formation,
not simply strengthening PhaseLock's later preservation.
