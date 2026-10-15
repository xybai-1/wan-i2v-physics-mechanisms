# Causal routing intervention v1

## Scope and validation

- Backbone: Wan2.1-I2V, seed 42.
- Cases: one single-object control, one pairwise collision, and two causal-chain cases.
- Intervention variants: forward, reverse-time, and same-time feature transport with equal routing mass.
- Frozen generation protocol: denoising steps 0--7, entity layers 0/1/32/37, self-attention layers 1/2/33/38, total strength 0.08.
- Smoke job: 4680. Full jobs: 4685 and 4686.
- Slurm outcome: all three jobs completed with exit code 0.
- Structural validation: 12/12 full runs valid. Every run has an exact baseline two-step-prior hash, a distinct and decodable final video, a finite 50-step latent trace, finite attention arrays, 64 diagnostic events, and 32 intervention events.

The contact-sheet row order is baseline, PhaseLock, forward, reverse-time, and same-time. Columns show frames 0, 20, 40, 60, and 80.

## Main result

The tested static feature-transport operator has causal leverage over the sampling trajectory, but it does **not** produce a forward-direction-specific physical improvement. It therefore does not yet constitute a successful training-free causal-attention method.

The intervention changed every final-video hash while leaving the two-step prior exactly unchanged. However, forward transport did not outperform the equal-mass reverse-time and same-time controls on the physical events:

| Case | Visual result |
|---|---|
| Single moving ball | Baseline-like motion is retained in all three variants. This is a useful no-regression result, but not a forward-specific gain. |
| Two-ball collision | The balls approach and interact, but identity/visibility after interaction remains unstable. Forward is not visibly better than reverse-time or same-time. |
| Domino gap chain | The first group falls, while the separated second row remains standing. None of the three routes closes the missing causal transfer across the gap. |
| Burning ball to paper | The ball briefly burns and then extinguishes without igniting the paper. All three routes preserve the original formation failure. PhaseLock instead produces a large, implausible paper deformation, not correct ignition. |

Thus, the easy single-object behavior is mostly preserved, while the contact and multi-stage causal failures remain.

## Quantitative evidence

Final cosine similarity to the fixed two-step prior:

| Case | Baseline | PhaseLock | Forward | Reverse-time | Same-time |
|---|---:|---:|---:|---:|---:|
| Single ball | 0.344379 | 0.192736 | 0.345437 | 0.345142 | 0.343576 |
| Ball collision | 0.225063 | 0.201590 | 0.227966 | 0.232638 | 0.230368 |
| Domino gap | 0.172566 | 0.192352 | 0.171840 | 0.176909 | 0.177048 |
| Fire to paper | 0.381251 | 0.196748 | 0.386975 | 0.382660 | 0.378179 |

The forward variant is not consistently the largest or smallest deviation. Reverse-time or same-time changes the trajectory by a comparable amount in the collision and domino cases.

Mean intervention RMS relative to the receiver feature RMS:

| Case | Forward | Reverse-time | Same-time |
|---|---:|---:|---:|
| Single ball | 0.021787 | 0.022015 | 0.021727 |
| Ball collision | 0.007763 | 0.007715 | 0.007303 |
| Domino gap | 0.008299 | 0.008126 | 0.007484 |
| Fire to paper | 0.017186 | 0.017145 | 0.016759 |

These closely matched magnitudes confirm that the direction controls are fair enough for this pilot: the negative result is not explained by the forward route receiving much less perturbation.

At denoising step 8, directed-attention statistics also show no consistent forward advantage. For example, future-key mass is 0.259/0.264/0.237 for forward/reverse/same in the collision case, 0.216/0.226/0.220 in the domino case, and 0.290/0.279/0.281 in the fire case. The fire case has the strongest forward-specific shift, yet still does not ignite the paper. Attention redistribution alone is therefore insufficient evidence of successful causal utilization.

## Interpretation

This experiment narrows the hypothesis rather than rejecting causal attention in general:

1. Entity-localized features can be moved during early denoising without destroying ordinary single-object motion.
2. A fixed temporal offset and fixed source-to-receiver blend changes the trajectory, but does not encode when contact occurs or when a downstream state transition should be activated.
3. Forward temporal direction by itself is insufficient. A useful method likely needs event-gated routing: detect an interaction onset, route only during the relevant denoising/time window, and update the receiver state according to the event rather than copying a generic source feature.
4. PhaseLock and this routing operator address different failure modes. PhaseLock preserves an early prior; the present operator perturbs information flow. Neither forms the missing fire-to-paper or cross-gap event in this pilot.

The next mechanism experiment should therefore test **event-conditioned routing**, with contact/onset gating and adaptive layer/timestep strength, while retaining reverse-time and same-time controls and the single-object no-regression case. It should not scale to all three backbones until a forward-specific effect appears on at least one contact or causal-chain case.

## Artifacts

- `review/contact_sheets/`: visual comparison grids.
- `wan21_i2v/*/seed_42/*/final.mp4`: intervention videos.
- `wan21_i2v/*/seed_42/*/attention/`: raw attention arrays and intervention metadata.
- `wan21_i2v/*/seed_42/*/visualizations/`: entity and directed-attention maps.
- `slurm/`: full-job logs.

