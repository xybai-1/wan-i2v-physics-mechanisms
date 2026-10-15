# Causal-attention diagnostics v1

This is a read-only follow-up to `formation_preservation_matrix_v1`. It asks
where object/event evidence is localized before any training-free intervention
is designed.

## Frozen subset

The study uses seed 42 and both baseline and PhaseLock for six parent-matrix
cases:

- single-entity control: `physics_iq_single_ball_free`;
- two-entity contact: `phyground_pair_balls_collide`;
- material contact: `phygenbench_pair_egg_rock`;
- multi-stage contact chain: `phyground_chain_domino_gap`;
- cross-object state propagation: `phygenbench_chain_fire_paper`;
- multi-object shielding: `physics_iq_chain_weight_protects_duck`.

Across Wan2.1-I2V, CogVideoX-I2V, and LTX-Video-I2V this is 36 diagnostic
runs. Denoising steps 1, 2, 4, and 8 are observed. Layer selections come from
the full-layer aggregate scan in the parent matrix.

## Measurements

- Entity localization: scaled, head-mean video-query/text-key QK affinity.
- Directed propagation: exact sparse self-attention mass from top source-object
  query tokens to top target-object key tokens, summarized by query/key frame.
- Preservation: the existing cosine-to-two-step-prior trace.

The probe delegates every generation call to the original Diffusers attention
processor and performs a second, detached Q/K calculation. Its `few_step.mp4`
and `final.mp4` SHA-256 hashes must match the corresponding frozen parent run.
This verifies that observation has not changed the sampled video.

Attention is treated as correlational localization, not causal proof. Only a
later activation-patching or attention-transport intervention can establish
causal use of a localized pathway.

## Execution gate

The smoke gate runs the ball-collision baseline once per backbone. The full
36-run batch is submitted only if all three smoke runs:

1. match parent video hashes;
2. contain the expected finite entity and sparse-self arrays;
3. produce decodable videos; and
4. yield interpretable visualizations.
