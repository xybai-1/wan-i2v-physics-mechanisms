# Attention instrumentation study

## Question

For a matched Wan2.1-I2V baseline and `+PhaseLock`, determine whether a failed
physical event is absent from the early denoising representation (**formation
failure**) or appears early and is later lost (**preservation failure**).

Attention is treated as a correlational diagnostic, not as causal proof.

## Sparse probes

The Wan video grid has 32,760 tokens (`21 x 30 x 52`), so full self-attention
is not materialized. The probe records, on the conditional CFG branch only:

- denoising indices `0, 1, 2, 4, 9, 19` (displayed as steps 1, 2, 3, 5, 10, 20);
- exact head-mean cross-attention maps at layers `0, 9, 19, 29, 38` for two
  entity phrases and one event phrase;
- sparse self-attention at layers `1, 10, 20, 30, 39`, using four
  primary-entity queries and sixteen entity supports per latent frame;
- latent snapshots and latent-delta similarity to the two-step motion prior.

The processor first calls Diffusers' original Wan processor and computes the
probe as a side-channel read. The single-case gate requires the instrumented
video SHA-256 to equal the corresponding uninstrumented video SHA-256.

## Eight-case contrast set

The frozen manifest is `configs/phyground_attention_8.json`. It contains four
low/high interaction-complexity pairs:

1. one bowling pin vs. a rack of pins;
2. two-ball collision vs. three-ball momentum transfer;
3. direct domino chain vs. a chain that must bridge a gap;
4. impact-driven gate collapse vs. an unstable multi-block stack.

All prompts and first images come from the pinned PhyGround dataset revision.
Both arms use the same first image, prompt, seed, dimensions, frame count,
guidance scale, and denoising schedule. Only the PhaseLock latent-delta update
differs.

## Execution gate

The eight-case run is submitted only after both single-case arms satisfy:

- 150 recorded arrays and 60 probe events;
- expected array shapes and finite values;
- exact video-hash equality with the uninstrumented run;
- readable visual summaries for cross-attention trajectories and sparse
  frame-to-frame self-attention.
