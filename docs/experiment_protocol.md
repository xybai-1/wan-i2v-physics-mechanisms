# Experiment protocol

## Scope lock

The first round answers one binary mechanism question, not whether four papers
can be combined into a better generator.

| Item | First round |
|---|---|
| Backbone | One pinned Wan2.1-I2V revision |
| Generation arms | baseline; PhaseLock |
| Benchmark | selected PhyGround cases only |
| Seeds | identical fixed seed set across arms |
| Interventions | none beyond PhaseLock |
| Diagnostics | attention and latent traces, read-only |

## Required case strata

The selected manifest should balance five mechanisms:

1. initial-frame leakage / visual dominance;
2. simple motion;
3. collision and momentum transfer;
4. material or semantic state transition;
5. multi-stage causal propagation.

Selection must be made before inspecting new generation outputs. Each entry
must point to the official PhyGround prompt and first-frame identity.

## Four recorded signals

### Image/text attention balance

Measure the relative influence of image- and text-conditioning tokens across
denoising steps and transformer layers. Do not interpret raw attention mass as
causality without an ablation.

### Event/frame attention

Map prompt event spans to frames and measure whether event semantics localize
to the expected temporal region. Event spans and expected partial order must be
specified in the manifest before generation.

### Early trajectory attention

For trackable entities, export object-token attention maps at selected early
steps. Candidate trajectories are diagnostic hypotheses; causal claims require
head intervention or attribution.

### Latent-delta evolution

For each denoising step `k`, record a similarity between the current temporal
latent difference and the 2-step prior. Preserve raw trace metadata and the
exact tensor layout used for the calculation.

## Acceptance gates

A smoke run is accepted only if:

- baseline and PhaseLock consume byte-identical conditioning inputs;
- both use the same seed and generation parameters;
- final videos decode successfully;
- the 2-step artifact and latent-delta trace are saved for PhaseLock;
- attention exports declare step, layer, head aggregation, token span, frame
  layout, dtype, and tensor shape;
- run metadata contains all pinned revisions and the Slurm job ID;
- rerunning the same arm does not overwrite a completed result silently.

## Interpretation boundary

- Wrong at 2 steps: evidence for formation failure.
- Reasonable at 2 steps but wrong at completion: evidence for preservation
  failure.
- Attention visualization alone is correlational.
- An intervention that changes backbone, prompt, input frame, seed, scheduler,
  or resolution is not a matched mechanism comparison.

