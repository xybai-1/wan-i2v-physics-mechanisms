# Causal-attention diagnostics: formation versus preservation

## Outcome

The diagnostic does **not** support the claim that failed physical events are
caused by an absence of object/event attention. All three backbones contain
readable, non-degenerate entity-token affinity and directed temporal-attention
structure from the first denoising step. The more precise result is that these
signals are routed and altered differently by each architecture, and that
PhaseLock does not consistently turn them into forward causal propagation.

The clearest mechanism signal is in Wan2.1: by step 8, PhaseLock increases the
mean same-frame diagonal attention fraction from 0.396 to 0.468 while reducing
future-target mass from 0.248 to 0.225. This is consistent with over-locking:
identity/appearance may be anchored while cross-time interaction is suppressed.
In LTX, PhaseLock instead increases the temporal span of entity affinity by
+0.554 standardized units at step 8 and preserves the early latent prior much
more successfully, but the source-query/target-key lookup becomes more
past-key-biased rather than showing a new causal route. CogVideoX changes are
smaller and distributed.

Together with the parent 162-video matrix, this refines the earlier conclusion:
hard contact and chain cases are still formation failures, but not because the
network lacks any internal entity or temporal structure. The likely gap is
**causal routing/utilization along the sampling trajectory**, followed by a
separate architecture-dependent preservation problem.

## Frozen design and integrity

- Models: Wan2.1-I2V-14B, CogVideoX-5B-I2V, and LTX-Video-0.9.5-I2V.
- Cases: six seed-42 representatives: a single-object control, two contact
  cases, and three multi-object or multi-stage chains.
- Arms: baseline and PhaseLock, for 36 total runs.
- Probes: denoising steps 1, 2, 4, and 8; architecture-specific early/late
  layers; conditional branch only.
- Signals: scaled head-mean text-token QK affinity and exact head-mean sparse
  self-attention probability from a source-entity support to a target-entity
  support, indexed by query/key video frame.
- Intervention: none. Raw latents were not saved.

`analysis/validation.json` verifies 36/36 runs. Every diagnostic `few_step.mp4`
and `final.mp4` is byte-identical to its parent-matrix reference, every expected
array/event/shape is present and finite, and every video is non-empty and
decodable. Jobs 4670 and 4671 completed with exit `0:0` in 02:27:30 and
00:42:33, respectively.

## Quantitative findings

`Entity selectivity` is `(spatial p95 - mean) / spatial std`; it measures
contrast in a token-affinity map, not overlap with a ground-truth object mask.
`Future mass` is attention from a source-labelled query to target-labelled keys
at later frames. This is a query-key lookup orientation, not source-to-target
information flow: key/value information flows into the query output. Positive
deltas below mean PhaseLock minus baseline.

| Backbone | Step-8 entity selectivity delta | Step-8 temporal-span delta | Step-8 future-mass delta | Step-8 forward-bias delta |
| --- | ---: | ---: | ---: | ---: |
| CogVideoX-I2V | +0.058 | -0.081 | +0.005 | +0.011 |
| LTX-Video-I2V | +0.030 | +0.554 | +0.013 | -0.039 |
| Wan2.1-I2V | -0.033 | +0.242 | -0.023 | +0.017 |

All step-1 arm deltas are exactly zero. The first probed forward pass is shared;
PhaseLock differences appear only after its first latent update. This is a useful
implementation check and prevents attributing pre-intervention differences to
the method.

Absolute pathway structure also differs substantially:

- **CogVideoX** has broadly distributed off-diagonal temporal communication:
  baseline step-8 future/past/diagonal fractions are 0.423/0.441/0.136.
  PhaseLock slightly increases spatial selectivity and future mass, but its
  effect is small relative to the existing broad temporal bands.
- **LTX** begins with the highest affinity selectivity (1.838 at step 1) and a
  sparse, past-biased temporal pattern. PhaseLock raises step-8 entity
  selectivity from 1.815 to 1.845 and temporal span from 0.811 to 1.365, while
  future/past/diagonal fractions become 0.285/0.390/0.324. This looks like
  stronger temporally varying entity structure, not a new forward chain.
- **Wan** has stable token-affinity selectivity but becomes increasingly
  same-frame under PhaseLock. At step 8, future/past/diagonal fractions change
  from 0.248/0.357/0.396 to 0.225/0.307/0.468. This supplies a plausible
  mechanism for why Wan's final prior-similarity delta was negative in the
  parent matrix: the fixed preservation force suppresses off-diagonal temporal
  exchange instead of maintaining a useful event trajectory.

The role-level deltas are not universal. At step 8, the two-ball contact role
shows +0.043 entity selectivity and +0.012 future mass, while the domino-chain
role shows -0.007 and -0.009 and the shielding chain shows -0.022 and -0.003.
The fire-to-paper role gains +0.026 selectivity and +0.006 future mass. Since
each role contains only one case per model in this diagnostic, these are
directional mechanism observations, not benchmark-level estimates.

## Visual review

The three contact sheets in `analysis/review_contact_sheets/` cover every
backbone and all six case roles, comparing baseline and PhaseLock entity maps
and the first directed edge at steps 1/2/4/8.

- All maps are readable and non-degenerate. Baseline and PhaseLock are visually
  identical at step 1 and diverge afterward, matching the quantitative audit.
- Wan's directed maps are sparse and diagonal; PhaseLock progressively removes
  off-diagonal structure and strengthens same-frame bands, especially in later
  diagnostic layers.
- CogVideoX retains broad diagonal and off-diagonal bands. PhaseLock causes
  modest redistribution rather than a clear new edge.
- LTX keeps sparse temporal bands while its entity-affinity maps change more
  strongly across denoising steps under PhaseLock.
- These patterns recur in the single-object control, both contact cases, and all
  three causal-chain roles; no single attention signature uniquely separates
  the physically hard cases.

Each heatmap is independently normalized to its 2nd-98th percentiles for
readability, so colors must not be used for absolute cross-panel comparison.
The CSV/JSON aggregates use the underlying arrays.

## Implication for a training-free method

The next method should be a **gated causal-routing intervention**, not a stronger
global PhaseLock and not generic text/image rebalancing:

1. Use entity-token affinity only to identify spatial supports for source,
   receiver, and event tokens.
2. During early/mid denoising, selectively boost source-at-earlier-frame to
   receiver-at-later-frame communication on the required interaction edges.
3. Preserve identity with a separate residual path, but gate its strength when
   same-frame diagonal mass rises and off-diagonal communication collapses.
4. Use architecture-specific layer groups: Wan around early blocks 1-2 and
   late blocks 33/38, CogVideoX blocks 35-39, and LTX blocks 0-2.
5. Keep the single-entity control as a no-regression gate; already-correct
   motion must not be frozen or distorted.

Before implementing a full method, the smallest causal proof experiment is a
3-6-case intervention matrix at steps 1/2/4/8:

- restore Wan PhaseLock off-diagonal mass from its matched baseline;
- boost only the specified forward entity edge, with reverse-edge,
  shuffled-edge, and equal-mass same-frame controls;
- patch architecture-specific attention/residual blocks independently;
- cross the best patch with a small solver/timestep perturbation;
- score event occurrence, event order, object identity/count, and simple-motion
  no-regression, not just latent similarity.

If the correct edge changes the physical event while the controls do not, the
attention pathway has causal leverage and can justify a general training-free
method. If it only changes appearance or motion magnitude, the next target
should be the denoising trajectory/solver rather than attention routing.

## Interpretation boundary

Attention is correlational localization. In particular, query-to-key arrows do
not indicate the direction of feature transport. These measurements establish
where entity and temporal signals exist and how PhaseLock redistributes them;
they do not prove that any map causes the rendered outcome. Activation patching
or a controlled attention intervention is required for that claim.

## Files

- `analysis/validation.json`: exhaustive 36-run integrity and decode audit.
- `analysis/aggregate_metrics.json`: model/step and role/step aggregates.
- `analysis/run_step_metrics.csv`: 144 run-step records.
- `analysis/phaselock_minus_baseline.csv`: 72 matched arm deltas.
- `analysis/AGGREGATES.md`: automatically generated compact tables.
- `analysis/review_contact_sheets/`: visual review sheets for all backbones and
  roles.
- Each arm directory: videos, metadata, latent-delta trace, attention arrays,
  and per-label/per-edge visualization grids.
