# Three-backbone formation-versus-preservation matrix

## Outcome

The completed matrix supports a narrower conclusion than "PhaseLock solves
physics": **the selected contact and causal-chain failures are usually already
absent from the decoded two-step prior, and preserving that prior cannot create
the missing event.** PhaseLock's ability to preserve the prior is strongly
backbone-dependent at the frozen setting used here.

This is a mechanism study, not an official benchmark leaderboard. The cosine
statistics below measure similarity to the model's own two-step latent prior;
they do not measure physical correctness. The attention files contain reduced
activation statistics, not attention weights and not causal evidence.

## Frozen design

- Backbones: Wan2.1-I2V-14B, CogVideoX-5B-I2V, and LTX-Video-0.9.5-I2V.
- Benchmarks: PhyGround, PhyGenBench, and Physics-IQ Verified.
- Cases: one single-entity, one two-entity contact, and one multi-stage causal
  chain per benchmark.
- Seeds: 42, 123, and 777.
- Arms: baseline and PhaseLock, with an identical first frame, prompt, seed,
  scheduler, and two-step prior in every pair.
- Scale: 3 models x 9 cases x 3 seeds x 2 arms = 162 runs (81 matched pairs).
- PhaseLock: strength 0.05, applied during denoising steps 1-25 of 50.
- Decodes: steps 1, 2, 4, 8, 25, and 50.

The exact manifest is `configs/mechanism_matrix_v1.json`; every run records the
study revision `d4c515bd88ab9cef397e664bcc794f49685fc4ed`.

## Integrity audit

`analysis/validation.json` reports a complete and valid matrix:

- 162/162 `run.json` records and 81/81 matched pairs.
- 81/81 baseline/PhaseLock pairs have byte-identical `few_step.mp4` priors.
- All six checkpoint videos exist for every run.
- All 324 `few_step.mp4` and `final.mp4` files are non-empty and decodable.
- Every latent trace has 50 finite records.
- Full-layer reduced activation statistics cover every denoising step:
  4,200 records/run for CogVideoX (42 modules), 5,600 for LTX (56 modules),
  and 8,000 for Wan (80 modules).
- No raw latent tensors, attention matrices, model weights, caches, or secrets
  are included in this result bundle.

GOOSE jobs 4648 and 4649 finished `COMPLETED`, exit `0:0`, in 11:16:30 and
03:39:36. They reused six already-complete pilot cells from jobs 4646/4647;
all pilot and full-run cells use the same frozen study revision.

## Quantitative mechanism result

The table reports PhaseLock minus baseline cosine similarity to the shared
two-step prior. `Early` averages steps 1-25, `late` steps 26-50, and `final`
is step 50. `P(final > 0)` is the fraction of 27 pairs per backbone for which
PhaseLock ends closer to the early prior.

| Backbone | Early delta | Late delta | Final delta | P(final > 0) | Early/late residual-change ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| CogVideoX-I2V | +0.102 | +0.171 | +0.024 | 59.3% | 3.21 |
| LTX-Video-I2V | +0.189 | +0.358 | +0.302 | 100.0% | 1.10 |
| Wan2.1-I2V | +0.001 | -0.004 | -0.089 | 18.5% | 0.95 |

Three findings follow.

1. **Preservation is not architecture-agnostic.** LTX strongly and consistently
   retains the two-step prior. CogVideoX is pulled toward it while guidance is
   active, but much of the effect disappears by the final step. Under this
   frozen strength and schedule, Wan does not accumulate the intervention and
   usually ends farther from the early prior.
2. **Preserving a prior is not the same as making it physical.** LTX has the
   strongest latent-preservation signal, yet representative contact and chain
   videos still lack the required collision, propagation, or ordered event.
3. **There is no shared layer-depth signature.** The largest early residual
   changes repeatedly occur in CogVideoX blocks 35-39, LTX blocks 0-2, and Wan
   blocks 33/38 with a secondary early-block component. These are useful
   activation-patching targets, not proof that they encode physics.

Pooling all models, the final prior-similarity delta is +0.032 for single
motion, +0.132 for two-entity contact, and +0.074 for causal chains. This does
not imply that contact is solved: the pooled means are dominated by LTX's
strong preservation behavior and should not be read as benchmark scores.

## Visual review

Representative seed-42 final and two-step contact sheets were reviewed across
all three backbones and all three benchmarks. The complete sheets are in
`analysis/contact_sheets/` and `analysis/few_step_contact_sheets/`.

- **Two-ball contact:** all three two-step priors blur or fade the balls without
  forming a readable approach-contact-separation trajectory. CogVideoX
  PhaseLock often preserves two identities better than baseline, but tends to
  freeze them instead of producing a collision. LTX loses one ball; Wan creates
  motion but not a clean, identity-preserving collision.
- **Domino chain:** no reviewed two-step prior contains the required ordered
  first-row fall, gap bridge, and second-row fall. CogVideoX and LTX remain
  mostly static; Wan produces large motion and object proliferation rather
  than the specified chain. PhaseLock does not supply the missing order.
- **Burning paper to paper pile:** the early prior does not form the complete
  flight-contact-ignition chain. CogVideoX can render fire without transferring
  it to the pile; LTX PhaseLock preserves a mostly static ball/pile; Wan
  PhaseLock disrupts the pile's identity and geometry.
- **Egg-rock impact:** outputs show disappearance, deformation, stylistic
  debris, or object multiplication, but not a stable egg trajectory followed
  by contact-conditioned egg fracture while the rock remains consistent.
- **Single-object motion:** PhaseLock can reduce late appearance drift, but in
  several reviewed outputs this becomes motion suppression rather than a more
  physically correct trajectory.

The representative evidence therefore favors **formation failure** for the
hard interaction cases, plus an independent preservation/identity problem
during later refinement. PhaseLock addresses the latter only on some
backbones; it does not create causal event structure that is missing at two
steps.

## Implication for the next method

The next intervention should not be a stronger universal pull toward the
two-step prior. It should first form an object- and event-aware causal signal,
then preserve it adaptively without damaging already-correct simple motion.
Before implementing a full method, the smallest decisive follow-up is:

1. Activation-patch 3-6 representative runs at the architecture-specific
   module groups above and at steps 1/2/4/8.
2. Test whether patches alter **event order and inter-object response**, not
   just cosine similarity or appearance.
3. Cross the patches with a small solver/timestep intervention to separate
   information availability from sampling-trajectory accessibility.
4. Gate any intervention by an event/interaction confidence signal and retain
   an identity path so that single-entity cases cannot regress.
5. Use blinded physical-event scoring before claiming a benchmark improvement.

Recommended diagnostic cases are the PhyGround ball collision, PhyGround
domino gap, PhyGenBench fire-to-paper chain, PhyGenBench egg-rock impact, and
Physics-IQ weight-protects-duck. Together they distinguish missing event
formation, preservation-without-motion, and hallucinated motion.

## Files

- `analysis/validation.json`: exhaustive output and decode audit.
- `analysis/mechanism_analysis.json`: all 81 pair-level measurements and
  aggregate tables.
- `analysis/slurm_jobs.json`: scheduler provenance.
- `analysis/contact_sheets/`: final-video baseline/PhaseLock comparisons.
- `analysis/few_step_contact_sheets/`: shared two-step-prior comparisons.
- Each arm directory: `run.json`, `latent_delta_trace.json`, reduced attention
  activation arrays/metadata, `few_step.mp4`, and `final.mp4`.
