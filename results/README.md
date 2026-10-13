# Result index

Raw model weights, tensors, caches, and unreviewed experiment trees are
excluded from Git. A completed smoke bundle may be committed after its inputs,
provenance, videos, metrics, and visual review have been validated.

## Published bundles

### PhyGround `mechanics_411`, seed 42

Matched Wan2.1-I2V baseline versus Wan2.1-I2V + PhaseLock:

- [`baseline/few_step.mp4`](goose/mechanics_411/seed_42/baseline/few_step.mp4)
- [`baseline/final.mp4`](goose/mechanics_411/seed_42/baseline/final.mp4)
- [`phaselock/few_step.mp4`](goose/mechanics_411/seed_42/phaselock/few_step.mp4)
- [`phaselock/final.mp4`](goose/mechanics_411/seed_42/phaselock/final.mp4)
- [`analysis/temporal_contact_sheet.png`](goose/mechanics_411/seed_42/analysis/temporal_contact_sheet.png)
- [`analysis/latent_delta_similarity.png`](goose/mechanics_411/seed_42/analysis/latent_delta_similarity.png)
- [`analysis/analysis_summary.json`](goose/mechanics_411/seed_42/analysis/analysis_summary.json)
- [`analysis/review.md`](goose/mechanics_411/seed_42/analysis/review.md)

Each arm also includes its 40-step latent trace and `run.json` provenance.
The scientific interpretation is documented in
[`docs/results/mechanics_411_smoke.md`](../docs/results/mechanics_411_smoke.md).

