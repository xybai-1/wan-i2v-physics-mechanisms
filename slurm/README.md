# GOOSE jobs

GOOSE allows at most two submitted/running jobs per user.

The first deployment sequence is deliberately gated:

1. `setup_sources.sbatch`: clone and pin source repositories only;
2. freeze and validate the selected-case manifest;
3. implement and unit-test the common Wan2.1-I2V adapter;
4. run exactly one matched baseline/PhaseLock smoke case;
5. inspect artifacts before submitting the selected subset.

No SwitchCraft or RoPE intervention job belongs in the first generation round.

# GOOSE submission order

The first-round deployment is deliberately staged:

1. `setup_sources.sbatch` clones and records upstream source revisions.
2. `setup_generation.sbatch` creates the pinned Python environment, downloads
   the pinned Wan checkpoint, and verifies the frozen conditioning image.
3. Submit exactly one smoke arm at a time:

   ```bash
   sbatch --export=ALL,ARM=baseline slurm/run_smoke.sbatch
   sbatch --export=ALL,ARM=phaselock slurm/run_smoke.sbatch
   ```

GOOSE permits at most two concurrent Slurm jobs. Do not submit the generation
jobs until the environment job completes successfully. The smoke runner refuses
to overwrite an existing run directory.

Expected artifacts per arm are `few_step.mp4`, `final.mp4`,
`latent_delta_trace.json`, and `run.json`.

After both arms complete, submit `visualize_smoke.sbatch`. It creates a latent
similarity curve, a four-row temporal contact sheet, a machine-readable summary,
and a manual-review checklist. Attention-map extraction is intentionally a
separate validation gate because token/frame layout must be verified for the
Wan I2V implementation before a heatmap can be interpreted.
