# GOOSE jobs

GOOSE allows at most two submitted/running jobs per user.

The first deployment sequence is deliberately gated:

1. `setup_sources.sbatch`: clone and pin source repositories only;
2. freeze and validate the selected-case manifest;
3. implement and unit-test the common Wan2.1-I2V adapter;
4. run exactly one matched baseline/PhaseLock smoke case;
5. inspect artifacts before submitting the selected subset.

No SwitchCraft or RoPE intervention job belongs in the first generation round.

