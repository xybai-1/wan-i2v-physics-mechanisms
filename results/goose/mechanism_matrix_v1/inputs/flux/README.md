# FLUX conditioning inputs

These three images are the frozen PhyGenBench I2V conditioning inputs for the
formation-versus-preservation mechanism matrix. They were generated on GOOSE
by Slurm job `4645` with pinned
`black-forest-labs/FLUX.1-schnell@741f7c3ce8b383c54771c7003378a50191e9efe9`.

The images use seeds 4242--4244 and are shared unchanged by all three video
backbones, all three experiment seeds, and both baseline/PhaseLock arms. The
video-generation prompts remain the original benchmark prompts; the image
prompts in `input_lock_flux.json` are used only to construct pre-action I2V
frames.

Manual review passed on 2026-10-14:

- `prompt_025.png`: one intact ball above a surface before contact;
- `prompt_130.png`: one intact egg and one rock, visibly separated before impact;
- `prompt_140.png`: one burning paper ball separated from an intact, unlit paper pile.

No image exposes the target bounce, fracture, or propagated ignition outcome.
See `review.json` for exact hashes and structured acceptance decisions.
