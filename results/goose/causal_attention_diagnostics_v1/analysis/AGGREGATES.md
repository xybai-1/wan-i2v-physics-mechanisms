# Causal-attention diagnostic aggregates

These are non-invasive, correlational diagnostics over 36 matched runs. Positive deltas mean PhaseLock minus baseline.

## Model × denoising step

| Model | Step | Δ entity selectivity | Δ temporal span | Δ future mass | Δ forward bias |
| --- | --- | --- | --- | --- | --- |
| cogvideox_i2v | 1 | +0.000 | +0.000 | +0.000 | +0.000 |
| cogvideox_i2v | 2 | +0.005 | -0.001 | -0.001 | -0.002 |
| cogvideox_i2v | 4 | +0.019 | -0.022 | +0.002 | -0.001 |
| cogvideox_i2v | 8 | +0.058 | -0.081 | +0.005 | +0.011 |
| ltx_video_i2v | 1 | +0.000 | +0.000 | +0.000 | +0.000 |
| ltx_video_i2v | 2 | +0.009 | +0.054 | +0.005 | +0.001 |
| ltx_video_i2v | 4 | +0.021 | +0.304 | +0.002 | -0.024 |
| ltx_video_i2v | 8 | +0.030 | +0.554 | +0.013 | -0.039 |
| wan21_i2v | 1 | +0.000 | +0.000 | +0.000 | +0.000 |
| wan21_i2v | 2 | -0.043 | +0.298 | -0.005 | +0.018 |
| wan21_i2v | 4 | -0.044 | +0.305 | -0.017 | +0.011 |
| wan21_i2v | 8 | -0.033 | +0.242 | -0.023 | +0.017 |

## Case role at early and later diagnostic steps

| Role | Step | Δ entity selectivity | Δ future mass | Δ forward bias |
| --- | --- | --- | --- | --- |
| cross_object_state_propagation | 2 | -0.037 | +0.001 | +0.025 |
| cross_object_state_propagation | 8 | +0.026 | +0.006 | +0.029 |
| material_contact | 2 | +0.011 | -0.010 | -0.007 |
| material_contact | 8 | +0.050 | -0.009 | -0.008 |
| multi_object_shielding | 2 | -0.009 | -0.001 | -0.004 |
| multi_object_shielding | 8 | -0.022 | -0.003 | -0.024 |
| multi_stage_contact_chain | 2 | -0.014 | -0.001 | -0.007 |
| multi_stage_contact_chain | 8 | -0.007 | -0.009 | -0.040 |
| single_entity_control | 2 | -0.007 | +0.001 | +0.008 |
| single_entity_control | 8 | +0.019 | -0.007 | +0.004 |
| two_entity_contact | 2 | -0.003 | +0.009 | +0.018 |
| two_entity_contact | 8 | +0.043 | +0.012 | +0.018 |

## Interpretation boundary

Entity QK maps are token-affinity maps without ground-truth object masks. Directed matrices measure source-query to target-key temporal mass. Neither establishes that a pathway causes the rendered physical outcome; causal claims require intervention.
