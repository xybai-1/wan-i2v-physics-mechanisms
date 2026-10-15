#!/usr/bin/env python3
"""Small deterministic unit test for temporal feature-routing orientation."""

from __future__ import annotations

import torch

from causal_attention_probe import CausalAttentionProbe


def make_probe(mode: str) -> CausalAttentionProbe:
    probe = CausalAttentionProbe(
        torch.nn.Identity(),
        model_kind="wan",
        token_positions={"source": [0], "receiver": [1]},
        token_metadata={},
        grid_thw=(3, 1, 2),
        steps=[0],
        entity_map_layers=[0],
        self_source_layer_pairs=[[1, 0]],
        calls_per_step=1,
        cfg_layout="separate_calls",
        query_topk_per_frame=1,
        support_topk_per_frame=1,
        query_chunk_size=1,
        cross_query_chunk_size=1,
        intervention_plan={
            "mode": mode,
            "steps_zero_based": [0],
            "self_layers": [1],
            "support_topk_per_frame": 1,
            "temporal_lag_frames": 1,
            "total_strength": 1.0,
            "edges": [{"source": "source", "receiver": "receiver"}],
        },
    )
    probe.step_index = 0
    source_map = torch.tensor([[[2.0, 0.0]], [[2.0, 0.0]], [[2.0, 0.0]]])
    receiver_map = torch.tensor([[[0.0, 2.0]], [[0.0, 2.0]], [[0.0, 2.0]]])
    probe.entity_maps[(0, 0, "source")] = source_map
    probe.entity_maps[(0, 0, "receiver")] = receiver_map
    return probe


def main() -> None:
    # Token order is [source_f0, receiver_f0, source_f1, receiver_f1, ...].
    output = torch.tensor(
        [[[10.0], [1.0], [20.0], [2.0], [30.0], [3.0]]], dtype=torch.float32
    )
    forward = make_probe("forward").route_wan_features(1, output)
    assert forward[0, 3, 0].item() == 10.0
    assert forward[0, 5, 0].item() == 3.0
    assert forward[0, 1, 0].item() == 1.0

    reverse = make_probe("reverse_time").route_wan_features(1, output)
    assert reverse[0, 3, 0].item() == 30.0
    assert reverse[0, 1, 0].item() == 1.0
    assert reverse[0, 5, 0].item() == 3.0

    same = make_probe("same_time").route_wan_features(1, output)
    assert same[0, 1, 0].item() == 1.0
    assert same[0, 3, 0].item() == 20.0
    assert same[0, 5, 0].item() == 3.0
    print("causal routing orientation: PASS")


if __name__ == "__main__":
    main()
