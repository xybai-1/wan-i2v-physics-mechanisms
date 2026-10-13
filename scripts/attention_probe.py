#!/usr/bin/env python3
"""Sparse, non-invasive attention probes for Diffusers Wan2.1-I2V.

The full Wan video self-attention matrix is prohibitively large.  This module
therefore records exact head-mean cross-attention maps for selected text spans
and computes self-attention only for a small set of object-guided query tokens.
The wrapped processor still delegates the generation output to Diffusers'
original processor; all probe calculations are side-channel reads.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from diffusers.models.transformers.transformer_wan import WanAttnProcessor


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def locate_text_spans(tokenizer, prompt: str, targets: dict[str, str]) -> tuple[dict[str, list[int]], dict]:
    encoded = tokenizer(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded["offset_mapping"]
    lower_prompt = prompt.lower()
    positions: dict[str, list[int]] = {}
    for label, phrase in targets.items():
        start = lower_prompt.find(phrase.lower())
        if start < 0:
            raise ValueError(f"attention target {phrase!r} not found in prompt")
        end = start + len(phrase)
        hits = [
            index
            for index, (left, right) in enumerate(offsets)
            if right > left and left < end and right > start
        ]
        if not hits:
            raise ValueError(f"attention target {phrase!r} has no tokenizer overlap")
        positions[label] = hits
    metadata = {
        "prompt": prompt,
        "targets": targets,
        "positions": positions,
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
    }
    return positions, metadata


def _apply_rotary(hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


@dataclass(frozen=True)
class ProbePlan:
    steps: tuple[int, ...] = (0, 1, 2, 4, 9, 19)
    cross_layers: tuple[int, ...] = (0, 9, 19, 29, 38)
    self_layers: tuple[int, ...] = (1, 10, 20, 30, 39)
    query_topk_per_frame: int = 4
    support_topk_per_frame: int = 16
    query_chunk_size: int = 8
    cross_query_chunk_size: int = 256


class WanAttentionProbeState:
    def __init__(
        self,
        token_positions: dict[str, list[int]],
        token_metadata: dict,
        grid_thw: tuple[int, int, int],
        plan: ProbePlan,
    ):
        self.token_positions = token_positions
        self.token_metadata = token_metadata
        self.grid_thw = grid_thw
        self.plan = plan
        self.forward_index = -1
        self.step_index = -1
        self.branch = "unknown"
        self.cross_maps: dict[tuple[int, int, str], torch.Tensor] = {}
        self.ball_connectivity: dict[tuple[int, int], torch.Tensor] = {}
        self.ball_to_pins: dict[tuple[int, int], torch.Tensor] = {}
        self.events: list[dict] = []

    def begin_transformer_forward(self) -> None:
        self.forward_index += 1
        self.step_index = self.forward_index // 2
        self.branch = "cond" if self.forward_index % 2 == 0 else "uncond"

    def should_collect(self) -> bool:
        return self.branch == "cond" and self.step_index in self.plan.steps

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=False)
        arrays: dict[str, np.ndarray] = {}
        for (step, layer, label), value in self.cross_maps.items():
            arrays[f"cross_s{step:02d}_l{layer:02d}_{_safe_name(label)}"] = value.numpy()
        for (step, layer), value in self.ball_connectivity.items():
            arrays[f"self_ball_s{step:02d}_l{layer:02d}"] = value.numpy()
        for (step, layer), value in self.ball_to_pins.items():
            arrays[f"self_ball_to_pins_s{step:02d}_l{layer:02d}"] = value.numpy()
        np.savez_compressed(output_dir / "attention_maps.npz", **arrays)
        metadata = {
            "schema_version": 1,
            "step_indexing": "zero-based; displayed denoising step is step_index + 1",
            "branch": "conditional CFG branch only",
            "grid_thw": list(self.grid_thw),
            "plan": {
                "steps": list(self.plan.steps),
                "cross_layers": list(self.plan.cross_layers),
                "self_layers": list(self.plan.self_layers),
                "query_topk_per_frame": self.plan.query_topk_per_frame,
                "support_topk_per_frame": self.plan.support_topk_per_frame,
            },
            "tokenization": self.token_metadata,
            "events": self.events,
            "interpretation": {
                "cross_maps": "Exact text-normalized, head-mean video-query attention for selected text spans.",
                "self_ball": "Attention mass from ball-guided queries to ball-guided key supports by query/key frame.",
                "self_ball_to_pins": "Attention mass from ball-guided queries to pins-guided key supports by query/key frame.",
                "caveat": "Attention is a correlational diagnostic, not causal proof.",
            },
        }
        (output_dir / "attention_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


class ProbingWanAttnProcessor:
    def __init__(self, state: WanAttentionProbeState, layer: int, kind: str):
        self.state = state
        self.layer = layer
        self.kind = kind
        self.base = WanAttnProcessor()

    @torch.no_grad()
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        if self.kind == "self" and self.layer == 0:
            self.state.begin_transformer_forward()
        output = self.base(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if not self.state.should_collect():
            return output
        if self.kind == "cross" and self.layer in self.state.plan.cross_layers:
            self._collect_cross(attn, hidden_states, encoder_hidden_states)
        elif self.kind == "self" and self.layer in self.state.plan.self_layers:
            self._collect_self(attn, hidden_states, rotary_emb)
        return output

    def _collect_cross(self, attn, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor) -> None:
        text_states = encoder_hidden_states
        if attn.add_k_proj is not None:
            text_states = encoder_hidden_states[:, encoder_hidden_states.shape[1] - 512 :]
        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(text_states)).unflatten(2, (attn.heads, -1))
        scale = 1.0 / math.sqrt(query.shape[-1])
        maps = {
            label: torch.empty(query.shape[1], dtype=torch.float32, device="cpu")
            for label in self.state.token_positions
        }
        chunk_size = self.state.plan.cross_query_chunk_size
        for start in range(0, query.shape[1], chunk_size):
            stop = min(start + chunk_size, query.shape[1])
            logits = torch.einsum(
                "bqhd,bkhd->bhqk", query[:, start:stop].float(), key.float()
            ).mul_(scale)
            probabilities = logits.softmax(dim=-1)
            for label, positions in self.state.token_positions.items():
                selected = probabilities[..., positions]
                maps[label][start:stop] = selected.mean(dim=(0, 1, 3)).cpu()
            del logits, probabilities
        t, h, w = self.state.grid_thw
        if query.shape[1] != t * h * w:
            raise RuntimeError(f"video token/grid mismatch: {query.shape[1]} vs {(t, h, w)}")
        for label, values in maps.items():
            self.state.cross_maps[(self.state.step_index, self.layer, label)] = values.reshape(t, h, w)
        self.state.events.append(
            {"kind": "cross", "step": self.state.step_index, "layer": self.layer}
        )

    @staticmethod
    def _topk_by_frame(map_thw: torch.Tensor, topk: int) -> list[torch.Tensor]:
        t, h, w = map_thw.shape
        flattened = map_thw.reshape(t, h * w)
        k = min(topk, h * w)
        return [
            torch.topk(flattened[frame], k=k).indices + frame * h * w
            for frame in range(t)
        ]

    def _collect_self(self, attn, hidden_states: torch.Tensor, rotary_emb) -> None:
        source_layer = self.layer - 1
        ball_key = (self.state.step_index, source_layer, "ball")
        pins_key = (self.state.step_index, source_layer, "pins")
        if ball_key not in self.state.cross_maps or pins_key not in self.state.cross_maps:
            raise RuntimeError(
                f"self probe layer {self.layer} requires cross maps from layer {source_layer}"
            )
        ball_map = self.state.cross_maps[ball_key]
        pins_map = self.state.cross_maps[pins_key]
        ball_queries = self._topk_by_frame(ball_map, self.state.plan.query_topk_per_frame)
        ball_support = self._topk_by_frame(ball_map, self.state.plan.support_topk_per_frame)
        pins_support = self._topk_by_frame(pins_map, self.state.plan.support_topk_per_frame)

        query = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1))
        query = _apply_rotary(query, rotary_emb)
        key = _apply_rotary(key, rotary_emb)
        device = query.device
        query_indices = torch.cat(ball_queries).to(device)
        query_frames = torch.cat(
            [torch.full((len(indices),), frame, dtype=torch.long) for frame, indices in enumerate(ball_queries)]
        )
        t = self.state.grid_thw[0]
        ball_matrix = torch.zeros((t, t), dtype=torch.float64)
        pins_matrix = torch.zeros((t, t), dtype=torch.float64)
        counts = torch.zeros(t, dtype=torch.float64)
        scale = 1.0 / math.sqrt(query.shape[-1])
        chunk_size = self.state.plan.query_chunk_size
        for start in range(0, len(query_indices), chunk_size):
            stop = min(start + chunk_size, len(query_indices))
            selected_query = query.index_select(1, query_indices[start:stop])
            logits = torch.einsum("bqhd,bkhd->bhqk", selected_query.float(), key.float()).mul_(scale)
            mean_probabilities = logits.softmax(dim=-1).mean(dim=(0, 1))
            for local_index, query_frame in enumerate(query_frames[start:stop].tolist()):
                counts[query_frame] += 1
                row = mean_probabilities[local_index]
                for key_frame in range(t):
                    ball_matrix[query_frame, key_frame] += float(
                        row.index_select(0, ball_support[key_frame].to(device)).sum().cpu()
                    )
                    pins_matrix[query_frame, key_frame] += float(
                        row.index_select(0, pins_support[key_frame].to(device)).sum().cpu()
                    )
            del logits, mean_probabilities
        counts = counts.clamp_min(1).unsqueeze(1)
        self.state.ball_connectivity[(self.state.step_index, self.layer)] = (ball_matrix / counts).float()
        self.state.ball_to_pins[(self.state.step_index, self.layer)] = (pins_matrix / counts).float()
        self.state.events.append(
            {"kind": "self", "step": self.state.step_index, "layer": self.layer, "source_cross_layer": source_layer}
        )


def install_wan_attention_probes(transformer, state: WanAttentionProbeState) -> None:
    for layer, block in enumerate(transformer.blocks):
        block.attn1.set_processor(ProbingWanAttnProcessor(state, layer, "self"))
        block.attn2.set_processor(ProbingWanAttnProcessor(state, layer, "cross"))

