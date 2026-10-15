#!/usr/bin/env python3
"""Entity and temporal attention diagnostics for three I2V backbones.

The wrappers always delegate generation to the already-installed Diffusers
processor and calculate diagnostics from a second set of Q/K projections.  No
tensor on the model's generation path is modified unless an explicit Wan-only
``intervention_plan`` is supplied. Entity maps are scaled QK affinities (not
probabilities); sparse self-attention matrices are exact head-mean probability
mass for a small set of entity-guided query/key tokens.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.models.embeddings import apply_rotary_emb as apply_diffusers_rotary
from diffusers.models.transformers.transformer_ltx import (
    apply_rotary_emb as apply_ltx_rotary,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def locate_text_spans(
    tokenizer, prompt: str, targets: dict[str, str], max_length: int
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    common = {
        "padding": "max_length",
        "max_length": max_length,
        "truncation": True,
        "add_special_tokens": True,
    }
    positions: dict[str, list[int]] = {}
    location_method = "character_offsets"
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, **common)
        offsets = encoded["offset_mapping"]
        lower_prompt = prompt.lower()
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
    except (NotImplementedError, TypeError):
        # CogVideoX and some LTX releases ship a Python SentencePiece tokenizer,
        # which cannot return character offsets.  Exact token-id subsequences
        # retain the same model-facing indexing without replacing the tokenizer.
        location_method = "token_id_subsequence"
        encoded = tokenizer(prompt, **common)
        input_ids = list(encoded["input_ids"])
        for label, phrase in targets.items():
            if phrase.lower() not in prompt.lower():
                raise ValueError(f"attention target {phrase!r} not found in prompt")
            candidates = []
            for text in (phrase, f" {phrase}"):
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                if ids and ids not in candidates:
                    candidates.append(list(ids))
            hits = None
            for candidate in candidates:
                for start in range(len(input_ids) - len(candidate) + 1):
                    if input_ids[start : start + len(candidate)] == candidate:
                        hits = list(range(start, start + len(candidate)))
                        break
                if hits is not None:
                    break
            if hits is None:
                raise ValueError(
                    f"attention target {phrase!r} token sequence not found; "
                    f"candidates={candidates}"
                )
            positions[label] = hits
    return positions, {
        "prompt": prompt,
        "targets": targets,
        "positions": positions,
        "max_length": max_length,
        "location_method": location_method,
        "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
    }


def _apply_wan_rotary(hidden_states: torch.Tensor, rotary_emb) -> torch.Tensor:
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output.type_as(hidden_states)


class CausalAttentionProbe:
    """State and processor management for one model/case generation."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        model_kind: str,
        token_positions: dict[str, list[int]],
        token_metadata: dict[str, Any],
        grid_thw: tuple[int, int, int],
        steps: list[int],
        entity_map_layers: list[int],
        self_source_layer_pairs: list[list[int]],
        calls_per_step: int,
        cfg_layout: str,
        query_topk_per_frame: int,
        support_topk_per_frame: int,
        query_chunk_size: int,
        cross_query_chunk_size: int,
        intervention_plan: dict[str, Any] | None = None,
    ) -> None:
        self.transformer = transformer
        self.model_kind = model_kind
        self.token_positions = token_positions
        self.token_metadata = token_metadata
        self.grid_thw = grid_thw
        self.steps = set(steps)
        self.entity_map_layers = set(entity_map_layers)
        self.self_to_source = {self_layer: source for self_layer, source in self_source_layer_pairs}
        self.calls_per_step = calls_per_step
        self.cfg_layout = cfg_layout
        self.query_topk_per_frame = query_topk_per_frame
        self.support_topk_per_frame = support_topk_per_frame
        self.query_chunk_size = query_chunk_size
        self.cross_query_chunk_size = cross_query_chunk_size
        self.intervention_plan = intervention_plan
        self.forward_index = -1
        self.step_index = -1
        self.timestep = float("nan")
        self.entity_maps: dict[tuple[int, int, str], torch.Tensor] = {}
        self.connectivity: dict[tuple[int, int, str, str], torch.Tensor] = {}
        self.events: list[dict[str, Any]] = []
        self.intervention_events: list[dict[str, Any]] = []
        self._handle = None
        self._originals: list[tuple[Any, Any]] = []

    def install(self) -> "CausalAttentionProbe":
        self._handle = self.transformer.register_forward_pre_hook(
            self._begin_forward, with_kwargs=True
        )
        if self.model_kind == "wan":
            blocks = self.transformer.blocks
            for layer, block in enumerate(blocks):
                self._wrap(block.attn1, layer, "self", WanProbeProcessor)
                self._wrap(block.attn2, layer, "cross", WanProbeProcessor)
        elif self.model_kind == "cogvideox":
            blocks = self.transformer.transformer_blocks
            for layer, block in enumerate(blocks):
                self._wrap(block.attn1, layer, "joint", CogProbeProcessor)
        elif self.model_kind == "ltx":
            blocks = self.transformer.transformer_blocks
            for layer, block in enumerate(blocks):
                self._wrap(block.attn1, layer, "self", LTXProbeProcessor)
                self._wrap(block.attn2, layer, "cross", LTXProbeProcessor)
        else:
            raise ValueError(f"unsupported model kind: {self.model_kind}")
        return self

    def _intervention_active(self, layer: int) -> bool:
        plan = self.intervention_plan
        return bool(
            plan
            and self.should_collect()
            and self.step_index in set(plan["steps_zero_based"])
            and layer in set(plan["self_layers"])
        )

    def route_wan_features(self, layer: int, output: torch.Tensor) -> torch.Tensor:
        """Transport source features into receiver supports under equal-mass controls.

        Information in attention flows from key/value positions into query
        outputs. This intervention therefore edits receiver output tokens
        directly, rather than treating a source-query/target-key diagnostic as
        proof of source-to-target causal flow.
        """
        if not self._intervention_active(layer):
            return output
        if self.model_kind != "wan":
            raise RuntimeError("feature routing intervention is currently Wan-only")
        plan = self.intervention_plan
        assert plan is not None
        source_layer = self.self_to_source[layer]
        labels = {
            label
            for edge in plan["edges"]
            for label in (edge["source"], edge["receiver"])
        }
        missing = [
            label
            for label in sorted(labels)
            if (self.step_index, source_layer, label) not in self.entity_maps
        ]
        if missing:
            raise RuntimeError(
                f"intervention layer {layer} requires entity maps from layer "
                f"{source_layer}; missing {missing}"
            )

        supports = {
            label: self._topk_by_frame(
                self.entity_maps[(self.step_index, source_layer, label)],
                int(plan["support_topk_per_frame"]),
            )
            for label in labels
        }
        mode = plan["mode"]
        lag = int(plan["temporal_lag_frames"])
        frames = self.grid_thw[0]
        receiver_frames = range(lag, frames - lag)
        if mode == "forward":
            frame_pairs = [
                (receiver_frame - lag, receiver_frame)
                for receiver_frame in receiver_frames
            ]
        elif mode == "reverse_time":
            frame_pairs = [
                (receiver_frame + lag, receiver_frame)
                for receiver_frame in receiver_frames
            ]
        elif mode == "same_time":
            frame_pairs = [
                (receiver_frame, receiver_frame)
                for receiver_frame in receiver_frames
            ]
        else:
            raise ValueError(f"unknown feature-routing mode: {mode}")

        modified = output.clone()
        conditional = self.conditional(modified)
        reference = self.conditional(output)
        layer_count = max(1, len(plan["self_layers"]))
        edge_count = max(1, len(plan["edges"]))
        alpha = float(plan["total_strength"]) / (layer_count * edge_count)
        squared_delta = 0.0
        edited_tokens = 0
        for edge in plan["edges"]:
            source_label = edge["source"]
            receiver_label = edge["receiver"]
            for source_frame, receiver_frame in frame_pairs:
                source_indices = supports[source_label][source_frame].to(output.device)
                receiver_indices = supports[receiver_label][receiver_frame].to(
                    output.device
                )
                source_feature = reference.index_select(1, source_indices).mean(
                    dim=1, keepdim=True
                )
                receiver_feature = conditional.index_select(1, receiver_indices)
                update = (source_feature - receiver_feature) * alpha
                conditional[:, receiver_indices] = receiver_feature + update
                squared_delta += float(update.float().square().mean().cpu())
                edited_tokens += int(receiver_indices.numel())
        reference_rms = float(reference.float().square().mean().sqrt().cpu())
        delta_rms = math.sqrt(squared_delta / max(1, len(plan["edges"]) * len(frame_pairs)))
        self.intervention_events.append(
            {
                "step": self.step_index,
                "layer": layer,
                "source_entity_layer": source_layer,
                "mode": mode,
                "alpha_per_layer_edge": alpha,
                "edge_count": len(plan["edges"]),
                "frame_pair_count": len(frame_pairs),
                "edited_token_visits": edited_tokens,
                "delta_rms": delta_rms,
                "reference_rms": reference_rms,
                "delta_to_reference_rms": delta_rms / max(reference_rms, 1e-12),
            }
        )
        return modified

    def _wrap(self, attention, layer: int, kind: str, wrapper_type) -> None:
        original = attention.get_processor()
        self._originals.append((attention, original))
        attention.set_processor(wrapper_type(original, self, layer, kind))

    def remove(self) -> None:
        for attention, original in self._originals:
            attention.set_processor(original)
        self._originals.clear()
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _begin_forward(self, _module, args, kwargs) -> None:
        self.forward_index += 1
        self.step_index = self.forward_index // self.calls_per_step
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
            timestep = args[1]
        if isinstance(timestep, torch.Tensor):
            self.timestep = float(timestep.detach().float().mean().cpu())

    def should_collect(self) -> bool:
        if self.step_index not in self.steps:
            return False
        if self.cfg_layout == "separate_calls":
            return self.forward_index % self.calls_per_step == 0
        return True

    def conditional(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg_layout == "batched_uncond_cond" and tensor.shape[0] >= 2:
            return tensor[tensor.shape[0] // 2 :]
        return tensor

    def collect_entity_affinity(
        self, layer: int, query: torch.Tensor, text_key: torch.Tensor
    ) -> None:
        """Collect scaled QK affinity from video queries to selected text keys.

        Both inputs use B,H,S,D layout.
        """
        if not self.should_collect() or layer not in self.entity_map_layers:
            return
        query = self.conditional(query)
        text_key = self.conditional(text_key)
        token_count = text_key.shape[2]
        outputs = {
            label: torch.empty(query.shape[2], dtype=torch.float32, device="cpu")
            for label in self.token_positions
        }
        scale = 1.0 / math.sqrt(query.shape[-1])
        for start in range(0, query.shape[2], self.cross_query_chunk_size):
            stop = min(start + self.cross_query_chunk_size, query.shape[2])
            q = query[:, :, start:stop].float()
            for label, positions in self.token_positions.items():
                valid = [position for position in positions if position < token_count]
                if not valid:
                    raise RuntimeError(
                        f"no valid token positions for {label!r}: {positions} >= {token_count}"
                    )
                selected = text_key[:, :, valid].float()
                affinity = torch.einsum("bhqd,bhkd->bhqk", q, selected).mul_(scale)
                outputs[label][start:stop] = affinity.mean(dim=(0, 1, 3)).cpu()
                del affinity
        expected = math.prod(self.grid_thw)
        if query.shape[2] != expected:
            raise RuntimeError(
                f"video token/grid mismatch for {self.model_kind}: "
                f"{query.shape[2]} vs {self.grid_thw}={expected}"
            )
        for label, values in outputs.items():
            self.entity_maps[(self.step_index, layer, label)] = values.reshape(
                self.grid_thw
            ).to(torch.float16)
        self.events.append(
            {
                "kind": "entity_qk_affinity",
                "step": self.step_index,
                "layer": layer,
                "timestep": self.timestep,
            }
        )

    @staticmethod
    def _topk_by_frame(map_thw: torch.Tensor, topk: int) -> list[torch.Tensor]:
        t, h, w = map_thw.shape
        flat = map_thw.float().reshape(t, h * w)
        k = min(topk, h * w)
        return [
            torch.topk(flat[frame], k=k).indices + frame * h * w
            for frame in range(t)
        ]

    def collect_sparse_self(
        self,
        layer: int,
        query: torch.Tensor,
        key: torch.Tensor,
        *,
        video_offset: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Collect exact sparse self-attention mass in B,H,S,D layout."""
        if not self.should_collect() or layer not in self.self_to_source:
            return
        if attention_mask is not None:
            raise RuntimeError("diagnostic self-attention mask handling is not implemented")
        source_layer = self.self_to_source[layer]
        missing = [
            label
            for label in self.token_positions
            if (self.step_index, source_layer, label) not in self.entity_maps
        ]
        if missing:
            raise RuntimeError(
                f"self layer {layer} requires entity maps from layer {source_layer}; "
                f"missing {missing}"
            )
        query = self.conditional(query)
        key = self.conditional(key)
        maps = {
            label: self.entity_maps[(self.step_index, source_layer, label)]
            for label in self.token_positions
        }
        query_indices = {
            label: self._topk_by_frame(value, self.query_topk_per_frame)
            for label, value in maps.items()
        }
        support_indices = {
            label: self._topk_by_frame(value, self.support_topk_per_frame)
            for label, value in maps.items()
        }
        device = query.device
        frames = self.grid_thw[0]
        scale = 1.0 / math.sqrt(query.shape[-1])
        for source_label, by_frame in query_indices.items():
            relative_queries = torch.cat(by_frame)
            selected_indices = (relative_queries + video_offset).to(device)
            query_frames = torch.cat(
                [
                    torch.full((len(indices),), frame, dtype=torch.long)
                    for frame, indices in enumerate(by_frame)
                ]
            )
            matrices = {
                target: torch.zeros((frames, frames), dtype=torch.float64)
                for target in support_indices
            }
            counts = torch.zeros(frames, dtype=torch.float64)
            for start in range(0, len(selected_indices), self.query_chunk_size):
                stop = min(start + self.query_chunk_size, len(selected_indices))
                selected_query = query.index_select(2, selected_indices[start:stop])
                logits = torch.einsum(
                    "bhqd,bhkd->bhqk", selected_query.float(), key.float()
                ).mul_(scale)
                probabilities = logits.softmax(dim=-1).mean(dim=(0, 1))
                for local_index, query_frame in enumerate(
                    query_frames[start:stop].tolist()
                ):
                    counts[query_frame] += 1
                    row = probabilities[local_index]
                    for target_label, target_by_frame in support_indices.items():
                        for key_frame, indices in enumerate(target_by_frame):
                            keys = (indices + video_offset).to(device)
                            matrices[target_label][query_frame, key_frame] += float(
                                row.index_select(0, keys).sum().cpu()
                            )
                del logits, probabilities
            counts = counts.clamp_min(1).unsqueeze(1)
            for target_label, matrix in matrices.items():
                self.connectivity[
                    (self.step_index, layer, source_label, target_label)
                ] = (matrix / counts).to(torch.float32)
        self.events.append(
            {
                "kind": "sparse_self_attention",
                "step": self.step_index,
                "layer": layer,
                "source_entity_layer": source_layer,
                "timestep": self.timestep,
            }
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=False)
        arrays: dict[str, np.ndarray] = {}
        for (step, layer, label), value in sorted(self.entity_maps.items()):
            arrays[
                f"entity_qk_s{step:02d}_l{layer:02d}_{_safe_name(label)}"
            ] = value.numpy()
        for (step, layer, source, target), value in sorted(
            self.connectivity.items()
        ):
            arrays[
                f"self_s{step:02d}_l{layer:02d}_{_safe_name(source)}_to_{_safe_name(target)}"
            ] = value.numpy()
        if not arrays:
            raise RuntimeError("causal attention probe collected no arrays")
        np.savez_compressed(output_dir / "attention_maps.npz", **arrays)
        metadata = {
            "schema_version": 1,
            "model_kind": self.model_kind,
            "step_indexing": "zero-based; displayed denoising step is step + 1",
            "branch": "conditional CFG branch only",
            "grid_thw": list(self.grid_thw),
            "tokenization": self.token_metadata,
            "plan": {
                "steps": sorted(self.steps),
                "entity_map_layers": sorted(self.entity_map_layers),
                "self_source_layer_pairs": [
                    [layer, self.self_to_source[layer]]
                    for layer in sorted(self.self_to_source)
                ],
                "query_topk_per_frame": self.query_topk_per_frame,
                "support_topk_per_frame": self.support_topk_per_frame,
            },
            "events": self.events,
            "array_count": len(arrays),
            "interpretation": {
                "entity_qk": "Scaled, head-mean QK affinity from video queries to selected text tokens; not softmax probability.",
                "self": "Exact head-mean self-attention probability mass from source-entity query supports to target-entity key supports, indexed by query/key frame.",
                "caveat": "Attention is correlational localization. Causal claims require a later intervention experiment.",
            },
            "generation_path_modified": self.intervention_plan is not None,
            "intervention_plan": self.intervention_plan,
            "intervention_events": self.intervention_events,
        }
        (output_dir / "attention_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


class _BaseProbeProcessor:
    def __init__(self, base, state: CausalAttentionProbe, layer: int, kind: str):
        self.base = base
        self.state = state
        self.layer = layer
        self.kind = kind


class WanProbeProcessor(_BaseProbeProcessor):
    @torch.no_grad()
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb=None,
    ) -> torch.Tensor:
        output = self.base(
            attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb
        )
        if not self.state.should_collect():
            return output
        if self.kind == "cross" and self.layer in self.state.entity_map_layers:
            text_states = encoder_hidden_states
            if attn.add_k_proj is not None:
                text_states = encoder_hidden_states[:, encoder_hidden_states.shape[1] - 512 :]
            query = attn.norm_q(attn.to_q(hidden_states)).unflatten(
                2, (attn.heads, -1)
            ).permute(0, 2, 1, 3)
            key = attn.norm_k(attn.to_k(text_states)).unflatten(
                2, (attn.heads, -1)
            ).permute(0, 2, 1, 3)
            self.state.collect_entity_affinity(self.layer, query, key)
        elif self.kind == "self" and self.layer in self.state.self_to_source:
            query = attn.norm_q(attn.to_q(hidden_states)).unflatten(
                2, (attn.heads, -1)
            )
            key = attn.norm_k(attn.to_k(hidden_states)).unflatten(
                2, (attn.heads, -1)
            )
            query = _apply_wan_rotary(query, rotary_emb).permute(0, 2, 1, 3)
            key = _apply_wan_rotary(key, rotary_emb).permute(0, 2, 1, 3)
            self.state.collect_sparse_self(
                self.layer, query, key, attention_mask=attention_mask
            )
            output = self.state.route_wan_features(self.layer, output)
        return output


class CogProbeProcessor(_BaseProbeProcessor):
    @torch.no_grad()
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ):
        output = self.base(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
        if not self.state.should_collect() or (
            self.layer not in self.state.entity_map_layers
            and self.layer not in self.state.self_to_source
        ):
            return output
        text_length = encoder_hidden_states.shape[1]
        joint = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch = joint.shape[0]
        query = attn.to_q(joint)
        key = attn.to_k(joint)
        head_dim = key.shape[-1] // attn.heads
        query = query.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch, -1, attn.heads, head_dim).transpose(1, 2)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query[:, :, text_length:] = apply_diffusers_rotary(
                query[:, :, text_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_length:] = apply_diffusers_rotary(
                    key[:, :, text_length:], image_rotary_emb
                )
        if self.layer in self.state.entity_map_layers:
            self.state.collect_entity_affinity(
                self.layer, query[:, :, text_length:], key[:, :, :text_length]
            )
        if self.layer in self.state.self_to_source:
            prepared_mask = None
            if attention_mask is not None:
                prepared_mask = attn.prepare_attention_mask(
                    attention_mask, joint.shape[1], batch
                ).view(batch, attn.heads, -1, joint.shape[1])
            self.state.collect_sparse_self(
                self.layer,
                query,
                key,
                video_offset=text_length,
                attention_mask=prepared_mask,
            )
        return output


class LTXProbeProcessor(_BaseProbeProcessor):
    @torch.no_grad()
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        output = self.base(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
        )
        if not self.state.should_collect():
            return output
        states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(states))
        if image_rotary_emb is not None:
            query = apply_ltx_rotary(query, image_rotary_emb)
            key = apply_ltx_rotary(key, image_rotary_emb)
        query = query.unflatten(2, (attn.heads, -1)).permute(0, 2, 1, 3)
        key = key.unflatten(2, (attn.heads, -1)).permute(0, 2, 1, 3)
        if self.kind == "cross" and self.layer in self.state.entity_map_layers:
            self.state.collect_entity_affinity(self.layer, query, key)
        elif self.kind == "self" and self.layer in self.state.self_to_source:
            self.state.collect_sparse_self(
                self.layer, query, key, attention_mask=attention_mask
            )
        return output
