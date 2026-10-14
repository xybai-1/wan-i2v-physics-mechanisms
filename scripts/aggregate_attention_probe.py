#!/usr/bin/env python3
"""Non-invasive, all-layer scalar attention-activation instrumentation.

This deliberately does not materialize attention matrices.  For every
attention module and transformer call it retains only small sampled reductions
on GPU, then transfers the scalar arrays once after sampling finishes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _sample_flat(tensor: torch.Tensor, maximum: int) -> torch.Tensor:
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() <= maximum:
        return flat
    stride = max(flat.numel() // maximum, 1)
    return flat[::stride][:maximum]


@dataclass(frozen=True)
class AggregateProbePlan:
    sample_elements: int = 4096


class AggregateAttentionProbe:
    """Collect sampled output statistics from every attention module."""

    BRANCH_IDS = {"all": 0, "cond": 1, "uncond": 2}

    def __init__(
        self,
        transformer: torch.nn.Module,
        *,
        calls_per_step: int,
        cfg_layout: str,
        plan: AggregateProbePlan | None = None,
    ) -> None:
        self.transformer = transformer
        self.calls_per_step = calls_per_step
        self.cfg_layout = cfg_layout
        self.plan = plan or AggregateProbePlan()
        self.forward_index = -1
        self.step_index = -1
        self.timestep = float("nan")
        self.records: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.modules: list[str] = []

    def install(self) -> "AggregateAttentionProbe":
        self.handles.append(
            self.transformer.register_forward_pre_hook(self._begin_forward, with_kwargs=True)
        )
        for name, module in self.transformer.named_modules():
            if name and "attention" in module.__class__.__name__.lower():
                module_id = len(self.modules)
                self.modules.append(name)
                self.handles.append(
                    module.register_forward_hook(self._hook(module_id), with_kwargs=True)
                )
        if not self.modules:
            raise RuntimeError("no Attention modules found under transformer")
        return self

    def _begin_forward(self, _module, args, kwargs) -> None:
        self.forward_index += 1
        self.step_index = self.forward_index // self.calls_per_step
        timestep = kwargs.get("timestep")
        if timestep is None and len(args) > 1 and isinstance(args[1], torch.Tensor):
            timestep = args[1]
        if isinstance(timestep, torch.Tensor):
            self.timestep = float(timestep.detach().float().mean().cpu())

    def _branches(self, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        if self.cfg_layout == "separate_calls":
            branch = "cond" if self.forward_index % self.calls_per_step == 0 else "uncond"
            return [(branch, tensor)]
        if self.cfg_layout == "batched_uncond_cond" and tensor.ndim and tensor.shape[0] >= 2:
            midpoint = tensor.shape[0] // 2
            return [("uncond", tensor[:midpoint]), ("cond", tensor[midpoint:])]
        return [("all", tensor)]

    def _hook(self, module_id: int):
        @torch.no_grad()
        def hook(_module, inputs, kwargs, output) -> None:
            output_tensor = _first_tensor(output)
            input_tensor = _first_tensor(inputs)
            if input_tensor is None:
                input_tensor = _first_tensor(kwargs)
            if output_tensor is None:
                return
            input_branches = dict(self._branches(input_tensor)) if input_tensor is not None else {}
            for branch, branch_output in self._branches(output_tensor):
                sampled_output = _sample_flat(branch_output, self.plan.sample_elements)
                sampled_input = None
                if branch in input_branches:
                    sampled_input = _sample_flat(input_branches[branch], self.plan.sample_elements)
                    count = min(sampled_input.numel(), sampled_output.numel())
                    sampled_input = sampled_input[:count]
                    sampled_for_cosine = sampled_output[:count]
                    cosine = torch.nn.functional.cosine_similarity(
                        sampled_input.unsqueeze(0), sampled_for_cosine.unsqueeze(0), dim=1
                    ).squeeze(0)
                    delta_rms = (sampled_for_cosine - sampled_input).square().mean().sqrt()
                else:
                    cosine = sampled_output.new_tensor(float("nan"))
                    delta_rms = sampled_output.new_tensor(float("nan"))
                self.records.append(
                    {
                        "step": self.step_index,
                        "call": self.forward_index,
                        "module": module_id,
                        "branch": self.BRANCH_IDS[branch],
                        "mean": sampled_output.mean(),
                        "std": sampled_output.std(unbiased=False),
                        "rms": sampled_output.square().mean().sqrt(),
                        "input_output_cosine": cosine,
                        "residual_rms": delta_rms,
                    }
                )

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=False)
        if not self.records:
            raise RuntimeError("attention probe collected no records")
        integer_keys = ("step", "call", "module", "branch")
        float_keys = ("mean", "std", "rms", "input_output_cosine", "residual_rms")
        arrays: dict[str, np.ndarray] = {
            key: np.asarray([record[key] for record in self.records], dtype=np.int32)
            for key in integer_keys
        }
        for key in float_keys:
            arrays[key] = torch.stack([record[key] for record in self.records]).float().cpu().numpy()
        np.savez_compressed(output_dir / "attention_activation_stats.npz", **arrays)
        metadata = {
            "schema_version": 1,
            "record_count": len(self.records),
            "module_count": len(self.modules),
            "module_names": self.modules,
            "branch_ids": self.BRANCH_IDS,
            "calls_per_step": self.calls_per_step,
            "cfg_layout": self.cfg_layout,
            "sample_elements_per_tensor": self.plan.sample_elements,
            "collection": "Every Attention module at every full-trajectory transformer call.",
            "interpretation": "Sampled activation reductions are correlational diagnostics, not attention weights and not causal proof.",
            "raw_attention_or_latents_saved": False,
        }
        (output_dir / "attention_activation_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
