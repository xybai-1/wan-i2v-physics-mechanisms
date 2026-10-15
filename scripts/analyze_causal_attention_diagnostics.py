#!/usr/bin/env python3
"""Aggregate non-invasive causal-attention diagnostics into compact tables.

The resulting statistics are descriptive. They quantify text/video affinity
contrast and directed temporal attention, but do not establish causal effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


ENTITY_RE = re.compile(r"^entity_qk_s(\d+)_l(\d+)_(.+)$")
SELF_RE = re.compile(r"^self_s(\d+)_l(\d+)_(.+)_to_(.+)$")


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def finite(value: float) -> float:
    return float(value) if math.isfinite(value) else 0.0


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    out_dir = args.out_dir or args.output_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for summary_path in sorted(
        args.output_root.glob("*/*/seed_*/**/visualizations/attention_summary.json")
    ):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_dir = summary_path.parent.parent
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        buckets: dict[int, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for name, item in summary["arrays"].items():
            entity = ENTITY_RE.match(name)
            if entity:
                step = int(entity.group(1)) + 1
                std = max(float(item["std"]), 1e-8)
                profile = [float(value) for value in item["temporal_profile"]]
                third = max(1, len(profile) // 3)
                buckets[step]["entity_selectivity_z"].append(
                    finite((float(item["p95"]) - float(item["mean"])) / std)
                )
                buckets[step]["entity_temporal_span_z"].append(
                    finite((max(profile) - min(profile)) / std)
                )
                buckets[step]["entity_late_minus_early_z"].append(
                    finite((mean(profile[-third:]) - mean(profile[:third])) / std)
                )
                continue
            directed = SELF_RE.match(name)
            if directed:
                step = int(directed.group(1)) + 1
                future = float(item["future_key_fraction"])
                past = float(item["past_key_fraction"])
                offdiag = max(future + past, 1e-8)
                buckets[step]["directed_future_fraction"].append(future)
                buckets[step]["directed_past_fraction"].append(past)
                buckets[step]["directed_diagonal_fraction"].append(
                    float(item["diagonal_fraction"])
                )
                buckets[step]["directed_forward_bias"].append(
                    finite((future - past) / offdiag)
                )

        for step, metrics in sorted(buckets.items()):
            records.append(
                {
                    "model": run["model_key"],
                    "case": run["case"]["case_id"],
                    "benchmark": run["case"]["benchmark"],
                    "role": run["diagnostic_case"]["role"],
                    "arm": run["arm"],
                    "step": step,
                    **{key: mean(values) for key, values in metrics.items()},
                }
            )

    if len(records) != 36 * 4:
        raise SystemExit(f"expected 144 run-step rows, found {len(records)}")

    metric_names = [
        "entity_selectivity_z",
        "entity_temporal_span_z",
        "entity_late_minus_early_z",
        "directed_future_fraction",
        "directed_past_fraction",
        "directed_diagonal_fraction",
        "directed_forward_bias",
    ]
    csv_path = out_dir / "run_step_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["model", "case", "benchmark", "role", "arm", "step"]
            + metric_names,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)

    paired: list[dict] = []
    by_key: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    for record in records:
        by_key[(record["model"], record["case"], record["step"])][
            record["arm"]
        ] = record
    for (model, case, step), arms in sorted(by_key.items()):
        if set(arms) != {"baseline", "phaselock"}:
            raise SystemExit(f"unpaired diagnostic row: {model}/{case}/step{step}")
        base = arms["baseline"]
        phase = arms["phaselock"]
        paired.append(
            {
                "model": model,
                "case": case,
                "benchmark": base["benchmark"],
                "role": base["role"],
                "step": step,
                **{
                    f"delta_{metric}": phase[metric] - base[metric]
                    for metric in metric_names
                },
            }
        )

    pair_csv = out_dir / "phaselock_minus_baseline.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(paired[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(paired)

    def aggregate(rows: list[dict], group_keys: list[str], metrics: list[str]) -> list[dict]:
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row[key] for key in group_keys)].append(row)
        result = []
        for key, items in sorted(grouped.items()):
            result.append(
                {
                    **dict(zip(group_keys, key)),
                    "n": len(items),
                    **{metric: mean([float(item[metric]) for item in items]) for metric in metrics},
                }
            )
        return result

    model_step = aggregate(records, ["model", "arm", "step"], metric_names)
    delta_model_step = aggregate(
        paired,
        ["model", "step"],
        [f"delta_{metric}" for metric in metric_names],
    )
    delta_role_step = aggregate(
        paired,
        ["role", "step"],
        [f"delta_{metric}" for metric in metric_names],
    )

    report = {
        "schema_version": 1,
        "checked_runs": 36,
        "run_step_rows": len(records),
        "paired_rows": len(paired),
        "definitions": {
            "entity_selectivity_z": "(spatial p95 - spatial mean) / spatial std of text-token QK affinity; higher means a more spatially selective affinity map, not verified object-mask localization.",
            "entity_temporal_span_z": "range of frame-mean token affinity divided by full-map std.",
            "entity_late_minus_early_z": "late-third minus early-third frame-mean affinity divided by full-map std.",
            "directed_future_fraction": "source-entity query attention mass to target-entity keys at later frames.",
            "directed_past_fraction": "source-entity query attention mass to target-entity keys at earlier frames.",
            "directed_forward_bias": "(future - past) / (future + past); positive means later-target preference.",
        },
        "model_step": model_step,
        "phaselock_minus_baseline_by_model_step": delta_model_step,
        "phaselock_minus_baseline_by_role_step": delta_role_step,
        "caveat": "All statistics are correlational. Attention localization and directionality do not prove causal influence on generated motion; activation patching or controlled attention intervention is required.",
    }
    (out_dir / "aggregate_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    model_rows = []
    for row in delta_model_step:
        model_rows.append(
            [
                row["model"],
                row["step"],
                f"{row['delta_entity_selectivity_z']:+.3f}",
                f"{row['delta_entity_temporal_span_z']:+.3f}",
                f"{row['delta_directed_future_fraction']:+.3f}",
                f"{row['delta_directed_forward_bias']:+.3f}",
            ]
        )
    role_rows = []
    for row in delta_role_step:
        if row["step"] not in (2, 8):
            continue
        role_rows.append(
            [
                row["role"],
                row["step"],
                f"{row['delta_entity_selectivity_z']:+.3f}",
                f"{row['delta_directed_future_fraction']:+.3f}",
                f"{row['delta_directed_forward_bias']:+.3f}",
            ]
        )
    markdown = [
        "# Causal-attention diagnostic aggregates",
        "",
        "These are non-invasive, correlational diagnostics over 36 matched runs. Positive deltas mean PhaseLock minus baseline.",
        "",
        "## Model × denoising step",
        "",
        markdown_table(
            ["Model", "Step", "Δ entity selectivity", "Δ temporal span", "Δ future mass", "Δ forward bias"],
            model_rows,
        ),
        "",
        "## Case role at early and later diagnostic steps",
        "",
        markdown_table(
            ["Role", "Step", "Δ entity selectivity", "Δ future mass", "Δ forward bias"],
            role_rows,
        ),
        "",
        "## Interpretation boundary",
        "",
        "Entity QK maps are token-affinity maps without ground-truth object masks. Directed matrices measure source-query to target-key temporal mass. Neither establishes that a pathway causes the rendered physical outcome; causal claims require intervention.",
        "",
    ]
    (out_dir / "AGGREGATES.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps({"runs": 36, "rows": len(records), "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
