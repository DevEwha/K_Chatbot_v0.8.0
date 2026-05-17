#!/usr/bin/env python3
"""
Build revised KV-repair figures and summary artifacts for the ASPLOS paper.

This version is tailored to the requested paper rewrite:
  - model order: LLaMA-2-7B, Falcon-7B, Gemma-7B, LLaMA-2-13B
  - correctness reported as delta PPL, not multiplicative ratio
  - figures split into separate, larger PDFs for readability

Inputs:
  - results_ppl_lossless_logs/eval_ppl_lossless_*.log
  - results_scaling_truee2e_originfirst_*.json

Outputs:
  - figures/kv_delta_ppl_correctness_revision.pdf
  - figures/kv_request_ttft_revision.pdf
  - figures/kv_repair_overhead_revision.pdf
  - figures/kv_repair_overhead_stage12_revision.pdf
  - figures/kv_repair_overhead_stage23_revision.pdf
  - figures/kv_delta_ppl_stage12_revision.pdf
  - figures/kv_delta_ppl_stage23_revision.pdf
  - figures/kv_request_ttft_stage12_revision.pdf
  - figures/kv_request_ttft_stage23_revision.pdf
  - figures/kv_revision_metrics.json
  - figures/kv_revision_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_LOSSLESS_DIR = SCRIPT_DIR / "results_ppl_lossless_logs"

MODEL_ORDER = [
    "llama2-7b",
    "falcon-7b",
    "gemma-7b",
    "llama2-13b",
]

DISPLAY_NAMES = {
    "llama2-7b": "LLaMA-2-7B",
    "falcon-7b": "Falcon-7B",
    "gemma-7b": "Gemma-7B",
    "llama2-13b": "LLaMA-2-13B",
}

BAR_STYLE_REPAIR = {
    "facecolor": "#bcd6f0",
    "edgecolor": "#264b73",
    "linewidth": 1.2,
    "hatch": "///",
}

BAR_STYLE_STALE = {
    "facecolor": "#f0b7b3",
    "edgecolor": "#8a302c",
    "linewidth": 1.2,
    "hatch": "\\\\",
}

LINE_STYLE_FULL_RESET = {
    "fmt": "o-",
    "linewidth": 2.2,
    "markersize": 6.5,
    "capsize": 4,
    "color": "#cf5b52",
    "markerfacecolor": "#cf5b52",
    "markeredgecolor": "#5c221d",
    "label": "Full-Reset",
}

LINE_STYLE_REPAIR = {
    "fmt": "s--",
    "linewidth": 2.2,
    "markersize": 6.5,
    "capsize": 4,
    "color": "#4c78a8",
    "markerfacecolor": "#ffffff",
    "markeredgecolor": "#24486b",
    "label": "ProgressiveServe-Repair",
}

MODEL_ALIASES = {
    "llama": "llama2-7b",
    "llama-7b": "llama2-7b",
    "llama2-7b": "llama2-7b",
    "falcon": "falcon-7b",
    "falcon-7b": "falcon-7b",
    "gemma": "gemma-7b",
    "gemma7b": "gemma-7b",
    "gemma-7b": "gemma-7b",
    "llama-13b": "llama2-13b",
    "llama2-13b": "llama2-13b",
}

SCALING_FILES = {
    ("llama2-7b", "1to2"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_llama-7b_1to2_20260414_175506.json",
    ("llama2-7b", "2to3"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_llama-7b_2to3_20260414_175506.json",
    ("falcon-7b", "1to2"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_falcon-7b_1to2_20260415_183646.json",
    ("falcon-7b", "2to3"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_falcon-7b_2to3_20260415_183646.json",
    ("gemma-7b", "1to2"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_gemma-7b_1to2_20260414_233039.json",
    ("gemma-7b", "2to3"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_gemma-7b_2to3_20260414_233039.json",
    ("llama2-13b", "1to2"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_llama-13b_1to2_20260414_222721.json",
    ("llama2-13b", "2to3"): SCRIPT_DIR / "results_scaling_truee2e_originfirst_llama-13b_2to3_20260414_222721.json",
}

LOG_MODEL_RE = re.compile(r"^\s*model=([^\s|]+)", re.MULTILINE)
SUMMARY_BLOCK_RE = re.compile(
    r"Multi-Mode Comparison \(corpus_ppl\)\s*"
    r"Mode\s+Stage1\(A\)\s+Stage2\(B\)\s+Stage3\(C\)\s*"
    r"-+\s*"
    r"full_recompute\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*"
    r"naive\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*"
    r"surgery\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.MULTILINE | re.DOTALL,
)


def canonical_model_name(name: str) -> str:
    return MODEL_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())


def display_name(name: str) -> str:
    return DISPLAY_NAMES[canonical_model_name(name)]


def get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["hatch.linewidth"] = 1.1
    return plt


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_lossless_log(path: Path) -> tuple[str, dict[str, dict[str, float]]] | None:
    text = path.read_text(encoding="utf-8")
    model_match = LOG_MODEL_RE.search(text)
    summary_match = SUMMARY_BLOCK_RE.search(text)
    if model_match is None or summary_match is None:
        return None

    model = canonical_model_name(model_match.group(1))
    vals = [float(item) for item in summary_match.groups()]
    return (
        model,
        {
            "full_recompute": {"A": vals[0], "B": vals[1], "C": vals[2]},
            "naive": {"A": vals[3], "B": vals[4], "C": vals[5]},
            "surgery": {"A": vals[6], "B": vals[7], "C": vals[8]},
        },
    )


def load_correctness_map(lossless_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    correctness_map: dict[str, dict[str, dict[str, float]]] = {}
    for path in sorted(lossless_dir.glob("eval_ppl_lossless_*.log")):
        parsed = parse_lossless_log(path)
        if parsed is None:
            continue
        model, results = parsed
        correctness_map[model] = results
    return correctness_map


def compute_delta_metrics(results: dict[str, dict[str, float]]) -> dict[str, float]:
    oracle_b = results["full_recompute"]["B"]
    oracle_c = results["full_recompute"]["C"]
    repair_b = results["surgery"]["B"]
    repair_c = results["surgery"]["C"]
    stale_b = results["naive"]["B"]
    stale_c = results["naive"]["C"]
    return {
        "repair_delta_B": repair_b - oracle_b,
        "stale_delta_B": stale_b - oracle_b,
        "repair_delta_C": repair_c - oracle_c,
        "stale_delta_C": stale_c - oracle_c,
        "oracle_B": oracle_b,
        "oracle_C": oracle_c,
        "repair_B": repair_b,
        "repair_C": repair_c,
        "stale_B": stale_b,
        "stale_C": stale_c,
    }


def repair_cost_us_per_token(point: dict[str, Any]) -> float:
    actual_t = max(1, int(point.get("actual_T", 1)))
    sync_s = float(point["partial"].get("sync_s", 0.0))
    reconcile_s = float(point["partial"].get("reconcile_s", 0.0))
    return ((sync_s + reconcile_s) / actual_t) * 1e6


def annotate_bars(ax, bars) -> None:
    ymin, ymax = ax.get_ylim()
    span = max(ymax - ymin, 1e-6)
    offset = span * 0.025
    for bar in bars:
        value = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"
        ax.text(x, y, f"{value:+.2f}", ha="center", va=va, fontsize=10)


def delta_ppl_panel_spec(suffix: str) -> tuple[str, str, tuple[float, float]]:
    if suffix == "B":
        return (
            "Delta PPL on span B after Stage 1->2",
            "Delta PPL vs Oracle-Recompute",
            (-1.15, 3.4),
        )
    if suffix == "C":
        return (
            "Delta PPL on span C after Stage 2->3",
            "Delta PPL vs Oracle-Recompute",
            (-0.8, 8.0),
        )
    raise ValueError(f"Unsupported suffix: {suffix}")


def draw_delta_ppl_panel(
    ax,
    correctness_map: dict[str, dict[str, dict[str, float]]],
    suffix: str,
    include_legend: bool = False,
) -> None:
    xs = list(range(len(MODEL_ORDER)))
    labels = [display_name(model) for model in MODEL_ORDER]
    width = 0.34

    _title, ylabel, ylim = delta_ppl_panel_spec(suffix)
    repair_key = f"repair_delta_{suffix}"
    stale_key = f"stale_delta_{suffix}"
    metrics = [compute_delta_metrics(correctness_map[model]) for model in MODEL_ORDER]
    repair_vals = [row[repair_key] for row in metrics]
    stale_vals = [row[stale_key] for row in metrics]

    repair_bars = ax.bar(
        [x - width / 2 for x in xs],
        repair_vals,
        width=width,
        label="ProgressiveServe-Repair",
        **BAR_STYLE_REPAIR,
    )
    stale_bars = ax.bar(
        [x + width / 2 for x in xs],
        stale_vals,
        width=width,
        label="Stale-Reuse",
        **BAR_STYLE_STALE,
    )

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    annotate_bars(ax, repair_bars)
    annotate_bars(ax, stale_bars)

    if include_legend:
        ax.legend(
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.14),
            handlelength=1.8,
        )


def build_delta_ppl_figure(
    correctness_map: dict[str, dict[str, dict[str, float]]],
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.0), constrained_layout=True)
    draw_delta_ppl_panel(axes[0], correctness_map, "B")
    draw_delta_ppl_panel(axes[1], correctness_map, "C")

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=1.8,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def ttft_panel_title(stage_key: str) -> str:
    if stage_key == "1to2":
        return "First post-promotion TTFT: Stage 1->2"
    if stage_key == "2to3":
        return "First post-promotion TTFT: Stage 2->3"
    raise ValueError(f"Unsupported stage key: {stage_key}")


def draw_request_ttft_panel(
    ax,
    scaling_data: dict[str, Any],
    stage_key: str,
    include_legend: bool = False,
) -> None:
    scaling = scaling_data["scaling"]
    ts = [int(point["actual_T"]) for point in scaling]
    full_reset_ms = [float(point["origin"]["request_ttft_s"]) * 1000.0 for point in scaling]
    repair_ms = [float(point["partial"]["request_ttft_s"]) * 1000.0 for point in scaling]
    full_reset_err = [
        float(point["origin"].get("request_ttft_s_stats", {}).get("std", 0.0)) * 1000.0
        for point in scaling
    ]
    repair_err = [
        float(point["partial"].get("request_ttft_s_stats", {}).get("std", 0.0)) * 1000.0
        for point in scaling
    ]

    ax.errorbar(ts, full_reset_ms, yerr=full_reset_err, **LINE_STYLE_FULL_RESET)
    ax.errorbar(ts, repair_ms, yerr=repair_err, **LINE_STYLE_REPAIR)
    ax.set_xlabel("Accumulated prefix length T")
    ax.set_ylabel("Request TTFT (ms)")
    ax.grid(True, alpha=0.28, linewidth=0.8)

    saved_ms = full_reset_ms[-1] - repair_ms[-1]
    ax.annotate(
        f"-{saved_ms:.1f} ms at T={ts[-1]}",
        xy=(ts[-1], repair_ms[-1]),
        xytext=(ts[-1] - 360, repair_ms[-1] + 32),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        fontsize=11,
    )

    if include_legend:
        ax.legend(
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.14),
            handlelength=2.2,
        )


def build_request_ttft_figure(
    scaling_1to2: dict[str, Any],
    scaling_2to3: dict[str, Any],
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.0), constrained_layout=True, sharey=True)

    draw_request_ttft_panel(axes[0], scaling_1to2, "1to2")
    draw_request_ttft_panel(axes[1], scaling_2to3, "2to3")

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_legend,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=2.2,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_delta_ppl_single_panel_figure(
    correctness_map: dict[str, dict[str, dict[str, float]]],
    suffix: str,
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    draw_delta_ppl_panel(ax, correctness_map, suffix, include_legend=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_request_ttft_single_panel_figure(
    scaling_data: dict[str, Any],
    stage_key: str,
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    draw_request_ttft_panel(ax, scaling_data, stage_key, include_legend=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def repair_overhead_panel_spec(stage_key: str) -> tuple[str, str]:
    if stage_key == "1to2":
        return ("Normalized repair work: Stage 1->2", "#2E8B57")
    if stage_key == "2to3":
        return ("Normalized repair work: Stage 2->3", "#8C6BB1")
    raise ValueError(f"Unsupported stage key: {stage_key}")


def draw_repair_overhead_panel(
    ax,
    scaling_data: dict[str, Any],
    stage_key: str,
    include_title: bool = False,
    include_ylabel: bool = True,
) -> None:
    title, color = repair_overhead_panel_spec(stage_key)
    scaling = scaling_data["scaling"]
    ts = [int(point["actual_T"]) for point in scaling]
    overheads = [repair_cost_us_per_token(point) for point in scaling]

    ax.plot(
        ts,
        overheads,
        "o-",
        linewidth=2.3,
        markersize=6.5,
        color=color,
    )
    if include_title:
        ax.set_title(title)
    ax.set_xlabel("Accumulated prefix length T")
    if include_ylabel:
        ax.set_ylabel("Repair overhead (us/token)")
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.annotate(
        f"{overheads[-1]:.1f} us/token",
        xy=(ts[-1], overheads[-1]),
        xytext=(ts[-1] - 250, overheads[-1] + 22),
        arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
        fontsize=11,
    )


def build_repair_overhead_figure(
    scaling_1to2: dict[str, Any],
    scaling_2to3: dict[str, Any],
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 3.0), constrained_layout=True, sharey=True)
    draw_repair_overhead_panel(
        axes[0],
        scaling_1to2,
        "1to2",
        include_title=True,
        include_ylabel=True,
    )
    draw_repair_overhead_panel(
        axes[1],
        scaling_2to3,
        "2to3",
        include_title=True,
        include_ylabel=False,
    )
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_repair_overhead_single_panel_figure(
    scaling_data: dict[str, Any],
    stage_key: str,
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(6.8, 3.0), constrained_layout=True)
    draw_repair_overhead_panel(ax, scaling_data, stage_key, include_title=False, include_ylabel=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def summarize_scaling_tmax(scaling_data: dict[str, Any]) -> dict[str, float]:
    point = max(scaling_data["scaling"], key=lambda row: int(row["actual_T"]))
    origin_ttft_ms = float(point["origin"]["request_ttft_s"]) * 1000.0
    repair_ttft_ms = float(point["partial"]["request_ttft_s"]) * 1000.0
    return {
        "tmax": int(point["actual_T"]),
        "origin_ttft_ms": origin_ttft_ms,
        "repair_ttft_ms": repair_ttft_ms,
        "ttft_saved_ms": origin_ttft_ms - repair_ttft_ms,
        "ttft_speedup_x": float(point["request_only_speedup_ratio"]),
        "repair_overhead_us_per_token": repair_cost_us_per_token(point),
        "fast_path_success_pct": float(point["partial"].get("surgery_success_rate", 0.0)) * 100.0,
    }


def build_metrics_rows(
    correctness_map: dict[str, dict[str, dict[str, float]]],
    scaling_map: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        deltas = compute_delta_metrics(correctness_map[model])
        stage_1to2 = summarize_scaling_tmax(scaling_map[(model, "1to2")])
        stage_2to3 = summarize_scaling_tmax(scaling_map[(model, "2to3")])
        rows.append(
            {
                "model": model,
                "display_name": display_name(model),
                **{key: round(value, 4) for key, value in deltas.items()},
                "stage_1to2": {key: round(value, 4) for key, value in stage_1to2.items()},
                "stage_2to3": {key: round(value, 4) for key, value in stage_2to3.items()},
            }
        )
    return rows


def write_metrics(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "kv_revision_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "kv_revision_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "repair_delta_B",
                "stale_delta_B",
                "repair_delta_C",
                "stale_delta_C",
                "stage1to2_tmax",
                "stage1to2_origin_ttft_ms",
                "stage1to2_repair_ttft_ms",
                "stage1to2_ttft_saved_ms",
                "stage1to2_repair_overhead_us_per_token",
                "stage2to3_tmax",
                "stage2to3_origin_ttft_ms",
                "stage2to3_repair_ttft_ms",
                "stage2to3_ttft_saved_ms",
                "stage2to3_repair_overhead_us_per_token",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["display_name"],
                    row["repair_delta_B"],
                    row["stale_delta_B"],
                    row["repair_delta_C"],
                    row["stale_delta_C"],
                    row["stage_1to2"]["tmax"],
                    row["stage_1to2"]["origin_ttft_ms"],
                    row["stage_1to2"]["repair_ttft_ms"],
                    row["stage_1to2"]["ttft_saved_ms"],
                    row["stage_1to2"]["repair_overhead_us_per_token"],
                    row["stage_2to3"]["tmax"],
                    row["stage_2to3"]["origin_ttft_ms"],
                    row["stage_2to3"]["repair_ttft_ms"],
                    row["stage_2to3"]["ttft_saved_ms"],
                    row["stage_2to3"]["repair_overhead_us_per_token"],
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build revised KV paper artifacts.")
    parser.add_argument(
        "--lossless-dir",
        type=str,
        default=str(DEFAULT_LOSSLESS_DIR),
        help="Directory containing eval_ppl_lossless_*.log files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the revised figure PDFs and summaries will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir).resolve()
    lossless_dir = Path(args.lossless_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    correctness_map = load_correctness_map(lossless_dir)
    missing_models = [model for model in MODEL_ORDER if model not in correctness_map]
    if missing_models:
        raise FileNotFoundError(f"Missing correctness logs for: {missing_models}")

    scaling_map: dict[tuple[str, str], dict[str, Any]] = {}
    for key, path in SCALING_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing scaling JSON: {path}")
        scaling_map[key] = load_json(path)

    build_delta_ppl_figure(
        correctness_map=correctness_map,
        out_path=output_dir / "kv_delta_ppl_correctness_revision.pdf",
    )
    build_delta_ppl_single_panel_figure(
        correctness_map=correctness_map,
        suffix="B",
        out_path=output_dir / "kv_delta_ppl_stage12_revision.pdf",
    )
    build_delta_ppl_single_panel_figure(
        correctness_map=correctness_map,
        suffix="C",
        out_path=output_dir / "kv_delta_ppl_stage23_revision.pdf",
    )
    build_request_ttft_figure(
        scaling_1to2=scaling_map[("llama2-7b", "1to2")],
        scaling_2to3=scaling_map[("llama2-7b", "2to3")],
        out_path=output_dir / "kv_request_ttft_revision.pdf",
    )
    build_request_ttft_single_panel_figure(
        scaling_data=scaling_map[("llama2-7b", "1to2")],
        stage_key="1to2",
        out_path=output_dir / "kv_request_ttft_stage12_revision.pdf",
    )
    build_request_ttft_single_panel_figure(
        scaling_data=scaling_map[("llama2-7b", "2to3")],
        stage_key="2to3",
        out_path=output_dir / "kv_request_ttft_stage23_revision.pdf",
    )
    build_repair_overhead_figure(
        scaling_1to2=scaling_map[("llama2-7b", "1to2")],
        scaling_2to3=scaling_map[("llama2-7b", "2to3")],
        out_path=output_dir / "kv_repair_overhead_revision.pdf",
    )
    build_repair_overhead_single_panel_figure(
        scaling_data=scaling_map[("llama2-7b", "1to2")],
        stage_key="1to2",
        out_path=output_dir / "kv_repair_overhead_stage12_revision.pdf",
    )
    build_repair_overhead_single_panel_figure(
        scaling_data=scaling_map[("llama2-7b", "2to3")],
        stage_key="2to3",
        out_path=output_dir / "kv_repair_overhead_stage23_revision.pdf",
    )

    rows = build_metrics_rows(correctness_map, scaling_map)
    write_metrics(rows, output_dir)

    print("Wrote revised KV artifacts:")
    print(f"  - {output_dir / 'kv_delta_ppl_correctness_revision.pdf'}")
    print(f"  - {output_dir / 'kv_delta_ppl_stage12_revision.pdf'}")
    print(f"  - {output_dir / 'kv_delta_ppl_stage23_revision.pdf'}")
    print(f"  - {output_dir / 'kv_request_ttft_revision.pdf'}")
    print(f"  - {output_dir / 'kv_request_ttft_stage12_revision.pdf'}")
    print(f"  - {output_dir / 'kv_request_ttft_stage23_revision.pdf'}")
    print(f"  - {output_dir / 'kv_repair_overhead_revision.pdf'}")
    print(f"  - {output_dir / 'kv_repair_overhead_stage12_revision.pdf'}")
    print(f"  - {output_dir / 'kv_repair_overhead_stage23_revision.pdf'}")
    print(f"  - {output_dir / 'kv_revision_metrics.json'}")
    print(f"  - {output_dir / 'kv_revision_metrics.csv'}")


if __name__ == "__main__":
    main()
