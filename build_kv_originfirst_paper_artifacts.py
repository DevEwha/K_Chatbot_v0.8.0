#!/usr/bin/env python3
"""
Build paper-ready KV evaluation artifacts from the currently completed logs.

This script is intentionally scoped to the user-confirmed inputs for the
ASPLOS draft:
  - correctness from results_ppl_lossless_logs/*.log
  - representative repair-cost / post-promotion latency from the completed
    LLaMA-2-7B origin-first true-e2e runs

It writes:
  - figures/kv_correctness.pdf
  - figures/kv_post_transition_ttft.pdf
  - figures/kv_repair_cost_scaling.pdf
  - figures/kv_summary_metrics.json
  - figures/kv_summary_table.csv
  - figures/kv_summary_rows.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LOSSLESS_DIR = SCRIPT_DIR / "results_ppl_lossless_logs"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_REP_SCALING_1TO2 = (
    SCRIPT_DIR / "results_scaling_truee2e_originfirst_llama-7b_1to2_20260414_175506.json"
)
DEFAULT_REP_SCALING_2TO3 = (
    SCRIPT_DIR / "results_scaling_truee2e_originfirst_llama-7b_2to3_20260414_175506.json"
)
DEFAULT_MODELS = ["llama2-7b", "llama2-13b", "falcon-7b", "gemma-7b"]
ORIGINFIRST_SCALING_PREFIX = {
    "llama2-7b": "llama-7b",
    "llama2-13b": "llama-13b",
    "falcon-7b": "falcon-7b",
    "gemma-7b": "gemma-7b",
}

MODEL_ALIASES = {
    "llama": "llama2-7b",
    "llama-7b": "llama2-7b",
    "llama2-7b": "llama2-7b",
    "llama2-13b": "llama2-13b",
    "llama-13b": "llama2-13b",
    "falcon": "falcon-7b",
    "falcon-7b": "falcon-7b",
    "gemma": "gemma-7b",
    "gemma7b": "gemma-7b",
    "gemma-7b": "gemma-7b",
}

DISPLAY_NAMES = {
    "llama2-7b": "LLaMA-2-7B",
    "llama2-13b": "LLaMA-2-13B",
    "falcon-7b": "Falcon-7B",
    "gemma-7b": "Gemma-7B",
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


def display_model_name(name: str) -> str:
    canonical = canonical_model_name(name)
    return DISPLAY_NAMES.get(canonical, canonical)


def get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    return plt


def load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_latest_originfirst_scaling_path(
    results_dir: Path,
    model: str,
    stage: str,
) -> Path | None:
    prefix = ORIGINFIRST_SCALING_PREFIX.get(model)
    if prefix is None:
        return None
    candidates = sorted(
        results_dir.glob(f"results_scaling_truee2e_originfirst_{prefix}_{stage}_*.json")
    )
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_lossless_log(path: Path) -> tuple[str, dict[str, dict[str, Any]]] | None:
    text = path.read_text(encoding="utf-8")
    model_match = LOG_MODEL_RE.search(text)
    summary_match = SUMMARY_BLOCK_RE.search(text)
    if model_match is None or summary_match is None:
        return None

    model = canonical_model_name(model_match.group(1))
    vals = [float(item) for item in summary_match.groups()]
    mode_rows = {
        "full_recompute": vals[0:3],
        "naive": vals[3:6],
        "surgery": vals[6:9],
    }
    result: dict[str, dict[str, Any]] = {}
    for mode, (a_val, b_val, c_val) in mode_rows.items():
        result[mode] = {
            "model": model,
            "mode": mode,
            "aggregate": {
                "stage1_turn1_A": {"corpus_ppl": a_val},
                "stage2_turn2_B": {"corpus_ppl": b_val},
                "stage3_turn3_C": {"corpus_ppl": c_val},
            },
            "source_log": str(path),
        }
    return model, result


def load_correctness_map(lossless_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    correctness_map: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(lossless_dir.glob("eval_ppl_lossless_*.log")):
        parsed = parse_lossless_log(path)
        if parsed is None:
            continue
        model, mode_map = parsed
        correctness_map[model] = mode_map
    return correctness_map


def get_turn_corpus_ppl(result: dict[str, Any], turn_key: str) -> float:
    aggregate = result.get("aggregate", {})
    turn = aggregate.get(turn_key, {})
    return float(turn.get("corpus_ppl", 0.0))


def correctness_ratios(per_mode_data: dict[str, dict[str, Any]]) -> dict[str, float]:
    oracle = per_mode_data["full_recompute"]
    repair = per_mode_data["surgery"]
    stale = per_mode_data["naive"]

    oracle_b = get_turn_corpus_ppl(oracle, "stage2_turn2_B")
    oracle_c = get_turn_corpus_ppl(oracle, "stage3_turn3_C")
    repair_b = get_turn_corpus_ppl(repair, "stage2_turn2_B")
    repair_c = get_turn_corpus_ppl(repair, "stage3_turn3_C")
    stale_b = get_turn_corpus_ppl(stale, "stage2_turn2_B")
    stale_c = get_turn_corpus_ppl(stale, "stage3_turn3_C")

    def safe_ratio(num: float, den: float) -> float:
        return (num / den) if den > 0 else 0.0

    return {
        "repair_oracle_B": safe_ratio(repair_b, oracle_b),
        "stale_oracle_B": safe_ratio(stale_b, oracle_b),
        "repair_oracle_C": safe_ratio(repair_c, oracle_c),
        "stale_oracle_C": safe_ratio(stale_c, oracle_c),
    }


def relative_pct(per_mode_data: dict[str, dict[str, Any]]) -> dict[str, float]:
    ratios = correctness_ratios(per_mode_data)
    return {
        key: (value - 1.0) * 100.0
        for key, value in ratios.items()
    }


def build_kv_correctness_figure(
    correctness_map: dict[str, dict[str, dict[str, Any]]],
    out_path: Path,
) -> None:
    plt = get_plt()
    models = [model for model in DEFAULT_MODELS if model in correctness_map]
    labels = [display_model_name(model) for model in models]
    metrics = {model: correctness_ratios(correctness_map[model]) for model in models}

    xs = list(range(len(models)))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)

    repair_b = [metrics[model]["repair_oracle_B"] for model in models]
    stale_b = [metrics[model]["stale_oracle_B"] for model in models]
    repair_c = [metrics[model]["repair_oracle_C"] for model in models]
    stale_c = [metrics[model]["stale_oracle_C"] for model in models]

    axes[0].bar(
        [x - width / 2 for x in xs],
        repair_b,
        width=width,
        color="#8fb2ff",
        edgecolor="#1f3c88",
        hatch="//",
        label="ProgressiveServe-Repair / Oracle",
    )
    axes[0].bar(
        [x + width / 2 for x in xs],
        stale_b,
        width=width,
        color="#ff9aa2",
        edgecolor="#8a1c2c",
        hatch="\\\\",
        label="Stale-Reuse / Oracle",
    )
    axes[0].axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Oracle baseline")
    axes[0].set_title("B after 1->2")
    axes[0].set_ylabel("Perplexity / Oracle")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=22, ha="right")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(
        [x - width / 2 for x in xs],
        repair_c,
        width=width,
        color="#8fb2ff",
        edgecolor="#1f3c88",
        hatch="//",
        label="ProgressiveServe-Repair / Oracle",
    )
    axes[1].bar(
        [x + width / 2 for x in xs],
        stale_c,
        width=width,
        color="#ff9aa2",
        edgecolor="#8a1c2c",
        hatch="\\\\",
        label="Stale-Reuse / Oracle",
    )
    axes[1].axhline(1.0, color="gray", linestyle=":", linewidth=1.5)
    axes[1].set_title("C after 2->3")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=22, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def repair_cost_runs(point: dict[str, Any]) -> list[float]:
    actual_t = max(1, int(point.get("actual_T", 1)))
    runs = point.get("partial_runs", [])
    vals: list[float] = []
    for row in runs:
        sync_s = float(row.get("sync_s", 0.0))
        reconcile_s = float(row.get("reconcile_s", 0.0))
        vals.append(((sync_s + reconcile_s) / actual_t) * 1e6)
    if vals:
        return vals
    sync_s = float(point["partial"].get("sync_s", 0.0))
    reconcile_s = float(point["partial"].get("reconcile_s", 0.0))
    return [((sync_s + reconcile_s) / actual_t) * 1e6]


def build_kv_post_transition_ttft_figure(
    representative_model: str,
    scaling_1to2: dict[str, Any],
    scaling_2to3: dict[str, Any],
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)

    for ax, scaling_data, title in [
        (axes[0], scaling_1to2, "1->2"),
        (axes[1], scaling_2to3, "2->3"),
    ]:
        scaling = scaling_data.get("scaling", [])
        ts = [int(point["actual_T"]) for point in scaling]
        repair = [float(point["partial"]["request_ttft_s"]) for point in scaling]
        reset = [float(point["origin"]["request_ttft_s"]) for point in scaling]
        repair_err = [
            float(point["partial"].get("request_ttft_s_stats", {}).get("std", 0.0))
            for point in scaling
        ]
        reset_err = [
            float(point["origin"].get("request_ttft_s_stats", {}).get("std", 0.0))
            for point in scaling
        ]

        ax.errorbar(
            ts,
            reset,
            yerr=reset_err,
            fmt="o-",
            color="#d62728",
            linewidth=2,
            markersize=6,
            capsize=4,
            label="Full-Reset",
        )
        ax.errorbar(
            ts,
            repair,
            yerr=repair_err,
            fmt="s-",
            color="#1f77b4",
            linewidth=2,
            markersize=6,
            capsize=4,
            label="ProgressiveServe-Repair",
        )
        ax.set_title(title)
        ax.set_xlabel("Accumulated prefix length T")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("First post-promotion request TTFT (s)")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Post-Promotion Request TTFT ({display_model_name(representative_model)})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_kv_repair_cost_figure(
    representative_model: str,
    scaling_1to2: dict[str, Any],
    scaling_2to3: dict[str, Any],
    out_path: Path,
) -> None:
    plt = get_plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), sharey=True)

    for ax, scaling_data, title, color in [
        (axes[0], scaling_1to2, "1->2", "#2ca02c"),
        (axes[1], scaling_2to3, "2->3", "#9467bd"),
    ]:
        scaling = scaling_data.get("scaling", [])
        ts = [int(point["actual_T"]) for point in scaling]
        means = []
        stds = []
        for point in scaling:
            vals = repair_cost_runs(point)
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                std = math.sqrt(variance)
            else:
                std = 0.0
            means.append(mean)
            stds.append(std)

        ax.errorbar(
            ts,
            means,
            yerr=stds,
            fmt="o-",
            color=color,
            linewidth=2,
            markersize=6,
            capsize=4,
        )
        ax.set_title(title)
        ax.set_xlabel("Accumulated prefix length T")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Repair overhead (us/token)")
    fig.suptitle(f"Repair Cost Scaling ({display_model_name(representative_model)})", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def max_t_scaling_point(scaling_data: dict[str, Any]) -> dict[str, Any]:
    scaling = scaling_data.get("scaling", [])
    if not scaling:
        raise ValueError("No scaling data found.")
    return max(scaling, key=lambda point: int(point.get("actual_T", 0)))


def summarize_scaling_point(point: dict[str, Any]) -> dict[str, float | int]:
    vals = repair_cost_runs(point)
    return {
        "tmax": int(point.get("actual_T", 0)),
        "repair_cost_tmax_us_per_token": round(sum(vals) / len(vals), 2),
        "fast_path_success_pct": round(
            float(point["partial"].get("surgery_success_rate", 0.0)) * 100.0,
            1,
        ),
        "post_promo_gain_tmax_x": round(float(point.get("request_only_speedup_ratio", 0.0)), 2),
        "reset_ttft_tmax_s": round(float(point["origin"].get("request_ttft_s", 0.0)), 4),
        "repair_ttft_tmax_s": round(float(point["partial"].get("request_ttft_s", 0.0)), 4),
    }


def tex_cell(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return r"\textit{?}"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_summary_artifacts(metrics_rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "kv_summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "kv_summary_table.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "repair_oracle_B_x",
                "stale_oracle_B_x",
                "repair_oracle_C_x",
                "stale_oracle_C_x",
                "repair_cost_tmax_us_per_token",
                "post_promo_gain_tmax_x",
                "tmax",
                "stage_2to3_repair_cost_tmax_us_per_token",
                "stage_2to3_post_promo_gain_tmax_x",
                "stage_2to3_reset_ttft_tmax_s",
                "stage_2to3_repair_ttft_tmax_s",
            ]
        )
        for row in metrics_rows:
            writer.writerow(
                [
                    row["display_name"],
                    row.get("repair_oracle_B"),
                    row.get("stale_oracle_B"),
                    row.get("repair_oracle_C"),
                    row.get("stale_oracle_C"),
                    row.get("repair_cost_tmax_us_per_token"),
                    row.get("post_promo_gain_tmax_x"),
                    row.get("tmax"),
                    row.get("stage_2to3_repair_cost_tmax_us_per_token"),
                    row.get("stage_2to3_post_promo_gain_tmax_x"),
                    row.get("stage_2to3_reset_ttft_tmax_s"),
                    row.get("stage_2to3_repair_ttft_tmax_s"),
                ]
            )

    tex_path = output_dir / "kv_summary_rows.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        for row in metrics_rows:
            f.write(
                f"{row['display_name']} & "
                f"{tex_cell(row.get('repair_oracle_B'), 2)} & "
                f"{tex_cell(row.get('stale_oracle_B'), 2)} & "
                f"{tex_cell(row.get('repair_oracle_C'), 2)} & "
                f"{tex_cell(row.get('stale_oracle_C'), 2)} & "
                f"{tex_cell(row.get('post_promo_gain_tmax_x'), 2)} \\\\\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper-ready KV artifacts from the completed origin-first logs."
    )
    parser.add_argument(
        "--lossless-dir",
        type=str,
        default=str(DEFAULT_LOSSLESS_DIR),
        help="Directory containing eval_ppl_lossless_*.log files.",
    )
    parser.add_argument(
        "--rep-scaling-1to2",
        type=str,
        default=str(DEFAULT_REP_SCALING_1TO2),
        help="Representative LLaMA-2-7B 1->2 scaling JSON.",
    )
    parser.add_argument(
        "--rep-scaling-2to3",
        type=str,
        default=str(DEFAULT_REP_SCALING_2TO3),
        help="Representative LLaMA-2-7B 2->3 scaling JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the paper artifacts will be written.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    lossless_dir = Path(args.lossless_dir).resolve()
    rep_scaling_1to2_path = Path(args.rep_scaling_1to2).resolve()
    rep_scaling_2to3_path = Path(args.rep_scaling_2to3).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = SCRIPT_DIR.resolve()

    correctness_map = load_correctness_map(lossless_dir)
    rep_model = "llama2-7b"
    if rep_model not in correctness_map:
        raise FileNotFoundError(
            f"Representative correctness log for {rep_model} was not found in {lossless_dir}."
        )

    build_kv_correctness_figure(
        correctness_map=correctness_map,
        out_path=output_dir / "kv_correctness.pdf",
    )

    scaling_1to2 = load_json(rep_scaling_1to2_path)
    scaling_2to3 = load_json(rep_scaling_2to3_path)
    build_kv_post_transition_ttft_figure(
        representative_model=rep_model,
        scaling_1to2=scaling_1to2,
        scaling_2to3=scaling_2to3,
        out_path=output_dir / "kv_post_transition_ttft.pdf",
    )
    build_kv_repair_cost_figure(
        representative_model=rep_model,
        scaling_1to2=scaling_1to2,
        scaling_2to3=scaling_2to3,
        out_path=output_dir / "kv_repair_cost_scaling.pdf",
    )

    metrics_rows: list[dict[str, Any]] = []
    for model in DEFAULT_MODELS:
        row: dict[str, Any] = {
            "model": model,
            "display_name": display_model_name(model),
            "repair_oracle_B": None,
            "stale_oracle_B": None,
            "repair_oracle_C": None,
            "stale_oracle_C": None,
            "repair_cost_tmax_us_per_token": None,
            "fast_path_success_pct": None,
            "post_promo_gain_tmax_x": None,
            "tmax": None,
            "stage_2to3_repair_cost_tmax_us_per_token": None,
            "stage_2to3_post_promo_gain_tmax_x": None,
            "stage_2to3_reset_ttft_tmax_s": None,
            "stage_2to3_repair_ttft_tmax_s": None,
        }
        if model in correctness_map:
            row.update(correctness_ratios(correctness_map[model]))
            row.update(
                {
                    "repair_vs_oracle_B_pct": round(relative_pct(correctness_map[model])["repair_oracle_B"], 2),
                    "stale_vs_oracle_B_pct": round(relative_pct(correctness_map[model])["stale_oracle_B"], 2),
                    "repair_vs_oracle_C_pct": round(relative_pct(correctness_map[model])["repair_oracle_C"], 2),
                    "stale_vs_oracle_C_pct": round(relative_pct(correctness_map[model])["stale_oracle_C"], 2),
                }
            )
        stage_1to2_path = find_latest_originfirst_scaling_path(results_dir, model, "1to2")
        if stage_1to2_path is not None:
            stage_1to2_data = load_json(stage_1to2_path)
            row.update(summarize_scaling_point(max_t_scaling_point(stage_1to2_data)))

        stage_2to3_path = find_latest_originfirst_scaling_path(results_dir, model, "2to3")
        if stage_2to3_path is not None:
            stage_2to3_data = load_json(stage_2to3_path)
            stage_2to3_summary = summarize_scaling_point(max_t_scaling_point(stage_2to3_data))
            row.update(
                {
                    "stage_2to3_repair_cost_tmax_us_per_token": stage_2to3_summary[
                        "repair_cost_tmax_us_per_token"
                    ],
                    "stage_2to3_post_promo_gain_tmax_x": stage_2to3_summary[
                        "post_promo_gain_tmax_x"
                    ],
                    "stage_2to3_reset_ttft_tmax_s": stage_2to3_summary["reset_ttft_tmax_s"],
                    "stage_2to3_repair_ttft_tmax_s": stage_2to3_summary["repair_ttft_tmax_s"],
                }
            )
        metrics_rows.append(row)

    write_summary_artifacts(metrics_rows, output_dir=output_dir)

    print(f"Saved: {output_dir / 'kv_correctness.pdf'}")
    print(f"Saved: {output_dir / 'kv_post_transition_ttft.pdf'}")
    print(f"Saved: {output_dir / 'kv_repair_cost_scaling.pdf'}")
    print(f"Saved: {output_dir / 'kv_summary_metrics.json'}")
    print(f"Saved: {output_dir / 'kv_summary_table.csv'}")
    print(f"Saved: {output_dir / 'kv_summary_rows.tex'}")


if __name__ == "__main__":
    main()
