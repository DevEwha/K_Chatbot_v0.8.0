#!/usr/bin/env python3
"""
build_kv_paper_artifacts.py
===========================

논문용 KV evaluation figure/table 아티팩트 생성기.

입력:
  - results_ppl_lossless_{model}_{mode}_{timestamp}.json
  - results_scaling_truee2e_{model}_{stage}_{timestamp}.json

출력:
  - figures/kv_correctness.pdf
  - figures/kv_post_transition_ttft.pdf
  - figures/kv_repair_cost_scaling.pdf
  - figures/kv_summary_metrics.json
  - figures/kv_summary_table.csv
  - figures/kv_summary_rows.tex

기본 동작:
  - results-dir 에서 최신 correctness/scaling 결과를 자동 탐색
  - 대표 모델 1개(기본: llama2-7b)에 대해 본문용 3개 figure 생성
  - 4개 최종 모델에 대해 KV summary row 생성
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figures"
DEFAULT_REP_MODEL = "llama2-7b"
DEFAULT_STAGE = "1to2"
DEFAULT_SUMMARY_MODELS = ["llama2-7b", "llama2-13b", "falcon-7b", "gemma-7b"]
CORRECTNESS_MODES = ("full_recompute", "naive", "surgery")

MODEL_ALIASES = {
    "llama": "llama2-7b",
    "llama-7b": "llama2-7b",
    "llama_kd_ssd32": "llama2-7b",
    "llama2-7b": "llama2-7b",
    "llama-13b": "llama2-13b",
    "llama2-13b": "llama2-13b",
    "falcon": "falcon-7b",
    "falcon_kd_ssd32": "falcon-7b",
    "falcon-7b": "falcon-7b",
    "gemma7b": "gemma-7b",
    "gemma-7b": "gemma-7b",
}

DISPLAY_NAMES = {
    "llama2-7b": "LLaMA-2-7B",
    "llama2-13b": "LLaMA-2-13B",
    "falcon-7b": "Falcon-7B",
    "gemma-7b": "Gemma-7B",
}


def canonical_model_name(name: str) -> str:
    key = str(name).strip().lower()
    return MODEL_ALIASES.get(key, key)


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


def _correctness_result_is_paper_ready(data: dict[str, Any]) -> bool:
    paper = data.get("paper_readiness", {})
    if not isinstance(paper, dict):
        return False
    return bool(paper.get("eligible", False))


def _correctness_group_id(data: dict[str, Any]) -> str:
    group = data.get("comparison_group", {})
    if not isinstance(group, dict):
        return ""
    return str(group.get("group_id", "")).strip()


def _correctness_group_signature(data: dict[str, Any]) -> str:
    group = data.get("comparison_group", {})
    if not isinstance(group, dict):
        return ""
    return str(group.get("config_signature", "")).strip()


def find_latest_correctness_results(results_dir: Path) -> dict[str, dict[str, Path]]:
    grouped: dict[str, dict[str, dict[str, tuple[float, Path, dict[str, Any]]]]] = {}
    for path in sorted(results_dir.glob("results_ppl_lossless_*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        model = canonical_model_name(str(data.get("model", "")))
        mode = str(data.get("mode", "")).strip()
        if not model or not mode:
            continue
        if mode not in CORRECTNESS_MODES:
            continue
        if not _correctness_result_is_paper_ready(data):
            continue
        group_id = _correctness_group_id(data)
        group_signature = _correctness_group_signature(data)
        if not group_id or not group_signature:
            continue
        grouped.setdefault(model, {})
        grouped[model].setdefault(group_id, {})
        current = grouped[model][group_id].get(mode)
        mtime = path.stat().st_mtime
        if current is None or mtime >= current[0]:
            grouped[model][group_id][mode] = (mtime, path, data)

    resolved: dict[str, dict[str, Path]] = {}
    for model, per_group in grouped.items():
        best_group_id = ""
        best_group_mtime = -1.0
        for group_id, per_mode in per_group.items():
            if set(per_mode.keys()) != set(CORRECTNESS_MODES):
                continue
            signatures = {item[2]["comparison_group"]["config_signature"] for item in per_mode.values()}
            if len(signatures) != 1:
                continue
            group_mtime = max(item[0] for item in per_mode.values())
            if group_mtime >= best_group_mtime:
                best_group_id = group_id
                best_group_mtime = group_mtime
        if best_group_id:
            resolved[model] = {
                mode: per_group[best_group_id][mode][1]
                for mode in CORRECTNESS_MODES
            }
    return resolved


def find_latest_scaling_results(results_dir: Path, stage: str) -> dict[str, Path]:
    grouped: dict[str, tuple[float, Path]] = {}
    for path in sorted(results_dir.glob("results_scaling_truee2e_*.json")):
        try:
            data = load_json(path)
        except Exception:
            continue
        if str(data.get("measurement_mode", "")) != "true_e2e":
            continue
        if str(data.get("stage", "")) != stage:
            continue
        model = canonical_model_name(str(data.get("model", "")))
        if not model:
            continue
        mtime = path.stat().st_mtime
        current = grouped.get(model)
        if current is None or mtime >= current[0]:
            grouped[model] = (mtime, path)
    return {model: item[1] for model, item in grouped.items()}


def get_turn_corpus_ppl(result: dict[str, Any], turn_key: str) -> float:
    aggregate = result.get("aggregate", {})
    turn = aggregate.get(turn_key, {})
    return float(turn.get("corpus_ppl", 0.0))


def correctness_ratios(per_mode_data: dict[str, dict[str, Any]]) -> dict[str, float]:
    if "full_recompute" not in per_mode_data:
        raise KeyError("Missing full_recompute correctness result.")
    if "surgery" not in per_mode_data:
        raise KeyError("Missing surgery correctness result.")
    if "naive" not in per_mode_data:
        raise KeyError("Missing naive correctness result.")

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


def build_kv_correctness_figure(
    representative_model: str,
    per_mode_data: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    plt = get_plt()
    ratios = correctness_ratios(per_mode_data)
    xlabels = ["B after 1->2", "C after 2->3"]
    repair_vals = [ratios["repair_oracle_B"], ratios["repair_oracle_C"]]
    stale_vals = [ratios["stale_oracle_B"], ratios["stale_oracle_C"]]

    xs = [0, 1]
    width = 0.32
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(
        [x - width / 2 for x in xs],
        repair_vals,
        width=width,
        color="#1f77b4",
        label="ProgressiveServe-Repair / Oracle",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        stale_vals,
        width=width,
        color="#d62728",
        label="Stale-Reuse / Oracle",
    )
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.5, label="Oracle baseline")
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Perplexity / Oracle")
    ax.set_title(f"KV Correctness ({display_model_name(representative_model)})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_kv_post_transition_ttft_figure(
    representative_model: str,
    scaling_data: dict[str, Any],
    out_path: Path,
) -> None:
    plt = get_plt()
    scaling = scaling_data.get("scaling", [])
    if not scaling:
        raise ValueError("No scaling points found in representative scaling JSON.")

    ts = [int(point["actual_T"]) for point in scaling]
    repair = [float(point["partial"]["request_ttft_s"]) for point in scaling]
    reset = [float(point["origin"]["request_ttft_s"]) for point in scaling]
    repair_err = [
        float(point["partial"].get("request_ttft_s_stats", {}).get("std", 0.0)) for point in scaling
    ]
    reset_err = [
        float(point["origin"].get("request_ttft_s_stats", {}).get("std", 0.0)) for point in scaling
    ]

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
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
    ax.set_xlabel("Accumulated prefix length T")
    ax.set_ylabel("First post-promotion request TTFT (s)")
    ax.set_title(f"Post-Promotion Request TTFT ({display_model_name(representative_model)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
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


def build_kv_repair_cost_figure(
    representative_model: str,
    scaling_data: dict[str, Any],
    out_path: Path,
) -> None:
    plt = get_plt()
    scaling = scaling_data.get("scaling", [])
    if not scaling:
        raise ValueError("No scaling points found in representative scaling JSON.")

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

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.errorbar(
        ts,
        means,
        yerr=stds,
        fmt="o-",
        color="#2ca02c",
        linewidth=2,
        markersize=6,
        capsize=4,
    )
    ax.set_xlabel("Accumulated prefix length T")
    ax.set_ylabel("Repair overhead (us/token)")
    ax.set_title(f"Repair Cost Scaling ({display_model_name(representative_model)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def max_t_scaling_point(scaling_data: dict[str, Any]) -> dict[str, Any]:
    scaling = scaling_data.get("scaling", [])
    if not scaling:
        raise ValueError("No scaling data found.")
    return max(scaling, key=lambda point: int(point.get("actual_T", 0)))


def summary_metrics_for_model(
    model: str,
    correctness_map: dict[str, dict[str, Path]],
    scaling_map: dict[str, Path],
) -> dict[str, Any]:
    canonical = canonical_model_name(model)
    metrics: dict[str, Any] = {
        "model": canonical,
        "display_name": display_model_name(canonical),
    }

    correctness_paths = correctness_map.get(canonical, {})
    loaded_correctness = {
        mode: load_json(path)
        for mode, path in correctness_paths.items()
        if mode in ("full_recompute", "naive", "surgery")
    }
    try:
        ratios = correctness_ratios(loaded_correctness)
        metrics.update(ratios)
    except Exception:
        metrics.update(
            {
                "repair_oracle_B": None,
                "stale_oracle_B": None,
                "repair_oracle_C": None,
                "stale_oracle_C": None,
            }
        )

    scaling_path = scaling_map.get(canonical)
    if scaling_path is None:
        metrics.update(
            {
                "repair_cost_tmax_us_per_token": None,
                "fast_path_success_pct": None,
                "post_promo_gain_tmax_x": None,
                "tmax": None,
            }
        )
        return metrics

    scaling_data = load_json(scaling_path)
    point = max_t_scaling_point(scaling_data)
    vals = repair_cost_runs(point)
    metrics.update(
        {
            "repair_cost_tmax_us_per_token": round(sum(vals) / len(vals), 2),
            "fast_path_success_pct": round(
                float(point["partial"].get("surgery_success_rate", 0.0)) * 100.0,
                1,
            ),
            "post_promo_gain_tmax_x": round(float(point.get("request_only_speedup_ratio", 0.0)), 2),
            "tmax": int(point.get("actual_T", 0)),
        }
    )
    return metrics


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
                "fast_path_success_pct",
                "post_promo_gain_tmax_x",
                "tmax",
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
                    row.get("fast_path_success_pct"),
                    row.get("post_promo_gain_tmax_x"),
                    row.get("tmax"),
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
                f"{tex_cell(row.get('repair_cost_tmax_us_per_token'), 2)} & "
                f"{tex_cell(row.get('fast_path_success_pct'), 1)} & "
                f"{tex_cell(row.get('post_promo_gain_tmax_x'), 2)} \\\\\n"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build paper-ready KV correctness/repair figures and table rows."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing results_ppl_lossless_*.json and results_scaling_truee2e_*.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where kv_*.pdf and summary files will be written",
    )
    parser.add_argument(
        "--representative-model",
        type=str,
        default=DEFAULT_REP_MODEL,
        help="Representative model for the 3 paper figures",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=DEFAULT_STAGE,
        choices=["1to2", "2to3"],
        help="Scaling stage to use for true-e2e plots and summary",
    )
    parser.add_argument(
        "--representative-scaling",
        type=str,
        default="",
        help="Optional explicit results_scaling_truee2e_*.json path for the representative model",
    )
    parser.add_argument(
        "--summary-models",
        type=str,
        default=",".join(DEFAULT_SUMMARY_MODELS),
        help="Comma-separated model list for kv_summary_rows.tex",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    representative_model = canonical_model_name(args.representative_model)
    summary_models = [
        canonical_model_name(item)
        for item in args.summary_models.split(",")
        if item.strip()
    ]

    correctness_map = find_latest_correctness_results(results_dir)
    scaling_map = find_latest_scaling_results(results_dir, stage=args.stage)

    rep_correctness_paths = correctness_map.get(representative_model, {})
    rep_loaded_correctness = {
        mode: load_json(path)
        for mode, path in rep_correctness_paths.items()
        if mode in ("full_recompute", "naive", "surgery")
    }

    if {"full_recompute", "naive", "surgery"} <= set(rep_loaded_correctness.keys()):
        try:
            build_kv_correctness_figure(
                representative_model=representative_model,
                per_mode_data=rep_loaded_correctness,
                out_path=output_dir / "kv_correctness.pdf",
            )
            print(f"Saved: {output_dir / 'kv_correctness.pdf'}")
        except ModuleNotFoundError as exc:
            print(f"Skip kv_correctness.pdf: plotting dependency missing ({exc}).")
    else:
        print(
            f"Skip kv_correctness.pdf: missing representative correctness results for "
            f"{representative_model} (need full_recompute, naive, surgery)."
        )

    rep_scaling_path = (
        Path(args.representative_scaling).resolve()
        if args.representative_scaling.strip()
        else scaling_map.get(representative_model)
    )
    if rep_scaling_path is not None and rep_scaling_path.exists():
        rep_scaling_data = load_json(rep_scaling_path)
        try:
            build_kv_post_transition_ttft_figure(
                representative_model=representative_model,
                scaling_data=rep_scaling_data,
                out_path=output_dir / "kv_post_transition_ttft.pdf",
            )
            build_kv_repair_cost_figure(
                representative_model=representative_model,
                scaling_data=rep_scaling_data,
                out_path=output_dir / "kv_repair_cost_scaling.pdf",
            )
            print(f"Saved: {output_dir / 'kv_post_transition_ttft.pdf'}")
            print(f"Saved: {output_dir / 'kv_repair_cost_scaling.pdf'}")
        except ModuleNotFoundError as exc:
            print(f"Skip scaling figures: plotting dependency missing ({exc}).")
    else:
        print(
            f"Skip scaling figures: no representative scaling JSON found for "
            f"{representative_model} stage={args.stage}."
        )

    metrics_rows = [
        summary_metrics_for_model(model, correctness_map=correctness_map, scaling_map=scaling_map)
        for model in summary_models
    ]
    write_summary_artifacts(metrics_rows, output_dir=output_dir)
    print(f"Saved: {output_dir / 'kv_summary_metrics.json'}")
    print(f"Saved: {output_dir / 'kv_summary_table.csv'}")
    print(f"Saved: {output_dir / 'kv_summary_rows.tex'}")


if __name__ == "__main__":
    main()
