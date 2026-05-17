#!/usr/bin/env python3
"""
Origin-first variant of benchmark_scaling_true_e2e.py.

This keeps the original worker logic intact and only changes the top-level
measurement order from:
  partial -> origin
to:
  origin -> partial
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import benchmark_scaling_true_e2e as base

SCRIPT_DIR = Path(__file__).resolve().parent


def run_true_e2e_scaling_origin_first(args) -> dict[str, Any]:
    model_name = base.canonical_model_name(args.model)
    cache_condition = base.resolve_cache_condition(args)
    strict_partial_path = base.DEFAULT_STRICT_PARTIAL_PATH and not bool(args.allow_partial_fallback)
    model_cfg = base.MODELS[model_name]
    max_model_len = (
        int(args.max_model_len)
        if int(args.max_model_len) > 0
        else int(model_cfg.get("max_model_len", base.DEFAULT_MAX_MODEL_LEN))
    )
    if args.token_counts.strip():
        token_counts = sorted(set(int(x.strip()) for x in args.token_counts.split(",") if x.strip()))
    else:
        token_counts = base.model_default_token_counts(model_name, max_model_len)
    if not token_counts:
        raise ValueError("No valid token counts resolved. Please provide --token-counts explicitly.")

    impl_name, impl_path = base.resolve_progressive_impl(args.progressive_impl)
    requested_gpu_memory_utilization = base.resolve_requested_gpu_memory_utilization(
        args.gpu_memory_utilization,
        model_cfg,
    )
    effective_gpu_memory_utilization = requested_gpu_memory_utilization
    gpu_mem_plan_preview = base.build_nonintrusive_gpu_utilization_plan(
        requested_gpu_memory_utilization,
    )
    effective_tensor_parallel_size = (
        max(1, int(args.tensor_parallel_size))
        if int(args.tensor_parallel_size) > 0
        else max(1, int(model_cfg.get("tensor_parallel_size", 1)))
    )

    print("\n" + "=" * 78)
    print(f"  TRUE E2E TOKEN SCALING BENCHMARK  (model={model_name}, stage={args.stage})")
    print("=" * 78)
    print(f"  Token counts:         {token_counts}")
    print(f"  Runs per T:           {args.num_runs}")
    print(f"  max_model_len:        {max_model_len}")
    print(f"  gpu_mem_util req:     {requested_gpu_memory_utilization}")
    print(f"  gpu_mem_util preview: {effective_gpu_memory_utilization}")
    print(f"  tp_size:              {effective_tensor_parallel_size}")
    print(f"  progressive_impl:     {impl_name} ({impl_path})")
    print(f"  cache_condition:      {cache_condition}")
    print(f"  drop_caches timeout:  {float(args.drop_caches_timeout_s):g}s")
    print(f"  strict_partial_path:  {strict_partial_path}")
    print(f"  source warmup runs:   {args.stage1_warmup_runs}")
    print(f"  stage timeout (s):    {args.stage_transition_timeout_s}")
    print("  branch order:         origin -> partial")
    print("=" * 78)
    base.log_gpu_memory_utilization_plan("  [GPU preview]", gpu_mem_plan_preview)

    scaling: list[dict[str, Any]] = []
    worker_gpu_memory_utils: list[float] = []
    worker_gpu_memory_clamps: list[bool] = []
    for target_t in token_counts:
        partial_runs: list[dict[str, Any]] = []
        origin_runs: list[dict[str, Any]] = []
        skip_reason = None

        print(f"\n  [T={target_t}]")
        for run_index in range(int(args.num_runs)):
            print(f"    run {run_index + 1}/{args.num_runs}: origin...", flush=True)
            origin = base.run_worker_subprocess(args, "origin", target_t, run_index)
            if origin.get("skip"):
                skip_reason = origin.get("skip_reason", "worker requested skip")
                break
            origin_runs.append(origin)
            worker_gpu_memory_utils.append(float(origin["gpu_memory_utilization_effective"]))
            worker_gpu_memory_clamps.append(bool(origin.get("gpu_memory_utilization_clamped", False)))

            print(f"    run {run_index + 1}/{args.num_runs}: partial...", flush=True)
            partial = base.run_worker_subprocess(args, "partial", target_t, run_index)
            if partial.get("skip"):
                skip_reason = partial.get("skip_reason", "worker requested skip")
                break
            partial_runs.append(partial)
            worker_gpu_memory_utils.append(float(partial["gpu_memory_utilization_effective"]))
            worker_gpu_memory_clamps.append(bool(partial.get("gpu_memory_utilization_clamped", False)))

        if skip_reason is not None:
            print(f"    skip - {skip_reason}")
            continue

        point = base.summarize_scaling_point(target_t, partial_runs, origin_runs)
        scaling.append(point)

        p = point["partial"]
        o = point["origin"]
        print(
            f"    request_ttft: partial={base._fmt_stat(p['request_ttft_s_stats'], int(args.num_runs))}  "
            f"origin={base._fmt_stat(o['request_ttft_s_stats'], int(args.num_runs))}  "
            f"({point['request_only_speedup_ratio']:.2f}x)"
        )
        print(
            f"    e2e_request:  partial={base._fmt_stat(p['e2e_from_request_s_stats'], int(args.num_runs))}  "
            f"origin={base._fmt_stat(o['e2e_from_request_s_stats'], int(args.num_runs))}  "
            f"({point['e2e_speedup_ratio']:.2f}x)"
        )
        print(
            f"    partial path: prefetch={p['prefetch_s']:.3f}s  activation={p['activation_s']:.3f}s  "
            f"sync={p['sync_s']:.4f}s  reconcile={p['reconcile_s']:.4f}s"
        )

    results = {
        "measurement_mode": "true_e2e",
        "model": model_name,
        "stage": args.stage,
        "num_runs": int(args.num_runs),
        "timestamp": datetime.datetime.now().isoformat(),
        "cache_condition": cache_condition,
        "strict_partial_path": strict_partial_path,
        "max_model_len": max_model_len,
        "gpu_name": base.torch.cuda.get_device_name(0) if base.torch.cuda.is_available() else "N/A",
        "gpu_total_mem_gb": (
            round(base.torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            if base.torch.cuda.is_available()
            else 0.0
        ),
        "gpu_memory_utilization_requested": requested_gpu_memory_utilization,
        "gpu_memory_utilization_effective": effective_gpu_memory_utilization,
        "gpu_memory_utilization_worker_min": (
            round(min(worker_gpu_memory_utils), 6)
            if worker_gpu_memory_utils
            else effective_gpu_memory_utilization
        ),
        "gpu_memory_utilization_worker_max": (
            round(max(worker_gpu_memory_utils), 6)
            if worker_gpu_memory_utils
            else effective_gpu_memory_utilization
        ),
        "gpu_memory_utilization_clamped_any": any(worker_gpu_memory_clamps),
        "gpu_memory_reserve_gb": float(base.DEFAULT_GPU_FREE_RESERVE_GB),
        "tensor_parallel_size_effective": effective_tensor_parallel_size,
        "progressive_impl": impl_name,
        "progressive_impl_path": impl_path,
        "progressive_path": model_cfg["progressive_path"],
        "stage_b_checkpoint": model_cfg.get("stage_b_checkpoint"),
        "stage_c_checkpoint": model_cfg.get("stage_c_checkpoint"),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "enable_prefix_caching": bool(model_cfg.get("enable_prefix_caching", True)),
        "disable_sliding_window": bool(model_cfg.get("disable_sliding_window", False)),
        "enforce_eager": bool(args.enforce_eager),
        "require_drop_caches": cache_condition == "cold",
        "skip_drop_caches": cache_condition == "warm",
        "drop_caches_timeout_s": float(args.drop_caches_timeout_s),
        "stage1_warmup_runs": int(args.stage1_warmup_runs),
        "stage_transition_timeout_s": float(args.stage_transition_timeout_s),
        "fixed_question": base.FIXED_QUESTION,
        "scaling": scaling,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved -> {args.output}")
    return results


def build_parser():
    parser = base.build_parser()
    parser.description = (
        "Origin-first variant of the paper-grade true E2E token scaling benchmark "
        "for stage transition timing"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.cache_condition = base.resolve_cache_condition(args)

    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode.")
        result = base.worker_measure_branch(args)
        with open(args.worker_output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    if args.plot:
        with open(args.plot) as f:
            data = json.load(f)
        save_path = args.plot.replace(".json", ".png")
        base.plot_results(data, save_path=save_path)
        return

    model_name = base.canonical_model_name(args.model)
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(
            SCRIPT_DIR / f"results_scaling_truee2e_originfirst_{model_name}_{args.stage}_{ts}.json"
        )

    results = run_true_e2e_scaling_origin_first(args)
    if not args.no_plot:
        plot_path = args.output.replace(".json", ".png")
        base.plot_results(results, save_path=plot_path)


if __name__ == "__main__":
    main()
