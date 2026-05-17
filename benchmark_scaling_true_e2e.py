#!/usr/bin/env python3
"""
benchmark_scaling_true_e2e.py — paper-grade true E2E stage-transition scaling
=============================================================================

기존 benchmark_scaling.py 가 transition 비용을 분해해서 보는 microbenchmark라면,
이 스크립트는 각 branch(partial/origin)를 fresh source-stage 상태에서 다시 시작해
"transition request 이후 first token까지"를 진짜 end-to-end로 재는 본문용 벤치마크다.

핵심 원칙:
  - 각 T, 각 run, 각 branch는 별도 Python worker 프로세스에서 실행
  - worker는 fresh source stage(Stage1 or Stage2)를 준비한 뒤 prefix를 다시 채움
  - partial branch:
      prefetch -> activate -> sync -> reconcile -> next request first token
  - origin branch:
      prefetch -> activate -> cache invalidate -> next request full-prefill first token
  - 즉, partial/origin 모두 동일한 source-stage prefix 상태에서 출발한다

주의:
  - model load / stage1 warmup은 timed region에 포함되지 않는다.
  - timed E2E는 "transition request start -> next request first token" 이다.
  - 필요 시 --require-drop-caches 로 cold prefetch 조건을 강제할 수 있다.
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

from shared_model_configs import (
    BENCHMARK_MODELS as MODELS,
    DEFAULT_MODEL,
    MODEL_CHOICES,
    canonical_model_name,
    model_default_token_counts,
)

import vllm.config

vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_scaling import (  # noqa: E402
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_PROGRESSIVE_IMPL,
    FIXED_QUESTION,
    STAGE_CONFIG,
    _fmt_stat,
    _stats,
    build_context_of_length,
    build_prompt_with_question,
    clear_hidden_cache,
    clear_runtime_state,
    cpu_mem_gb,
    get_model_handle,
    gpu_mem_gb,
    gpu_temp_c,
    kv_cache_gb_theoretical,
    prepare_progressive_imports,
    reconcile_after_transition,
    reset_prefix_cache,
    resolve_progressive_impl,
    resolve_transition_seq_len,
    warmup_generate,
)

DEFAULT_NUM_RUNS = 5
DEFAULT_STAGE1_WARMUP_RUNS = 3
DEFAULT_STAGE_TIMEOUT_S = 300.0
DEFAULT_DROP_CACHES_TIMEOUT_S = 60.0
DEFAULT_REQUEST_MAX_TOKENS = 1
DEFAULT_CACHE_CONDITION = "cold"
DEFAULT_STRICT_PARTIAL_PATH = True
DEFAULT_GPU_FREE_RESERVE_GB = 1.0
CACHE_CONDITIONS = ("cold", "warm")
WORKER_BRANCHES = ("partial", "origin")


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_requested_gpu_memory_utilization(
    override_value: float,
    config: dict[str, Any],
) -> float:
    return (
        float(override_value)
        if float(override_value) > 0
        else float(config.get("gpu_memory_utilization", 0.4))
    )


def clamp_gpu_memory_utilization_to_free_vram(
    requested_utilization: float,
    reserve_gb: float = DEFAULT_GPU_FREE_RESERVE_GB,
) -> tuple[float, dict[str, Any]]:
    requested = float(requested_utilization)
    info: dict[str, Any] = {
        "requested_utilization": requested,
        "effective_utilization": requested,
        "clamped": False,
        "free_gb": None,
        "total_gb": None,
        "reserve_gb": float(reserve_gb),
        "max_util_from_free": None,
        "reason": "cuda_unavailable",
        "error": None,
    }
    if not torch.cuda.is_available():
        return requested, info

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except RuntimeError as exc:
        info.update(
            {
                "reason": "cuda_mem_get_info_failed",
                "error": str(exc),
            }
        )
        return requested, info
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    reserve_bytes = max(0, int(float(reserve_gb) * (1024 ** 3)))
    usable_bytes = max(0, free_bytes - reserve_bytes)
    max_util_from_free = usable_bytes / total_bytes if total_bytes > 0 else requested
    effective = min(requested, max_util_from_free)

    info.update(
        {
            "effective_utilization": effective,
            "clamped": effective + 1e-9 < requested,
            "free_gb": free_gb,
            "total_gb": total_gb,
            "max_util_from_free": max_util_from_free,
            "reason": "ok",
        }
    )
    return effective, info


def build_nonintrusive_gpu_utilization_plan(
    requested_utilization: float,
    reason: str = "parent_probe_skipped",
) -> dict[str, Any]:
    requested = float(requested_utilization)
    return {
        "requested_utilization": requested,
        "effective_utilization": requested,
        "clamped": False,
        "free_gb": None,
        "total_gb": None,
        "reserve_gb": float(DEFAULT_GPU_FREE_RESERVE_GB),
        "max_util_from_free": None,
        "reason": reason,
        "error": None,
    }


def log_gpu_memory_utilization_plan(prefix: str, plan: dict[str, Any]) -> None:
    requested = float(plan["requested_utilization"])
    effective = float(plan["effective_utilization"])
    if plan.get("free_gb") is None or plan.get("total_gb") is None:
        reason = str(plan.get("reason", "unknown"))
        if reason == "parent_probe_skipped":
            print(
                f"{prefix} gpu_memory_utilization={effective:.4f} "
                "(parent skipped live GPU probe; worker will decide)"
            )
        elif reason == "cuda_mem_get_info_failed":
            print(
                f"{prefix} gpu_memory_utilization={effective:.4f} "
                f"(live GPU probe failed; no clamp applied: {plan.get('error', '')})"
            )
        else:
            print(f"{prefix} gpu_memory_utilization={effective:.4f} (CUDA unavailable; no clamp applied)")
        return

    free_gb = float(plan["free_gb"])
    total_gb = float(plan["total_gb"])
    reserve_gb = float(plan["reserve_gb"])
    if bool(plan.get("clamped")):
        print(
            f"{prefix} gpu_memory_utilization clamped {requested:.4f} -> {effective:.4f} "
            f"(free={free_gb:.2f}GiB / total={total_gb:.2f}GiB, reserve={reserve_gb:.2f}GiB)"
        )
    else:
        print(
            f"{prefix} gpu_memory_utilization={effective:.4f} "
            f"(free={free_gb:.2f}GiB / total={total_gb:.2f}GiB, reserve={reserve_gb:.2f}GiB)"
        )


def get_prefetch_status_snapshot(model: Any) -> dict[str, Any]:
    getter = getattr(model, "get_prefetch_status", None)
    if getter is None:
        return {}
    try:
        status = getter()
    except Exception as exc:
        return {"status_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(status, dict):
        return status
    return {"raw_status": repr(status)}


def format_prefetch_status(status: dict[str, Any]) -> str:
    if not status:
        return "<unavailable>"
    try:
        return json.dumps(status, sort_keys=True, default=str)
    except TypeError:
        return repr(status)


def wait_for_prefetch_or_raise(
    model: Any,
    timeout_s: float,
    *,
    checkpoint_path: str,
    context: str,
) -> dict[str, Any]:
    ready = model.wait_for_prefetch(timeout_s=timeout_s)
    status = get_prefetch_status_snapshot(model)
    if not ready:
        raise TimeoutError(
            f"{context}: prefetch did not become ready within {float(timeout_s):.1f}s "
            f"for checkpoint={checkpoint_path}. status={format_prefetch_status(status)}"
        )
    return status


def paper_drop_caches(
    require_success: bool,
    timeout_s: float = DEFAULT_DROP_CACHES_TIMEOUT_S,
) -> bool:
    error_detail = ""
    timeout_s = max(1.0, float(timeout_s))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    try:
        if os.geteuid() == 0:
            cmd = ["sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
        else:
            cmd = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        ok = out.returncode == 0
        error_detail = (out.stderr or out.stdout or "").strip()
    except Exception as exc:
        ok = False
        error_detail = str(exc)

    if not ok and require_success:
        detail_suffix = f" Details: {error_detail}" if error_detail else ""
        raise RuntimeError(
            "OS page cache drop failed while enforcing cold cache condition. "
            "If this is a long-running job, your sudo credential may have expired; "
            "refresh it with 'sudo -v' or use the launcher keepalive. "
            f"(timeout={timeout_s:g}s)"
            f"{detail_suffix}"
        )

    time.sleep(2)
    return ok


def paper_warm_file_cache(path: str, require_success: bool) -> bool:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found for warm-cache preparation: {path}")
    if not ckpt_path.is_file():
        raise RuntimeError(f"Warm-cache preparation expects a file checkpoint: {path}")

    ok = False
    try:
        with ckpt_path.open("rb", buffering=0) as handle:
            while True:
                chunk = handle.read(64 * 1024 * 1024)
                if not chunk:
                    break
        ok = True
    except Exception:
        ok = False

    if not ok and require_success:
        raise RuntimeError("Warm file-cache preparation failed while enforcing warm cache condition.")
    return ok


def resolve_cache_condition(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "cache_condition", "") or "").strip().lower()
    if explicit and explicit not in CACHE_CONDITIONS:
        raise ValueError(
            f"Unsupported cache condition: {explicit}. Expected one of {CACHE_CONDITIONS}."
        )
    if bool(args.skip_drop_caches) and bool(args.require_drop_caches):
        raise ValueError(
            "--skip-drop-caches and --require-drop-caches cannot be used together."
        )
    if bool(args.skip_drop_caches):
        if explicit and explicit != "warm":
            raise ValueError("--skip-drop-caches conflicts with --cache-condition cold.")
        return "warm"
    if bool(args.require_drop_caches):
        if explicit and explicit != "cold":
            raise ValueError("--require-drop-caches conflicts with --cache-condition warm.")
        return "cold"
    return explicit or DEFAULT_CACHE_CONDITION


def enforce_cache_condition(
    cache_condition: str,
    checkpoint_path: str,
    *,
    drop_caches_timeout_s: float = DEFAULT_DROP_CACHES_TIMEOUT_S,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if cache_condition == "cold":
        ok = paper_drop_caches(require_success=True, timeout_s=drop_caches_timeout_s)
        return {
            "cache_condition": cache_condition,
            "method": "drop_caches",
            "ok": bool(ok),
            "elapsed_s": time.perf_counter() - t0,
        }
    if cache_condition == "warm":
        ok = paper_warm_file_cache(checkpoint_path, require_success=True)
        return {
            "cache_condition": cache_condition,
            "method": "warm_file_cache",
            "ok": bool(ok),
            "elapsed_s": time.perf_counter() - t0,
        }
    raise ValueError(f"Unsupported cache condition: {cache_condition}")


def enforce_strict_partial_path(
    *,
    inner_model,
    sync_api_available: bool,
    sync_seq_len: int,
    reconcile: dict[str, Any],
) -> None:
    if inner_model is None:
        raise RuntimeError("Strict partial-path benchmark requires model.model to be available.")
    if not sync_api_available:
        raise RuntimeError(
            "Strict partial-path benchmark requires sync_persistent_cache()."
        )
    if int(sync_seq_len) <= 0:
        raise RuntimeError(
            "Strict partial-path benchmark requires a positive transition seq_len."
        )

    reconcile_path = str(reconcile.get("reconcile_path", "none"))
    boundary = reconcile.get("boundary")
    surgery_ok = bool(reconcile.get("surgery_ok", False))
    partial_profile = reconcile.get("partial_profile")

    if boundary is None:
        raise RuntimeError(
            "Strict partial-path benchmark requires a valid recompute boundary, "
            f"but got reconcile_path={reconcile_path!r}."
        )
    if reconcile_path != "surgery":
        raise RuntimeError(
            "Strict partial-path benchmark forbids fallback/degraded reconcile paths, "
            f"but got {reconcile_path!r}."
        )
    if not surgery_ok:
        raise RuntimeError("Strict partial-path benchmark requires successful KV surgery.")
    if partial_profile is not None:
        raise RuntimeError(
            "Strict partial-path benchmark expected surgery-only reconciliation but "
            "observed partial recompute fallback metadata."
        )


def invalidate_caches_for_origin(llm, model, prefix_caching_enabled: bool) -> float:
    cuda_sync()
    t0 = time.perf_counter()
    if prefix_caching_enabled:
        reset_prefix_cache(llm)
    clear_hidden_cache(model)
    cuda_sync()
    return time.perf_counter() - t0


def load_progressive_llm(
    model_name: str,
    progressive_impl: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    enforce_eager: bool,
) -> tuple[LLM, Any, Any, dict[str, Any], str, str]:
    impl_name, impl_path = resolve_progressive_impl(progressive_impl)
    prepare_progressive_imports(impl_path)
    from progressive_for_causal_lm import ProgressiveForCausalLM  # noqa: E402

    config = dict(MODELS[model_name])
    model_path = config["progressive_path"]
    prefix_caching_enabled = bool(config.get("enable_prefix_caching", True))
    trust_remote_code = bool(config.get("trust_remote_code", True))
    disable_sliding_window = bool(config.get("disable_sliding_window", False))

    with open(os.path.join(model_path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    try:
        ModelRegistry.register_model(arch, ProgressiveForCausalLM)
    except Exception:
        pass

    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        enable_prefix_caching=prefix_caching_enabled,
        disable_sliding_window=disable_sliding_window,
    )
    model = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()

    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()

    return llm, model, tokenizer, config, impl_name, impl_path


def warm_current_stage(llm, model, tokenizer, warmup_runs: int):
    if warmup_runs > 0:
        warmup_generate(llm, tokenizer, n=warmup_runs, label="")
    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        try:
            model.model.clear_persistent_buffers()
        except Exception:
            pass
    clear_runtime_state(llm, model)


def prepare_source_stage(
    llm,
    model,
    tokenizer,
    config: dict[str, Any],
    stage: str,
    warmup_runs: int,
    timeout_s: float,
):
    stage_cfg = STAGE_CONFIG[stage]
    if stage_cfg["setup_stage"] is None:
        warm_current_stage(llm, model, tokenizer, warmup_runs)
        return

    setup_cfg = STAGE_CONFIG[stage_cfg["setup_stage"]]
    setup_ckpt = config[setup_cfg["checkpoint_key"]]
    if not os.path.exists(setup_ckpt):
        raise FileNotFoundError(f"Source stage checkpoint not found: {setup_ckpt}")

    getattr(model, setup_cfg["prefetch_fn"])(setup_ckpt)
    prefetch_status = wait_for_prefetch_or_raise(
        model,
        timeout_s=timeout_s,
        checkpoint_path=setup_ckpt,
        context=(
            "Untimed source-stage setup "
            f"({stage_cfg['setup_stage']} -> {stage}, {setup_cfg['advance_fn']})"
        ),
    )
    ok = getattr(model, setup_cfg["advance_fn"])(wait_if_needed=False)
    if not ok:
        raise RuntimeError(
            "Untimed source-stage setup activation failed even though prefetch "
            f"reported ready. checkpoint={setup_ckpt}. "
            f"status={format_prefetch_status(prefetch_status)}"
        )
    reset_prefix_cache(llm)
    warm_current_stage(llm, model, tokenizer, warmup_runs)


def worker_measure_branch(args: argparse.Namespace) -> dict[str, Any]:
    model_name = canonical_model_name(args.model)
    cache_condition = resolve_cache_condition(args)
    strict_partial_path = DEFAULT_STRICT_PARTIAL_PATH and not bool(args.allow_partial_fallback)
    config = dict(MODELS[model_name])
    max_model_len = (
        int(args.max_model_len)
        if int(args.max_model_len) > 0
        else int(config.get("max_model_len", DEFAULT_MAX_MODEL_LEN))
    )
    requested_gpu_memory_utilization = resolve_requested_gpu_memory_utilization(
        args.gpu_memory_utilization,
        config,
    )
    gpu_memory_utilization, gpu_mem_plan = clamp_gpu_memory_utilization_to_free_vram(
        requested_gpu_memory_utilization
    )
    tensor_parallel_size = (
        max(1, int(args.tensor_parallel_size))
        if int(args.tensor_parallel_size) > 0
        else max(1, int(config.get("tensor_parallel_size", 1)))
    )
    prefix_caching_enabled = bool(config.get("enable_prefix_caching", True))
    request_params = SamplingParams(temperature=0.0, max_tokens=DEFAULT_REQUEST_MAX_TOKENS)
    fill_params = SamplingParams(temperature=0.0, max_tokens=1)

    llm = model = tokenizer = None
    try:
        log_gpu_memory_utilization_plan(
            f"  [Worker {args.branch} T={int(args.target_t)} run={int(args.run_index) + 1}]",
            gpu_mem_plan,
        )
        llm, model, tokenizer, config, impl_name, impl_path = load_progressive_llm(
            model_name=model_name,
            progressive_impl=args.progressive_impl,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=bool(args.enforce_eager),
        )
        prepare_source_stage(
            llm=llm,
            model=model,
            tokenizer=tokenizer,
            config=config,
            stage=args.stage,
            warmup_runs=int(args.stage1_warmup_runs),
            timeout_s=float(args.stage_transition_timeout_s),
        )

        content, prompt_fill, actual_T = build_context_of_length(tokenizer, int(args.target_t))
        if actual_T + 80 > max_model_len:
            return {
                "skip": True,
                "skip_reason": f"actual_T+80 exceeds max_model_len={max_model_len}",
                "target_T": int(args.target_t),
                "actual_T": actual_T,
            }

        temp_c = gpu_temp_c()
        clear_runtime_state(llm, model)

        cuda_sync()
        t0 = time.perf_counter()
        setup_out = llm.generate([prompt_fill], fill_params)
        cuda_sync()
        source_fill_s = time.perf_counter() - t0

        dummy_response = setup_out[0].outputs[0].text.strip() or "."
        prompt_with_q = build_prompt_with_question(tokenizer, content, dummy_response)
        n_total = len(tokenizer.encode(prompt_with_q))
        n_Q = n_total - actual_T
        if n_total > max_model_len:
            return {
                "skip": True,
                "skip_reason": f"prompt_with_q exceeds max_model_len={max_model_len}",
                "target_T": int(args.target_t),
                "actual_T": actual_T,
                "n_total_tokens": n_total,
            }

        stage_cfg = STAGE_CONFIG[args.stage]
        ckpt = config[stage_cfg["checkpoint_key"]]
        prefetch_fn = getattr(model, stage_cfg["prefetch_fn"])
        advance_fn = getattr(model, stage_cfg["advance_fn"])
        cache_prep = enforce_cache_condition(
            cache_condition,
            ckpt,
            drop_caches_timeout_s=float(args.drop_caches_timeout_s),
        )

        result: dict[str, Any] = {
            "skip": False,
            "branch": args.branch,
            "model": model_name,
            "stage": args.stage,
            "cache_condition": cache_condition,
            "strict_partial_path": strict_partial_path,
            "cache_prepare_method": cache_prep["method"],
            "cache_prepare_ok": bool(cache_prep["ok"]),
            "cache_prepare_s": round(float(cache_prep["elapsed_s"]), 6),
            "drop_caches_timeout_s": float(args.drop_caches_timeout_s),
            "run_index": int(args.run_index),
            "target_T": int(args.target_t),
            "actual_T": actual_T,
            "n_Q_tokens": n_Q,
            "n_total_tokens": n_total,
            "source_fill_s": round(source_fill_s, 6),
            "progressive_impl": impl_name,
            "progressive_impl_path": impl_path,
            "progressive_path": config["progressive_path"],
            "max_model_len": max_model_len,
            "gpu_memory_utilization_requested": requested_gpu_memory_utilization,
            "gpu_memory_utilization_effective": gpu_memory_utilization,
            "gpu_memory_utilization_clamped": bool(gpu_mem_plan["clamped"]),
            "gpu_free_mem_before_load_gb": (
                round(float(gpu_mem_plan["free_gb"]), 3)
                if gpu_mem_plan["free_gb"] is not None
                else None
            ),
            "gpu_total_mem_before_load_gb": (
                round(float(gpu_mem_plan["total_gb"]), 3)
                if gpu_mem_plan["total_gb"] is not None
                else None
            ),
            "gpu_memory_reserve_gb": round(float(gpu_mem_plan["reserve_gb"]), 3),
            "tensor_parallel_size_effective": tensor_parallel_size,
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
            "enable_prefix_caching": prefix_caching_enabled,
            "drop_caches_requested": cache_condition == "cold",
            "drop_caches_ok": cache_condition != "cold" or bool(cache_prep["ok"]),
            "warm_cache_requested": cache_condition == "warm",
            "warm_cache_ok": cache_condition != "warm" or bool(cache_prep["ok"]),
            "gpu_temp_c_start": temp_c,
        }

        cuda_sync()
        t_req_start = time.perf_counter()

        cuda_sync()
        t0 = time.perf_counter()
        prefetch_fn(ckpt)
        prefetch_status = wait_for_prefetch_or_raise(
            model,
            timeout_s=float(args.stage_transition_timeout_s),
            checkpoint_path=ckpt,
            context=(
                f"Timed transition prefetch failed for branch={args.branch}, "
                f"stage={args.stage}, T={int(args.target_t)}, run={int(args.run_index)}"
            ),
        )
        cuda_sync()
        t_prefetch = time.perf_counter() - t0

        cuda_sync()
        t0 = time.perf_counter()
        ok = advance_fn(wait_if_needed=False)
        cuda_sync()
        t_activation = time.perf_counter() - t0
        if not ok:
            raise RuntimeError(
                f"{stage_cfg['advance_fn']} returned False even though prefetch "
                f"reported ready for checkpoint={ckpt}. "
                f"status={format_prefetch_status(prefetch_status)}"
            )

        result["prefetch_s"] = round(t_prefetch, 6)
        result["prefetch_status"] = prefetch_status
        result["activation_s"] = round(t_activation, 6)

        if args.branch == "partial":
            inner_model = getattr(model, "model", None)
            t_sync = 0.0
            sync_api_available = inner_model is not None and hasattr(
                inner_model, "sync_persistent_cache"
            )
            sync_seq_len = 0
            if sync_api_available:
                sync_seq_len = resolve_transition_seq_len(tokenizer, prompt_fill, inner_model)
                if strict_partial_path and sync_seq_len <= 0:
                    raise RuntimeError(
                        "Strict partial-path benchmark could not resolve a positive transition seq_len."
                    )
                if sync_seq_len > 0:
                    cuda_sync()
                    t0 = time.perf_counter()
                    inner_model.sync_persistent_cache(sync_seq_len)
                    cuda_sync()
                    t_sync = time.perf_counter() - t0
            elif strict_partial_path:
                raise RuntimeError(
                    "Strict partial-path benchmark requires sync_persistent_cache()."
                )
            reconcile = reconcile_after_transition(
                llm=llm,
                model=model,
                prompt_fill=prompt_fill,
                stage_cfg=stage_cfg,
                minimal_params=fill_params,
                prefix_caching_enabled=prefix_caching_enabled,
            )
            if strict_partial_path:
                enforce_strict_partial_path(
                    inner_model=inner_model,
                    sync_api_available=sync_api_available,
                    sync_seq_len=sync_seq_len,
                    reconcile=reconcile,
                )
            cuda_sync()
            t0 = time.perf_counter()
            llm.generate([prompt_with_q], request_params)
            cuda_sync()
            request_ttft_s = time.perf_counter() - t0
            e2e_from_request_s = time.perf_counter() - t_req_start

            result.update(
                {
                    "sync_s": round(t_sync, 6),
                    "reconcile_s": round(float(reconcile["elapsed"]), 6),
                    "request_ttft_s": round(request_ttft_s, 6),
                    "e2e_from_request_s": round(e2e_from_request_s, 6),
                    "e2e_from_source_fill_s": round(source_fill_s + e2e_from_request_s, 6),
                    "surgery_ok": bool(reconcile["surgery_ok"]),
                    "boundary": reconcile["boundary"],
                    "reconcile_path": reconcile["reconcile_path"],
                    "surgery_profile": reconcile["surgery_profile"],
                    "partial_recompute_profile": reconcile["partial_profile"],
                }
            )
        else:
            cache_invalidate_s = invalidate_caches_for_origin(
                llm=llm,
                model=model,
                prefix_caching_enabled=prefix_caching_enabled,
            )
            cuda_sync()
            t0 = time.perf_counter()
            llm.generate([prompt_with_q], request_params)
            cuda_sync()
            request_ttft_s = time.perf_counter() - t0
            e2e_from_request_s = time.perf_counter() - t_req_start

            result.update(
                {
                    "cache_invalidate_s": round(cache_invalidate_s, 6),
                    "request_ttft_s": round(request_ttft_s, 6),
                    "e2e_from_request_s": round(e2e_from_request_s, 6),
                    "e2e_from_source_fill_s": round(source_fill_s + e2e_from_request_s, 6),
                }
            )

        result["kv_cache_gb_theoretical"] = kv_cache_gb_theoretical(actual_T, llm)
        result["cpu_mem_gb"] = round(cpu_mem_gb(), 3)
        result["gpu_mem_gb"] = round(gpu_mem_gb(), 3)
        return result
    finally:
        if llm is not None and model is not None:
            try:
                clear_runtime_state(llm, model)
            except Exception:
                pass
        del llm, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def numeric_summary(runs: list[dict[str, Any]], field: str) -> dict[str, Any]:
    st = _stats([float(r.get(field, 0.0)) for r in runs])
    return {
        "mean": st["mean"],
        "stats": st,
    }


def summarize_scaling_point(
    target_t: int,
    partial_runs: list[dict[str, Any]],
    origin_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    ref = partial_runs[0]
    cache_conditions = {
        str(r.get("cache_condition", "")).strip()
        for r in (partial_runs + origin_runs)
        if str(r.get("cache_condition", "")).strip()
    }
    if len(cache_conditions) > 1:
        raise RuntimeError(f"Inconsistent cache conditions within one scaling point: {cache_conditions}")
    cache_condition = next(iter(cache_conditions), DEFAULT_CACHE_CONDITION)
    strict_modes = {
        bool(r.get("strict_partial_path", DEFAULT_STRICT_PARTIAL_PATH))
        for r in partial_runs
    }
    if len(strict_modes) > 1:
        raise RuntimeError(f"Inconsistent strict partial-path modes within one scaling point: {strict_modes}")
    strict_partial_path = next(iter(strict_modes), DEFAULT_STRICT_PARTIAL_PATH)
    partial_numeric = [
        "source_fill_s",
        "prefetch_s",
        "activation_s",
        "sync_s",
        "reconcile_s",
        "request_ttft_s",
        "e2e_from_request_s",
        "e2e_from_source_fill_s",
    ]
    origin_numeric = [
        "source_fill_s",
        "prefetch_s",
        "activation_s",
        "cache_invalidate_s",
        "request_ttft_s",
        "e2e_from_request_s",
        "e2e_from_source_fill_s",
    ]

    partial_summary: dict[str, Any] = {}
    origin_summary: dict[str, Any] = {}
    for field in partial_numeric:
        st = numeric_summary(partial_runs, field)
        partial_summary[field] = st["mean"]
        partial_summary[f"{field}_stats"] = st["stats"]
    for field in origin_numeric:
        st = numeric_summary(origin_runs, field)
        origin_summary[field] = st["mean"]
        origin_summary[f"{field}_stats"] = st["stats"]

    partial_paths = Counter(str(r.get("reconcile_path", "none")) for r in partial_runs)
    boundaries = [r.get("boundary") for r in partial_runs if r.get("boundary") is not None]

    request_only_speedup = (
        round(origin_summary["request_ttft_s"] / partial_summary["request_ttft_s"], 2)
        if partial_summary["request_ttft_s"] > 0
        else 0.0
    )
    e2e_speedup = (
        round(origin_summary["e2e_from_request_s"] / partial_summary["e2e_from_request_s"], 2)
        if partial_summary["e2e_from_request_s"] > 0
        else 0.0
    )
    source_speedup = (
        round(origin_summary["e2e_from_source_fill_s"] / partial_summary["e2e_from_source_fill_s"], 2)
        if partial_summary["e2e_from_source_fill_s"] > 0
        else 0.0
    )

    return {
        "target_T": int(target_t),
        "actual_T": int(ref["actual_T"]),
        "cache_condition": cache_condition,
        "strict_partial_path": strict_partial_path,
        "n_Q_tokens": int(ref["n_Q_tokens"]),
        "n_total_tokens": int(ref["n_total_tokens"]),
        "kv_cache_gb_theoretical": float(ref.get("kv_cache_gb_theoretical", 0.0)),
        "partial": {
            **partial_summary,
            "surgery_success_rate": round(
                sum(1.0 if bool(r.get("surgery_ok", False)) else 0.0 for r in partial_runs)
                / max(1, len(partial_runs)),
                4,
            ),
            "reconcile_path_counts": dict(partial_paths),
            "boundary_values": boundaries,
            "last_surgery_profile": partial_runs[-1].get("surgery_profile"),
            "last_partial_recompute_profile": partial_runs[-1].get("partial_recompute_profile"),
            "drop_caches_all_ok": all(bool(r.get("drop_caches_ok", True)) for r in partial_runs),
            "warm_cache_all_ok": all(bool(r.get("warm_cache_ok", True)) for r in partial_runs),
            "cache_prepare_all_ok": all(bool(r.get("cache_prepare_ok", True)) for r in partial_runs),
        },
        "origin": {
            **origin_summary,
            "drop_caches_all_ok": all(bool(r.get("drop_caches_ok", True)) for r in origin_runs),
            "warm_cache_all_ok": all(bool(r.get("warm_cache_ok", True)) for r in origin_runs),
            "cache_prepare_all_ok": all(bool(r.get("cache_prepare_ok", True)) for r in origin_runs),
        },
        "request_only_speedup_ratio": request_only_speedup,
        "e2e_speedup_ratio": e2e_speedup,
        "source_fill_inclusive_speedup_ratio": source_speedup,
        "request_only_savings_s": round(
            origin_summary["request_ttft_s"] - partial_summary["request_ttft_s"], 6
        ),
        "e2e_savings_s": round(
            origin_summary["e2e_from_request_s"] - partial_summary["e2e_from_request_s"], 6
        ),
        "source_fill_inclusive_savings_s": round(
            origin_summary["e2e_from_source_fill_s"] - partial_summary["e2e_from_source_fill_s"], 6
        ),
        "partial_runs": partial_runs,
        "origin_runs": origin_runs,
    }


def build_worker_command(
    args: argparse.Namespace,
    branch: str,
    target_t: int,
    run_index: int,
    output_path: str,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-output",
        output_path,
        "--branch",
        branch,
        "--run-index",
        str(run_index),
        "--target-t",
        str(target_t),
        "--model",
        args.model,
        "--stage",
        args.stage,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--stage1-warmup-runs",
        str(args.stage1_warmup_runs),
        "--stage-transition-timeout-s",
        str(args.stage_transition_timeout_s),
        "--progressive-impl",
        args.progressive_impl,
        "--cache-condition",
        args.cache_condition,
        "--drop-caches-timeout-s",
        str(args.drop_caches_timeout_s),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.allow_partial_fallback:
        cmd.append("--allow-partial-fallback")
    return cmd


def run_worker_subprocess(
    args: argparse.Namespace,
    branch: str,
    target_t: int,
    run_index: int,
) -> dict[str, Any]:
    tmp = tempfile.NamedTemporaryFile(prefix="scaling_truee2e_", suffix=".json", delete=False)
    tmp.close()
    cmd = build_worker_command(args, branch, target_t, run_index, tmp.name)
    proc = subprocess.run(
        cmd,
        cwd=str(SCRIPT_DIR),
        text=True,
        capture_output=True,
    )
    try:
        if proc.returncode != 0:
            raise RuntimeError(
                f"Worker failed for branch={branch}, T={target_t}, run={run_index}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        with open(tmp.name) as f:
            return json.load(f)
    finally:
        try:
            os.unlink(tmp.name)
        except FileNotFoundError:
            pass


def run_true_e2e_scaling(args: argparse.Namespace) -> dict[str, Any]:
    model_name = canonical_model_name(args.model)
    cache_condition = resolve_cache_condition(args)
    strict_partial_path = DEFAULT_STRICT_PARTIAL_PATH and not bool(args.allow_partial_fallback)
    model_cfg = MODELS[model_name]
    max_model_len = (
        int(args.max_model_len)
        if int(args.max_model_len) > 0
        else int(model_cfg.get("max_model_len", DEFAULT_MAX_MODEL_LEN))
    )
    if args.token_counts.strip():
        token_counts = sorted(set(int(x.strip()) for x in args.token_counts.split(",") if x.strip()))
    else:
        token_counts = model_default_token_counts(model_name, max_model_len)
    if not token_counts:
        raise ValueError("No valid token counts resolved. Please provide --token-counts explicitly.")

    impl_name, impl_path = resolve_progressive_impl(args.progressive_impl)
    requested_gpu_memory_utilization = resolve_requested_gpu_memory_utilization(
        args.gpu_memory_utilization,
        model_cfg,
    )
    effective_gpu_memory_utilization = requested_gpu_memory_utilization
    gpu_mem_plan_preview = build_nonintrusive_gpu_utilization_plan(
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
    print("=" * 78)
    log_gpu_memory_utilization_plan("  [GPU preview]", gpu_mem_plan_preview)

    scaling: list[dict[str, Any]] = []
    worker_gpu_memory_utils: list[float] = []
    worker_gpu_memory_clamps: list[bool] = []
    for target_t in token_counts:
        partial_runs: list[dict[str, Any]] = []
        origin_runs: list[dict[str, Any]] = []
        skip_reason = None

        print(f"\n  [T={target_t}]")
        for run_index in range(int(args.num_runs)):
            print(f"    run {run_index + 1}/{args.num_runs}: partial...", flush=True)
            partial = run_worker_subprocess(args, "partial", target_t, run_index)
            if partial.get("skip"):
                skip_reason = partial.get("skip_reason", "worker requested skip")
                break
            partial_runs.append(partial)
            worker_gpu_memory_utils.append(float(partial["gpu_memory_utilization_effective"]))
            worker_gpu_memory_clamps.append(bool(partial.get("gpu_memory_utilization_clamped", False)))

            print(f"    run {run_index + 1}/{args.num_runs}: origin...", flush=True)
            origin = run_worker_subprocess(args, "origin", target_t, run_index)
            if origin.get("skip"):
                skip_reason = origin.get("skip_reason", "worker requested skip")
                break
            origin_runs.append(origin)
            worker_gpu_memory_utils.append(float(origin["gpu_memory_utilization_effective"]))
            worker_gpu_memory_clamps.append(bool(origin.get("gpu_memory_utilization_clamped", False)))

        if skip_reason is not None:
            print(f"    skip — {skip_reason}")
            continue

        point = summarize_scaling_point(target_t, partial_runs, origin_runs)
        scaling.append(point)

        p = point["partial"]
        o = point["origin"]
        print(
            f"    request_ttft: partial={_fmt_stat(p['request_ttft_s_stats'], int(args.num_runs))}  "
            f"origin={_fmt_stat(o['request_ttft_s_stats'], int(args.num_runs))}  "
            f"({point['request_only_speedup_ratio']:.2f}x)"
        )
        print(
            f"    e2e_request:  partial={_fmt_stat(p['e2e_from_request_s_stats'], int(args.num_runs))}  "
            f"origin={_fmt_stat(o['e2e_from_request_s_stats'], int(args.num_runs))}  "
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
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
          "gpu_total_mem_gb": (
              round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
              if torch.cuda.is_available()
              else 0.0
          ),
          "gpu_memory_utilization_requested": requested_gpu_memory_utilization,
          "gpu_memory_utilization_effective": effective_gpu_memory_utilization,
          "gpu_memory_utilization_worker_min": (
              round(min(worker_gpu_memory_utils), 6) if worker_gpu_memory_utils else effective_gpu_memory_utilization
          ),
          "gpu_memory_utilization_worker_max": (
              round(max(worker_gpu_memory_utils), 6) if worker_gpu_memory_utils else effective_gpu_memory_utilization
          ),
          "gpu_memory_utilization_clamped_any": any(worker_gpu_memory_clamps),
          "gpu_memory_reserve_gb": float(DEFAULT_GPU_FREE_RESERVE_GB),
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
        "fixed_question": FIXED_QUESTION,
        "scaling": scaling,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved → {args.output}")
    return results


def plot_results(data: dict[str, Any], save_path: str | None = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not installed: pip install matplotlib")
        return

    scaling = data.get("scaling", [])
    if not scaling:
        print("  ⚠️  No scaling data to plot.")
        return

    Ts = [r["actual_T"] for r in scaling]
    p_req = [r["partial"]["request_ttft_s"] for r in scaling]
    o_req = [r["origin"]["request_ttft_s"] for r in scaling]
    p_e2e = [r["partial"]["e2e_from_request_s"] for r in scaling]
    o_e2e = [r["origin"]["e2e_from_request_s"] for r in scaling]
    req_speed = [r["request_only_speedup_ratio"] for r in scaling]
    e2e_speed = [r["e2e_speedup_ratio"] for r in scaling]
    p_prefetch = [r["partial"]["prefetch_s"] for r in scaling]
    p_activation = [r["partial"]["activation_s"] for r in scaling]
    p_sync = [r["partial"]["sync_s"] for r in scaling]
    p_reconcile = [r["partial"]["reconcile_s"] for r in scaling]

    err_p_req = [r["partial"]["request_ttft_s_stats"].get("std", 0.0) for r in scaling]
    err_o_req = [r["origin"]["request_ttft_s_stats"].get("std", 0.0) for r in scaling]
    err_p_e2e = [r["partial"]["e2e_from_request_s_stats"].get("std", 0.0) for r in scaling]
    err_o_e2e = [r["origin"]["e2e_from_request_s_stats"].get("std", 0.0) for r in scaling]
    has_err = any(e > 0 for e in err_p_req + err_o_req + err_p_e2e + err_o_e2e)
    ekw = dict(capsize=4, elinewidth=1.5) if has_err else {}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    fig.suptitle(
        f"True E2E Transition Scaling  —  Model: {data['model'].upper()}  "
        f"Stage: {data['stage']}  runs={data.get('num_runs', 1)}  "
        f"cache={data.get('cache_condition', DEFAULT_CACHE_CONDITION)}\n"
        f"GPU: {data.get('gpu_name', '')}",
        fontsize=11,
    )

    ax1.errorbar(Ts, o_req, yerr=err_o_req if has_err else None,
                 fmt="ro-", label="Origin request TTFT", lw=2, ms=7, **ekw)
    ax1.errorbar(Ts, p_req, yerr=err_p_req if has_err else None,
                 fmt="bs-", label="Partial request TTFT", lw=2, ms=7, **ekw)
    ax1.set_ylabel("Request-only TTFT (s)")
    ax1.set_title("① Request TTFT after Transition")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.errorbar(Ts, o_e2e, yerr=err_o_e2e if has_err else None,
                 fmt="ro-", label="Origin E2E", lw=2, ms=7, **ekw)
    ax2.errorbar(Ts, p_e2e, yerr=err_p_e2e if has_err else None,
                 fmt="bs-", label="Partial E2E", lw=2, ms=7, **ekw)
    ax2.set_ylabel("Transition request -> first token (s)")
    ax2.set_title("② True E2E from Transition Request")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax3.plot(Ts, req_speed, "m^-", lw=2, ms=8, label="Request-only speedup")
    ax3.plot(Ts, e2e_speed, "g^-", lw=2, ms=8, label="E2E speedup")
    ax3.axhline(y=1.0, color="gray", ls=":", lw=1.5, label="Baseline (1×)")
    ax3.set_xlabel("Accumulated tokens (T)")
    ax3.set_ylabel("Speedup (×)")
    ax3.set_title("③ Partial vs Origin Speedup")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    ax4.plot(Ts, p_prefetch, "ko-", lw=1.8, ms=6, label="prefetch")
    ax4.plot(Ts, p_activation, "co-", lw=1.8, ms=6, label="activation")
    ax4.plot(Ts, p_sync, "yo-", lw=1.8, ms=6, label="sync")
    ax4.plot(Ts, p_reconcile, "go-", lw=1.8, ms=6, label="reconcile")
    ax4.set_xlabel("Accumulated tokens (T)")
    ax4.set_ylabel("Component time (s)")
    ax4.set_title("④ Partial-Path Transition Breakdown")
    ax4.legend(loc="upper left")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-grade true E2E token scaling benchmark for stage transition timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_scaling_true_e2e.py --model llama2-7b --num-runs 5 --cache-condition cold
  python benchmark_scaling_true_e2e.py --model llama2-13b --stage 2to3 --num-runs 5 --cache-condition cold
  python benchmark_scaling_true_e2e.py --model gemma-7b --num-runs 7 --cache-condition warm
  python benchmark_scaling_true_e2e.py --model llama2-7b --allow-partial-fallback
  python benchmark_scaling_true_e2e.py --plot results_scaling_truee2e_llama2-7b_1to2_20260406_160000.json
        """,
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("--stage", choices=list(STAGE_CONFIG.keys()), default="1to2")
    parser.add_argument(
        "--token-counts",
        type=str,
        default="",
        help="쉼표 구분 토큰 수. 비우면 모델별 기본 토큰 수 사용",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"각 T당 반복 측정 횟수 (paper default: {DEFAULT_NUM_RUNS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="출력 JSON 경로 (default: results_scaling_truee2e_{model}_{stage}_{timestamp}.json)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="vLLM max_model_len override. 0이면 모델별 기본값 사용",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.0,
        help=(
            "gpu_memory_utilization override. 0이면 모델별 기본값 사용. "
            "Worker는 실행 시점의 실제 free VRAM 기준으로 이 값을 자동 clamp할 수 있음"
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=0,
        help="tensor_parallel_size override. 0이면 모델별 기본값 사용",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="vLLM enforce_eager=True 로 실행",
    )
    parser.add_argument(
        "--progressive-impl",
        type=str,
        default=DEFAULT_PROGRESSIVE_IMPL,
        help="progressive implementation 디렉터리",
    )
    parser.add_argument(
        "--cache-condition",
        choices=CACHE_CONDITIONS,
        default="",
        help=(
            "OS page-cache condition for each worker run. "
            "'cold' forces drop_caches success before timing; "
            "'warm' untimed-reads the checkpoint to warm page cache before timing. "
            f"Default: {DEFAULT_CACHE_CONDITION}."
        ),
    )
    parser.add_argument(
        "--drop-caches-timeout-s",
        type=float,
        default=DEFAULT_DROP_CACHES_TIMEOUT_S,
        help=(
            "timeout in seconds for the cold-cache sync/drop_caches helper "
            f"(default: {DEFAULT_DROP_CACHES_TIMEOUT_S})"
        ),
    )
    parser.add_argument(
        "--stage1-warmup-runs",
        type=int,
        default=DEFAULT_STAGE1_WARMUP_RUNS,
        help=f"source stage steady-state warmup 횟수 (default: {DEFAULT_STAGE1_WARMUP_RUNS})",
    )
    parser.add_argument(
        "--stage-transition-timeout-s",
        type=float,
        default=DEFAULT_STAGE_TIMEOUT_S,
        help=f"prefetch wait timeout in seconds (default: {DEFAULT_STAGE_TIMEOUT_S})",
    )
    parser.add_argument(
        "--skip-drop-caches",
        action="store_true",
        help="deprecated compatibility flag; equivalent to --cache-condition warm",
    )
    parser.add_argument(
        "--require-drop-caches",
        action="store_true",
        help="deprecated compatibility flag; equivalent to --cache-condition cold",
    )
    parser.add_argument(
        "--allow-partial-fallback",
        action="store_true",
        help=(
            "Allow degraded partial paths (e.g. fallback partial recompute) instead of "
            "failing the worker. Default is strict surgery-only fail-fast."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="측정 후 그래프 생성 스킵",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        metavar="JSON_FILE",
        help="기존 JSON 결과를 불러와 그래프만 출력",
    )

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument("--branch", choices=WORKER_BRANCHES, default="partial", help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--target-t", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.cache_condition = resolve_cache_condition(args)

    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode.")
        result = worker_measure_branch(args)
        with open(args.worker_output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    if args.plot:
        with open(args.plot) as f:
            data = json.load(f)
        save_path = args.plot.replace(".json", ".png")
        plot_results(data, save_path=save_path)
        return

    model_name = canonical_model_name(args.model)
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = str(SCRIPT_DIR / f"results_scaling_truee2e_{model_name}_{args.stage}_{ts}.json")

    results = run_true_e2e_scaling(args)
    if not args.no_plot:
        plot_path = args.output.replace(".json", ".png")
        plot_results(results, save_path=plot_path)


if __name__ == "__main__":
    main()
