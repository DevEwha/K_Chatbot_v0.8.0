#!/usr/bin/env python3
"""
benchmark_scaling.py — 토큰 수에 따른 stage 전환 시간 스케일링 측정
======================================================================

단일 모델 로드로 여러 누적 토큰 수(T)에서 stage 전환 비용을 측정:

  - t_sync(T)               : GPU persistent buffer → hidden cache 동기화 시간
  - t_reconcile(T)          : stage 전환 후 cache reconciliation 시간
                              우선 경로: inject_upper_layer_kv(boundary)
                              fallback: set_partial_recompute(boundary)
                                        + reset_prefix_cache()
                                        + generate(max_tokens=1)
  - t_first_chat_origin(T)  : full prefill 시간 (T+Q 토큰 전체 재계산)
  - t_first_chat_partial(T) : prefix hit 후 신규 Q 토큰만 처리 시간
  - t_total_effective(T)    : 상수 항(prefetch+activation) + sync/reconcile 포함 총 체감 시간

  측정 흐름 (T 하나당, num_runs회 반복):
  1. KV cache 비우기 + GPU 메모리 클리어 (gc + empty_cache)
  2. T 토큰짜리 합성 컨텍스트 generate → KV cache 채우기
  3. [Partial] t_sync 측정: sync_persistent_cache(T)
  4. [Partial] t_reconcile 측정: KV surgery 또는 fallback partial recompute
  5. [Partial] prompt_with_q generate → t_first_chat_partial (prefix hit + Q tokens)
  6. KV cache 비우기 + GPU 메모리 클리어
  7. [Origin]  prompt_with_q generate → t_first_chat_origin (full prefill)

정확도 보장:
  - --num-runs N 으로 반복 측정 후 mean±std 리포트 (논문용 권장: 3 이상)
  - 각 run 전후 torch.cuda.empty_cache() + gc.collect() 로 GPU 메모리 클리어
  - Stage 전환 후 warmup generate 3회 (CUDA graph re-capture 흡수)
  - 모든 타이밍: torch.cuda.synchronize() + time.perf_counter() 사용
  - GPU 온도 기록 (thermal throttling 감지용)

t_prefetch + t_activation 은 1회만 측정 후 모든 T에 공통 적용.
t_sync / t_reconcile 은 T에 비례하므로 각 T마다 개별 측정한다.

Usage:
  python benchmark_scaling.py --model llama2-7b
  python benchmark_scaling.py --model falcon-7b --stage 2to3
  python benchmark_scaling.py --model gemma-7b --token-counts 64,128,256,384,448
  python benchmark_scaling.py --model llama2-13b --num-runs 3
  python benchmark_scaling.py --plot results_scaling_llama2-7b_1to2_20260406_120000.json

Note:
  --max-model-len 0 이면 모델별 기본값을 사용한다.
  긴 context를 직접 지정할 때는 필요 시:
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python benchmark_scaling.py ...
"""

import gc
import json
import os
import statistics
import subprocess
import sys
import time
import argparse
import datetime
from collections import Counter
import psutil

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROGRESSIVE_IMPL = os.environ.get("ASPLOS_PROGRESSIVE_IMPL", "progressive_serve5")
_PROGRESSIVE_MODULES = (
    "progressive_for_causal_lm",
    "progressive_model_dual_path",
    "model_config",
    "universal_bypass_layer",
)

# =========================================================
# vLLM v0.8.0 버그 우회 (Prefix Caching 강제 활성화)
# =========================================================
import vllm.config
vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)
# =========================================================


STAGE_CONFIG = {
    "1to2": {
        "checkpoint_key":  "stage_b_checkpoint",
        "prefetch_fn":     "prefetch_stage2",
        "advance_fn":      "advance_to_stage2_instant",
        "get_indices_fn":  "_get_b_indices",
        "setup_stage":     None,
    },
    "2to3": {
        "checkpoint_key":  "stage_c_checkpoint",
        "prefetch_fn":     "prefetch_stage3",
        "advance_fn":      "advance_to_stage3_instant",
        "get_indices_fn":  "_get_c_indices",
        "setup_stage":     "1to2",
    },
}

DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_NUM_RUNS = 1

# 모든 T에 공통으로 추가하는 짧은 질문 (~15 tokens)
FIXED_QUESTION = "Briefly summarize the key points discussed so far."

# 합성 컨텍스트용 베이스 텍스트 (~120 tokens, 반복해서 목표 길이 달성)
_BASE_TEXT = (
    "The history of computing spans centuries, from mechanical calculators "
    "to modern microprocessors. The transistor, invented at Bell Laboratories "
    "in 1947, displaced the vacuum tube and enabled reliable computation. "
    "Integrated circuits miniaturized hardware, and Moore's Law guided decades "
    "of exponential density growth. Programming languages evolved from machine "
    "code through FORTRAN, LISP, C, and Java to modern Python and Rust. "
    "Operating systems like UNIX introduced file abstraction and modularity. "
    "Networking advances led from ARPANET to the global internet. "
)


# ============================================================================
# 유틸리티
# ============================================================================

def drop_all_caches():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    try:
        if os.geteuid() == 0:
            cmd = ["sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
        else:
            cmd = ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"]
        out = subprocess.run(cmd, capture_output=True, timeout=15, check=False, text=True)
        if out.returncode == 0:
            print("  [Cache] OS page cache dropped")
        else:
            print("  [Cache] OS page cache drop skipped (sudo unavailable/non-interactive)")
    except Exception as e:
        print(f"  [Cache] OS page cache drop skipped ({e})")
    time.sleep(2)


def gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 3)


def gpu_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / (1024 ** 3)


def cpu_mem_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def kv_cache_gb_theoretical(actual_T: int, llm) -> float:
    """
    vLLM pre-allocated KV pool 우회: 모델 config + block 수로 이론값 계산.

    KV cache bytes = num_blocks × block_size × num_kv_heads × head_dim
                     × 2 (K+V) × 2 (fp16) × num_layers
    """
    try:
        hf_cfg    = llm.llm_engine.model_config.hf_config
        cache_cfg = llm.llm_engine.cache_config

        num_layers   = hf_cfg.num_hidden_layers
        num_kv_heads = getattr(hf_cfg, "num_key_value_heads",
                               hf_cfg.num_attention_heads)
        head_dim     = hf_cfg.hidden_size // hf_cfg.num_attention_heads
        block_size   = cache_cfg.block_size      # 토큰/block (보통 16)
        dtype_bytes  = 2                         # fp16

        num_blocks = (actual_T + block_size - 1) // block_size
        bytes_total = (num_blocks * block_size * num_kv_heads
                       * head_dim * 2 * dtype_bytes * num_layers)
        return round(bytes_total / (1024 ** 3), 3)
    except Exception:
        return 0.0


def gpu_temp_c() -> int | None:
    """nvidia-smi 로 GPU 온도(°C) 읽기. 실패 시 None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


def get_model_handle(llm):
    engine = llm.llm_engine
    if hasattr(engine, "engine_core"):
        raise RuntimeError("V1 engine detected. Set VLLM_USE_V1=0.")
    try:
        return engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError("Could not resolve v0 model handle.") from exc


def resolve_progressive_impl(progressive_impl: str) -> tuple[str, str]:
    impl_name = progressive_impl.strip() or DEFAULT_PROGRESSIVE_IMPL
    if os.path.isabs(impl_name):
        impl_path = impl_name
    else:
        impl_path = os.path.join(SCRIPT_DIR, impl_name)
    impl_path = os.path.abspath(impl_path)
    if not os.path.isdir(impl_path):
        raise FileNotFoundError(
            f"Progressive implementation path not found: {impl_path} "
            f"(progressive_impl={impl_name})"
        )
    return impl_name, impl_path


def prepare_progressive_imports(progressive_impl_path: str):
    if progressive_impl_path not in sys.path:
        sys.path.insert(0, progressive_impl_path)
    progressive_impl_path = os.path.abspath(progressive_impl_path)
    for module_name in _PROGRESSIVE_MODULES:
        cached = sys.modules.get(module_name)
        if cached is None:
            continue
        cached_file = os.path.abspath(str(getattr(cached, "__file__", "")))
        if cached_file and not cached_file.startswith(progressive_impl_path):
            del sys.modules[module_name]


def reset_prefix_cache(llm):
    if hasattr(llm, "reset_prefix_cache"):
        llm.reset_prefix_cache()


def clear_hidden_cache(model):
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "clear_hidden_cache"):
        try:
            inner_model.clear_hidden_cache()
        except Exception:
            pass


def clear_runtime_state(llm, model):
    reset_prefix_cache(llm)
    clear_hidden_cache(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def resolve_transition_seq_len(tokenizer, prompt_fill: str, inner_model) -> int:
    if inner_model is not None:
        seq_tensor = getattr(inner_model, "_surgery_seq_lens_tensor", None)
        if seq_tensor is not None and getattr(seq_tensor, "numel", lambda: 0)() > 0:
            seq_len = int(seq_tensor[0].item())
            if seq_len > 0:
                return seq_len
    return len(tokenizer.encode(prompt_fill))


def _stats(vals: list[float]) -> dict:
    """리스트의 통계량(mean/std/median/min/max) 계산."""
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    mean   = sum(vals) / n
    std    = statistics.stdev(vals) if n > 1 else 0.0
    median = statistics.median(vals)
    return {
        "mean":   round(mean,   4),
        "std":    round(std,    4),
        "median": round(median, 4),
        "min":    round(min(vals), 4),
        "max":    round(max(vals), 4),
        "n":      n,
    }


def _fmt_stat(st: dict, num_runs: int) -> str:
    if num_runs == 1:
        return f"{st['mean']:.3f}s"
    return f"{st['mean']:.3f}±{st['std']:.3f}s"


# ============================================================================
# 합성 프롬프트 생성
# ============================================================================

def build_context_of_length(tokenizer, target_T: int) -> tuple[str, str, int]:
    """
    target_T 토큰에 가까운 합성 유저 메시지 프롬프트 생성.

    Returns:
        content      : chat template 없는 순수 텍스트 내용
        prompt_fill  : chat template 적용 완성 프롬프트 (KV cache 채우기용)
        actual_T     : prompt_fill의 실제 토큰 수
    """
    try:
        test = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        overhead = len(tokenizer.encode(test)) - len(
            tokenizer.encode("x", add_special_tokens=False)
        )
    except Exception:
        overhead = 10

    content_target = max(target_T - overhead, 10)

    base_tok = tokenizer.encode(_BASE_TEXT, add_special_tokens=False)
    toks = []
    while len(toks) < content_target:
        toks.extend(base_tok)
    content = tokenizer.decode(toks[:content_target])

    try:
        prompt_fill = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt_fill = f"User: {content}\nAssistant: "

    actual_T = len(tokenizer.encode(prompt_fill))
    return content, prompt_fill, actual_T


def build_prompt_with_question(tokenizer, content: str, dummy_response: str) -> str:
    """
    기존 컨텍스트(content) + dummy_response + FIXED_QUESTION 으로
    multi-turn 프롬프트 구성.

    prompt_fill 이 이 프롬프트의 prefix 이므로 partial 모드에서 prefix cache hit 발생.
    """
    conv = [
        {"role": "user",      "content": content},
        {"role": "assistant", "content": dummy_response},
        {"role": "user",      "content": FIXED_QUESTION},
    ]
    try:
        return tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return (
            f"User: {content}\nAssistant: {dummy_response}\n"
            f"User: {FIXED_QUESTION}\nAssistant: "
        )


# ============================================================================
# Stage 전환 후 Warmup (CUDA Graph re-capture 흡수)
# ============================================================================

def warmup_generate(llm, tokenizer, n: int = 3, label: str = ""):
    """
    Stage 전환 후 첫 generate의 CUDA graph 재캡처 지연을 흡수하는 warmup.
    n회 generate 후 prefix cache 초기화.
    이 시간은 측정에 포함되지 않음.
    """
    try:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        prompt = "Hello"

    wp = SamplingParams(temperature=0.0, max_tokens=2)
    for _ in range(n):
        reset_prefix_cache(llm)
        llm.generate([prompt], wp)
    reset_prefix_cache(llm)
    torch.cuda.synchronize()
    if label:
        print(f"  [Warmup ×{n}] {label} ✅")


# ============================================================================
# 전환 상수 측정 (1회)
# ============================================================================

def measure_transition_constants(
    llm, model, tokenizer, config: dict, stage_cfg: dict
) -> dict:
    """
    t_prefetch + t_activation 1회 측정.
    실행 후 모델 weights 는 stage_cfg 의 stage 로 전환된 상태가 됨.
    측정 완료 후 warmup generate 3회로 CUDA graph re-capture를 흡수함.
    """
    ckpt = config[stage_cfg["checkpoint_key"]]
    assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"

    prefetch_fn = getattr(model, stage_cfg["prefetch_fn"])
    advance_fn  = getattr(model, stage_cfg["advance_fn"])

    print(f"\n  [Constants] t_prefetch + t_activation 측정 중...")

    # t_prefetch
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prefetch_fn(ckpt)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    t_prefetch = round(time.perf_counter() - t0, 3)

    # t_activation
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    t_activation = round(time.perf_counter() - t0, 3)
    if not ok:
        raise RuntimeError(f"{stage_cfg['advance_fn']} returned False")

    # t_cache_clear
    t0 = time.perf_counter()
    reset_prefix_cache(llm)
    t_cache_clear = round(time.perf_counter() - t0, 4)

    c = {
        "t_prefetch_s":    t_prefetch,
        "t_activation_s":  t_activation,
        "t_cache_clear_s": t_cache_clear,
    }
    print(
        f"  [Constants] t_prefetch={t_prefetch:.3f}s | "
        f"t_activation={t_activation:.3f}s | "
        f"t_cache_clear={t_cache_clear:.4f}s"
    )

    # Stage 전환 후 warmup: CUDA graph re-capture + 초기 실행 지연 흡수
    # 이 시간은 t_constants 에 포함되지 않음
    warmup_generate(llm, tokenizer, n=3, label="post-activation warmup done")

    return c


def reconcile_after_transition(
    llm,
    model,
    prompt_fill: str,
    stage_cfg: dict,
    minimal_params: SamplingParams,
    prefix_caching_enabled: bool,
) -> dict:
    inner_model = getattr(model, "model", None)
    get_indices_fn = stage_cfg["get_indices_fn"]
    surgery_profile = None
    partial_profile = None
    boundary = None
    surgery_ok = False
    reconcile_path = "none"

    def _reset_prefix_cache(reason: str):
        if prefix_caching_enabled:
            reset_prefix_cache(llm)
            print(f"      [Cache] Prefix cache reset ({reason})")

    if inner_model is None:
        return {
            "elapsed": 0.0,
            "surgery_ok": False,
            "boundary": None,
            "reconcile_path": "skip_no_inner_model",
            "surgery_profile": None,
            "partial_profile": None,
        }

    if not hasattr(model, "get_recompute_boundary") or not hasattr(model, get_indices_fn):
        _reset_prefix_cache("missing boundary api")
        return {
            "elapsed": 0.0,
            "surgery_ok": False,
            "boundary": None,
            "reconcile_path": "skip_missing_boundary_api",
            "surgery_profile": None,
            "partial_profile": None,
        }

    indices = getattr(model, get_indices_fn)()
    boundary = model.get_recompute_boundary(indices) if indices else None
    if boundary is None:
        _reset_prefix_cache("boundary unavailable")
        return {
            "elapsed": 0.0,
            "surgery_ok": False,
            "boundary": None,
            "reconcile_path": "skip_no_boundary",
            "surgery_profile": None,
            "partial_profile": None,
        }

    do_fallback = False
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    if hasattr(inner_model, "inject_upper_layer_kv"):
        surgery_ok = inner_model.inject_upper_layer_kv(boundary)
        if hasattr(inner_model, "get_last_surgery_profile"):
            surgery_profile = inner_model.get_last_surgery_profile()
        if surgery_ok:
            reconcile_path = "surgery"
        else:
            do_fallback = True
            reconcile_path = "fallback_after_surgery_fail"
    else:
        do_fallback = True
        reconcile_path = "fallback_no_surgery_api"

    if do_fallback and not surgery_ok:
        if not hasattr(inner_model, "set_partial_recompute"):
            _reset_prefix_cache("fallback_missing_partial_api")
            torch.cuda.synchronize()
            return {
                "elapsed": time.perf_counter() - t0,
                "surgery_ok": False,
                "boundary": boundary,
                "reconcile_path": "fallback_missing_partial_api",
                "surgery_profile": surgery_profile,
                "partial_profile": None,
            }
        inner_model.set_partial_recompute(boundary)
        _reset_prefix_cache("fallback")
        llm.generate([prompt_fill], minimal_params)
        if hasattr(inner_model, "get_last_partial_recompute_profile"):
            partial_profile = inner_model.get_last_partial_recompute_profile()

    torch.cuda.synchronize()
    return {
        "elapsed": time.perf_counter() - t0,
        "surgery_ok": surgery_ok,
        "boundary": boundary,
        "reconcile_path": reconcile_path,
        "surgery_profile": surgery_profile,
        "partial_profile": partial_profile,
    }


# ============================================================================
# 단일 run 측정
# ============================================================================

def _measure_single_run(
    llm,
    model,
    tokenizer,
    prompt_fill: str,
    prompt_with_q: str,
    stage_cfg: dict,
    sampling_params: SamplingParams,
    minimal_params: SamplingParams,
    prefix_caching_enabled: bool,
) -> dict:
    """
    단일 run: reset+clear → fill_kv → t_sync → t_reconcile → t_partial
               → reset+clear → t_origin.

    각 타이밍은 torch.cuda.synchronize() + time.perf_counter() 로 정확하게 측정.
    partial/origin 모두 fill_kv 이후 warm GPU 상태에서 측정 (공정 비교).
    """
    inner_model = getattr(model, "model", None)

    # ── 초기화: KV cache + GPU 메모리 클리어 ─────────────────────────
    clear_runtime_state(llm, model)

    gpu_mem_base = gpu_mem_gb()    # KV 없는 기저 메모리 (모델 weights만)

    # ── fill_kv: T 토큰 KV cache 채우기 ──────────────────────────────
    # KV cache에 T토큰을 채우고 surgery/fallback용 block table 상태를 갱신
    llm.generate([prompt_fill], minimal_params)
    torch.cuda.synchronize()

    gpu_mem_filled = gpu_mem_gb()   # KV cache 채워진 후

    # ── t_sync ────────────────────────────────────────────────────────
    # GPU persistent buffer → hidden cache (fallback partial recompute 준비)
    t_sync = 0.0
    if inner_model is not None and hasattr(inner_model, "sync_persistent_cache"):
        seq_len = resolve_transition_seq_len(tokenizer, prompt_fill, inner_model)
        if seq_len > 0:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            inner_model.sync_persistent_cache(seq_len)
            torch.cuda.synchronize()
            t_sync = time.perf_counter() - t0

    # ── t_reconcile (= surgery or fallback partial recompute) ───────
    reconcile = reconcile_after_transition(
        llm=llm,
        model=model,
        prompt_fill=prompt_fill,
        stage_cfg=stage_cfg,
        minimal_params=minimal_params,
        prefix_caching_enabled=prefix_caching_enabled,
    )
    t_reconcile = reconcile["elapsed"]

    # ── t_first_chat_partial (prefix cache hit) ───────────────────────
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([prompt_with_q], sampling_params)
    torch.cuda.synchronize()
    t_partial = time.perf_counter() - t0

    # ── origin 측정 전: KV cache + GPU 메모리 클리어 ─────────────────
    clear_runtime_state(llm, model)

    # ── t_first_chat_origin (full prefill) ────────────────────────────
    t0 = time.perf_counter()
    llm.generate([prompt_with_q], sampling_params)
    torch.cuda.synchronize()
    t_origin = time.perf_counter() - t0

    return {
        "t_sync":            t_sync,
        "t_reconcile":       t_reconcile,
        "surgery_ok":        reconcile["surgery_ok"],
        "boundary":          reconcile["boundary"],
        "reconcile_path":    reconcile["reconcile_path"],
        "surgery_profile":   reconcile["surgery_profile"],
        "partial_profile":   reconcile["partial_profile"],
        "t_partial":         t_partial,
        "t_origin":          t_origin,
        "gpu_mem_base_gb":   round(gpu_mem_base,   3),
        "gpu_mem_filled_gb": round(gpu_mem_filled, 3),
        "kv_cache_gb":       round(gpu_mem_filled - gpu_mem_base, 3),
    }


# ============================================================================
# 단일 T에서 측정
# ============================================================================

def measure_at_token_count(
    llm, model, tokenizer,
    target_T: int,
    sampling_params: SamplingParams,
    minimal_params:  SamplingParams,
    stage_cfg: dict,
    t_constants: dict,
    max_model_len: int,
    prefix_caching_enabled: bool,
    num_runs: int = 1,
) -> dict | None:
    """
    target_T 토큰 누적 시점에서 origin / partial 전환 시간 측정.
    num_runs > 1 이면 반복 측정하여 통계(mean±std)를 리포트.
    Returns None 이면 이 T 는 skip.
    """
    if target_T + 80 > max_model_len:
        print(f"  [T={target_T}] skip — exceeds max_model_len={max_model_len}")
        return None

    result: dict = {"target_T": target_T}

    # ── 합성 컨텍스트 빌드 ──────────────────────────────────────────────
    content, prompt_fill, actual_T = build_context_of_length(tokenizer, target_T)
    result["actual_T"] = actual_T

    temp = gpu_temp_c()
    temp_str = f"  GPU_temp={temp}°C" if temp is not None else ""
    print(f"\n  [T={actual_T} (target={target_T})]  GPU={gpu_mem_gb():.2f}GB{temp_str}")
    if temp is not None:
        result["gpu_temp_c_start"] = temp

    # ── dummy_response 확정 (모든 run에서 동일한 prompt_with_q 사용) ───
    # setup generate 자체는 타이밍에 포함되지 않음
    clear_runtime_state(llm, model)
    setup_out      = llm.generate([prompt_fill], minimal_params)
    dummy_response = setup_out[0].outputs[0].text.strip() or "."
    prompt_with_q  = build_prompt_with_question(tokenizer, content, dummy_response)

    n_total = len(tokenizer.encode(prompt_with_q))
    n_Q     = n_total - actual_T
    result["n_Q_tokens"]     = n_Q
    result["n_total_tokens"] = n_total
    if n_total > max_model_len:
        print(f"  [T={actual_T}] skip — prompt_with_q exceeds max_model_len={max_model_len}")
        clear_runtime_state(llm, model)
        return None

    # ── num_runs 반복 측정 ──────────────────────────────────────────────
    runs_sync    = []
    runs_reconcile = []
    runs_partial = []
    runs_origin  = []
    runs_kv_gb   = []
    reconcile_paths = Counter()
    boundary_last = None
    surgery_profile_last = None
    partial_profile_last = None
    surgery_ok_any = False

    for run_i in range(num_runs):
        if num_runs > 1:
            print(f"    run {run_i + 1}/{num_runs} ...", flush=True)

        r = _measure_single_run(
            llm, model,
            tokenizer,
            prompt_fill, prompt_with_q,
            stage_cfg, sampling_params, minimal_params,
            prefix_caching_enabled,
        )
        runs_sync.append(r["t_sync"])
        runs_reconcile.append(r["t_reconcile"])
        runs_partial.append(r["t_partial"])
        runs_origin.append(r["t_origin"])
        runs_kv_gb.append(r["kv_cache_gb"])
        surgery_ok_any = surgery_ok_any or r["surgery_ok"]
        reconcile_paths[r["reconcile_path"]] += 1
        boundary_last = r["boundary"]
        surgery_profile_last = r["surgery_profile"]
        partial_profile_last = r["partial_profile"]

        # run 간 GPU 메모리 정리
        clear_runtime_state(llm, model)

    # ── 통계 계산 ─────────────────────────────────────────────────────
    st_sync    = _stats(runs_sync)
    st_reconcile = _stats(runs_reconcile)
    st_partial = _stats(runs_partial)
    st_origin  = _stats(runs_origin)
    st_kv_gb   = _stats(runs_kv_gb)

    # 단일 값 (mean): downstream plotting/table code가 바로 읽는 대표값
    result["t_sync_s"]               = st_sync["mean"]
    result["t_reconcile_s"]          = st_reconcile["mean"]
    result["surgery_ok"]             = surgery_ok_any
    result["reconcile_path"]         = reconcile_paths.most_common(1)[0][0] if reconcile_paths else "none"
    result["reconcile_path_counts"]  = dict(reconcile_paths)
    result["boundary"]               = boundary_last
    result["surgery_profile"]        = surgery_profile_last
    result["partial_recompute_profile"] = partial_profile_last
    result["t_first_chat_partial_s"] = st_partial["mean"]
    result["t_first_chat_origin_s"]  = st_origin["mean"]

    # 통계 필드 (num_runs > 1 시 의미 있음, num_runs=1 에서는 std=0)
    result["t_sync_stats"]               = st_sync
    result["t_reconcile_stats"]          = st_reconcile
    result["t_first_chat_partial_stats"] = st_partial
    result["t_first_chat_origin_stats"]  = st_origin

    # ── t_total_effective 계산 ─────────────────────────────────────────
    tp = t_constants["t_prefetch_s"]
    ta = t_constants["t_activation_s"]
    tc = t_constants.get("t_cache_clear_s", 0.0)

    result["t_total_effective_partial_s"] = round(
        tp + ta + result["t_sync_s"] + result["t_reconcile_s"] + result["t_first_chat_partial_s"], 3
    )
    result["t_total_effective_origin_s"] = round(
        tp + ta + tc + result["t_first_chat_origin_s"], 3
    )

    # ── 파생 지표 (논문 Table 직접 사용) ──────────────────────────────
    t_orig_mean = st_origin["mean"]
    t_part_mean = st_partial["mean"]

    speedup = round(t_orig_mean / t_part_mean, 2) if t_part_mean > 0 else 0.0
    savings_s   = round(t_orig_mean - t_part_mean, 4)
    savings_pct = round(savings_s / t_orig_mean * 100, 1) if t_orig_mean > 0 else 0.0

    # t_reconcile 당 토큰당 비용 (μs/token) — linear scaling 확인용
    t_reconcile_per_token_us = round(
        result["t_reconcile_s"] / actual_T * 1e6, 2) if actual_T > 0 else 0.0

    result["speedup_ratio"]           = speedup
    result["ttft_savings_s"]          = savings_s
    result["ttft_savings_pct"]        = savings_pct
    result["t_reconcile_per_token_us"] = t_reconcile_per_token_us

    # ── 메모리 지표 ────────────────────────────────────────────────────
    # vLLM이 KV pool을 init 시 통째로 pre-allocate하므로 memory_allocated() 차이는
    # 항상 0. 대신 모델 config + block 수로 이론값을 정확하게 계산.
    result["kv_cache_gb_mean"]       = kv_cache_gb_theoretical(actual_T, llm)
    result["kv_cache_gb_theoretical"] = kv_cache_gb_theoretical(actual_T, llm)
    result["kv_cache_gb_stats"]      = st_kv_gb   # 참고용 (항상 0)
    result["cpu_mem_gb"]             = round(cpu_mem_gb(), 3)

    # ── 출력 ─────────────────────────────────────────────────────────
    status = "✅" if surgery_ok_any else "⚠️ failed"
    path_info = ", ".join(f"{k}:{v}" for k, v in sorted(reconcile_paths.items())) or "none"

    print(f"    t_sync:       {_fmt_stat(st_sync, num_runs)}")
    print(
        f"    t_reconcile:  {_fmt_stat(st_reconcile, num_runs)} ({status})  "
        f"[{t_reconcile_per_token_us:.1f} μs/token]  paths={path_info}"
    )
    print(
        f"    t_first_chat: partial={_fmt_stat(st_partial, num_runs)}  "
        f"origin={_fmt_stat(st_origin, num_runs)}  "
        f"({speedup:.1f}x speedup, -{savings_pct:.0f}%)"
    )
    print(
        f"    t_total_eff:  partial={result['t_total_effective_partial_s']:.3f}s  "
        f"origin={result['t_total_effective_origin_s']:.3f}s"
    )
    print(f"    KV cache:     {result['kv_cache_gb_mean']:.3f} GB (theoretical)  "
          f"CPU RAM: {result['cpu_mem_gb']:.2f} GB")

    return result


# ============================================================================
# 메인 벤치마크
# ============================================================================

def run_scaling(
    model_name: str,
    token_counts: list[int],
    output_path: str,
    stage: str,
    max_model_len: int,
    progressive_impl: str,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    enforce_eager: bool,
    num_runs: int = 1,
) -> dict:
    print("\n" + "=" * 65)
    print(f"  TOKEN SCALING BENCHMARK  (model={model_name}, stage={stage}, runs={num_runs})")
    print("=" * 65)

    impl_name, impl_path = resolve_progressive_impl(progressive_impl)
    prepare_progressive_imports(impl_path)
    from progressive_for_causal_lm import ProgressiveForCausalLM as PartialPFCLM  # noqa: F401

    drop_all_caches()

    config = dict(MODELS[model_name])
    model_path = config["progressive_path"]
    prefix_caching_enabled = bool(config.get("enable_prefix_caching", True))
    effective_gpu_memory_utilization = (
        float(gpu_memory_utilization)
        if float(gpu_memory_utilization) > 0
        else float(config.get("gpu_memory_utilization", 0.4))
    )
    effective_tensor_parallel_size = (
        max(1, int(tensor_parallel_size))
        if int(tensor_parallel_size) > 0
        else max(1, int(config.get("tensor_parallel_size", 1)))
    )
    trust_remote_code = bool(config.get("trust_remote_code", True))
    disable_sliding_window = bool(config.get("disable_sliding_window", False))

    with open(os.path.join(model_path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    try:
        ModelRegistry.register_model(arch, PartialPFCLM)
        print(f"  Registered PartialPFCLM as: {arch}")
    except Exception as e:
        print(f"  ModelRegistry: {e}")

    print(f"\n  Progressive implementation: {impl_name} ({impl_path})")
    print(f"  Loading model: {model_path}")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=effective_gpu_memory_utilization,
        max_model_len=max_model_len,
        tensor_parallel_size=effective_tensor_parallel_size,
        enforce_eager=enforce_eager,
        enable_prefix_caching=prefix_caching_enabled,
        disable_sliding_window=disable_sliding_window,
    )
    t_load = round(time.time() - t0, 2)

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()

    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()
        print("  ✅ Persistent GPU buffers cleared")

    print(f"  ✅ Loaded in {t_load:.1f}s  GPU={gpu_mem_gb():.2f}GB")

    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    minimal_params  = SamplingParams(temperature=0.0, max_tokens=1)

    stage_cfg = STAGE_CONFIG[stage]

    # 2to3: 먼저 Stage 2 활성화 (시간 미측정)
    if stage_cfg["setup_stage"] is not None:
        setup_cfg  = STAGE_CONFIG[stage_cfg["setup_stage"]]
        setup_ckpt = config[setup_cfg["checkpoint_key"]]
        assert os.path.exists(setup_ckpt), f"Stage 2 checkpoint not found: {setup_ckpt}"
        print(f"\n  [Setup] Activating Stage 2 first (required for 2to3)...")
        getattr(model, setup_cfg["prefetch_fn"])(setup_ckpt)
        model.wait_for_prefetch(timeout_s=120.0)
        ok = getattr(model, setup_cfg["advance_fn"])(wait_if_needed=False)
        if not ok:
            raise RuntimeError("Stage 2 setup activation failed")
        reset_prefix_cache(llm)
        warmup_generate(llm, tokenizer, n=3, label="Stage 2 setup warmup done")
        print("  [Setup] Stage 2 active ✅")

    # ── 전환 상수 1회 측정 (prefetch + activation + warmup 포함) ───────
    t_constants = measure_transition_constants(llm, model, tokenizer, config, stage_cfg)

    # ── 각 T에서 측정 ──────────────────────────────────────────────────
    print(f"\n  Token counts to measure: {token_counts}")
    scaling = []
    for T in sorted(set(token_counts)):
        r = measure_at_token_count(
            llm, model, tokenizer,
            target_T=T,
            sampling_params=sampling_params,
            minimal_params=minimal_params,
            stage_cfg=stage_cfg,
            t_constants=t_constants,
            max_model_len=max_model_len,
            prefix_caching_enabled=prefix_caching_enabled,
            num_runs=num_runs,
        )
        if r is not None:
            scaling.append(r)

        # T 간 GPU 메모리 클리어 (잔류 텐서 + 단편화 해소)
        clear_runtime_state(llm, model)

    # ── 결과 저장 ──────────────────────────────────────────────────────
    gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    gpu_total = (torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                 if torch.cuda.is_available() else 0.0)

    results = {
        "model":            model_name,
        "stage":            stage,
        "num_runs":         num_runs,
        "progressive_impl": impl_name,
        "progressive_impl_path": impl_path,
        "max_model_len":    max_model_len,
        "timestamp":        datetime.datetime.now().isoformat(),
        "gpu_name":         gpu_name,
        "gpu_total_mem_gb": round(gpu_total, 2),
        "t_load_s":         t_load,
        "gpu_mem_gb":       round(gpu_mem_gb(), 3),
        "cpu_mem_gb":       round(cpu_mem_gb(), 3),
        "progressive_path": model_path,
        "stage_b_checkpoint": config.get("stage_b_checkpoint"),
        "stage_c_checkpoint": config.get("stage_c_checkpoint"),
        "trust_remote_code": trust_remote_code,
        "enable_prefix_caching": prefix_caching_enabled,
        "disable_sliding_window": disable_sliding_window,
        "gpu_memory_utilization_effective": effective_gpu_memory_utilization,
        "tensor_parallel_size_effective": effective_tensor_parallel_size,
        "enforce_eager": enforce_eager,
        "t_constants":      t_constants,
        "fixed_question":   FIXED_QUESTION,
        "scaling":          scaling,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved → {output_path}")
    return results


# ============================================================================
# 플롯
# ============================================================================

def plot_results(data: dict, save_path: str | None = None):
    """스케일링 결과 그래프 생성 (4-panel). num_runs > 1 시 오차막대 표시."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not installed: pip install matplotlib")
        return

    scaling = data.get("scaling", [])
    if not scaling:
        print("  ⚠️  No scaling data to plot.")
        return

    Ts          = [r["actual_T"] for r in scaling]
    t_orig      = [r["t_first_chat_origin_s"] for r in scaling]
    t_part      = [r["t_first_chat_partial_s"] for r in scaling]
    t_reconcile = [r.get("t_reconcile_s", 0.0) for r in scaling]
    t_eff_orig  = [r["t_total_effective_origin_s"] for r in scaling]
    t_eff_part  = [r["t_total_effective_partial_s"] for r in scaling]
    speedups   = [r.get("speedup_ratio",
                        (r["t_first_chat_origin_s"] / r["t_first_chat_partial_s"]
                         if r["t_first_chat_partial_s"] > 0 else 0.0))
                  for r in scaling]
    kv_gb = [r.get("kv_cache_gb_mean", 0.0) for r in scaling]
    reconcile_us_tok = [
        r.get("t_reconcile_per_token_us", 0.0)
        for r in scaling
    ]

    # 오차막대 (num_runs > 1 시)
    err_orig = [r.get("t_first_chat_origin_stats", {}).get("std", 0.0) for r in scaling]
    err_part = [r.get("t_first_chat_partial_stats", {}).get("std", 0.0) for r in scaling]
    err_reconcile = [
        r.get("t_reconcile_stats", {}).get("std", 0.0)
        for r in scaling
    ]
    has_err  = any(e > 0 for e in err_orig + err_part + err_reconcile)

    tc       = data["t_constants"]
    floor    = tc["t_prefetch_s"] + tc["t_activation_s"]
    num_runs = data.get("num_runs", 1)
    gpu_name = data.get("gpu_name", "")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    fig.suptitle(
        f"Stage Transition Scaling  —  "
        f"Model: {data['model'].upper()}  Stage: {data['stage']}  runs={num_runs}\n"
        f"GPU: {gpu_name}  |  "
        f"t_prefetch={tc['t_prefetch_s']:.2f}s  t_activation={tc['t_activation_s']:.3f}s",
        fontsize=11,
    )

    ekw = dict(capsize=4, elinewidth=1.5) if has_err else {}

    # ── [1] t_first_chat (TTFT) vs T ────────────────────────────────
    ax1.errorbar(Ts, t_orig, yerr=err_orig if has_err else None,
                 fmt="ro-", label="Origin (full prefill)",   lw=2, ms=7, **ekw)
    ax1.errorbar(Ts, t_part, yerr=err_part if has_err else None,
                 fmt="bs-", label="Partial (prefix hit + Q)", lw=2, ms=7, **ekw)
    ax1.set_ylabel("TTFT / t_first_chat (s)")
    ax1.set_title("① TTFT after Stage Transition")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax1r = ax1.twinx()
    ax1r.errorbar(Ts, t_reconcile, yerr=err_reconcile if has_err else None,
                  fmt="g^--", label="t_reconcile", lw=1.5, ms=6, alpha=0.75, **ekw)
    ax1r.set_ylabel("t_reconcile (s)", color="green")
    ax1r.tick_params(axis="y", labelcolor="green")
    ax1r.legend(loc="center right")

    # ── [2] Speedup ratio vs T (핵심 Figure) ────────────────────────
    ax2.plot(Ts, speedups, "m^-", lw=2, ms=8, label="TTFT speedup (Origin/Partial)")
    ax2.axhline(y=1.0, color="gray", ls=":", lw=1.5, label="Baseline (1×)")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("② TTFT Speedup  [Partial vs Origin]")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    # 각 점에 값 표시
    for x, y in zip(Ts, speedups):
        ax2.annotate(f"{y:.1f}×", xy=(x, y), xytext=(0, 6),
                     textcoords="offset points", ha="center", fontsize=8)

    # ── [3] t_total_effective vs T ──────────────────────────────────
    ax3.plot(Ts, t_eff_orig, "ro-", label="Origin  total_effective", lw=2, ms=7)
    ax3.plot(Ts, t_eff_part, "bs-", label="Partial total_effective", lw=2, ms=7)
    ax3.axhline(y=floor, color="gray", ls=":", lw=1.5,
                label=f"floor = prefetch+activ ({floor:.2f}s)")
    ax3.set_xlabel("Accumulated tokens (T)")
    ax3.set_ylabel("t_total_effective (s)")
    ax3.set_title("③ t_total_effective = transition + TTFT")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # ── [4] KV cache memory & t_reconcile/token vs T ────────────────
    ax4.bar(Ts, kv_gb, width=[max((Ts[-1] - Ts[0]) / (len(Ts) * 1.5), 50)] * len(Ts),
            color="steelblue", alpha=0.7, label="KV cache (GB)")
    ax4.set_xlabel("Accumulated tokens (T)")
    ax4.set_ylabel("KV cache size (GB)", color="steelblue")
    ax4.tick_params(axis="y", labelcolor="steelblue")
    ax4.set_title("④ KV Cache Memory  &  t_reconcile / token")
    ax4.grid(True, alpha=0.3, axis="y")

    if any(v > 0 for v in reconcile_us_tok):
        ax4r = ax4.twinx()
        ax4r.plot(Ts, reconcile_us_tok, "rs--", lw=1.5, ms=6, alpha=0.85,
                  label="t_reconcile/token (μs)")
        ax4r.set_ylabel("t_reconcile per token (μs)", color="red")
        ax4r.tick_params(axis="y", labelcolor="red")
        ax4r.legend(loc="upper right")

    ax4.legend(loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✅ Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Token scaling benchmark for stage transition timing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_scaling.py --model llama2-7b
  python benchmark_scaling.py --model falcon-7b --stage 2to3
  python benchmark_scaling.py --model gemma-7b --token-counts 64,128,256,384,448
  python benchmark_scaling.py --model llama2-13b --num-runs 3
  python benchmark_scaling.py --plot results_scaling_llama2-7b_1to2_20260406_120000.json

Note:
  --max-model-len 0 이면 모델별 기본값 사용.
  긴 context를 직접 지정할 때는 필요 시:
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 python benchmark_scaling.py ...
        """,
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL)
    parser.add_argument("--stage",  choices=list(STAGE_CONFIG.keys()), default="1to2")
    parser.add_argument(
        "--token-counts", type=str,
        default="",
        help="쉼표 구분 토큰 수. 비우면 모델별 기본 토큰 수 사용",
    )
    parser.add_argument(
        "--num-runs", type=int, default=DEFAULT_NUM_RUNS,
        help=f"각 T당 반복 측정 횟수 (default: {DEFAULT_NUM_RUNS}). "
             "논문용 정확한 mean±std를 원하면 3 이상 권장",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="출력 JSON 경로 (default: results_scaling_{model}_{stage}_{timestamp}.json)",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=0,
        help="vLLM max_model_len override. 0이면 모델별 기본값 사용",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.0,
        help="gpu_memory_utilization override. 0이면 모델별 기본값 사용",
    )
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=0,
        help="tensor_parallel_size override. 0이면 모델별 기본값 사용",
    )
    parser.add_argument(
        "--enforce-eager", action="store_true",
        help="vLLM enforce_eager=True 로 실행",
    )
    parser.add_argument(
        "--progressive-impl", type=str, default=DEFAULT_PROGRESSIVE_IMPL,
        help=f"progressive implementation 디렉터리 (default: {DEFAULT_PROGRESSIVE_IMPL})",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="측정 후 그래프 생성 스킵",
    )
    parser.add_argument(
        "--plot", type=str, default=None, metavar="JSON_FILE",
        help="기존 JSON 결과를 불러와 그래프만 출력",
    )

    args = parser.parse_args()

    # ── plot only 모드 ──────────────────────────────────────────────
    if args.plot:
        with open(args.plot) as f:
            data = json.load(f)
        save_path = args.plot.replace(".json", ".png")
        plot_results(data, save_path=save_path)
        return

    # ── 벤치마크 실행 ────────────────────────────────────────────────
    model_name = canonical_model_name(args.model)
    model_cfg = MODELS[model_name]
    effective_max_model_len = (
        int(args.max_model_len)
        if int(args.max_model_len) > 0
        else int(model_cfg.get("max_model_len", DEFAULT_MAX_MODEL_LEN))
    )

    if args.token_counts.strip():
        token_counts = sorted(set(int(x.strip()) for x in args.token_counts.split(",") if x.strip()))
    else:
        token_counts = model_default_token_counts(model_name, effective_max_model_len)
    if not token_counts:
        raise ValueError("No valid token counts resolved. Please provide --token-counts explicitly.")

    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results_scaling_{model_name}_{args.stage}_{ts}.json"

    print(f"\n{'='*65}")
    print(f"  Token Scaling Benchmark")
    print(f"  Model:         {model_name}")
    print(f"  Stage:         {args.stage}")
    print(f"  Token counts:  {token_counts}")
    print(f"  Runs per T:    {args.num_runs}")
    print(f"  max_model_len: {effective_max_model_len}")
    print(f"  gpu_mem_util:  {args.gpu_memory_utilization or model_cfg.get('gpu_memory_utilization', 0.4)}")
    print(f"  tp_size:       {args.tensor_parallel_size or model_cfg.get('tensor_parallel_size', 1)}")
    print(f"  impl:          {args.progressive_impl}")
    print(f"  enforce_eager: {args.enforce_eager}")
    print(f"  GPU:           {torch.cuda.get_device_name(0)}")
    print(f"  Output:        {args.output}")
    print(f"{'='*65}")

    results = run_scaling(
        model_name=model_name,
        token_counts=token_counts,
        output_path=args.output,
        stage=args.stage,
        max_model_len=effective_max_model_len,
        progressive_impl=args.progressive_impl,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        num_runs=args.num_runs,
    )

    if not args.no_plot:
        plot_path = args.output.replace(".json", ".png")
        plot_results(results, save_path=plot_path)


if __name__ == "__main__":
    main()
