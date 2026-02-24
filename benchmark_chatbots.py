#!/usr/bin/env python3
"""
Benchmark: chatbot_origin vs chatbot_partial_cache
=====================================================

각 모드를 **별도 프로세스**로 실행하여 공정한 비교.
(동일 프로세스에서 두 모드를 순차 실행하면 GPU/OS page cache가 공유되어 불공정)

Usage:
  # 1) Origin 모드 실행 (별도 터미널 / GPU 0)
  python benchmark_chatbots.py --mode origin --model llama --output results_origin.json

  # 2) Partial 모드 실행 (별도 터미널 / GPU 0, origin 완전 종료 후)
  python benchmark_chatbots.py --mode partial --model llama --output results_partial.json

  # 3) 결과 비교
  python benchmark_chatbots.py --compare results_origin.json results_partial.json

측정 항목:
  [Origin]   t_prefetch | t_activation | t_cache_clear
             → t_total_transition (빠름)
             → t_first_chat  ← 여기서 FULL PREFILL 발생 (느림)
             → t_total_effective = transition + first_chat

  [Partial / Method A - GPU-resident]
             t_sync≈0 | t_prefetch | t_activation(+block_reset) | t_recompute(back-layers only)
             → t_total_transition
             → t_first_chat  ← KV 이미 계산됨 (빠름)

  [Partial / Method B - CPU-based (legacy)]
             t_sync | t_prefetch | t_activation | t_recompute(kv-only+back-layers)
             → t_total_transition (recompute 포함, 느림)
             → t_first_chat  ← KV 이미 계산됨 (빠름)

핵심: t_total_effective 가 진짜 사용자 체감 비용
"""

import os
import sys
import gc
import json
import time
import argparse
import subprocess

os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================================
# 모델 설정
# ============================================================================

MODELS = {
    "llama": {
        "progressive_path":   "/acpl-ssd30/7b_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage3_layers_C.safetensors",
    },
    "mistral": {
        "progressive_path":   "/home/devewha/entropy_routing/25_mistral_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage3_layers_C.safetensors",
    },
}


# ============================================================================
# 고정 대화 스크립트 (두 모드에 동일하게 적용)
# ============================================================================

# Stage 1에서 3번 대화 (KV cache 누적)
STAGE1_PROMPTS = [
    "Tell me about the history of computing briefly.",
    "What were the key innovations of the 1970s in computing?",
    "How did personal computers change society?",
]

# Stage 2에서 사용할 질문들 (첫 번째 = transition 직후 첫 채팅)
STAGE2_PROMPTS = [
    "What is machine learning in simple terms?",   # ← transition 후 첫 채팅 (핵심 측정)
    "Can you explain neural networks briefly?",
]

# Stage 3에서 사용할 질문들
STAGE3_PROMPTS = [
    "What is the future of artificial intelligence?",  # ← transition 후 첫 채팅
    "How will AI affect employment in the next decade?",
]


# ============================================================================
# 유틸리티
# ============================================================================

def drop_all_caches():
    """
    Python GC + CUDA cache + OS page cache 완전 제거.

    OS page cache 제거 이유:
    - safetensors checkpoint 파일이 이전 실행에서 RAM에 캐싱됨
    - 제거 안 하면 두 번째 실행이 디스크 I/O 없이 빠르게 로드 → 불공정
    - sudo 권한 필요: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
    """
    print("\n  [Cache] Clearing all caches...")

    # 1. Python GC
    gc.collect()
    print("  [Cache]   ✅ Python GC collected")

    # 2. CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  [Cache]   ✅ CUDA allocator cache cleared")

    # 3. OS page cache
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True, capture_output=True, timeout=15
        )
        print("  [Cache]   ✅ OS page cache dropped (sync + echo 3)")
    except subprocess.CalledProcessError as e:
        print(f"  [Cache]   ⚠️  OS page cache: sudo failed (returncode={e.returncode})")
        print("             → 수동으로 실행 필요: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    except FileNotFoundError:
        print("  [Cache]   ⚠️  OS page cache: sudo not found")
    except Exception as e:
        print(f"  [Cache]   ⚠️  OS page cache: {e}")

    time.sleep(2)  # cache settle 대기


def gpu_mem_gb() -> float:
    return torch.cuda.memory_allocated() / (1024 ** 3)


def get_model_handle(llm):
    """v0 엔진 progressive model handle 가져오기"""
    engine = llm.llm_engine
    if hasattr(engine, "engine_core"):
        raise RuntimeError("V1 engine detected. Set VLLM_USE_V1=0.")
    try:
        return engine.model_executor.driver_worker.worker.model_runner.model
    except AttributeError as exc:
        raise RuntimeError("Could not resolve v0 model handle.") from exc


def build_prompt(tokenizer, conversation: list) -> str:
    """대화 기록 → 프롬프트 문자열"""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    prompt = ""
    for msg in conversation:
        prefix = "User: " if msg["role"] == "user" else "Assistant: "
        prompt += prefix + msg["content"] + "\n"
    return prompt + "Assistant: "


def do_chat(llm, tokenizer, conversation, user_input, sampling_params):
    """
    대화 1턴 수행 + 타이밍/토큰 수 측정.
    conversation 리스트를 in-place로 업데이트.
    """
    conversation.append({"role": "user", "content": user_input})
    prompt = build_prompt(tokenizer, conversation)
    n_input = len(tokenizer.encode(prompt))

    torch.cuda.synchronize()
    t0 = time.time()
    outputs = llm.generate([prompt], sampling_params)
    torch.cuda.synchronize()
    t_chat = time.time() - t0

    response = outputs[0].outputs[0].text.strip()
    n_gen = len(tokenizer.encode(response))
    conversation.append({"role": "assistant", "content": response})

    return {
        "question": user_input[:60],
        "t_chat_s": round(t_chat, 3),
        "n_input_tokens": n_input,
        "n_gen_tokens": n_gen,
        "tokens_per_sec": round(n_gen / t_chat, 1) if t_chat > 0 else 0.0,
        "gpu_mem_gb": round(gpu_mem_gb(), 3),
    }


def _measure_transition_origin(llm, model, tokenizer, config, stage_key,
                                advance_fn_name, prefetch_fn_name,
                                conversation, sampling_params,
                                first_prompt):
    """
    Origin 모드 stage 전환 타이밍 측정.

    단계: prefetch → activation → cache_clear
    이후: 첫 채팅 (full prefill 발생)
    """
    tr = {}
    checkpoint_path = config[stage_key]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    prefetch_fn = getattr(model, prefetch_fn_name)
    advance_fn  = getattr(model, advance_fn_name)

    # t_prefetch: checkpoint CPU 로드 + 대기
    torch.cuda.synchronize()
    t0 = time.time()
    prefetch_fn(checkpoint_path)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    tr["t_prefetch_s"] = round(time.time() - t0, 3)

    # t_activation: GPU weight copy + alpha 변경
    torch.cuda.synchronize()
    t0 = time.time()
    ok = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    tr["t_activation_s"] = round(time.time() - t0, 3)
    if not ok:
        raise RuntimeError(f"{advance_fn_name} returned False")

    # t_cache_clear: prefix cache 초기화 (다음 turn에서 full prefill 자동)
    t0 = time.time()
    llm.reset_prefix_cache()
    tr["t_cache_clear_s"] = round(time.time() - t0, 3)

    tr["t_total_transition_s"] = round(
        tr["t_prefetch_s"] + tr["t_activation_s"] + tr["t_cache_clear_s"], 3
    )
    tr["gpu_mem_after_transition_gb"] = round(gpu_mem_gb(), 3)

    # 첫 채팅 (FULL PREFILL: KV cache가 모두 비워졌으므로)
    print(f"    → t_prefetch={tr['t_prefetch_s']:.3f}s | "
          f"t_activation={tr['t_activation_s']:.3f}s | "
          f"t_cache_clear={tr['t_cache_clear_s']:.3f}s | "
          f"t_transition={tr['t_total_transition_s']:.3f}s")
    print(f"    → [First chat] FULL PREFILL (KV cache cleared)...")

    r_first = do_chat(llm, tokenizer, conversation, first_prompt, sampling_params)
    tr["t_first_chat_s"]     = r_first["t_chat_s"]
    tr["first_chat_n_input"]  = r_first["n_input_tokens"]
    tr["first_chat_n_gen"]    = r_first["n_gen_tokens"]
    tr["t_total_effective_s"] = round(tr["t_total_transition_s"] + tr["t_first_chat_s"], 3)

    print(f"    → t_first_chat={tr['t_first_chat_s']:.3f}s  "
          f"t_total_effective={tr['t_total_effective_s']:.3f}s")

    return tr


def _measure_transition_partial(llm, model, tokenizer, config, stage_key,
                                 advance_fn_name, prefetch_fn_name,
                                 conversation, sampling_params,
                                 first_prompt):
    """
    Partial 모드 stage 전환 타이밍 측정.

    [Method A - GPU-resident]: (block_reset) → prefetch → activation → recompute(back-layers only)
    [Method B - CPU-based]:    sync → prefetch → activation → recompute(kv-only+back-layers)
    이후: 첫 채팅 (KV 이미 계산됨 → 빠름)

    Method A 감지 기준: model.model._persistent_buffers_initialized == True
                       AND model.model.set_recompute_from_boundary_gpu 존재
    """
    tr = {}
    checkpoint_path = config[stage_key]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    prefetch_fn = getattr(model, prefetch_fn_name)
    advance_fn  = getattr(model, advance_fn_name)
    minimal_params = SamplingParams(temperature=0.0, max_tokens=1)

    # Method A (GPU-resident) vs Method B (CPU-based) 감지
    inner_model = getattr(model, "model", None)
    use_gpu_resident = (
        inner_model is not None
        and hasattr(inner_model, "_persistent_buffers_initialized")
        and inner_model._persistent_buffers_initialized
        and hasattr(inner_model, "set_recompute_from_boundary_gpu")
    )
    tr["method"] = "A_gpu_resident" if use_gpu_resident else "B_cpu_based"
    method_label = "A:GPU-only" if use_gpu_resident else "B:CPU-sync"
    print(f"    → Recompute method: {tr['method']}")

    # t_sync: [Method B] GPU persistent buffer → CPU cache 동기화
    #         [Method A] 불필요 (GPU buffer에 이미 존재) → 0.000s 기록
    t0 = time.time()
    if not use_gpu_resident:
        if inner_model is not None and hasattr(inner_model, "sync_persistent_cache"):
            prompt_now = build_prompt(tokenizer, conversation)
            seq_len = len(tokenizer.encode(prompt_now))
            torch.cuda.synchronize()
            inner_model.sync_persistent_cache(seq_len)
            torch.cuda.synchronize()
            print(f"    → [B] sync_persistent_cache({seq_len} tokens)")
    else:
        print(f"    → [A] No CPU sync needed (GPU-resident buffers)")
    tr["t_sync_s"] = round(time.time() - t0, 3)

    # t_prefetch: checkpoint CPU 로드 + 대기
    torch.cuda.synchronize()
    t0 = time.time()
    prefetch_fn(checkpoint_path)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    tr["t_prefetch_s"] = round(time.time() - t0, 3)

    # t_activation: GPU weight copy + boundary 설정
    # [Method A] advance_fn 이후 block computed 리셋 포함:
    #   → prefix caching이 context_len=0으로 판단 → 전체 토큰 fresh prefill
    #   → front layers forward 스킵 (KV 유지), back layers GPU buffer에서 재계산
    torch.cuda.synchronize()
    t0 = time.time()
    ok = advance_fn(wait_if_needed=False)
    if use_gpu_resident:
        # benchmark는 chatbot_partial_cache.py를 거치지 않으므로 직접 block reset
        engine = llm.llm_engine
        if hasattr(engine, 'scheduler') and engine.scheduler:
            block_manager = engine.scheduler[0].block_manager
            if hasattr(block_manager, 'mark_all_blocks_as_uncomputed'):
                block_manager.mark_all_blocks_as_uncomputed()
                print(f"    → [A] All KV blocks marked as uncomputed (forcing fresh prefill)")
    torch.cuda.synchronize()
    tr["t_activation_s"] = round(time.time() - t0, 3)
    if not ok:
        raise RuntimeError(f"{advance_fn_name} returned False")

    # t_recompute: partial recompute (generate max_tokens=1)
    # [Method A] back-layers only: GPU buffer에서 boundary hidden states 직접 사용
    #            front KV cache 유지, CPU↔GPU 전송 없음
    # [Method B] KV-only pass(front layers) + full forward(back layers)
    torch.cuda.synchronize()
    t0 = time.time()
    if len(conversation) > 0:
        prompt_now = build_prompt(tokenizer, conversation)
        llm.generate([prompt_now], minimal_params)
    torch.cuda.synchronize()
    tr["t_recompute_s"] = round(time.time() - t0, 3)

    tr["t_total_transition_s"] = round(
        tr["t_sync_s"] + tr["t_prefetch_s"] + tr["t_activation_s"] + tr["t_recompute_s"], 3
    )
    tr["gpu_mem_after_transition_gb"] = round(gpu_mem_gb(), 3)

    print(f"    → [{method_label}] t_sync={tr['t_sync_s']:.3f}s | "
          f"t_prefetch={tr['t_prefetch_s']:.3f}s | "
          f"t_activation={tr['t_activation_s']:.3f}s | "
          f"t_recompute={tr['t_recompute_s']:.3f}s | "
          f"t_transition={tr['t_total_transition_s']:.3f}s")
    print(f"    → [First chat] KV already updated (should be fast)...")

    # 첫 채팅 (KV 이미 partial recompute 완료 → 새 user 입력만 처리)
    r_first = do_chat(llm, tokenizer, conversation, first_prompt, sampling_params)
    tr["t_first_chat_s"]     = r_first["t_chat_s"]
    tr["first_chat_n_input"]  = r_first["n_input_tokens"]
    tr["first_chat_n_gen"]    = r_first["n_gen_tokens"]
    tr["t_total_effective_s"] = round(tr["t_total_transition_s"] + tr["t_first_chat_s"], 3)

    print(f"    → t_first_chat={tr['t_first_chat_s']:.3f}s  "
          f"t_total_effective={tr['t_total_effective_s']:.3f}s")

    return tr


# ============================================================================
# Origin 모드 벤치마크
# ============================================================================

def run_origin(model_name: str, output_path: str):
    """
    Origin 모드 벤치마크.

    Stage 전환: reset_prefix_cache()만 호출 → 다음 chat에서 full prefill.
    사용 모듈: origin_progressive_serve/
    """
    print("\n" + "=" * 65)
    print(f"  BENCHMARK: Origin Mode  (model={model_name})")
    print("=" * 65)

    # origin_progressive_serve import
    # model_config를 먼저 import해야 progressive_model_dual_path.py의
    # hardcoded sys.path(/home/devewha/v08/...)가 우회됨
    origin_dir = os.path.join(SCRIPT_DIR, "origin_progressive_serve")
    sys.path.insert(0, origin_dir)
    import model_config as _mc_origin  # noqa: F401
    from progressive_for_causal_lm import ProgressiveForCausalLM as OriginPFCLM

    # 캐시 완전 제거 (OS page cache 포함)
    drop_all_caches()

    config = MODELS[model_name]
    model_path = config["progressive_path"]

    with open(os.path.join(model_path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    try:
        ModelRegistry.register_model(arch, OriginPFCLM)
        print(f"  Registered OriginPFCLM as: {arch}")
    except Exception as e:
        print(f"  ModelRegistry.register_model: {e}")

    # 모델 로드
    print(f"\n  Loading model from: {model_path}")
    t_load_start = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    t_load = time.time() - t_load_start

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    print(f"  ✅ Loaded in {t_load:.1f}s  GPU={gpu_mem_gb():.2f}GB")

    results = {
        "mode": "origin",
        "model": model_name,
        "t_load_s": round(t_load, 2),
        "gpu_mem_after_load_gb": round(gpu_mem_gb(), 3),
        "stage1_chats": [],
        "stage1_to_2": {},
        "stage2_chats": [],
        "stage2_to_3": {},
        "stage3_chats": [],
    }

    conversation = []

    # ── Stage 1 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 1] {len(STAGE1_PROMPTS)} turns")
    for i, q in enumerate(STAGE1_PROMPTS):
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage1_chats"].append(r)
        print(f"    Turn {i+1}: {r['t_chat_s']:.2f}s  "
              f"({r['n_input_tokens']}→{r['n_gen_tokens']} tok, {r['tokens_per_sec']} tok/s)")

    # ── Stage 1 → 2 전환 ─────────────────────────────────────
    print(f"\n  [Stage 1 → 2] Transition...")
    tr12 = _measure_transition_origin(
        llm, model, tokenizer, config,
        stage_key="stage_b_checkpoint",
        prefetch_fn_name="prefetch_stage2",
        advance_fn_name="advance_to_stage2_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE2_PROMPTS[0],
    )
    results["stage1_to_2"] = tr12

    # ── Stage 2 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 2] {len(STAGE2_PROMPTS) - 1} more turn(s)")
    for q in STAGE2_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage2_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    # ── Stage 2 → 3 전환 ─────────────────────────────────────
    print(f"\n  [Stage 2 → 3] Transition...")
    tr23 = _measure_transition_origin(
        llm, model, tokenizer, config,
        stage_key="stage_c_checkpoint",
        prefetch_fn_name="prefetch_stage3",
        advance_fn_name="advance_to_stage3_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE3_PROMPTS[0],
    )
    results["stage2_to_3"] = tr23

    # ── Stage 3 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 3] {len(STAGE3_PROMPTS) - 1} more turn(s)")
    for q in STAGE3_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage3_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved → {output_path}")


# ============================================================================
# Partial 모드 벤치마크
# ============================================================================

def run_partial(model_name: str, output_path: str):
    """
    Partial 모드 벤치마크.

    Stage 전환 [Method A - GPU-resident]: prefetch → activation(+block_reset) → recompute(back-layers only)
    Stage 전환 [Method B - CPU-based]:    sync → prefetch → activation → recompute(kv-only+back-layers)
    사용 모듈: progressive_serve/
    Method 자동 감지: model.model._persistent_buffers_initialized 여부로 판단
    """
    print("\n" + "=" * 65)
    print(f"  BENCHMARK: Partial Mode  (model={model_name})")
    print("=" * 65)

    # progressive_serve import
    partial_dir = os.path.join(SCRIPT_DIR, "progressive_serve")
    sys.path.insert(0, partial_dir)
    from progressive_for_causal_lm import ProgressiveForCausalLM as PartialPFCLM

    # 캐시 완전 제거
    drop_all_caches()

    config = MODELS[model_name]
    model_path = config["progressive_path"]

    with open(os.path.join(model_path, "config.json")) as f:
        arch = json.load(f)["architectures"][0]
    try:
        ModelRegistry.register_model(arch, PartialPFCLM)
        print(f"  Registered PartialPFCLM as: {arch}")
    except Exception as e:
        print(f"  ModelRegistry.register_model: {e}")

    print(f"\n  Loading model from: {model_path}")
    t_load_start = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    t_load = time.time() - t_load_start

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    # Warmup 중 쌓인 쓰레기 데이터 제거
    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()
        print(f"  ✅ Persistent GPU buffers cleared (warmup residue removed)")

    print(f"  ✅ Loaded in {t_load:.1f}s  GPU={gpu_mem_gb():.2f}GB")

    results = {
        "mode": "partial",
        "model": model_name,
        "t_load_s": round(t_load, 2),
        "gpu_mem_after_load_gb": round(gpu_mem_gb(), 3),
        "stage1_chats": [],
        "stage1_to_2": {},
        "stage2_chats": [],
        "stage2_to_3": {},
        "stage3_chats": [],
    }

    conversation = []

    # ── Stage 1 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 1] {len(STAGE1_PROMPTS)} turns")
    for i, q in enumerate(STAGE1_PROMPTS):
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage1_chats"].append(r)
        print(f"    Turn {i+1}: {r['t_chat_s']:.2f}s  "
              f"({r['n_input_tokens']}→{r['n_gen_tokens']} tok, {r['tokens_per_sec']} tok/s)")

    # ── Stage 1 → 2 전환 ─────────────────────────────────────
    print(f"\n  [Stage 1 → 2] Transition...")
    tr12 = _measure_transition_partial(
        llm, model, tokenizer, config,
        stage_key="stage_b_checkpoint",
        prefetch_fn_name="prefetch_stage2",
        advance_fn_name="advance_to_stage2_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE2_PROMPTS[0],
    )
    results["stage1_to_2"] = tr12

    # ── Stage 2 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 2] {len(STAGE2_PROMPTS) - 1} more turn(s)")
    for q in STAGE2_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage2_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    # ── Stage 2 → 3 전환 ─────────────────────────────────────
    print(f"\n  [Stage 2 → 3] Transition...")
    tr23 = _measure_transition_partial(
        llm, model, tokenizer, config,
        stage_key="stage_c_checkpoint",
        prefetch_fn_name="prefetch_stage3",
        advance_fn_name="advance_to_stage3_instant",
        conversation=conversation,
        sampling_params=sampling_params,
        first_prompt=STAGE3_PROMPTS[0],
    )
    results["stage2_to_3"] = tr23

    # ── Stage 3 채팅 ──────────────────────────────────────────
    print(f"\n  [Stage 3] {len(STAGE3_PROMPTS) - 1} more turn(s)")
    for q in STAGE3_PROMPTS[1:]:
        r = do_chat(llm, tokenizer, conversation, q, sampling_params)
        results["stage3_chats"].append(r)
        print(f"    {r['t_chat_s']:.2f}s  ({r['n_input_tokens']}→{r['n_gen_tokens']} tok)")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✅ Results saved → {output_path}")


# ============================================================================
# 결과 비교
# ============================================================================

def compare(path_a: str, path_b: str):
    """두 JSON 결과 파일을 읽어 세부 단계 시간 비교 테이블 출력"""
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    label_a = f"{a['mode'].upper()} ({path_a})"
    label_b = f"{b['mode'].upper()} ({path_b})"

    def fmt_row(label, va, vb, unit="s"):
        """비교 행 포맷. None이면 N/A."""
        if va is None and vb is None:
            return f"  {label:<45}  {'N/A':>8}  {'N/A':>8}  {'':>12}"
        va_s = f"{va:.3f}{unit}" if va is not None else "N/A"
        vb_s = f"{vb:.3f}{unit}" if vb is not None else "N/A"
        if va is not None and vb is not None and va > 0 and vb > 0:
            diff = vb - va
            winner = "A" if va < vb else "B"
            pct = abs(diff) / max(va, vb) * 100
            arrow = "▼" if diff < 0 else "▲"
            diff_s = f"{arrow}{abs(diff):.3f}s ({pct:.0f}%) [{winner}↑]"
        else:
            diff_s = ""
        return f"  {label:<45}  {va_s:>8}  {vb_s:>8}  {diff_s}"

    W = 80
    print("\n" + "=" * W)
    print(f"  COMPARISON")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"  Model: {a['model']}")
    print("=" * W)
    print(f"  {'Metric':<45}  {'A':>8}  {'B':>8}  {'Delta (winner↑)':>20}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*20}")

    # ── 로드 시간 ──
    print(fmt_row("Model load time", a["t_load_s"], b["t_load_s"]))

    # ── Stage 1 채팅 ──
    print(f"\n  {'── Stage 1 Chats ──':}")
    chats_a = a.get("stage1_chats", [])
    chats_b = b.get("stage1_chats", [])
    for i in range(max(len(chats_a), len(chats_b))):
        ca = chats_a[i] if i < len(chats_a) else None
        cb = chats_b[i] if i < len(chats_b) else None
        label = f"  Turn {i+1} chat"
        va = ca["t_chat_s"] if ca else None
        vb = cb["t_chat_s"] if cb else None
        print(fmt_row(label, va, vb))

    def print_transition(label_12, ta, tb):
        print(f"\n  {'── ' + label_12 + ' ──':}")

        # sync (partial Method B only; Method A는 0.000s)
        va = ta.get("t_sync_s", 0.0)
        vb = tb.get("t_sync_s", 0.0)
        print(fmt_row("  t_sync [Method B: GPU→CPU; Method A: 0]", va, vb))

        print(fmt_row("  t_prefetch [ckpt CPU load]",
                      ta.get("t_prefetch_s"), tb.get("t_prefetch_s")))
        print(fmt_row("  t_activation [GPU weight copy]",
                      ta.get("t_activation_s"), tb.get("t_activation_s")))

        # cache_clear (origin only)
        va_cc = ta.get("t_cache_clear_s", 0.0)
        vb_cc = tb.get("t_cache_clear_s", 0.0)
        print(fmt_row("  t_cache_clear [origin only]", va_cc, vb_cc))

        # recompute (partial only)
        va_rc = ta.get("t_recompute_s", 0.0)
        vb_rc = tb.get("t_recompute_s", 0.0)
        print(fmt_row("  t_recompute [partial only]", va_rc, vb_rc))

        print(fmt_row("  t_total_transition",
                      ta.get("t_total_transition_s"), tb.get("t_total_transition_s")))

        n_in_a = ta.get("first_chat_n_input", "?")
        n_in_b = tb.get("first_chat_n_input", "?")
        print(fmt_row(f"  t_first_chat [KEY] "
                      f"(A:{n_in_a}tok, B:{n_in_b}tok)",
                      ta.get("t_first_chat_s"), tb.get("t_first_chat_s")))

        print(fmt_row("  ★ t_total_effective [transition+first_chat]",
                      ta.get("t_total_effective_s"), tb.get("t_total_effective_s")))

    print_transition("Stage 1 → 2 Transition",
                     a.get("stage1_to_2", {}), b.get("stage1_to_2", {}))

    # Stage 2 chats
    print(f"\n  {'── Stage 2 Chats ──':}")
    chats_a = a.get("stage2_chats", [])
    chats_b = b.get("stage2_chats", [])
    for i in range(max(len(chats_a), len(chats_b))):
        ca = chats_a[i] if i < len(chats_a) else None
        cb = chats_b[i] if i < len(chats_b) else None
        print(fmt_row(f"  Turn {i+1} chat",
                      ca["t_chat_s"] if ca else None,
                      cb["t_chat_s"] if cb else None))

    print_transition("Stage 2 → 3 Transition",
                     a.get("stage2_to_3", {}), b.get("stage2_to_3", {}))

    # Stage 3 chats
    print(f"\n  {'── Stage 3 Chats ──':}")
    chats_a = a.get("stage3_chats", [])
    chats_b = b.get("stage3_chats", [])
    for i in range(max(len(chats_a), len(chats_b))):
        ca = chats_a[i] if i < len(chats_a) else None
        cb = chats_b[i] if i < len(chats_b) else None
        print(fmt_row(f"  Turn {i+1} chat",
                      ca["t_chat_s"] if ca else None,
                      cb["t_chat_s"] if cb else None))

    # ── 핵심 요약 ──
    print("\n" + "=" * W)
    print("  ★ KEY INSIGHT (t_total_effective = transition + first_chat)")
    print(f"  {'Transition':<20}  {'A':>10}  {'B':>10}  Winner")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*8}")

    for label, key in [("Stage 1→2", "stage1_to_2"), ("Stage 2→3", "stage2_to_3")]:
        ta = a.get(key, {})
        tb = b.get(key, {})
        va = ta.get("t_total_effective_s")
        vb = tb.get("t_total_effective_s")
        if va is not None and vb is not None:
            winner = "A" if va < vb else "B"
            saving = abs(va - vb)
            pct = saving / max(va, vb) * 100
            print(f"  {label:<20}  {va:>9.2f}s  {vb:>9.2f}s  "
                  f"[{winner}] faster by {saving:.2f}s ({pct:.0f}%)")

    print("\n  NOTE:")
    print("  - t_first_chat in Origin  = FULL PREFILL (all tokens recomputed, KV cache cleared)")
    print("  - t_first_chat in Partial = only new user tokens processed (KV cache preserved)")
    print("  - t_sync in Partial       = 0.000s if Method A (GPU-resident); >0 if Method B (CPU sync)")
    print("  - t_recompute in Partial:")
    print("      Method A (GPU-resident): back layers only, GPU buffer → boundary hidden states,")
    print("                               front KV cache in-place, no CPU↔GPU transfer")
    print("      Method B (CPU-based):    KV-only pass (front layers) + full forward (back layers)")
    print("=" * W)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chatbot_origin vs chatbot_partial_cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run origin mode
  python benchmark_chatbots.py --mode origin --model llama --output results_origin.json

  # Run partial mode (separate process, after origin finishes)
  python benchmark_chatbots.py --mode partial --model llama --output results_partial.json

  # Compare results
  python benchmark_chatbots.py --compare results_origin.json results_partial.json
        """,
    )
    parser.add_argument(
        "--mode", choices=["origin", "partial"],
        help="Which chatbot mode to benchmark"
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()), default="llama",
        help="Model to use (default: llama)"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output JSON path for benchmark results"
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
        help="Compare two result JSON files"
    )

    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])

    elif args.mode:
        if not args.output:
            parser.error("--output is required with --mode")

        print(f"\n{'='*65}")
        print(f"  Chatbot Benchmark")
        print(f"  Mode:  {args.mode}")
        print(f"  Model: {args.model}")
        print(f"  GPU:   {torch.cuda.get_device_name(0)}")
        print(f"  Out:   {args.output}")
        print(f"{'='*65}")

        if args.mode == "origin":
            run_origin(args.model, args.output)
        else:
            run_partial(args.model, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
