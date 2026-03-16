#!/usr/bin/env python3
"""
Benchmark: chatbot_origin vs chatbot_partial_cache (일반 프롬프트 버전)
=====================================================

benchmark_chatbots.py 와 동일한 구조지만, 일반적인 길이의 프롬프트를 사용.
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 없이 max_model_len=4096 으로 실행 가능.

각 모드를 **별도 프로세스**로 실행하여 공정한 비교.
(동일 프로세스에서 두 모드를 순차 실행하면 GPU/OS page cache가 공유되어 불공정)

Usage:
  # 1) Origin 모드 실행 (별도 터미널 / GPU 0)
  python benchmark_chatbots_normal.py --mode origin --model llama --output results_origin.json

  # 2) Partial 모드 실행 (별도 터미널 / GPU 0, origin 완전 종료 후)
  python benchmark_chatbots_normal.py --mode partial --model llama --output results_partial.json

  # 3) 결과 비교
  python benchmark_chatbots_normal.py --compare results_origin.json results_partial.json

프롬프트 토큰 설계 (max_model_len=4096):
  Stage 1: 6턴, 각 ~80 토큰 → 전환 시점 ~600 토큰 누적
  Stage 2: 3턴, 각 ~80 토큰 → 전환 시점 ~900 토큰 누적
  Stage 3: 3턴, 각 ~80 토큰 → ~1200 토큰 누적 (4096 이내)

측정 항목:
  [Origin]  t_prefetch | t_activation | t_cache_clear
            → t_total_transition (빠름)
            → t_first_chat  ← 여기서 FULL PREFILL 발생 (느림)
            → t_total_effective = transition + first_chat

  [Partial] t_sync | t_prefetch | t_activation | t_skbi (~20ms, Selective KV Block Injection (SKBI))
            → t_total_transition (sync+SKBI 포함, 빠름)
            → t_first_chat  ← KV 이미 업데이트됨, prefix cache 유지 (빠름)
            → t_total_effective = transition + first_chat

핵심: t_total_effective 가 진짜 사용자 체감 비용
"""

import os
import sys
import gc
import json
import time
import argparse
import subprocess
import threading

import psutil

os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# 🔥 vLLM v0.8.0 버그 우회 (Prefix Caching 강제 활성화 패치)
# =========================================================
import vllm.config
# vLLM이 이 모델을 멀티모달로 착각하지 않도록 속임
vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)
# =========================================================

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

# Stage 1에서 6번 대화 (KV cache 누적)
# 각 프롬프트 ~60 단어(≈80 토큰), 6턴 + chat template ≈ 600 토큰
STAGE1_PROMPTS = [
    (
        "Give me a brief overview of the history of computing, "
        "from early mechanical calculators to modern microprocessors. "
        "Focus on the most important milestones."
    ),
    (
        "How did the invention of the transistor change computing? "
        "Why was it such a significant improvement over vacuum tubes? "
        "Mention a few key developments it enabled."
    ),
    (
        "What is the von Neumann architecture and why is it still "
        "relevant today? Briefly explain the stored-program concept "
        "and how it differs from earlier computing designs."
    ),
    (
        "Can you summarize the evolution of programming languages? "
        "Start from assembly and machine code, and trace the key "
        "steps up to modern high-level languages like Python and Rust."
    ),
    (
        "What were the most important contributions of UNIX to modern "
        "operating systems? How did its design principles influence "
        "Linux and other systems we use today?"
    ),
    (
        "How did the personal computer revolution of the 1970s and 1980s "
        "change society? Briefly describe the roles of Apple, IBM, and "
        "Microsoft in making PCs widely accessible."
    ),
]

# Stage 2에서 3번 대화 (첫 번째 = transition 직후 핵심 측정)
# S1 6턴 후 ~600 토큰, S2 3턴 후 ~900 토큰 누적
STAGE2_PROMPTS = [
    (
        "Explain the basics of machine learning: what is supervised learning, "
        "unsupervised learning, and reinforcement learning? Give a simple "
        "example of each."
    ),
    (
        "What is a neural network and how does backpropagation work? "
        "Briefly explain gradient descent and why it is used to train "
        "deep learning models."
    ),
    (
        "What is the transformer architecture and why did it revolutionize "
        "natural language processing? Explain the self-attention mechanism "
        "in simple terms."
    ),
]

# Stage 3에서 3번 대화 (첫 번째 = transition 직후 핵심 측정)
# S2 3턴 후 ~900 토큰, S3 3턴 후 ~1200 토큰 < 4096
STAGE3_PROMPTS = [
    (
        "What does the near-term future of AI look like over the next "
        "5 to 10 years? Briefly discuss trends in model scaling, "
        "multimodal capabilities, and AI agents."
    ),
    (
        "How might AI automation affect jobs over the next decade? "
        "Which types of work are most at risk and which new roles "
        "might emerge as a result?"
    ),
    (
        "What are the main geopolitical issues around AI development today? "
        "Briefly discuss the US-China technology competition and the "
        "debate around AI regulation and governance."
    ),
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


def gpu_reserved_gb() -> float:
    return torch.cuda.memory_reserved() / (1024 ** 3)


def gpu_peak_allocated_gb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


def gpu_peak_reserved_gb() -> float:
    return torch.cuda.max_memory_reserved() / (1024 ** 3)


def cpu_mem_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def ckpt_size_gb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 3)


class ResourceSampler:
    """
    백그라운드에서 CPU/GPU 메모리 및 CPU 활용률을 주기적으로 샘플링.
    prefetch처럼 오래 걸리는 단계의 피크 자원 사용량 측정에 사용.
    """
    def __init__(self, interval_s: float = 0.05):
        self.interval = interval_s
        self._samples: list = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()

    def start(self):
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._samples:
            return {}
        cpu_mems   = [s["cpu_mem_gb"]  for s in self._samples]
        gpu_mems   = [s["gpu_mem_gb"]  for s in self._samples]
        cpu_pcts   = [s["cpu_pct"]     for s in self._samples]
        return {
            "n_samples":        len(self._samples),
            "cpu_mem_peak_gb":  round(max(cpu_mems), 3),
            "cpu_mem_mean_gb":  round(sum(cpu_mems) / len(cpu_mems), 3),
            "gpu_mem_peak_gb":  round(max(gpu_mems), 3),
            "cpu_pct_mean":     round(sum(cpu_pcts) / len(cpu_pcts), 1),
            "cpu_pct_peak":     round(max(cpu_pcts), 1),
        }

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self._samples.append({
                    "cpu_mem_gb": self._proc.memory_info().rss / (1024 ** 3),
                    "gpu_mem_gb": torch.cuda.memory_allocated() / (1024 ** 3),
                    "cpu_pct":    self._proc.cpu_percent(),
                })
            except Exception:
                pass


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
        "gpu_allocated_gb": round(gpu_mem_gb(), 3),
        "gpu_reserved_gb":  round(gpu_reserved_gb(), 3),
        "cpu_mem_gb":       round(cpu_mem_gb(), 3),
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

    _ckpt_gb = ckpt_size_gb(checkpoint_path)
    tr["ckpt_size_gb"] = round(_ckpt_gb, 3)

    # 전환 전 스냅샷
    torch.cuda.reset_peak_memory_stats()
    tr["cpu_mem_before_gb"]        = round(cpu_mem_gb(), 3)
    tr["gpu_allocated_before_gb"]  = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_before_gb"]   = round(gpu_reserved_gb(), 3)

    # t_prefetch: checkpoint CPU 로드 + 대기 (백그라운드 샘플링)
    sampler = ResourceSampler(interval_s=0.05)
    sampler.start()
    torch.cuda.synchronize()
    t0 = time.time()
    prefetch_fn(checkpoint_path)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    tr["t_prefetch_s"] = round(time.time() - t0, 3)
    tr["prefetch_resources"] = sampler.stop()

    # prefetch 직후 스냅샷 (CPU에 ckpt 올라온 상태)
    tr["cpu_mem_after_prefetch_gb"] = round(cpu_mem_gb(), 3)

    # t_activation: GPU weight copy + alpha 변경
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    ok = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    tr["t_activation_s"] = round(time.time() - t0, 3)
    if not ok:
        raise RuntimeError(f"{advance_fn_name} returned False")

    # H2D 대역폭: ckpt 크기 / activation 시간
    tr["h2d_bandwidth_gb_s"] = round(
        _ckpt_gb / tr["t_activation_s"], 2) if tr["t_activation_s"] > 0 else 0.0
    tr["gpu_allocated_after_activation_gb"] = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_after_activation_gb"]  = round(gpu_reserved_gb(), 3)
    tr["gpu_peak_allocated_activation_gb"]  = round(gpu_peak_allocated_gb(), 3)

    # t_cache_clear: prefix cache 초기화 (다음 turn에서 full prefill 자동)
    t0 = time.time()
    llm.reset_prefix_cache()
    tr["t_cache_clear_s"] = round(time.time() - t0, 3)

    tr["t_total_transition_s"] = round(
        tr["t_prefetch_s"] + tr["t_activation_s"] + tr["t_cache_clear_s"], 3
    )
    tr["gpu_allocated_after_transition_gb"] = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_after_transition_gb"]  = round(gpu_reserved_gb(), 3)
    tr["cpu_mem_after_transition_gb"]       = round(cpu_mem_gb(), 3)

    # 첫 채팅 (FULL PREFILL: KV cache가 모두 비워졌으므로)
    print(f"    → t_prefetch={tr['t_prefetch_s']:.3f}s | "
          f"t_activation={tr['t_activation_s']:.3f}s | "
          f"t_cache_clear={tr['t_cache_clear_s']:.3f}s | "
          f"t_transition={tr['t_total_transition_s']:.3f}s")
    print(f"    → H2D bw={tr['h2d_bandwidth_gb_s']:.2f} GB/s | "
          f"CPU RAM peak={tr['prefetch_resources'].get('cpu_mem_peak_gb', '?'):.3f} GB | "
          f"CPU util={tr['prefetch_resources'].get('cpu_pct_mean', '?'):.1f}% avg")
    print(f"    → [First chat] FULL PREFILL (KV cache cleared)...")

    r_first = do_chat(llm, tokenizer, conversation, first_prompt, sampling_params)
    tr["t_first_chat_s"]     = r_first["t_chat_s"]
    tr["first_chat_n_input"]  = r_first["n_input_tokens"]
    tr["first_chat_n_gen"]    = r_first["n_gen_tokens"]
    tr["t_total_effective_s"] = round(tr["t_total_transition_s"] + tr["t_first_chat_s"], 3)
    tr["gpu_allocated_after_first_chat_gb"] = r_first["gpu_allocated_gb"]
    tr["gpu_reserved_after_first_chat_gb"]  = r_first["gpu_reserved_gb"]

    print(f"    → t_first_chat={tr['t_first_chat_s']:.3f}s  "
          f"t_total_effective={tr['t_total_effective_s']:.3f}s")

    return tr


def _measure_transition_partial(llm, model, tokenizer, config, stage_key,
                                 advance_fn_name, prefetch_fn_name,
                                 get_indices_fn_name,
                                 conversation, sampling_params,
                                 first_prompt):
    """
    Partial 모드 stage 전환 타이밍 측정 (Selective KV Block Injection (SKBI)).

    단계: sync → prefetch → activation → Selective KV Block Injection (SKBI) (~20ms)
    이후: 첫 채팅 (prefix cache 유지 → prefill 스킵 → 빠름)
    """
    tr = {}
    checkpoint_path = config[stage_key]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    prefetch_fn = getattr(model, prefetch_fn_name)
    advance_fn  = getattr(model, advance_fn_name)

    _ckpt_gb = ckpt_size_gb(checkpoint_path)
    tr["ckpt_size_gb"] = round(_ckpt_gb, 3)

    # 전환 전 스냅샷
    torch.cuda.reset_peak_memory_stats()
    tr["cpu_mem_before_gb"]        = round(cpu_mem_gb(), 3)
    tr["gpu_allocated_before_gb"]  = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_before_gb"]   = round(gpu_reserved_gb(), 3)

    # t_sync: GPU persistent buffer → _layer_output_cache (SKBI fallback용)
    torch.cuda.synchronize()
    t0 = time.time()
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "sync_persistent_cache"):
        seq_len = 0
        if (getattr(inner_model, "_skbi_seq_lens_tensor", None) is not None
                and inner_model._skbi_seq_lens_tensor.numel() > 0):
            seq_len = int(inner_model._skbi_seq_lens_tensor[0].item())
        if seq_len > 0:
            print(f"    → [Sync] Caching {seq_len} tokens for SKBI fallback...")
            inner_model.sync_persistent_cache(seq_len)
    torch.cuda.synchronize()
    tr["t_sync_s"] = round(time.time() - t0, 3)
    tr["gpu_allocated_after_sync_gb"] = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_after_sync_gb"]  = round(gpu_reserved_gb(), 3)

    # t_prefetch: checkpoint CPU 로드 + 대기 (백그라운드 샘플링)
    sampler = ResourceSampler(interval_s=0.05)
    sampler.start()
    torch.cuda.synchronize()
    t0 = time.time()
    prefetch_fn(checkpoint_path)
    model.wait_for_prefetch(timeout_s=120.0)
    torch.cuda.synchronize()
    tr["t_prefetch_s"] = round(time.time() - t0, 3)
    tr["prefetch_resources"] = sampler.stop()

    # prefetch 직후 스냅샷 (CPU에 ckpt 올라온 상태)
    tr["cpu_mem_after_prefetch_gb"] = round(cpu_mem_gb(), 3)

    # t_activation: GPU weight copy + boundary 설정
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    ok = advance_fn(wait_if_needed=False)
    torch.cuda.synchronize()
    tr["t_activation_s"] = round(time.time() - t0, 3)
    if not ok:
        raise RuntimeError(f"{advance_fn_name} returned False")

    # H2D 대역폭
    tr["h2d_bandwidth_gb_s"] = round(
        _ckpt_gb / tr["t_activation_s"], 2) if tr["t_activation_s"] > 0 else 0.0
    tr["gpu_allocated_after_activation_gb"] = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_after_activation_gb"]  = round(gpu_reserved_gb(), 3)
    tr["gpu_peak_allocated_activation_gb"]  = round(gpu_peak_allocated_gb(), 3)

    # t_skbi: Selective KV Block Injection (SKBI)
    torch.cuda.synchronize()
    t0 = time.time()
    skbi_ok = False
    if (hasattr(model, "get_recompute_boundary")
            and hasattr(model, get_indices_fn_name)
            and hasattr(model, "model")):
        indices = getattr(model, get_indices_fn_name)()
        boundary = model.get_recompute_boundary(indices)
        if boundary is not None:
            skbi_ok = model.model.apply_skbi(boundary=boundary)

    if not skbi_ok:
        # Fallback: prefix cache 초기화 후 full prefill
        print("    → [SKBI] fallback: reset_prefix_cache + full prefill")
        minimal_params = SamplingParams(temperature=0.0, max_tokens=1)
        if len(conversation) > 0:
            prompt_now = build_prompt(tokenizer, conversation)
            llm.reset_prefix_cache()
            llm.generate([prompt_now], minimal_params)

    torch.cuda.synchronize()
    tr["t_skbi_s"] = round(time.time() - t0, 3)
    tr["skbi_ok"] = skbi_ok
    tr["gpu_allocated_after_skbi_gb"] = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_after_skbi_gb"]  = round(gpu_reserved_gb(), 3)

    tr["t_total_transition_s"] = round(
        tr["t_sync_s"] + tr["t_prefetch_s"] + tr["t_activation_s"] + tr["t_skbi_s"], 3
    )
    tr["cpu_mem_after_transition_gb"]       = round(cpu_mem_gb(), 3)
    tr["gpu_allocated_after_transition_gb"] = round(gpu_mem_gb(), 3)
    tr["gpu_reserved_after_transition_gb"]  = round(gpu_reserved_gb(), 3)

    status = "✅ SKBI" if skbi_ok else "⚠️ fallback(full prefill)"
    print(f"    → t_sync={tr['t_sync_s']:.3f}s | "
          f"t_prefetch={tr['t_prefetch_s']:.3f}s | "
          f"t_activation={tr['t_activation_s']:.3f}s | "
          f"t_skbi={tr['t_skbi_s']:.3f}s ({status}) | "
          f"t_transition={tr['t_total_transition_s']:.3f}s")
    print(f"    → H2D bw={tr['h2d_bandwidth_gb_s']:.2f} GB/s | "
          f"CPU RAM peak={tr['prefetch_resources'].get('cpu_mem_peak_gb', '?'):.3f} GB | "
          f"CPU util={tr['prefetch_resources'].get('cpu_pct_mean', '?'):.1f}% avg")
    cache_status = "preserved (prefix hit expected)" if skbi_ok else "cleared (full prefill)"
    print(f"    → [First chat] prefix cache {cache_status}...")

    # 첫 채팅 (SKBI 성공 시 → new user tokens만 처리, 매우 빠름)
    r_first = do_chat(llm, tokenizer, conversation, first_prompt, sampling_params)
    tr["t_first_chat_s"]     = r_first["t_chat_s"]
    tr["first_chat_n_input"]  = r_first["n_input_tokens"]
    tr["first_chat_n_gen"]    = r_first["n_gen_tokens"]
    tr["t_total_effective_s"] = round(tr["t_total_transition_s"] + tr["t_first_chat_s"], 3)
    tr["gpu_allocated_after_first_chat_gb"] = r_first["gpu_allocated_gb"]
    tr["gpu_reserved_after_first_chat_gb"]  = r_first["gpu_reserved_gb"]

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
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    t_load = time.time() - t_load_start

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1)

    print(f"  ✅ Loaded in {t_load:.1f}s  "
          f"GPU alloc={gpu_mem_gb():.2f}GB  reserved={gpu_reserved_gb():.2f}GB  "
          f"CPU RAM={cpu_mem_gb():.2f}GB")

    results = {
        "mode": "origin",
        "model": model_name,
        "t_load_s": round(t_load, 2),
        "gpu_allocated_after_load_gb": round(gpu_mem_gb(), 3),
        "gpu_reserved_after_load_gb":  round(gpu_reserved_gb(), 3),
        "cpu_mem_after_load_gb":       round(cpu_mem_gb(), 3),
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

    Stage 전환: GPU buffer sync → prefetch → activation → partial recompute
    사용 모듈: progressive_serve/
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
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    t_load = time.time() - t_load_start

    model     = get_model_handle(llm)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1)

    # Warmup 중 쌓인 쓰레기 데이터 제거
    if hasattr(model, "model") and hasattr(model.model, "clear_persistent_buffers"):
        model.model.clear_persistent_buffers()
        print(f"  ✅ Persistent GPU buffers cleared (warmup residue removed)")

    print(f"  ✅ Loaded in {t_load:.1f}s  "
          f"GPU alloc={gpu_mem_gb():.2f}GB  reserved={gpu_reserved_gb():.2f}GB  "
          f"CPU RAM={cpu_mem_gb():.2f}GB")

    results = {
        "mode": "partial",
        "model": model_name,
        "t_load_s": round(t_load, 2),
        "gpu_allocated_after_load_gb": round(gpu_mem_gb(), 3),
        "gpu_reserved_after_load_gb":  round(gpu_reserved_gb(), 3),
        "cpu_mem_after_load_gb":       round(cpu_mem_gb(), 3),
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
        get_indices_fn_name="_get_b_indices",
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
        get_indices_fn_name="_get_c_indices",
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

    def fmt_row(label, va, vb, unit="s", lower_is_better=True):
        """비교 행 포맷. None이면 N/A."""
        if va is None and vb is None:
            return f"  {label:<45}  {'N/A':>8}  {'N/A':>8}  {'':>12}"
        va_s = f"{va:.3f}{unit}" if va is not None else "N/A"
        vb_s = f"{vb:.3f}{unit}" if vb is not None else "N/A"
        if va is not None and vb is not None and va > 0 and vb > 0:
            diff = vb - va
            winner = ("A" if va < vb else "B") if lower_is_better else ("A" if va > vb else "B")
            pct = abs(diff) / max(va, vb) * 100
            arrow = "▼" if diff < 0 else "▲"
            diff_s = f"{arrow}{abs(diff):.3f}{unit} ({pct:.0f}%) [{winner}↑]"
        else:
            diff_s = ""
        return f"  {label:<45}  {va_s:>8}  {vb_s:>8}  {diff_s}"

    def fmt_mem_row(label, va, vb):
        """메모리 비교 행 포맷 (GB 단위, 낮을수록 좋음)."""
        return fmt_row(label, va, vb, unit="GB", lower_is_better=True)

    W = 80
    print("\n" + "=" * W)
    print(f"  COMPARISON")
    print(f"  A = {label_a}")
    print(f"  B = {label_b}")
    print(f"  Model: {a['model']}")
    print("=" * W)
    print(f"  {'Metric':<45}  {'A':>8}  {'B':>8}  {'Delta (winner↑)':>20}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*8}  {'-'*20}")

    # ── 로드 시간 + 메모리 ──
    print(fmt_row("Model load time", a["t_load_s"], b["t_load_s"]))
    print(fmt_mem_row("  GPU allocated after load",
                      a.get("gpu_allocated_after_load_gb"), b.get("gpu_allocated_after_load_gb")))
    print(fmt_mem_row("  GPU reserved after load",
                      a.get("gpu_reserved_after_load_gb"), b.get("gpu_reserved_after_load_gb")))
    print(fmt_mem_row("  CPU RAM after load",
                      a.get("cpu_mem_after_load_gb"), b.get("cpu_mem_after_load_gb")))

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

        # sync (partial only)
        va = ta.get("t_sync_s", 0.0)
        vb = tb.get("t_sync_s", 0.0)
        print(fmt_row("  t_sync [GPU persistent buffer→cache, partial only]", va, vb))

        print(fmt_row("  t_prefetch [ckpt CPU load]",
                      ta.get("t_prefetch_s"), tb.get("t_prefetch_s")))
        print(fmt_row("  t_activation [GPU weight copy]",
                      ta.get("t_activation_s"), tb.get("t_activation_s")))

        # cache_clear (origin only)
        va_cc = ta.get("t_cache_clear_s", 0.0)
        vb_cc = tb.get("t_cache_clear_s", 0.0)
        print(fmt_row("  t_cache_clear [origin only]", va_cc, vb_cc))

        # SKBI (partial only) — 구 결과의 t_recompute_s도 호환
        va_rc = ta.get("t_skbi_s", ta.get("t_recompute_s", 0.0))
        vb_rc = tb.get("t_skbi_s", tb.get("t_recompute_s", 0.0))
        print(fmt_row("  t_skbi/recompute [partial only]", va_rc, vb_rc))

        print(fmt_row("  t_total_transition",
                      ta.get("t_total_transition_s"), tb.get("t_total_transition_s")))

        n_in_a = ta.get("first_chat_n_input", "?")
        n_in_b = tb.get("first_chat_n_input", "?")
        print(fmt_row(f"  t_first_chat [KEY] "
                      f"(A:{n_in_a}tok, B:{n_in_b}tok)",
                      ta.get("t_first_chat_s"), tb.get("t_first_chat_s")))

        print(fmt_row("  ★ t_total_effective [transition+first_chat]",
                      ta.get("t_total_effective_s"), tb.get("t_total_effective_s")))

        # ── 메모리 상세 ──
        print(f"\n  [Memory Detail]")
        print(fmt_mem_row("  ckpt_size",
                          ta.get("ckpt_size_gb"), tb.get("ckpt_size_gb")))
        print(fmt_mem_row("  CPU RAM before transition",
                          ta.get("cpu_mem_before_gb"), tb.get("cpu_mem_before_gb")))
        print(fmt_mem_row("  CPU RAM after prefetch (ckpt on CPU)",
                          ta.get("cpu_mem_after_prefetch_gb"), tb.get("cpu_mem_after_prefetch_gb")))
        print(fmt_mem_row("  CPU RAM after transition",
                          ta.get("cpu_mem_after_transition_gb"), tb.get("cpu_mem_after_transition_gb")))
        print(fmt_mem_row("  GPU alloc before transition",
                          ta.get("gpu_allocated_before_gb"), tb.get("gpu_allocated_before_gb")))
        print(fmt_mem_row("  GPU alloc after activation",
                          ta.get("gpu_allocated_after_activation_gb"),
                          tb.get("gpu_allocated_after_activation_gb")))
        print(fmt_mem_row("  GPU reserved after activation",
                          ta.get("gpu_reserved_after_activation_gb"),
                          tb.get("gpu_reserved_after_activation_gb")))
        print(fmt_mem_row("  GPU peak alloc during activation",
                          ta.get("gpu_peak_allocated_activation_gb"),
                          tb.get("gpu_peak_allocated_activation_gb")))
        # H2D 대역폭 (높을수록 좋음)
        va_bw = ta.get("h2d_bandwidth_gb_s")
        vb_bw = tb.get("h2d_bandwidth_gb_s")
        print(fmt_row("  H2D bandwidth [ckpt/t_activation]",
                      va_bw, vb_bw, unit="GB/s", lower_is_better=False))
        # prefetch 중 CPU 활용률
        pa = ta.get("prefetch_resources", {})
        pb = tb.get("prefetch_resources", {})
        print(fmt_mem_row("  CPU RAM peak during prefetch",
                          pa.get("cpu_mem_peak_gb"), pb.get("cpu_mem_peak_gb")))
        va_cpu = pa.get("cpu_pct_mean")
        vb_cpu = pb.get("cpu_pct_mean")
        if va_cpu is not None or vb_cpu is not None:
            va_s = f"{va_cpu:.1f}%" if va_cpu is not None else "N/A"
            vb_s = f"{vb_cpu:.1f}%" if vb_cpu is not None else "N/A"
            print(f"  {'  CPU utilization mean during prefetch':<45}  {va_s:>8}  {vb_s:>8}")

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

    # ── 메모리 요약 ──
    print("\n" + "=" * W)
    print("  ★ MEMORY SUMMARY (for paper table)")
    print(f"  {'Metric':<45}  {'A':>10}  {'B':>10}")
    print(f"  {'-'*45}  {'-'*10}  {'-'*10}")
    for mem_label, key_a, key_b in [
        ("GPU alloc after load",   "gpu_allocated_after_load_gb",  "gpu_allocated_after_load_gb"),
        ("GPU reserved after load","gpu_reserved_after_load_gb",   "gpu_reserved_after_load_gb"),
        ("CPU RAM after load",     "cpu_mem_after_load_gb",        "cpu_mem_after_load_gb"),
    ]:
        va = a.get(key_a)
        vb = b.get(key_b)
        va_s = f"{va:.3f} GB" if va is not None else "N/A"
        vb_s = f"{vb:.3f} GB" if vb is not None else "N/A"
        print(f"  {mem_label:<45}  {va_s:>10}  {vb_s:>10}")

    for tr_label, tr_key in [("Stage 1→2", "stage1_to_2"), ("Stage 2→3", "stage2_to_3")]:
        ta = a.get(tr_key, {})
        tb = b.get(tr_key, {})
        pa = ta.get("prefetch_resources", {})
        pb = tb.get("prefetch_resources", {})
        rows = [
            ("CPU RAM peak (prefetch)",     pa.get("cpu_mem_peak_gb"),               pb.get("cpu_mem_peak_gb")),
            ("CPU RAM after transition",    ta.get("cpu_mem_after_transition_gb"),   tb.get("cpu_mem_after_transition_gb")),
            ("GPU alloc after activation",  ta.get("gpu_allocated_after_activation_gb"), tb.get("gpu_allocated_after_activation_gb")),
            ("GPU reserved after activ.",   ta.get("gpu_reserved_after_activation_gb"),  tb.get("gpu_reserved_after_activation_gb")),
            ("H2D bandwidth (GB/s)",        ta.get("h2d_bandwidth_gb_s"),            tb.get("h2d_bandwidth_gb_s")),
        ]
        print(f"\n  [{tr_label} Memory]")
        for ml, va, vb in rows:
            va_s = f"{va:.3f}" if va is not None else "N/A"
            vb_s = f"{vb:.3f}" if vb is not None else "N/A"
            print(f"  {ml:<45}  {va_s:>10}  {vb_s:>10}")

    print("\n  NOTE:")
    print("  - t_first_chat in Origin = FULL PREFILL (all tokens recomputed, slow)")
    print("  - t_first_chat in Partial = only new user tokens processed (fast, prefix cache hit)")
    print("  - t_skbi in Partial = Selective KV Block Injection (SKBI): upper layers only (~20ms)")
    print("    (lower layers KV untouched; prefix cache preserved → next generate skips prefill)")
    print("  - CPU RAM peak during prefetch = staging buffer size (pinned, partial only)")
    print("  - H2D bandwidth = ckpt_size_gb / t_activation_s")
    print("=" * W)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark chatbot_origin vs chatbot_partial_cache (일반 프롬프트 버전)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run origin mode
  python benchmark_chatbots_normal.py --mode origin --model llama --output results_origin.json

  # Run partial mode (separate process, after origin finishes)
  python benchmark_chatbots_normal.py --mode partial --model llama --output results_partial.json

  # Compare results
  python benchmark_chatbots_normal.py --compare results_origin.json results_partial.json
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
        print(f"  Chatbot Benchmark (Normal Prompt Length)")
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
