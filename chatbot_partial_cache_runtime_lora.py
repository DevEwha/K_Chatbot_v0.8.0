#!/usr/bin/env python3
"""
./run_batch.sh llama2-7b both 0 chatbot_partial_cache_runtime_lora.py \
  --repeat 3 \
  --max-stage 3
  
Progressive Serving Chatbot Test (KV Block Surgery + Measurement)
=================================================================
06_test_prefetch.py 기반 + KV Block Surgery 방식 적용 버전 (progressive_serve5)

[두 방식 통합]
  그래프 보존 (progressive_serve5/)
    - Dual-path: Path A/B 항상 계산, alpha로 선택 → CUDA graph topology 불변
    - _layer_output_cache: GPU hidden states 보존 (.clone(), GPU→GPU)
    - boundary 이전 레이어: KV-only forward (hidden state 재사용)

  KV Block Surgery (새 방식)
    - inject_upper_layer_kv(boundary): upper-layer KV blocks를 in-place 덮어씀 (~20ms)
    - 실패 시 fallback: set_partial_recompute + reset_prefix_cache + generate(max_tokens=1)
    - 대화 없는 경우: reset_prefix_cache() 만 수행

[통합 흐름]
  _sync_cache_before_transition()    →  sync_persistent_cache()  (GPU→GPU clone)
  advance_to_stageN_instant()        →  alpha 0→1 in-place
  _do_surgery_or_fallback(boundary)  →  inject_upper_layer_kv()  (KV block surgery)
                                         └ 실패 시: reset_prefix_cache() + generate(max_tokens=1)

- 측정 요소는 03_test_prefetch.py에서 그대로 가져온다.
  * Background loading(prefetch + overlap serving)
  * TTFT(request-only + stage e2e from prefetch start)
  * Throughput(tok/s)
  * NVTX range 표기 (nsys용)
- 네트워크 측정은 제외한다.

Usage:
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
  # interactive chat
  python chatbot_partial_cache_runtime_lora.py --mode chat --model llama2-7b

  # progressive benchmark (stage1->2->3)
  python chatbot_partial_cache_runtime_lora.py --mode progressive --model llama2-7b --max-stage 3

  # baseline only
  python chatbot_partial_cache_runtime_lora.py --mode baseline --model llama2-7b

  # baseline vs progressive + comparison table
  python chatbot_partial_cache_runtime_lora.py --mode both --model llama2-7b --save-path runtime_lora_compare.json

nsys profiling:
  nsys profile -t cuda,nvtx -o runtime_lora_transition_report \
    python chatbot_partial_cache_runtime_lora.py --mode both --model llama2-7b --save-path runtime_lora_compare.json

Commands:
  /stage2   - Stage 2 백그라운드 준비 시작 (준비 완료 즉시 자동 전환 + 측정)
  /stage3   - Stage 3 백그라운드 준비 시작 (준비 완료 즉시 자동 전환 + 측정)
  /bench    - 현재 stage TTFT/Throughput 측정
  /status   - 현재 상태 출력
  /metrics  - 누적 측정 결과(JSON) 출력
  /save     - 누적 측정 결과 파일 저장
  /reset    - 대화/캐시 초기화
  /quit     - 종료
"""

import os
import sys
import json
import time
import argparse
import gc
import select
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

# vLLM v0 엔진 강제 사용 (모델 직접 접근 필요)
os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.model_executor.models.registry import ModelRegistry

# vLLM이 ProgressiveForCausalLM을 multimodal 모델로 오인하는 버그 방지
# → prefix caching이 정상 작동하게 됨
import vllm.config
vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)

from shared_model_configs import (
    CANONICAL_MODEL_CHOICES,
    CHATBOT_MODELS as MODELS,
    DEFAULT_MODEL,
)

# Progressive model (serve3: .clone() 기반 GPU→GPU 캐시 보존)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
VENDORED_PS5_ROOT = os.path.join(SCRIPT_DIR, "progressive_serve5")
PS5_ROOT = os.path.join(PROJECT_ROOT, "progressive_serve5")
if PS5_ROOT not in sys.path:
    sys.path.insert(0, PS5_ROOT)

# If a stale vendored progressive_serve5 copy was imported earlier in-process,
# clear those modules so benchmark_transition always picks up the root copy.
for module_name in (
    "progressive_for_causal_lm",
    "progressive_model_dual_path",
    "model_config",
    "universal_bypass_layer",
):
    cached = sys.modules.get(module_name)
    if cached is None:
        continue
    cached_file = str(getattr(cached, "__file__", ""))
    if cached_file.startswith(VENDORED_PS5_ROOT):
        del sys.modules[module_name]

from progressive_for_causal_lm import ProgressiveForCausalLM

PromptInput = Union[str, Dict[str, List[int]]]


def nvtx_push(name: str):
    """nsys NVTX range 시작"""
    torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    """nsys NVTX range 종료"""
    torch.cuda.nvtx.range_pop()


TTFT_PROMPT = "What is the capital of France?"
THROUGHPUT_PROMPT = "Explain quantum computing in simple terms."
WARMUP_PROMPT = "shape_ttft"

FIXED_MAX_TOKENS_DEFAULT = 50
THROUGHPUT_REQUESTS_DEFAULT = 3
AUTO_PROMPT_DEFAULT = "The Future of Ai is"


def gpu_memory_snapshot() -> Dict[str, float]:
    return {
        "allocated_gb": torch.cuda.memory_allocated() / (1024 ** 3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024 ** 3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


def reset_gpu_memory_stats():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def _drop_page_cache_non_interactive():
    """Drop OS page cache without interactive prompt (requires sudo -v beforehand)."""
    if os.geteuid() == 0:
        cmd = ["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
    else:
        cmd = ["sudo", "-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"]
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode
    if rc != 0:
        raise RuntimeError(
            "drop_caches failed for strict-cold pair. "
            "Run 'sudo -v' first, or disable --strict-cold-paired."
        )


def prepare_strict_cold_arm(arm_name: str):
    """Prepare one arm in strict-cold paired measurement."""
    print(f"\n  [StrictCold] Preparing {arm_name} arm: drop_caches + GPU clear")
    reset_gpu_memory_stats()
    _drop_page_cache_non_interactive()
    reset_gpu_memory_stats()
    time.sleep(0.2)


def summarize_latency_samples(samples: List[float], name: str = "latency") -> Dict[str, float]:
    if not samples:
        return {
            f"{name}_count": 0,
            f"{name}_mean_s": 0.0,
            f"{name}_p50_s": 0.0,
            f"{name}_p95_s": 0.0,
            f"{name}_p99_s": 0.0,
            f"{name}_max_s": 0.0,
        }

    vals = sorted(float(x) for x in samples)
    n = len(vals)

    def pct(p: float) -> float:
        idx = int(round((n - 1) * p))
        idx = max(0, min(n - 1, idx))
        return vals[idx]

    return {
        f"{name}_count": n,
        f"{name}_mean_s": sum(vals) / n,
        f"{name}_p50_s": pct(0.50),
        f"{name}_p95_s": pct(0.95),
        f"{name}_p99_s": pct(0.99),
        f"{name}_max_s": vals[-1],
    }


def compare_token_sequences(reference: List[int], candidate: List[int]) -> Dict[str, Any]:
    common = 0
    for a, b in zip(reference, candidate):
        if a != b:
            break
        common += 1
    denom = max(1, len(reference))
    return {
        "reference_tokens": len(reference),
        "candidate_tokens": len(candidate),
        "common_prefix_tokens": common,
        "prefix_match_ratio": common / denom,
        "exact_match": reference == candidate,
    }


class ProgressiveChatbotMeasured:
    """
    Progressive Serving 챗봇 + 측정 도구 (progressive_serve5 / KV Block Surgery).

    [그래프 보존 + KV Block Surgery 통합]
    그래프 보존:
    - Dual-path (Path A/B 항상 계산) → CUDA graph topology 불변
    - _layer_output_cache: GPU hidden states 보존 (GPU→GPU .clone())
    - boundary 이전 레이어: KV-only forward,

    KV Block Surgery:
    - inject_upper_layer_kv(boundary): upper-layer KV blocks 직접 in-place 덮어씀
    - 실패 시 fallback: set_partial_recompute + reset_prefix_cache + generate(max_tokens=1)

    측정 기능:
    - prefetch + overlap serving(Background loading)
    - TTFT
    - Throughput
    - NVTX range 표기
    """

    def __init__(
        self,
        model_name: str,
        fixed_max_tokens: int = FIXED_MAX_TOKENS_DEFAULT,
        overlap_rounds: int = 4,
        throughput_requests: int = THROUGHPUT_REQUESTS_DEFAULT,
        transition_mode: str = "live",
        reconcile_mode: str = "auto",
        forward_variant: str = "dualpath_inplace",
        instant_streams: int = 4,
        alpha_update_mode: str = "inplace",
        transition_window_samples: int = 64,
        consistency_probe: bool = False,
        consistency_probe_max_tokens: int = 16,
        consistency_probe_prompt: str = "Summarize cache consistency in one short sentence.",
        enable_runtime_lora: bool = False,
        runtime_lora_path: Optional[str] = None,
        runtime_lora_name: str = "stage12_runtime_lora",
        runtime_lora_stage2_path: Optional[str] = None,
        runtime_lora_stage2_name: Optional[str] = None,
        runtime_lora_int_id: int = 1,
        runtime_lora_max_rank: int = 64,
        runtime_lora_max_loras: int = 1,
        runtime_lora_stage3_policy: str = "off",
        runtime_lora_strict: bool = False,
        enforce_eager: bool = False,
    ):
        self.model_name = model_name
        self.config = MODELS[model_name]
        self.current_stage = 1
        self.conversation: List[Dict[str, str]] = []
        self.fixed_max_tokens = fixed_max_tokens
        self.transition_mode = transition_mode
        self.overlap_rounds = overlap_rounds if transition_mode == "live" else 0
        self.reconcile_mode = str(reconcile_mode).strip().lower()
        if self.reconcile_mode == "full_reset":
            self.reconcile_mode = "full_prefill"
        if self.reconcile_mode not in ("auto", "surgery", "fallback", "full_prefill", "none"):
            print(f"  [Warn] Unknown reconcile_mode='{reconcile_mode}', fallback to 'auto'")
            self.reconcile_mode = "auto"
        self.forward_variant = forward_variant
        self.instant_streams = max(1, int(instant_streams))
        self.alpha_update_mode = alpha_update_mode
        self.transition_window_samples = max(4, int(transition_window_samples))
        self.consistency_probe_enabled = bool(consistency_probe)
        self.consistency_probe_max_tokens = max(1, int(consistency_probe_max_tokens))
        self.consistency_probe_prompt = str(consistency_probe_prompt)
        self.enforce_eager = bool(enforce_eager)
        self.throughput_requests = max(1, throughput_requests)
        self.pending_transition: Optional[Dict[str, Any]] = None
        self.early_prefetch: Optional[Dict[str, Any]] = None
        self._auto_prefetch_after: Optional[int] = None  # instant transition 직후 자동 prefetch할 다음 stage
        self._request_latency_history_s: List[float] = []
        self._request_latency_cap = 4096
        self.runtime_lora_enabled = bool(enable_runtime_lora)
        runtime_lora_path = (
            str(runtime_lora_path).strip() or None
            if runtime_lora_path is not None
            else None
        )
        self.runtime_lora_path = runtime_lora_path or self.config.get("runtime_lora_default_path")
        self.runtime_lora_name = str(runtime_lora_name).strip() or "stage12_runtime_lora"
        stage2_target_path = self.config.get("runtime_lora_stage2_default_path")
        if stage2_target_path is not None:
            stage2_target_path = str(stage2_target_path).strip() or None
        runtime_lora_stage2_path = (
            str(runtime_lora_stage2_path).strip() or None
            if runtime_lora_stage2_path is not None
            else None
        )
        if runtime_lora_stage2_path:
            stage2_target_path = runtime_lora_stage2_path
        stage2_target_name = None
        if runtime_lora_stage2_name is not None:
            stage2_target_name = str(runtime_lora_stage2_name).strip() or None
        self.runtime_lora_stage2_path = stage2_target_path
        self.runtime_lora_stage2_name = stage2_target_name or (
            self.runtime_lora_name if stage2_target_path else None
        )
        self.runtime_lora_int_id = int(runtime_lora_int_id)
        self.runtime_lora_max_rank = max(1, int(runtime_lora_max_rank))
        self.runtime_lora_max_loras = max(1, int(runtime_lora_max_loras))
        self.runtime_lora_stage3_policy = str(runtime_lora_stage3_policy).strip().lower()
        if self.runtime_lora_stage3_policy not in ("off", "remove", "keep"):
            print(
                f"  [Warn] Unknown runtime_lora_stage3_policy='{runtime_lora_stage3_policy}', "
                "fallback to 'off'"
            )
            self.runtime_lora_stage3_policy = "off"
        self.runtime_lora_strict = bool(runtime_lora_strict)
        self._runtime_lora_request: Optional[LoRARequest] = None
        self._runtime_lora_loaded = False
        self._runtime_lora_active_for_generation = False
        self._runtime_lora_stage2_swapped = False
        self._runtime_lora_initial_path = self.runtime_lora_path
        self._runtime_lora_initial_name = self.runtime_lora_name

        # Progressive model behavior control (used inside progressive_serve5 code).
        os.environ["P2_FORWARD_VARIANT"] = self.forward_variant
        os.environ["P2_INSTANT_STREAMS"] = str(self.instant_streams)
        os.environ["P2_ALPHA_UPDATE_MODE"] = self.alpha_update_mode

        self.metrics: Dict[str, Any] = {
            "script": "chatbot_partial_cache_runtime_lora.py",
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "fixed_max_tokens": fixed_max_tokens,
            "throughput_requests": self.throughput_requests,
            "transition_mode": transition_mode,
            "reconcile_mode": self.reconcile_mode,
            "forward_variant": forward_variant,
            "instant_streams": self.instant_streams,
            "alpha_update_mode": alpha_update_mode,
            "enforce_eager": self.enforce_eager,
            "transition_window_samples": self.transition_window_samples,
            "consistency_probe_enabled": self.consistency_probe_enabled,
            "consistency_probe_max_tokens": self.consistency_probe_max_tokens,
            "ttft_definition": {
                "stage1_ttft_e2e": "cold_start_time + cold_first_request_to_first_token",
                "stage1_ttft_steady_request_only": "post_warmup_request_to_first_token",
            },
            "t2ft_definition": (
                "prefetch_wait_from_request + transition_time "
                "(instant_transition_time + runtime_lora_stage_policy_time) "
                "+ cache_sync_time + cache_reconciliation_time + request_to_first_token"
            ),
            "promotion_window_definition": {
                "before_window": f"last {self.transition_window_samples} request latencies before promotion",
                "during_window": "overlap-serving latencies during prefetch plus transition first-token request",
                "after_window": "post-promotion single-token latency samples",
            },
            "graph_event_counters": {
                "cuda_graph_exception_count": 0,
                "cuda_graph_recapture_hint_count": 0,
                "instant_transition_fail_count": 0,
            },
            "runtime_lora": {
                "enabled": self.runtime_lora_enabled,
                "adapter_path": self.runtime_lora_path,
                "adapter_name": self.runtime_lora_name,
                "initial_adapter_path": self._runtime_lora_initial_path,
                "initial_adapter_name": self._runtime_lora_initial_name,
                "stage2_target_path": self.runtime_lora_stage2_path,
                "stage2_target_name": self.runtime_lora_stage2_name,
                "stage2_swapped": self._runtime_lora_stage2_swapped,
                "adapter_int_id": self.runtime_lora_int_id,
                "max_lora_rank": self.runtime_lora_max_rank,
                "max_loras": self.runtime_lora_max_loras,
                "stage3_policy": self.runtime_lora_stage3_policy,
                "strict_mode": self.runtime_lora_strict,
                "loaded": False,
                "active_for_generation": False,
                "events": [],
            },
            "stages": {},
            "stage_transition_times": {},
        }
        self._prefix_caching_enabled = bool(
            self.config.get("enable_prefix_caching", True)
        )
        self.max_model_len = int(self.config.get("max_model_len", 2048))
        self.tensor_parallel_size = max(1, int(self.config.get("tensor_parallel_size", 1)))

        model_path = self.config["progressive_path"]
        with open(os.path.join(model_path, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        ModelRegistry.register_model(arch, ProgressiveForCausalLM)
        print(f"  Registered ProgressiveForCausalLM as: {arch}")

        print(f"\n  Loading {model_name} Stage 1...")
        nvtx_push("chatbot_stage1_cold_start")
        cold_t = time.time()
        llm_kwargs: Dict[str, Any] = {
            "model": model_path,
            "trust_remote_code": self.config.get("trust_remote_code", True),
            "gpu_memory_utilization": float(self.config.get("gpu_memory_utilization", 0.4)),
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager,
            "enable_prefix_caching": self._prefix_caching_enabled,
            "disable_sliding_window": self.config.get("disable_sliding_window", False),
        }
        if self.runtime_lora_enabled:
            llm_kwargs.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": self.runtime_lora_max_rank,
                    "max_loras": self.runtime_lora_max_loras,
                    "max_cpu_loras": self.runtime_lora_max_loras,
                }
            )
        self.llm = LLM(**llm_kwargs)
        self.stage1_cold_start = time.time() - cold_t
        nvtx_pop()

        self.model = self._get_model_handle()
        self._init_runtime_lora()

        if hasattr(self.model, "model") and hasattr(self.model.model, "clear_persistent_buffers"):
            self.model.model.clear_persistent_buffers()
            print("  ✅ Persistent GPU buffers cleared (warmup data removed)")

        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=self.fixed_max_tokens,
        )
        self.measure_ttft_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.fixed_max_tokens,
        )
        self.measure_tp_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=self.fixed_max_tokens,
        )
        self.single_token_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
        )

        stage_info = self.model.get_stage_info()
        gating_profile = self._estimate_gating_overhead(stage_info)
        self.metrics["stages"]["stage1"] = {
            "cold_start_time": self.stage1_cold_start,
            "active_layers": len(stage_info["active_layers"]),
            "inactive_layers": len(stage_info["inactive_layers"]),
            "activation_progress": stage_info["activation_progress"],
            "gpu_memory": gpu_memory_snapshot(),
            "runtime_lora": dict(self.metrics.get("runtime_lora", {})),
            **gating_profile,
        }

        print(f"  Stage 1 Cold Start: {self.stage1_cold_start:.2f}s")
        print("  ✅ Partial KV Cache Recomputation enabled (Graph + Cache)")
        print(
            f"  ✅ Prefix caching: "
            f"{'enabled' if self._prefix_caching_enabled else 'disabled'}"
        )
        print(
            f"  ✅ Variant: {self.forward_variant}, "
            f"alpha_update={self.alpha_update_mode}, "
            f"instant_streams={self.instant_streams}, "
            f"tp={self.tensor_parallel_size}, "
            f"reconcile_mode={self.reconcile_mode}, "
            f"enforce_eager={self.enforce_eager}"
        )
        runtime_lora_state = "disabled"
        if self.runtime_lora_enabled:
            runtime_lora_state = "on" if self._runtime_lora_active_for_generation else "off"
            runtime_lora_state += ", loaded" if self._runtime_lora_loaded else ", not_loaded"
        print(
            "  ✅ Runtime LoRA: "
            f"{runtime_lora_state} "
            f"(stage3_policy={self.runtime_lora_stage3_policy})"
        )

    def _get_model_handle(self):
        engine = self.llm.llm_engine
        if hasattr(engine, "engine_core"):
            raise RuntimeError(
                "V1 engine detected. This script is v0-only. Use VLLM_USE_V1=0."
            )
        try:
            return engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError as exc:
            raise RuntimeError("Could not resolve v0 model handle path.") from exc

    def _should_use_tokenized_chat_template(self) -> bool:
        # Gemma chat templates already emit a BOS token. Passing the rendered
        # string back through vLLM's tokenizer inserts a second BOS and can
        # cause the model to terminate with an empty decoded response.
        return self.model_name.startswith("gemma")

    def _normalize_prompt_token_ids(self, tokenized_prompt: Any) -> List[int]:
        if hasattr(tokenized_prompt, "get"):
            tokenized_prompt = tokenized_prompt.get("input_ids", tokenized_prompt)
        if hasattr(tokenized_prompt, "tolist"):
            tokenized_prompt = tokenized_prompt.tolist()
        if isinstance(tokenized_prompt, list) and tokenized_prompt and isinstance(
            tokenized_prompt[0], list
        ):
            tokenized_prompt = tokenized_prompt[0]
        if tokenized_prompt is None:
            raise ValueError("Chat template tokenization did not return input_ids.")
        return [int(token_id) for token_id in tokenized_prompt]

    def _build_chat_template_prompt(
        self, messages: List[Dict[str, str]]
    ) -> Optional[PromptInput]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                if self._should_use_tokenized_chat_template():
                    tokenized = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    return {
                        "prompt_token_ids": self._normalize_prompt_token_ids(tokenized)
                    }
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return None

    def _prompt_token_ids(self, prompt: PromptInput) -> List[int]:
        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            return list(prompt["prompt_token_ids"])
        return self.tokenizer.encode(prompt)

    def _build_prompt(self) -> PromptInput:
        templated_prompt = self._build_chat_template_prompt(self.conversation)
        if templated_prompt is not None:
            return templated_prompt
        prompt = ""
        for msg in self.conversation:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt

    def _build_prompt_with_user_turn(self, user_input: str) -> PromptInput:
        probe_messages = list(self.conversation)
        probe_messages.append({"role": "user", "content": user_input})
        templated_prompt = self._build_chat_template_prompt(probe_messages)
        if templated_prompt is not None:
            return templated_prompt

        prompt = ""
        for msg in probe_messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt

    def _estimate_gating_overhead(self, stage_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lightweight estimate of gating-induced extra compute.
        - skip_inactive_layers: active layers only (reference path)
        - others: conservatively treat active+inactive as executed layer-equivalent
        """
        active = len(stage_info.get("active_layers", []))
        inactive = len(stage_info.get("inactive_layers", []))
        total = active + inactive
        inactive_ratio = (inactive / total) if total > 0 else 0.0

        if self.forward_variant == "skip_inactive_layers":
            executed_layer_equiv = active
        else:
            executed_layer_equiv = total

        extra_layer_equiv = max(0, executed_layer_equiv - active)
        denom = float(max(1, active))
        extra_ratio = float(extra_layer_equiv) / denom
        return {
            "inactive_layer_ratio": inactive_ratio,
            "executed_layer_equiv_est": float(executed_layer_equiv),
            "extra_layer_equiv_est": float(extra_layer_equiv),
            "gating_extra_flops_ratio_est": extra_ratio,
            "gating_extra_flops_pct_est": extra_ratio * 100.0,
            "gating_flops_reference": "skip_inactive_layers",
            "gating_flops_estimation_note": "layer-equivalent proxy (not kernel-level FLOPs)",
        }

    def _record_request_latency(self, latency_s: float) -> None:
        self._request_latency_history_s.append(float(latency_s))
        if len(self._request_latency_history_s) > self._request_latency_cap:
            self._request_latency_history_s = self._request_latency_history_s[-self._request_latency_cap:]

    def _record_runtime_lora_event(
        self,
        action: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        lora_metrics = self.metrics.setdefault("runtime_lora", {})
        events = lora_metrics.setdefault("events", [])
        item: Dict[str, Any] = {
            "at": datetime.now().isoformat(),
            "stage": int(self.current_stage),
            "action": str(action),
        }
        if detail:
            item.update(detail)
        events.append(item)
        if len(events) > 128:
            lora_metrics["events"] = events[-128:]
        lora_metrics["last_action"] = str(action)

    def _sync_runtime_lora_metrics(self) -> None:
        lora_metrics = self.metrics.setdefault("runtime_lora", {})
        lora_metrics["enabled"] = bool(self.runtime_lora_enabled)
        lora_metrics["loaded"] = bool(self._runtime_lora_loaded)
        lora_metrics["active_for_generation"] = bool(self._runtime_lora_active_for_generation)
        lora_metrics["adapter_path"] = self.runtime_lora_path
        lora_metrics["adapter_name"] = self.runtime_lora_name
        lora_metrics["initial_adapter_path"] = self._runtime_lora_initial_path
        lora_metrics["initial_adapter_name"] = self._runtime_lora_initial_name
        lora_metrics["stage2_target_path"] = self.runtime_lora_stage2_path
        lora_metrics["stage2_target_name"] = self.runtime_lora_stage2_name
        lora_metrics["stage2_swapped"] = bool(self._runtime_lora_stage2_swapped)
        lora_metrics["adapter_int_id"] = self.runtime_lora_int_id
        lora_metrics["stage3_policy"] = self.runtime_lora_stage3_policy

    def _current_runtime_lora_request(self) -> Optional[LoRARequest]:
        if not self.runtime_lora_enabled:
            return None
        if not self._runtime_lora_loaded:
            return None
        if not self._runtime_lora_active_for_generation:
            return None
        return self._runtime_lora_request

    def _remove_runtime_lora(self, reason: str) -> bool:
        if not self._runtime_lora_loaded:
            return True
        removed = False
        try:
            removed = bool(self.llm.llm_engine.remove_lora(self.runtime_lora_int_id))
        except Exception as exc:
            self._record_runtime_lora_event(
                "remove_exception",
                {"reason": reason, "error": str(exc)},
            )
            self._sync_runtime_lora_metrics()
            if self.runtime_lora_strict:
                raise
            print(f"  [Warn] Runtime LoRA remove exception: {exc}")
            return False
        if removed:
            self._runtime_lora_loaded = False
            self._runtime_lora_request = None
            self._runtime_lora_active_for_generation = False
            self._record_runtime_lora_event("remove", {"reason": reason})
        else:
            self._record_runtime_lora_event("remove_failed", {"reason": reason})
            if self.runtime_lora_strict:
                raise RuntimeError(
                    f"Runtime LoRA remove failed (id={self.runtime_lora_int_id})"
                )
        self._sync_runtime_lora_metrics()
        return removed

    def _add_runtime_lora(
        self,
        lora_path: str,
        lora_name: str,
        *,
        success_action: str = "add",
        failure_action: str = "add_failed",
        skipped_action: str = "add_skipped",
        disable_on_failure: bool = False,
    ) -> bool:
        self._sync_runtime_lora_metrics()
        if not self.runtime_lora_enabled:
            return False

        if not lora_path:
            msg = "Runtime LoRA path is empty."
            self._record_runtime_lora_event(skipped_action, {"reason": msg})
            if self.runtime_lora_strict:
                raise RuntimeError(msg)
            print(f"  [Warn] {msg}")
            if disable_on_failure:
                self.runtime_lora_enabled = False
                self._sync_runtime_lora_metrics()
            return False

        if not os.path.exists(lora_path):
            msg = f"Runtime LoRA path not found: {lora_path}"
            self._record_runtime_lora_event(skipped_action, {"reason": msg})
            if self.runtime_lora_strict:
                raise RuntimeError(msg)
            print(f"  [Warn] {msg}")
            if disable_on_failure:
                self.runtime_lora_enabled = False
                self._sync_runtime_lora_metrics()
            return False

        try:
            req = LoRARequest(
                lora_name=lora_name,
                lora_int_id=self.runtime_lora_int_id,
                lora_path=lora_path,
            )
            added = bool(self.llm.llm_engine.add_lora(req))
            if not added:
                raise RuntimeError(
                    "LLMEngine.add_lora returned False "
                    f"(id={self.runtime_lora_int_id}, path={lora_path})"
                )
            self.runtime_lora_path = lora_path
            self.runtime_lora_name = lora_name
            self._runtime_lora_request = req
            self._runtime_lora_loaded = True
            self._runtime_lora_active_for_generation = True
            self._record_runtime_lora_event(
                success_action,
                {
                    "adapter_path": lora_path,
                    "adapter_name": lora_name,
                    "adapter_int_id": self.runtime_lora_int_id,
                },
            )
            self._sync_runtime_lora_metrics()
            print(
                "  ✅ Runtime LoRA active: "
                f"name={lora_name}, id={self.runtime_lora_int_id}, path={lora_path}"
            )
            return True
        except Exception as exc:
            self._record_runtime_lora_event(failure_action, {"error": str(exc)})
            self._runtime_lora_request = None
            self._runtime_lora_loaded = False
            self._runtime_lora_active_for_generation = False
            self._sync_runtime_lora_metrics()
            if self.runtime_lora_strict:
                raise
            print(f"  [Warn] Runtime LoRA add failed: {exc}")
            if disable_on_failure:
                self.runtime_lora_enabled = False
                self._sync_runtime_lora_metrics()
            return False

    def _swap_runtime_lora_for_stage2(self, trigger: str) -> bool:
        if not self.runtime_lora_enabled:
            self._sync_runtime_lora_metrics()
            return False
        if self._runtime_lora_stage2_swapped:
            self._sync_runtime_lora_metrics()
            return True
        if self.current_stage < 2:
            self._sync_runtime_lora_metrics()
            return False
        if not self.runtime_lora_stage2_path:
            self._sync_runtime_lora_metrics()
            return False

        target_path = self.runtime_lora_stage2_path
        target_name = self.runtime_lora_stage2_name or self.runtime_lora_name
        previous_path = self.runtime_lora_path
        previous_name = self.runtime_lora_name

        if (
            self._runtime_lora_loaded
            and previous_path == target_path
            and previous_name == target_name
        ):
            self._runtime_lora_stage2_swapped = True
            self._record_runtime_lora_event(
                "swap_stage2_skipped_same_adapter",
                {
                    "trigger": trigger,
                    "adapter_path": previous_path,
                    "adapter_name": previous_name,
                },
            )
            self._sync_runtime_lora_metrics()
            return True

        had_loaded = bool(self._runtime_lora_loaded)
        if had_loaded and not self._remove_runtime_lora(reason=f"stage2_swap_{trigger}"):
            self._record_runtime_lora_event(
                "swap_stage2_failed_remove",
                {
                    "trigger": trigger,
                    "from_path": previous_path,
                    "from_name": previous_name,
                    "to_path": target_path,
                    "to_name": target_name,
                },
            )
            self._sync_runtime_lora_metrics()
            return False

        if self._add_runtime_lora(
            target_path,
            target_name,
            success_action="swap_stage2_add",
            failure_action="swap_stage2_add_failed",
            skipped_action="swap_stage2_add_skipped",
            disable_on_failure=False,
        ):
            self._runtime_lora_stage2_swapped = True
            self._record_runtime_lora_event(
                "swap_stage2_complete",
                {
                    "trigger": trigger,
                    "from_path": previous_path,
                    "from_name": previous_name,
                    "to_path": target_path,
                    "to_name": target_name,
                },
            )
            self._sync_runtime_lora_metrics()
            return True

        rollback_ok = False
        if had_loaded and previous_path:
            rollback_ok = self._add_runtime_lora(
                previous_path,
                previous_name,
                success_action="swap_stage2_rollback_add",
                failure_action="swap_stage2_rollback_failed",
                skipped_action="swap_stage2_rollback_skipped",
                disable_on_failure=False,
            )
        if rollback_ok:
            self._record_runtime_lora_event(
                "swap_stage2_rollback",
                {
                    "trigger": trigger,
                    "restored_path": previous_path,
                    "restored_name": previous_name,
                },
            )
        else:
            self._record_runtime_lora_event(
                "swap_stage2_no_rollback",
                {
                    "trigger": trigger,
                    "from_path": previous_path,
                    "from_name": previous_name,
                    "to_path": target_path,
                    "to_name": target_name,
                },
            )
        self._sync_runtime_lora_metrics()
        return False

    def _apply_runtime_lora_stage_policy(self, trigger: str) -> None:
        if not self.runtime_lora_enabled:
            self._sync_runtime_lora_metrics()
            return
        if self.current_stage == 2:
            self._swap_runtime_lora_for_stage2(trigger=trigger)
        if not self._runtime_lora_loaded:
            self._runtime_lora_active_for_generation = False
            self._sync_runtime_lora_metrics()
            return

        if self.current_stage < 3:
            self._runtime_lora_active_for_generation = True
            self._record_runtime_lora_event(
                "policy_stage12_on",
                {"trigger": trigger, "active_for_generation": True},
            )
            self._sync_runtime_lora_metrics()
            return

        if self.runtime_lora_stage3_policy == "keep":
            self._runtime_lora_active_for_generation = True
            self._record_runtime_lora_event(
                "policy_stage3_keep",
                {"trigger": trigger, "active_for_generation": True},
            )
        elif self.runtime_lora_stage3_policy == "off":
            self._runtime_lora_active_for_generation = False
            self._record_runtime_lora_event(
                "policy_stage3_off",
                {"trigger": trigger, "active_for_generation": False},
            )
        else:
            self._runtime_lora_active_for_generation = False
            self._record_runtime_lora_event(
                "policy_stage3_remove",
                {"trigger": trigger},
            )
            self._remove_runtime_lora(reason=f"stage3_{trigger}")
        self._sync_runtime_lora_metrics()

    def _init_runtime_lora(self) -> None:
        self._sync_runtime_lora_metrics()
        if not self.runtime_lora_enabled:
            return
        if self._add_runtime_lora(
            self.runtime_lora_path,
            self.runtime_lora_name,
            success_action="add",
            failure_action="add_failed",
            skipped_action="add_skipped",
            disable_on_failure=True,
        ):
            self._apply_runtime_lora_stage_policy(trigger="init")

    def _timed_generate(
        self,
        prompts: List[PromptInput],
        params: SamplingParams,
    ):
        t0 = time.time()
        try:
            lora_request = self._current_runtime_lora_request()
            if lora_request is None:
                outputs = self.llm.generate(prompts, params)
            else:
                outputs = self.llm.generate(prompts, params, lora_request=lora_request)
        except Exception as exc:
            msg = str(exc).lower()
            if ("cuda graph" in msg) or ("graph replay" in msg) or ("graph capture" in msg):
                ge = self.metrics.setdefault("graph_event_counters", {})
                ge["cuda_graph_exception_count"] = int(ge.get("cuda_graph_exception_count", 0)) + 1
                if ("recapture" in msg) or ("capture" in msg):
                    ge["cuda_graph_recapture_hint_count"] = int(ge.get("cuda_graph_recapture_hint_count", 0)) + 1
            raise
        dur = time.time() - t0
        self._record_request_latency(dur)
        return outputs, dur

    def _run_consistency_probe(self, stage_key: str) -> Optional[Dict[str, Any]]:
        if not self.consistency_probe_enabled:
            return None

        prompt = self._build_prompt_with_user_turn(self.consistency_probe_prompt)
        params = SamplingParams(temperature=0.0, max_tokens=self.consistency_probe_max_tokens)

        outputs_cur, lat_cur = self._timed_generate([prompt], params)
        text_cur = outputs_cur[0].outputs[0].text.strip()
        tokens_cur = list(outputs_cur[0].outputs[0].token_ids)

        if self._prefix_caching_enabled and hasattr(self.llm, "reset_prefix_cache"):
            self.llm.reset_prefix_cache()
        if hasattr(self.model, "model") and hasattr(self.model.model, "clear_hidden_cache"):
            try:
                self.model.model.clear_hidden_cache()
            except Exception:
                pass

        outputs_ref, lat_ref = self._timed_generate([prompt], params)
        text_ref = outputs_ref[0].outputs[0].text.strip()
        tokens_ref = list(outputs_ref[0].outputs[0].token_ids)

        token_cmp = compare_token_sequences(tokens_ref, tokens_cur)
        probe = {
            "stage": stage_key,
            "prompt": self.consistency_probe_prompt,
            "candidate_latency_s": lat_cur,
            "reference_latency_s": lat_ref,
            "candidate_text": text_cur,
            "reference_text": text_ref,
            "text_exact_match": text_cur == text_ref,
            "token_comparison": token_cmp,
            "measured_at": datetime.now().isoformat(),
        }
        print(
            "  Consistency probe: "
            f"exact={probe['text_exact_match']}, "
            f"token_prefix_match={token_cmp['prefix_match_ratio']:.3f}"
        )
        return probe

    def _measure_prefix_cache_preservation(
        self,
        prompt: str,
        post_transition_request_only_s: float,
        surgery_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Prefix cache 보존 효과를 post-transition 요청 기준으로 측정.
        - reference_hit_s: transition 직후 request-only latency (이미 측정된 t2ft_request_only_live)
        - forced_miss_s: prefix cache reset 직후 동일 prompt 요청 latency
        - warm_after_rebuild_s: miss 이후 동일 prompt 재요청 latency
        """
        out: Dict[str, Any] = {
            "enabled": bool(self._prefix_caching_enabled and hasattr(self.llm, "reset_prefix_cache")),
            "reference_hit_s": float(post_transition_request_only_s),
            "forced_miss_s": None,
            "warm_after_rebuild_s": None,
            "miss_penalty_s": None,
            "hit_speedup_vs_forced_miss": None,
            "rebuild_speedup": None,
            "estimated_reused_block_ratio": None,
        }
        if not out["enabled"]:
            return out

        try:
            self.llm.reset_prefix_cache()
            _, miss_s = self._timed_generate([prompt], self.single_token_params)
            _, warm_s = self._timed_generate([prompt], self.single_token_params)
        except Exception as e:
            out["error"] = str(e)
            return out

        out["forced_miss_s"] = miss_s
        out["warm_after_rebuild_s"] = warm_s
        out["miss_penalty_s"] = miss_s - float(post_transition_request_only_s)
        if post_transition_request_only_s > 0:
            out["hit_speedup_vs_forced_miss"] = miss_s / float(post_transition_request_only_s)
        if warm_s > 0:
            out["rebuild_speedup"] = miss_s / warm_s

        if isinstance(surgery_profile, dict):
            need_raw = surgery_profile.get("num_blocks_needed")
            avail_raw = surgery_profile.get("available_blocks")
            need = float(need_raw) if isinstance(need_raw, (int, float)) else None
            avail = float(avail_raw) if isinstance(avail_raw, (int, float)) else None
            if need is not None and avail is not None and avail > 0:
                out["estimated_reused_block_ratio"] = max(0.0, min(1.0, need / avail))
        return out

    def _warmup_fixed_shapes(self, tag: str) -> float:
        nvtx_push(f"{tag}_shape_warmup")
        t0 = time.time()
        self._timed_generate([WARMUP_PROMPT], self.measure_ttft_params)
        for _ in range(self.throughput_requests):
            self._timed_generate([THROUGHPUT_PROMPT], self.measure_tp_params)
        elapsed = time.time() - t0
        nvtx_pop()
        return elapsed

    def _measure_ttft_request_only(
        self, tag: str, prompt: PromptInput = TTFT_PROMPT
    ) -> float:
        nvtx_push(f"{tag}_ttft")
        _, elapsed = self._timed_generate([prompt], self.measure_ttft_params)
        nvtx_pop()
        return elapsed

    def _get_live_prompt_or_default(self) -> PromptInput:
        if len(self.conversation) > 0:
            return self._build_prompt()
        return TTFT_PROMPT

    def _measure_single_token_latency(
        self,
        tag: str,
        prompt: PromptInput,
        repeats: int = 64,
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        repeats = max(1, repeats)
        nvtx_push(f"{tag}_single_token_latency")
        latencies: List[float] = []
        for _ in range(repeats):
            _, dur = self._timed_generate([prompt], self.single_token_params)
            latencies.append(dur)
        nvtx_pop()
        stats = summarize_latency_samples(latencies, name="single_token")
        result: Dict[str, Any] = {
            "single_token_repeats": int(stats["single_token_count"]),
            "single_token_p50_s": stats["single_token_p50_s"],
            "single_token_p95_s": stats["single_token_p95_s"],
            "single_token_p99_s": stats["single_token_p99_s"],
            "single_token_mean_s": stats["single_token_mean_s"],
            "single_token_max_s": stats["single_token_max_s"],
        }
        if return_samples:
            result["single_token_samples_s"] = list(latencies)
        return result

    def _measure_throughput(self, tag: str) -> Dict[str, float]:
        nvtx_push(f"{tag}_throughput")
        t0 = time.time()
        total_tokens = 0
        for _ in range(self.throughput_requests):
            outputs, _ = self._timed_generate([THROUGHPUT_PROMPT], self.measure_tp_params)
            total_tokens += len(outputs[0].outputs[0].token_ids)
        dur = time.time() - t0
        nvtx_pop()
        return {
            "throughput_tokens": total_tokens,
            "throughput_duration": dur,
            "throughput_tok_per_sec": total_tokens / dur if dur > 0 else 0.0,
            "throughput_requests": self.throughput_requests,
        }

    def _start_background_prefetch(self, target_stage: int, checkpoint_path: str, prefetch_fn, instant_fn) -> bool:
        if self.pending_transition is not None:
            pending_stage = self.pending_transition["target_stage"]
            print(f"  Stage {pending_stage} transition is already preparing.")
            return False
        if self.early_prefetch is not None:
            early_stage = self.early_prefetch["target_stage"]
            print(f"  Early prefetch for Stage {early_stage} is already active.")
            return False
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"  Stage checkpoint not found: {checkpoint_path}")
            return False

        prev_stage = target_stage - 1
        stage_key = f"stage{target_stage}"
        prev_stage_key = f"stage{prev_stage}"

        self._sync_cache_before_transition()

        print(
            f"  [Stage {prev_stage} -> {target_stage}] "
            "Background prefetch started. Transition will happen automatically when ready."
        )
        nvtx_push(f"chatbot_{stage_key}_prefetch")
        prefetch_start = time.time()
        prefetch_fn(checkpoint_path)
        prefetch_launch_time = time.time() - prefetch_start

        self.pending_transition = {
            "target_stage": target_stage,
            "stage_key": stage_key,
            "prev_stage_key": prev_stage_key,
            "checkpoint_path": checkpoint_path,
            "prefetch_start_time": prefetch_start,
            "prefetch_launch_time": prefetch_launch_time,
            "prefetch_nvtx_open": True,
            "overlap_rounds": 0,
            "overlap_serving_time": 0.0,
            "overlap_request_latencies": [],
            "instant_fn": instant_fn,
        }
        print(f"  Prefetch launch time: {prefetch_launch_time:.4f}s")
        self._maybe_complete_pending_transition(trigger="prefetch_started")
        return True

    def start_early_prefetch(self, target_stage: int, skip_sync: bool = False) -> bool:
        """
        Stage 전환 전 미리 prefetch만 시작한다.
        실제 전환은 transition_to_stage() 호출 시 수행.
        skip_sync: True면 _sync_cache_before_transition() 호출 생략
                   (_maybe_complete_pending_transition 내부에서 호출할 때 사용)
        """
        if self.pending_transition is not None:
            pending_stage = self.pending_transition["target_stage"]
            print(f"  Stage {pending_stage} transition is already preparing.")
            return False
        if self.early_prefetch is not None:
            early_stage = self.early_prefetch["target_stage"]
            print(f"  Early prefetch for Stage {early_stage} is already active.")
            return False

        if target_stage == 2:
            if self.current_stage >= 2:
                print("  Already at Stage 2 or higher.")
                return False
            checkpoint_path = self.config.get("stage_b_checkpoint")
            prefetch_fn = self.model.prefetch_stage2
            instant_fn = self.model.advance_to_stage2_instant
        elif target_stage == 3:
            if self.current_stage < 2:
                print("  Must be at Stage 2 first. Use /stage2.")
                return False
            if self.current_stage >= 3:
                print("  Already at Stage 3.")
                return False
            checkpoint_path = self.config.get("stage_c_checkpoint")
            prefetch_fn = self.model.prefetch_stage3
            instant_fn = self.model.advance_to_stage3_instant
        else:
            print(f"  Unsupported stage: {target_stage}")
            return False

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"  Stage checkpoint not found: {checkpoint_path}")
            return False

        prev_stage = target_stage - 1
        stage_key = f"stage{target_stage}"
        prev_stage_key = f"stage{prev_stage}"

        if not skip_sync:
            self._sync_cache_before_transition()
        print(
            f"  [Stage {prev_stage} -> {target_stage}] "
            "Early prefetch started. Transition will be triggered later."
        )
        nvtx_push(f"chatbot_{stage_key}_prefetch")
        prefetch_start = time.time()
        prefetch_fn(checkpoint_path)
        prefetch_launch_time = time.time() - prefetch_start

        self.early_prefetch = {
            "target_stage": target_stage,
            "stage_key": stage_key,
            "prev_stage_key": prev_stage_key,
            "checkpoint_path": checkpoint_path,
            "prefetch_start_time": prefetch_start,
            "prefetch_launch_time": prefetch_launch_time,
            "prefetch_nvtx_open": True,
            "overlap_rounds": 0,
            "overlap_serving_time": 0.0,
            "overlap_request_latencies": [],
            "instant_fn": instant_fn,
        }
        print(f"  Early prefetch launch time: {prefetch_launch_time:.4f}s")
        return True

    def _maybe_complete_pending_transition(self, trigger: str = "poll") -> bool:
        if self.pending_transition is None:
            return False
        if not self.model.is_prefetch_ready():
            return False

        p = self.pending_transition
        target_stage = p["target_stage"]
        stage_key = p["stage_key"]
        prev_stage_key = p["prev_stage_key"]
        before_samples = self._request_latency_history_s[-self.transition_window_samples:]
        before_stats = summarize_latency_samples(before_samples, name="before_promotion")
        prefetch_total_time = time.time() - p["prefetch_start_time"]
        # 실제 전환 요청 시점부터 prefetch 완료까지 대기한 시간 (early prefetch면 ≈0)
        prefetch_wait_from_request = (
            time.time() - p["transition_requested_at"]
            if "transition_requested_at" in p else prefetch_total_time
        )
        status_before_activation = self.model.get_prefetch_status()

        if p.get("prefetch_nvtx_open"):
            nvtx_pop()
            p["prefetch_nvtx_open"] = False

        nvtx_push(f"chatbot_{stage_key}_transition_instant")
        t0 = time.time()
        transitioned = p["instant_fn"](wait_if_needed=False)
        instant_transition_time = time.time() - t0
        nvtx_pop()
        if not transitioned:
            print(f"  Stage {target_stage} instant transition failed.")
            ge = self.metrics.setdefault("graph_event_counters", {})
            ge["instant_transition_fail_count"] = int(ge.get("instant_transition_fail_count", 0)) + 1
            self.pending_transition = None
            return False

        instant_activation_profile = None
        if hasattr(self.model, "get_last_instant_activation_profile"):
            try:
                instant_activation_profile = self.model.get_last_instant_activation_profile()
            except Exception:
                instant_activation_profile = None

        self.current_stage = target_stage
        runtime_lora_policy_t0 = time.time()
        self._apply_runtime_lora_stage_policy(trigger=f"{stage_key}_{trigger}")
        runtime_lora_stage_policy_time = time.time() - runtime_lora_policy_t0
        transition_total_time = instant_transition_time + runtime_lora_stage_policy_time
        runtime_lora_snapshot = dict(self.metrics.get("runtime_lora", {}))
        transition_key = f"{prev_stage_key}_to_{stage_key}"
        self.metrics["stage_transition_times"][transition_key] = transition_total_time

        # pending_transition을 먼저 클리어해야 start_early_prefetch의 guard를 통과할 수 있음
        self.pending_transition = None

        # ★ 다음 stage prefetch를 즉시 시작 → 아래 측정들(partial_recompute+TTFT+throughput) 동안 SSD I/O overlap
        if (
            self.transition_mode == "live"
            and self._auto_prefetch_after is not None
            and self._auto_prefetch_after == target_stage + 1
        ):
            _next = self._auto_prefetch_after
            self._auto_prefetch_after = None
            print(f"\n  [Early Prefetch] Stage {_next} fetch starting during Stage {target_stage} measurement (overlapping SSD I/O)...")
            self.start_early_prefetch(_next, skip_sync=True)  # _sync_cache_before_transition은 아래에서 수행

        print(
            f"\n  ✅ Stage {target_stage} auto-transition complete ({trigger})"
            f"\n     prefetch_total={prefetch_total_time:.4f}s (SSD I/O start to now, includes overlap time)"
            f"\n     prefetch_wait_from_request={prefetch_wait_from_request:.4f}s (actual wait after transition request)"
            f"\n     instant={instant_transition_time:.4f}s, overlap_rounds={p['overlap_rounds']}, "
            f"overlap_serving={p['overlap_serving_time']:.4f}s"
            f"\n     runtime_lora_policy={runtime_lora_stage_policy_time:.4f}s, "
            f"transition_total={transition_total_time:.4f}s"
        )
        if instant_activation_profile:
            print(
                "     instant_breakdown: "
                f"total={instant_activation_profile.get('total_s', 0.0):.4f}s, "
                f"extract={instant_activation_profile.get('extract_layers_data_s', 0.0):.4f}s, "
                f"apply={instant_activation_profile.get('apply_schedule_s', 0.0):.4f}s, "
                f"sync={instant_activation_profile.get('stream_sync_s', 0.0):.4f}s, "
                f"h2d_enqueue={instant_activation_profile.get('sum_h2d_enqueue_s', 0.0):.4f}s, "
                f"h2d_dma={instant_activation_profile.get('sum_h2d_dma_gpu_s', 0.0):.4f}s, "
                f"h2d_overhead={instant_activation_profile.get('sum_h2d_enqueue_overhead_s', 0.0):.4f}s, "
                f"gpu_copy_cpu={instant_activation_profile.get('sum_gpu_copy_apply_s', 0.0):.4f}s, "
                f"gpu_copy_gpu={instant_activation_profile.get('sum_gpu_copy_apply_gpu_s', 0.0):.4f}s, "
                f"h2d_bytes={instant_activation_profile.get('sum_h2d_bytes', 0) / (1024**2):.1f}MiB, "
                f"h2d_gbps={instant_activation_profile.get('effective_h2d_gbps', 0.0):.2f}, "
                f"gpu_copy_bytes={instant_activation_profile.get('sum_gpu_copy_bytes', 0) / (1024**2):.1f}MiB, "
                f"gpu_copy_gbps={instant_activation_profile.get('effective_gpu_copy_gbps', 0.0):.2f}, "
                f"alpha={instant_activation_profile.get('sum_alpha_flip_s', 0.0):.4f}s"
            )

        # overlap serving으로 conversation이 늘어났을 수 있으므로 재sync
        # _persistent_h_buffers는 overlap serving 중에도 계속 업데이트됨
        # layer 0~B-1 가중치는 불변이므로 alpha flip 이후에도 buffer 값 유효

        # ── KV Block Surgery: boundary 계산 ──
        if target_stage == 2:
            new_indices = self.model._get_b_indices()
        elif target_stage == 3:
            new_indices = self.model._get_c_indices()
        else:
            new_indices = []
        boundary = self.model.get_recompute_boundary(new_indices) if new_indices else None

        cache_sync_time = self._sync_cache_before_transition()

        reconcile_time, surgery_ok, surgery_profile, partial_profile, reconcile_path = self._do_surgery_or_fallback(boundary)

        live_prompt = self._get_live_prompt_or_default()
        t2ft_request_only_live = self._measure_ttft_request_only(
            f"chatbot_{stage_key}_t2ft_live",
            prompt=live_prompt,
        )
        # T2FT(E2E): 전환 요청 시점부터 다음 first token까지
        t2ft_e2e_from_request = (
            prefetch_wait_from_request
            + transition_total_time
            + cache_sync_time
            + reconcile_time
            + t2ft_request_only_live
        )
        # 참고용: prefetch 시작 시점 기준
        t2ft_e2e_from_prefetch_start = (
            prefetch_total_time
            + transition_total_time
            + cache_sync_time
            + reconcile_time
            + t2ft_request_only_live
        )
        print(
            f"  Stage {target_stage} T2FT(E2E, from transition request): {t2ft_e2e_from_request:.4f}s "
            f"(prefetch_wait={prefetch_wait_from_request:.4f}s, transition={transition_total_time:.4f}s, "
            f"instant={instant_transition_time:.4f}s, runtime_lora_policy={runtime_lora_stage_policy_time:.4f}s, "
            f"cache_sync={cache_sync_time:.4f}s, reconcile={reconcile_time:.4f}s, "
            f"request-only={t2ft_request_only_live:.4f}s)"
        )

        overlap_latencies = list(p.get("overlap_request_latencies", []))
        during_samples = overlap_latencies + [t2ft_request_only_live]
        during_stats = summarize_latency_samples(during_samples, name="during_promotion")
        decode_during_stats = summarize_latency_samples(
            overlap_latencies, name="decode_during_promotion"
        )

        latency_stats = self._measure_single_token_latency(
            f"chatbot_{stage_key}_post_transition",
            prompt=live_prompt,
            repeats=64,
            return_samples=True,
        )
        post_samples = list(latency_stats.get("single_token_samples_s", []))
        after_stats = summarize_latency_samples(post_samples, name="after_promotion")
        promotion_window = {
            "before": before_stats,
            "during": during_stats,
            "after": after_stats,
            "window_samples": self.transition_window_samples,
        }
        print(
            "  Post-transition 1-token latency: "
            f"p50={latency_stats['single_token_p50_s']:.4f}s, "
            f"p95={latency_stats['single_token_p95_s']:.4f}s, "
            f"p99={latency_stats['single_token_p99_s']:.4f}s"
        )
        print(
            "  Promotion window latency (p50/p95/p99): "
            f"before={before_stats['before_promotion_p50_s']:.4f}/"
            f"{before_stats['before_promotion_p95_s']:.4f}/"
            f"{before_stats['before_promotion_p99_s']:.4f}s, "
            f"during={during_stats['during_promotion_p50_s']:.4f}/"
            f"{during_stats['during_promotion_p95_s']:.4f}/"
            f"{during_stats['during_promotion_p99_s']:.4f}s, "
            f"after={after_stats['after_promotion_p50_s']:.4f}/"
            f"{after_stats['after_promotion_p95_s']:.4f}/"
            f"{after_stats['after_promotion_p99_s']:.4f}s"
        )

        throughput = self._measure_throughput(f"chatbot_{stage_key}")
        print(
            f"  Stage {target_stage} Throughput: {throughput['throughput_tok_per_sec']:.2f} tok/s "
            f"({throughput['throughput_tokens']} tokens in {throughput['throughput_duration']:.2f}s)"
        )
        prefix_cache_preservation = self._measure_prefix_cache_preservation(
            prompt=live_prompt,
            post_transition_request_only_s=t2ft_request_only_live,
            surgery_profile=surgery_profile,
        )
        if prefix_cache_preservation.get("enabled"):
            _hit = prefix_cache_preservation.get("reference_hit_s")
            _miss = prefix_cache_preservation.get("forced_miss_s")
            _spd = prefix_cache_preservation.get("hit_speedup_vs_forced_miss")
            print(
                "  Prefix-cache preservation: "
                f"hit={float(_hit) if isinstance(_hit, (int, float)) else 0.0:.4f}s, "
                f"forced_miss={float(_miss) if isinstance(_miss, (int, float)) else 0.0:.4f}s, "
                f"speedup={float(_spd) if isinstance(_spd, (int, float)) else 0.0:.2f}x"
            )
        consistency_probe = self._run_consistency_probe(stage_key)
        graph_events_snapshot = dict(self.metrics.get("graph_event_counters", {}))

        stage_info = self.model.get_stage_info()
        self._record_stage_measurement(
            stage_key=stage_key,
            stage_info=stage_info,
            ttft_request_only=t2ft_request_only_live,
            throughput=throughput,
            extra={
                "t2ft_e2e_from_request": t2ft_e2e_from_request,
                "t2ft_e2e_from_prefetch_start": t2ft_e2e_from_prefetch_start,
                "t2ft_request_only": t2ft_request_only_live,
                "t2ft_request_only_live": t2ft_request_only_live,
                "t2ft_definition": (
                    "prefetch_wait_from_request + transition_time "
                    "(instant_transition_time + runtime_lora_stage_policy_time) "
                    "+ cache_sync_time + cache_reconciliation_time + request_to_first_token"
                ),
                "cache_sync_time": cache_sync_time,
                "cache_reconciliation_time": reconcile_time,
                "cache_reconciliation_mode": self.reconcile_mode,
                "cache_reconciliation_path": reconcile_path,
                "transition_from": transition_key,
                "prefetch_duration": prefetch_total_time,
                "prefetch_wait_from_request": prefetch_wait_from_request,
                "prefetch_launch_time": p["prefetch_launch_time"],
                "prefetch_overlap_serving_time": p["overlap_serving_time"],
                "prefetch_overlap_rounds": p["overlap_rounds"],
                "prefetch_overlap_request_latencies_s": overlap_latencies,
                "decode_during_promotion_latency": decode_during_stats,
                "prefetch_wait_time": 0.0,
                "prefetch_ready_before_wait": True,
                "prefetch_ready_after_wait": True,
                "prefetch_status_before_activation": status_before_activation,
                "transition_time": transition_total_time,
                "instant_transition_time": instant_transition_time,
                "runtime_lora_stage_policy_time": runtime_lora_stage_policy_time,
                "instant_activation_profile": instant_activation_profile,
                "promotion_window_latency": promotion_window,
                "instant_h2d_bytes": (
                    int(instant_activation_profile.get("sum_h2d_bytes", 0))
                    if instant_activation_profile else 0
                ),
                "instant_gpu_copy_bytes": (
                    int(instant_activation_profile.get("sum_gpu_copy_bytes", 0))
                    if instant_activation_profile else 0
                ),
                "instant_h2d_gbps": (
                    float(instant_activation_profile.get("effective_h2d_gbps", 0.0))
                    if instant_activation_profile else 0.0
                ),
                "instant_gpu_copy_gbps": (
                    float(instant_activation_profile.get("effective_gpu_copy_gbps", 0.0))
                    if instant_activation_profile else 0.0
                ),
                "surgery_time": reconcile_time,
                "surgery_ok": surgery_ok,
                "surgery_profile": surgery_profile,
                "partial_recompute_profile": partial_profile,
                "single_token_latency": latency_stats,
                "prefix_cache_preservation": prefix_cache_preservation,
                "consistency_probe": consistency_probe,
                "graph_event_counters": graph_events_snapshot,
                "runtime_lora": runtime_lora_snapshot,
                "auto_transition_trigger": trigger,
                # backward-compatible aliases
                "ttft_e2e_from_request": t2ft_e2e_from_request,
                "ttft_e2e_from_prefetch_start": t2ft_e2e_from_prefetch_start,
            },
        )
        return True

    def _sync_cache_before_transition(self) -> float:
        """
        [그래프 보존] Stage 전환 직전 GPU persistent buffer → GPU _layer_output_cache
        GPU→GPU .clone() 사용 (PCIe D2H 병목 없음)
        """
        t0 = time.time()
        if not hasattr(self.model, "model"):
            return 0.0
        inner_model = self.model.model
        if not hasattr(inner_model, "sync_persistent_cache"):
            return 0.0

        prompt = self._build_prompt()
        token_ids = self._prompt_token_ids(prompt)
        seq_len = len(token_ids)
        print(f"  [Sync] GPU buffer → GPU cache ({seq_len} tokens, GPU→GPU clone)")
        inner_model.sync_persistent_cache(seq_len)
        return time.time() - t0

    def _trigger_partial_recompute(self) -> float:
        """
        [캐시 보존 + 그래프 보존 통합]
        1. reset_prefix_cache(): vLLM stale KV blocks 제거  (캐시 보존)
        2. generate(max_tokens=1): partial forward 트리거
             → forward() 내부에서 _layer_output_cache 활용  (그래프 보존)
               - layers 0~boundary-1: KV-only (cached hidden states)
               - layers boundary~N:   full forward (새 weights)
        """
        if len(self.conversation) == 0:
            print("  [PartialRecompute] No conversation history, skipping")
            return 0.0

        prompt = self._build_prompt()
        token_ids = self._prompt_token_ids(prompt)
        print(f"  [PartialRecompute] Prompt tokens: {len(token_ids)}")

        # [캐시 보존] vLLM stale KV blocks 제거
        # stage 전환 후 outdated prefix cache blocks가 남아있으면
        # vLLM이 잘못된 KV를 재사용하거나 프롬프트를 truncate할 수 있음
        if self._prefix_caching_enabled and hasattr(self.llm, "reset_prefix_cache"):
            self.llm.reset_prefix_cache()
            print("  [Cache] Prefix cache reset (stale blocks removed)")
        else:
            print("  [Cache] Prefix cache reset skipped (prefix caching disabled)")

        # [그래프 보존] boundary부터 partial forward 트리거
        # forward() 내부: _layer_output_cache로 boundary 이전 KV-only, 이후 full forward
        minimal_params = SamplingParams(temperature=0.0, max_tokens=1)
        nvtx_push("partial_recompute")
        t0 = time.time()
        self._timed_generate([prompt], minimal_params)
        elapsed = time.time() - t0
        nvtx_pop()

        print(f"  ✅ Partial recomputation complete ({elapsed:.2f}s)")
        return elapsed

    def _do_surgery_or_fallback(self, boundary: Optional[int]) -> tuple:
        """
        [KV Block Surgery] stage 전환 후 upper-layer KV blocks를 in-place 갱신.

        reconcile_mode:
          - auto: surgery 시도 후 실패 시 fallback partial recompute
          - surgery: surgery 우선 시도(실패 시 fallback)
          - fallback: surgery 스킵, fallback partial recompute 강제
          - full_prefill: prefix cache reset만 수행(다음 요청에서 full prefill)
          - none: reconciliation 생략 (의도적 stale-cache ablation)

        Returns:
            (elapsed: float, surgery_ok: bool, surgery_profile: Optional[Dict[str, Any]],
             partial_profile: Optional[Dict[str, Any]], reconcile_path: str)
        """
        surgery_profile = None
        partial_profile = None
        reconcile_path = "none"
        mode = self.reconcile_mode

        def _reset_prefix_cache(reason: str):
            if self._prefix_caching_enabled and hasattr(self.llm, "reset_prefix_cache"):
                self.llm.reset_prefix_cache()
                print(f"  [Cache] Prefix cache reset ({reason})")

        if len(self.conversation) == 0 or boundary is None:
            print("  [Surgery] No conversation or boundary=None, skipping KV surgery")
            _reset_prefix_cache("no surgery needed")
            return 0.0, False, surgery_profile, partial_profile, "skip_no_history_or_boundary"

        inner_model = self.model.model
        t0 = time.time()
        surgery_ok = False
        do_fallback = False

        if mode == "full_prefill":
            _reset_prefix_cache("full_prefill mode")
            print("  [Reconcile] full_prefill mode: next request will run full prefill")
            return time.time() - t0, False, surgery_profile, partial_profile, "full_prefill"
        if mode == "none":
            print("  [Reconcile] none mode: reconciliation intentionally skipped (ablation)")
            return time.time() - t0, False, surgery_profile, partial_profile, "none_no_reconcile"

        if mode in ("auto", "surgery") and hasattr(inner_model, "inject_upper_layer_kv"):
            print(f"  [Surgery] inject_upper_layer_kv(boundary={boundary}) 시도...")
            nvtx_push("kv_block_surgery")
            surgery_ok = inner_model.inject_upper_layer_kv(boundary)
            nvtx_pop()
            if hasattr(inner_model, "get_last_surgery_profile"):
                surgery_profile = inner_model.get_last_surgery_profile()
            if surgery_ok:
                print(f"  ✅ KV Block Surgery 성공 (boundary={boundary})")
                reconcile_path = "surgery"
            else:
                print(f"  ⚠️  KV Block Surgery 실패 → fallback (partial recompute)")
                do_fallback = True
                reconcile_path = "fallback_after_surgery_fail"
        elif mode in ("auto", "surgery"):
            print("  [Surgery] inject_upper_layer_kv 없음 → fallback")
            do_fallback = True
            reconcile_path = "fallback_no_surgery_api"
        elif mode == "fallback":
            print("  [Reconcile] fallback mode: surgery를 건너뛰고 partial recompute 강제")
            do_fallback = True
            reconcile_path = "fallback_forced"
        else:
            print(f"  [Reconcile] unknown mode '{mode}', fallback 실행")
            do_fallback = True
            reconcile_path = "fallback_unknown_mode"

        if do_fallback and not surgery_ok:
            # Fallback: set_partial_recompute + reset_prefix_cache + generate(max_tokens=1)
            inner_model.set_partial_recompute(boundary)
            _reset_prefix_cache("fallback")
            prompt = self._build_prompt()
            minimal_params = SamplingParams(temperature=0.0, max_tokens=1)
            nvtx_push("partial_recompute_fallback")
            self._timed_generate([prompt], minimal_params)
            nvtx_pop()
            if hasattr(inner_model, "get_last_partial_recompute_profile"):
                partial_profile = inner_model.get_last_partial_recompute_profile()
            print(f"  ✅ Fallback partial recompute complete (boundary={boundary})")

        elapsed = time.time() - t0
        return elapsed, surgery_ok, surgery_profile, partial_profile, reconcile_path

    def _record_stage_measurement(
        self,
        stage_key: str,
        stage_info: Dict[str, Any],
        ttft_request_only: float,
        throughput: Dict[str, float],
        extra: Optional[Dict[str, Any]] = None,
    ):
        item: Dict[str, Any] = {
            "ttft_request_only": ttft_request_only,
            "throughput_tokens": throughput["throughput_tokens"],
            "throughput_duration": throughput["throughput_duration"],
            "throughput_tok_per_sec": throughput["throughput_tok_per_sec"],
            "gpu_memory": gpu_memory_snapshot(),
            "active_layers": len(stage_info["active_layers"]),
            "inactive_layers": len(stage_info["inactive_layers"]),
            "activation_progress": stage_info["activation_progress"],
            "measured_at": datetime.now().isoformat(),
        }
        item.update(self._estimate_gating_overhead(stage_info))
        if extra:
            item.update(extra)

        base = self.metrics["stages"].get(stage_key, {})
        base.update(item)
        self.metrics["stages"][stage_key] = base

    def measure_current_stage(self):
        self._maybe_complete_pending_transition(trigger="before_bench")
        stage_key = f"stage{self.current_stage}"
        tag = f"chatbot_{stage_key}"
        print(f"\n  Measuring {stage_key}...")

        stage_extra: Dict[str, Any] = {}
        if self.current_stage == 1:
            ttft_cold_first = self._measure_ttft_request_only(f"{tag}_cold_first")
            ttft_e2e = self.stage1_cold_start + ttft_cold_first
            print(
                "  TTFT(E2E, cold-first): "
                f"{ttft_e2e:.4f}s (cold-start: {self.stage1_cold_start:.4f}s, "
                f"request-only: {ttft_cold_first:.4f}s)"
            )

            warmup_time = self._warmup_fixed_shapes(tag)
            print(f"  Shape warmup time: {warmup_time:.4f}s")

            ttft_steady = self._measure_ttft_request_only(f"{tag}_steady")
            print(f"  TTFT(request-only, steady): {ttft_steady:.4f}s")
            steady_latency = self._measure_single_token_latency(
                f"{tag}_steady",
                prompt=TTFT_PROMPT,
                repeats=64,
            )
            print(
                "  Stage1 steady 1-token latency: "
                f"p50={steady_latency['single_token_p50_s']:.4f}s, "
                f"p95={steady_latency['single_token_p95_s']:.4f}s, "
                f"p99={steady_latency['single_token_p99_s']:.4f}s"
            )

            ttft = ttft_cold_first
            stage_extra = {
                "shape_warmup_time": warmup_time,
                "ttft_request_only_cold_first": ttft_cold_first,
                "ttft_request_only_steady": ttft_steady,
                "steady_request_latency": ttft_steady,
                "ttft_e2e_from_cold_start": ttft_e2e,
                "single_token_latency_steady": steady_latency,
                "runtime_lora": dict(self.metrics.get("runtime_lora", {})),
                "ttft_definition": (
                    "cold_start_time + cold_first_request_to_first_token "
                    "(steady_request_latency is a post-warmup full-request latency, not TTFT)"
                ),
            }
        else:
            warmup_time = self._warmup_fixed_shapes(tag)
            print(f"  Shape warmup time: {warmup_time:.4f}s")

            ttft = self._measure_ttft_request_only(tag)
            print(f"  TTFT(request-only): {ttft:.4f}s")
            stage_latency = self._measure_single_token_latency(
                f"{tag}_request_only",
                prompt=self._get_live_prompt_or_default(),
                repeats=64,
            )
            print(
                "  1-token latency: "
                f"p50={stage_latency['single_token_p50_s']:.4f}s, "
                f"p95={stage_latency['single_token_p95_s']:.4f}s, "
                f"p99={stage_latency['single_token_p99_s']:.4f}s"
            )
            stage_extra = {
                "shape_warmup_time": warmup_time,
                "ttft_e2e_from_cold_start": None,
                "runtime_lora": dict(self.metrics.get("runtime_lora", {})),
                "ttft_definition": "request_to_first_token",
                "single_token_latency": stage_latency,
            }

        tp = self._measure_throughput(tag)
        print(
            f"  Throughput: {tp['throughput_tok_per_sec']:.2f} tok/s "
            f"({tp['throughput_tokens']} tokens in {tp['throughput_duration']:.2f}s)"
        )

        stage_info = self.model.get_stage_info()
        self._record_stage_measurement(
            stage_key=stage_key,
            stage_info=stage_info,
            ttft_request_only=ttft,
            throughput=tp,
            extra=stage_extra,
        )

    def _generate_with_transition_tracking(
        self, prompts: List[PromptInput], params: SamplingParams
    ):
        if self.pending_transition is not None:
            self._maybe_complete_pending_transition(trigger="before_generate")

        pending_overlap_track = (
            self.transition_mode == "live"
            and
            self.pending_transition is not None
            and not self.model.is_prefetch_ready()
            and self.pending_transition["overlap_rounds"] < self.overlap_rounds
        )
        early_overlap_track = (
            self.transition_mode == "live"
            and
            self.pending_transition is None
            and self.early_prefetch is not None
            and not self.model.is_prefetch_ready()
            and self.early_prefetch["overlap_rounds"] < self.overlap_rounds
        )

        if pending_overlap_track:
            p = self.pending_transition
            round_idx = p["overlap_rounds"] + 1
            nvtx_push(f"chatbot_{p['stage_key']}_overlap_infer_round{round_idx}")
            outputs, dur = self._timed_generate(prompts, params)
            nvtx_pop()
            if self.pending_transition is not None:
                self.pending_transition["overlap_rounds"] += 1
                self.pending_transition["overlap_serving_time"] += dur
                self.pending_transition.setdefault("overlap_request_latencies", []).append(dur)
        elif early_overlap_track:
            p = self.early_prefetch
            round_idx = p["overlap_rounds"] + 1
            nvtx_push(f"chatbot_{p['stage_key']}_early_overlap_infer_round{round_idx}")
            outputs, dur = self._timed_generate(prompts, params)
            nvtx_pop()
            if self.early_prefetch is not None:
                self.early_prefetch["overlap_rounds"] += 1
                self.early_prefetch["overlap_serving_time"] += dur
                self.early_prefetch.setdefault("overlap_request_latencies", []).append(dur)
        else:
            outputs, _ = self._timed_generate(prompts, params)

        if self.pending_transition is not None:
            self._maybe_complete_pending_transition(trigger="after_generate")
        return outputs

    def chat(self, user_input: str) -> str:
        self.conversation.append({"role": "user", "content": user_input})
        prompt = self._build_prompt()

        token_ids = self._prompt_token_ids(prompt)
        if len(token_ids) > 1800:
            print(
                f"  [Warning] Conversation length ({len(token_ids)} tokens) "
                "approaching limit. Consider /reset."
            )

        outputs = self._generate_with_transition_tracking([prompt], self.sampling_params)
        candidate = outputs[0].outputs[0]
        raw_text = candidate.text
        response = raw_text.strip()
        if response == "":
            raw_token_ids = list(candidate.token_ids)
            finish_reason = getattr(candidate, "finish_reason", None)
            stop_reason = getattr(candidate, "stop_reason", None)
            decoded_with_specials = ""
            if raw_token_ids:
                try:
                    decoded_with_specials = self.tokenizer.decode(
                        raw_token_ids, skip_special_tokens=False
                    )
                except Exception as exc:
                    decoded_with_specials = f"<decode_failed: {exc}>"
            print(
                "  [EmptyResponseDebug] "
                f"raw_text={raw_text!r}, "
                f"token_ids={raw_token_ids}, "
                f"finish_reason={finish_reason!r}, "
                f"stop_reason={stop_reason!r}, "
                f"decoded_with_specials={decoded_with_specials!r}"
            )
        self.conversation.append({"role": "assistant", "content": response})
        return response

    def advance_to_stage2(self) -> bool:
        if self.current_stage >= 2:
            print("  Already at Stage 2 or higher.")
            return False
        return self._start_background_prefetch(
            target_stage=2,
            checkpoint_path=self.config.get("stage_b_checkpoint"),
            prefetch_fn=self.model.prefetch_stage2,
            instant_fn=self.model.advance_to_stage2_instant,
        )

    def advance_to_stage3(self) -> bool:
        if self.current_stage < 2:
            print("  Must be at Stage 2 first. Use /stage2.")
            return False
        if self.current_stage >= 3:
            print("  Already at Stage 3.")
            return False
        return self._start_background_prefetch(
            target_stage=3,
            checkpoint_path=self.config.get("stage_c_checkpoint"),
            prefetch_fn=self.model.prefetch_stage3,
            instant_fn=self.model.advance_to_stage3_instant,
        )

    def wait_for_pending_transition(self, timeout_s: float = 180.0) -> bool:
        if self.pending_transition is None:
            return True

        start = time.time()
        while self.pending_transition is not None:
            self._maybe_complete_pending_transition(trigger="wait_loop")
            if self.pending_transition is None:
                return True

            p = self.pending_transition
            if (not self.model.is_prefetch_ready()) and p["overlap_rounds"] < self.overlap_rounds:
                self._generate_with_transition_tracking([THROUGHPUT_PROMPT], self.measure_tp_params)
                continue

            if time.time() - start > timeout_s:
                if p.get("prefetch_nvtx_open"):
                    nvtx_pop()
                print(f"  Stage {p['target_stage']} transition wait timed out.")
                self.pending_transition = None
                return False

            time.sleep(0.1)
        return True

    def transition_to_stage(self, stage: int, timeout_s: float = 180.0) -> bool:
        if self.pending_transition is not None:
            p = self.pending_transition
            if p["target_stage"] != stage:
                print(
                    f"  Stage {p['target_stage']} transition is already preparing. "
                    f"Complete it before requesting stage {stage}."
                )
                return False
            return self.wait_for_pending_transition(timeout_s=timeout_s)

        if self.early_prefetch is not None:
            ep = self.early_prefetch
            if ep["target_stage"] != stage:
                print(
                    f"  Early prefetch is prepared for stage{ep['target_stage']}, "
                    f"not stage{stage}."
                )
                return False
            self.pending_transition = {
                "target_stage": ep["target_stage"],
                "stage_key": ep["stage_key"],
                "prev_stage_key": ep["prev_stage_key"],
                "checkpoint_path": ep["checkpoint_path"],
                "prefetch_start_time": ep["prefetch_start_time"],
                "prefetch_launch_time": ep["prefetch_launch_time"],
                "prefetch_nvtx_open": ep["prefetch_nvtx_open"],
                "overlap_rounds": ep.get("overlap_rounds", 0),
                "overlap_serving_time": ep.get("overlap_serving_time", 0.0),
                "overlap_request_latencies": list(ep.get("overlap_request_latencies", [])),
                "instant_fn": ep["instant_fn"],
                "transition_requested_at": time.time(),  # 실제 전환 요청 시점
            }
            self.early_prefetch = None
            self._maybe_complete_pending_transition(trigger="use_early_prefetch")
            return self.wait_for_pending_transition(timeout_s=timeout_s)

        if stage == 2:
            started = self.advance_to_stage2()
        elif stage == 3:
            started = self.advance_to_stage3()
        else:
            print(f"  Unsupported stage: {stage}")
            return False
        if not started:
            return False
        return self.wait_for_pending_transition(timeout_s=timeout_s)

    def reset_conversation(self):
        self.conversation = []
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "clear_hidden_cache"):
                inner.clear_hidden_cache()
            if hasattr(inner, "clear_persistent_buffers"):
                inner.clear_persistent_buffers()
            print("  Conversation, hidden cache, and persistent buffers reset.")
        else:
            print("  Conversation reset.")

    def print_status(self):
        self._maybe_complete_pending_transition(trigger="status_poll")
        stage_info = self.model.get_stage_info()
        partial_mode = False
        if hasattr(self.model, "model"):
            inner_model = self.model.model
            if hasattr(inner_model, "_partial_recompute_boundary"):
                partial_mode = inner_model._partial_recompute_boundary is not None

        stage_key = f"stage{self.current_stage}"
        stage_metric = self.metrics["stages"].get(stage_key, {})
        ttft_txt = stage_metric.get("ttft_request_only")
        tp_txt = stage_metric.get("throughput_tok_per_sec")

        print(f"\n  {'='*56}")
        print(f"  Model:               {self.model_name}")
        print(f"  Stage:               {self.current_stage}")
        print(f"  Active Layers:       {len(stage_info['active_layers'])}")
        print(f"  Inactive Layers:     {len(stage_info['inactive_layers'])}")
        print(f"  Activation Progress: {stage_info['activation_progress']}")
        print(f"  Turns:               {len(self.conversation) // 2}")
        print(f"  GPU Mem:             {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Partial Recompute:   {'Active' if partial_mode else 'Idle'}")
        if self.pending_transition is not None:
            p = self.pending_transition
            print(
                f"  Pending Transition:  stage{p['target_stage']} "
                f"(overlap_rounds={p['overlap_rounds']}, overlap_time={p['overlap_serving_time']:.2f}s)"
            )
        elif self.early_prefetch is not None:
            p = self.early_prefetch
            print(
                f"  Early Prefetch:      stage{p['target_stage']} "
                f"(ready={self.model.is_prefetch_ready()}, "
                f"overlap_rounds={p['overlap_rounds']}, overlap_time={p['overlap_serving_time']:.2f}s)"
            )
        else:
            print("  Pending Transition:  None")
        if ttft_txt is not None:
            print(f"  Last TTFT:           {ttft_txt:.4f}s")
        if tp_txt is not None:
            print(f"  Last Throughput:     {tp_txt:.2f} tok/s")
        print(f"  {'='*56}")

    def print_metrics_json(self):
        print(json.dumps(self.metrics, indent=2, default=str))

    def save_metrics(self, path: str) -> str:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.metrics["saved_at"] = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=2, default=str)
        return path

    def cleanup(self):
        try:
            if self.pending_transition is not None and self.pending_transition.get("prefetch_nvtx_open"):
                nvtx_pop()
            if self.early_prefetch is not None and self.early_prefetch.get("prefetch_nvtx_open"):
                nvtx_pop()
            del self.llm
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


def _last_stage_key(stages: Dict[str, Any]) -> Optional[str]:
    if not stages:
        return None
    keys = sorted(stages.keys(), key=lambda s: int(s.replace("stage", "")))
    return keys[-1]


def measure_baseline(
    model_name: str,
    fixed_max_tokens: int,
    throughput_requests: int = THROUGHPUT_REQUESTS_DEFAULT,
    enforce_eager: bool = False,
) -> Dict[str, Any]:
    cfg = MODELS[model_name]
    path = cfg["baseline_path"]
    print("\n" + "=" * 72)
    print(f"BASELINE MEASUREMENT ({model_name})")
    print(f"  Path: {path}")
    print("=" * 72)

    reset_gpu_memory_stats()
    ttft_params = SamplingParams(temperature=0.0, max_tokens=fixed_max_tokens)
    tp_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=fixed_max_tokens)

    result: Dict[str, Any] = {
        "type": "baseline",
        "model_name": model_name,
        "model_path": path,
        "timestamp": datetime.now().isoformat(),
        "throughput_requests": max(1, throughput_requests),
        "enforce_eager": bool(enforce_eager),
        "tensor_parallel_size": max(1, int(cfg.get("tensor_parallel_size", 1))),
    }

    nvtx_push("baseline_cold_start")
    t0 = time.time()
    llm = LLM(
        model=path,
        trust_remote_code=cfg.get("trust_remote_code", True),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.4)),
        max_model_len=int(cfg.get("max_model_len", 2048)),
        tensor_parallel_size=max(1, int(cfg.get("tensor_parallel_size", 1))),
        enforce_eager=bool(enforce_eager),
        disable_sliding_window=cfg.get("disable_sliding_window", False),
    )
    cold_start = time.time() - t0
    nvtx_pop()
    result["cold_start_time"] = cold_start
    result["gpu_after_load"] = gpu_memory_snapshot()
    print(f"  Cold Start: {cold_start:.4f}s")

    nvtx_push("baseline_ttft")
    t0 = time.time()
    ttft_outputs = llm.generate([TTFT_PROMPT], ttft_params)
    ttft_request_only = time.time() - t0
    nvtx_pop()
    result["ttft_request_only_cold_first"] = ttft_request_only
    result["ttft_request_only"] = ttft_request_only
    result["ttft"] = cold_start + ttft_request_only
    result["sample_response_ttft_cold_first"] = (
        ttft_outputs[0].outputs[0].text.strip() if ttft_outputs and ttft_outputs[0].outputs else ""
    )
    print(f"  TTFT(E2E, cold-first): {result['ttft']:.4f}s (request-only: {ttft_request_only:.4f}s)")

    nvtx_push("baseline_shape_warmup")
    warm_t0 = time.time()
    llm.generate([WARMUP_PROMPT], ttft_params)
    for _ in range(max(1, throughput_requests)):
        llm.generate([THROUGHPUT_PROMPT], tp_params)
    warmup = time.time() - warm_t0
    nvtx_pop()
    result["shape_warmup_time"] = warmup
    print(f"  Shape warmup: {warmup:.4f}s")

    nvtx_push("baseline_ttft_steady")
    t0 = time.time()
    llm.generate([TTFT_PROMPT], ttft_params)
    ttft_request_only_steady = time.time() - t0
    nvtx_pop()
    result["ttft_request_only_steady"] = ttft_request_only_steady
    result["steady_request_latency"] = ttft_request_only_steady
    print(
        f"  Steady request latency (max_tokens={fixed_max_tokens}): "
        f"{ttft_request_only_steady:.4f}s"
    )

    baseline_single_token_params = SamplingParams(temperature=0.0, max_tokens=1)
    nvtx_push("baseline_steady_single_token_latency")
    baseline_latencies: List[float] = []
    for _ in range(64):
        t_single = time.time()
        llm.generate([TTFT_PROMPT], baseline_single_token_params)
        baseline_latencies.append(time.time() - t_single)
    nvtx_pop()
    baseline_steady_latency = summarize_latency_samples(baseline_latencies, name="single_token")
    result["single_token_latency_steady"] = baseline_steady_latency
    print(
        "  Baseline steady 1-token latency: "
        f"p50={baseline_steady_latency['single_token_p50_s']:.4f}s, "
        f"p95={baseline_steady_latency['single_token_p95_s']:.4f}s, "
        f"p99={baseline_steady_latency['single_token_p99_s']:.4f}s"
    )

    nvtx_push("baseline_throughput")
    t0 = time.time()
    total_tokens = 0
    for _ in range(max(1, throughput_requests)):
        outputs = llm.generate([THROUGHPUT_PROMPT], tp_params)
        total_tokens += len(outputs[0].outputs[0].token_ids)
    tp_dur = time.time() - t0
    nvtx_pop()
    result["throughput_tokens"] = total_tokens
    result["throughput_duration"] = tp_dur
    result["throughput_tok_per_sec"] = total_tokens / tp_dur if tp_dur > 0 else 0.0
    print(
        f"  Throughput: {result['throughput_tok_per_sec']:.2f} tok/s "
        f"({total_tokens} tokens in {tp_dur:.2f}s)"
    )

    result["gpu_final"] = gpu_memory_snapshot()

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1.0)

    return result


def measure_progressive_auto(
    model_name: str,
    fixed_max_tokens: int,
    overlap_rounds: int,
    throughput_requests: int,
    max_stage: int,
    auto_prompt: str,
    auto_turns: int,
    pre_transition_turns: int = 1,
    pre_transition_delay_s: float = 0.0,
    pre_transition_max_turns: int = 24,
    stage2_auto_promote_on_ready: bool = False,
    transition_mode: str = "live",
    reconcile_mode: str = "auto",
    forward_variant: str = "dualpath_inplace",
    instant_streams: int = 4,
    alpha_update_mode: str = "inplace",
    transition_window_samples: int = 64,
    consistency_probe: bool = False,
    consistency_probe_max_tokens: int = 16,
    consistency_probe_prompt: str = "Summarize cache consistency in one short sentence.",
    enable_runtime_lora: bool = False,
    runtime_lora_path: Optional[str] = None,
    runtime_lora_name: str = "stage12_runtime_lora",
    runtime_lora_stage2_path: Optional[str] = None,
    runtime_lora_stage2_name: Optional[str] = None,
    runtime_lora_int_id: int = 1,
    runtime_lora_max_rank: int = 64,
    runtime_lora_max_loras: int = 1,
    runtime_lora_stage3_policy: str = "off",
    runtime_lora_strict: bool = False,
    enforce_eager: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print(f"PROGRESSIVE MEASUREMENT ({model_name})")
    print(f"  Max Stage: {max_stage}")
    print(f"  Pre-transition turns: {pre_transition_turns}")
    print(f"  Pre-transition delay (s): {pre_transition_delay_s}")
    print(f"  Pre-transition max turns: {pre_transition_max_turns}")
    print(f"  Stage2 auto promote on ready: {stage2_auto_promote_on_ready}")
    print(f"  Transition mode: {transition_mode}")
    print(f"  Reconcile mode: {reconcile_mode}")
    print(f"  Forward variant: {forward_variant}")
    print(f"  Instant streams: {instant_streams}")
    print(f"  Alpha update: {alpha_update_mode}")
    print(f"  Transition window samples: {transition_window_samples}")
    print(f"  Consistency probe: {consistency_probe}")
    print(f"  Runtime LoRA: {enable_runtime_lora}")
    if enable_runtime_lora:
        resolved_runtime_lora_path = (
            runtime_lora_path or MODELS[model_name].get("runtime_lora_default_path")
        )
        resolved_stage2_lora_path = (
            runtime_lora_stage2_path
            or MODELS[model_name].get("runtime_lora_stage2_default_path")
        )
        print(f"    - path: {resolved_runtime_lora_path}")
        print(f"    - name: {runtime_lora_name}")
        if resolved_stage2_lora_path:
            print(f"    - stage2 swap path: {resolved_stage2_lora_path}")
            print(f"    - stage2 swap name: {runtime_lora_stage2_name or runtime_lora_name}")
        print(f"    - int_id: {runtime_lora_int_id}")
        print(f"    - max_rank: {runtime_lora_max_rank}, max_loras: {runtime_lora_max_loras}")
        print(f"    - stage3_policy: {runtime_lora_stage3_policy}, strict: {runtime_lora_strict}")
    print(f"  Enforce eager: {enforce_eager}")
    print("=" * 72)

    chatbot = ProgressiveChatbotMeasured(
        model_name=model_name,
        fixed_max_tokens=fixed_max_tokens,
        overlap_rounds=overlap_rounds,
        throughput_requests=throughput_requests,
        transition_mode=transition_mode,
        reconcile_mode=reconcile_mode,
        forward_variant=forward_variant,
        instant_streams=instant_streams,
        alpha_update_mode=alpha_update_mode,
        transition_window_samples=transition_window_samples,
        consistency_probe=consistency_probe,
        consistency_probe_max_tokens=consistency_probe_max_tokens,
        consistency_probe_prompt=consistency_probe_prompt,
        enable_runtime_lora=enable_runtime_lora,
        runtime_lora_path=runtime_lora_path,
        runtime_lora_name=runtime_lora_name,
        runtime_lora_stage2_path=runtime_lora_stage2_path,
        runtime_lora_stage2_name=runtime_lora_stage2_name,
        runtime_lora_int_id=runtime_lora_int_id,
        runtime_lora_max_rank=runtime_lora_max_rank,
        runtime_lora_max_loras=runtime_lora_max_loras,
        runtime_lora_stage3_policy=runtime_lora_stage3_policy,
        runtime_lora_strict=runtime_lora_strict,
        enforce_eager=enforce_eager,
    )
    live_mode = transition_mode == "live"

    def _idle_prefetch_delay(target_stage: int) -> Optional[Dict[str, Any]]:
        """Live mode: transition 요청 전 일정 시간 기다리며 prefetch readiness 시점을 관측."""
        if not live_mode:
            return None
        requested = max(0.0, float(pre_transition_delay_s))
        if requested <= 0.0:
            return None
        ep = chatbot.early_prefetch
        if ep is None or ep.get("target_stage") != target_stage:
            return None

        delay_start = time.time()
        ready_observed_at: Optional[float] = None
        poll_interval_s = 0.05
        while True:
            now = time.time()
            if chatbot.model.is_prefetch_ready() and ready_observed_at is None:
                ready_observed_at = now
            elapsed = now - delay_start
            remaining = requested - elapsed
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_s, remaining))

        delay_elapsed = time.time() - delay_start
        prefetch_start = ep.get("prefetch_start_time")
        ready_at_from_prefetch_start = None
        if ready_observed_at is not None and isinstance(prefetch_start, (int, float)):
            ready_at_from_prefetch_start = ready_observed_at - float(prefetch_start)

        return {
            "pre_transition_delay_requested_s": requested,
            "pre_transition_delay_elapsed_s": delay_elapsed,
            "prefetch_ready_observed_during_delay": ready_observed_at is not None,
            "prefetch_ready_observed_at_s": ready_at_from_prefetch_start,
        }

    response_trace: List[Dict[str, Any]] = []
    stage_last_response: Dict[str, Dict[str, Any]] = {}

    def _trim_text(text: str, max_chars: int = 220) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    def _chat_with_trace(
        prompt: str,
        phase: str,
        turn_index: Optional[int] = None,
    ) -> Tuple[str, float]:
        stage_before = int(chatbot.current_stage)
        t0 = time.time()
        response = chatbot.chat(prompt)
        elapsed = time.time() - t0
        stage_after = int(chatbot.current_stage)

        entry = {
            "phase": phase,
            "turn_index": turn_index,
            "served_stage": stage_before,
            "stage_after_generate": stage_after,
            "latency_s": elapsed,
            "prompt_preview": _trim_text(prompt),
            "response_preview": _trim_text(response),
            "response_chars": len(response),
        }
        response_trace.append(entry)
        stage_last_response[f"stage{stage_before}"] = dict(entry)
        return response, elapsed

    try:
        if max_stage >= 2 and live_mode and not stage2_auto_promote_on_ready:
            # ★ Stage 1 측정 시작 전에 Stage 2 prefetch 시작 → Stage 1 측정 시간 동안 SSD I/O overlap
            print(f"\n  [Early Prefetch] Stage 2 fetch starting before Stage 1 measurement (overlapping SSD I/O)...")
            chatbot.start_early_prefetch(2)

        chatbot.measure_current_stage()

        if max_stage >= 2:
            if stage2_auto_promote_on_ready and live_mode:
                if chatbot.early_prefetch is None or chatbot.early_prefetch.get("target_stage") != 2:
                    print("\n  [Early Prefetch] Stage 2 fetch starting at first user-turn window...")
                    chatbot.start_early_prefetch(2)

                max_turns = max(1, int(pre_transition_max_turns))
                first_user_request_at: Optional[float] = None
                prefetch_ready_turn: Optional[int] = None
                stage2_promoted_turn: Optional[int] = None
                prefetch_ready_elapsed_s: Optional[float] = None
                stage2_promoted_elapsed_s: Optional[float] = None
                auto_promotion_timed_out = False

                print(
                    f"\n  [Stage2 auto-promotion] Sending up to {max_turns} turn(s) "
                    "until prefetch-ready is observed..."
                )
                for i in range(max_turns):
                    if first_user_request_at is None:
                        first_user_request_at = time.time()
                    response, _ = _chat_with_trace(
                        auto_prompt,
                        phase="stage2_auto_promotion_probe",
                        turn_index=i + 1,
                    )
                    turn_idx = i + 1
                    print(f"  Turn {turn_idx}: {auto_prompt[:40]}... → {response[:40]}...")

                    if chatbot.model.is_prefetch_ready():
                        now = time.time()
                        prefetch_ready_turn = turn_idx
                        prefetch_ready_elapsed_s = (
                            now - first_user_request_at if first_user_request_at is not None else None
                        )
                        if max_stage >= 3 and live_mode:
                            chatbot._auto_prefetch_after = 3
                        ok2 = chatbot.transition_to_stage(2)
                        if not ok2:
                            raise RuntimeError("Stage 2 transition failed.")
                        stage2_promoted_turn = turn_idx
                        stage2_promoted_elapsed_s = (
                            time.time() - first_user_request_at if first_user_request_at is not None else None
                        )
                        break

                if stage2_promoted_turn is None:
                    auto_promotion_timed_out = True
                    print(
                        "  [Stage2 auto-promotion] Prefetch-ready not observed within turn budget; "
                        "forcing stage2 transition now."
                    )
                    if max_stage >= 3 and live_mode:
                        chatbot._auto_prefetch_after = 3
                    ok2 = chatbot.transition_to_stage(2)
                    if not ok2:
                        raise RuntimeError("Stage 2 transition failed.")
                    stage2_promoted_turn = max_turns
                    if first_user_request_at is not None:
                        stage2_promoted_elapsed_s = time.time() - first_user_request_at

                s2 = chatbot.metrics["stages"].setdefault("stage2", {})
                s2["stage2_auto_promote_on_ready"] = True
                s2["pre_transition_max_turns"] = max_turns
                s2["first_user_request_observed"] = first_user_request_at is not None
                s2["turns_until_prefetch_ready_from_first_request"] = prefetch_ready_turn
                s2["turns_until_stage2_promotion_from_first_request"] = stage2_promoted_turn
                s2["time_until_prefetch_ready_from_first_request_s"] = prefetch_ready_elapsed_s
                s2["time_until_stage2_promotion_from_first_request_s"] = stage2_promoted_elapsed_s
                s2["auto_promotion_turn_budget_exhausted"] = auto_promotion_timed_out
                s2["stage2_promotion_turn_definition"] = (
                    "Number of stage1 user turns from first user request until stage2 promotion is applied"
                )
                s2["stage2_promotion_time_definition"] = (
                    "Elapsed wall time from first user request issuance until stage2 promotion completion"
                )

            else:

                # pre-transition chat = Stage 2 fetch와 overlap
                if pre_transition_turns > 0:
                    print(f"\n  [Pre-transition chat] {pre_transition_turns} turn(s) before Stage 2 (overlapping prefetch)...")
                    for i in range(pre_transition_turns):
                        response, _ = _chat_with_trace(
                            auto_prompt,
                            phase="pre_stage2_overlap_chat",
                            turn_index=i + 1,
                        )
                        print(f"  Turn {i+1}: {auto_prompt[:40]}... → {response[:40]}...")
                delay2_info: Optional[Dict[str, Any]] = _idle_prefetch_delay(2)
                if delay2_info is not None:
                    print(
                        "  [Pre-transition delay] Stage 2: "
                        f"requested={delay2_info['pre_transition_delay_requested_s']:.2f}s, "
                        f"elapsed={delay2_info['pre_transition_delay_elapsed_s']:.2f}s, "
                        f"ready_during_delay={delay2_info['prefetch_ready_observed_during_delay']}"
                    )

                overlap2_s: Optional[float] = None
                ready2_before_transition: Optional[bool] = None
                if live_mode and chatbot.early_prefetch is not None and chatbot.early_prefetch["target_stage"] == 2:
                    overlap2_s = time.time() - chatbot.early_prefetch["prefetch_start_time"]
                    ready = chatbot.model.is_prefetch_ready()
                    print(f"  [Early Prefetch] Overlap available: {overlap2_s:.2f}s  |  prefetch ready: {ready}")
                    ready2_before_transition = ready

                # ★ Stage 2 instant transition 직후 자동으로 Stage 3 prefetch 시작되도록 설정
                if max_stage >= 3 and live_mode:
                    chatbot._auto_prefetch_after = 3

                ok2 = chatbot.transition_to_stage(2)
                if not ok2:
                    raise RuntimeError("Stage 2 transition failed.")

                # Stage 2 메트릭에 early prefetch 정보 추가
                if overlap2_s is not None:
                    s2 = chatbot.metrics["stages"].setdefault("stage2", {})
                    s2["early_prefetch_overlap_available_s"] = overlap2_s
                    s2["early_prefetch_ready_before_transition"] = ready2_before_transition
                if delay2_info is not None:
                    s2 = chatbot.metrics["stages"].setdefault("stage2", {})
                    s2.update(delay2_info)

        if max_stage >= 3:
            # pre-transition chat = Stage 3 fetch와 overlap (fetch는 Stage 2 측정 시작 시 이미 시작됨)
            if pre_transition_turns > 0:
                print(f"\n  [Pre-transition chat] {pre_transition_turns} turn(s) before Stage 3 (overlapping prefetch)...")
                for i in range(pre_transition_turns):
                    response, _ = _chat_with_trace(
                        auto_prompt,
                        phase="pre_stage3_overlap_chat",
                        turn_index=i + 1,
                    )
                    print(f"  Turn {i+1}: {auto_prompt[:40]}... → {response[:40]}...")
            delay3_info: Optional[Dict[str, Any]] = _idle_prefetch_delay(3)
            if delay3_info is not None:
                print(
                    "  [Pre-transition delay] Stage 3: "
                    f"requested={delay3_info['pre_transition_delay_requested_s']:.2f}s, "
                    f"elapsed={delay3_info['pre_transition_delay_elapsed_s']:.2f}s, "
                    f"ready_during_delay={delay3_info['prefetch_ready_observed_during_delay']}"
                )

            overlap3_s: Optional[float] = None
            ready3_before_transition: Optional[bool] = None
            if live_mode and chatbot.early_prefetch is not None and chatbot.early_prefetch["target_stage"] == 3:
                overlap3_s = time.time() - chatbot.early_prefetch["prefetch_start_time"]
                ready = chatbot.model.is_prefetch_ready()
                print(f"  [Early Prefetch] Overlap available: {overlap3_s:.2f}s  |  prefetch ready: {ready}")
                ready3_before_transition = ready

            ok3 = chatbot.transition_to_stage(3)
            if not ok3:
                raise RuntimeError("Stage 3 transition failed.")

            # Stage 3 메트릭에 early prefetch 정보 추가
            if overlap3_s is not None:
                s3 = chatbot.metrics["stages"].setdefault("stage3", {})
                s3["early_prefetch_overlap_available_s"] = overlap3_s
                s3["early_prefetch_ready_before_transition"] = ready3_before_transition
            if delay3_info is not None:
                s3 = chatbot.metrics["stages"].setdefault("stage3", {})
                s3.update(delay3_info)

        turns = max(1, auto_turns)
        print(f"\n  [Auto Chat] prompt='{auto_prompt}', turns={turns}")
        for i in range(turns):
            print(f"You [Stage {chatbot.current_stage}] (auto {i + 1}/{turns}): {auto_prompt}")
            response, dur = _chat_with_trace(
                auto_prompt,
                phase="post_measurement_auto_chat",
                turn_index=i + 1,
            )
            print(f"Assistant [Stage {chatbot.current_stage}] ({dur:.1f}s): {response}\n")

        result = dict(chatbot.metrics)
        last = _last_stage_key(result.get("stages", {}))
        result["type"] = "progressive"
        result["max_stage"] = max_stage
        result["final_stage"] = chatbot.current_stage
        result["transition_mode"] = transition_mode
        result["reconcile_mode"] = reconcile_mode
        result["forward_variant"] = forward_variant
        result["instant_streams"] = instant_streams
        result["alpha_update_mode"] = alpha_update_mode
        result["pre_transition_turns"] = int(pre_transition_turns)
        result["pre_transition_delay_s"] = float(pre_transition_delay_s)
        result["pre_transition_max_turns"] = int(pre_transition_max_turns)
        result["stage2_auto_promote_on_ready"] = bool(stage2_auto_promote_on_ready)
        result["enforce_eager"] = bool(enforce_eager)
        result["consistency_probe_enabled"] = bool(consistency_probe)
        result["runtime_lora_enabled"] = bool(enable_runtime_lora)
        result["runtime_lora_path"] = runtime_lora_path or MODELS[model_name].get("runtime_lora_default_path")
        result["runtime_lora_name"] = str(runtime_lora_name)
        result["runtime_lora_stage2_path"] = (
            str(runtime_lora_stage2_path) if runtime_lora_stage2_path else None
        )
        result["runtime_lora_stage2_name"] = (
            str(runtime_lora_stage2_name) if runtime_lora_stage2_name else None
        )
        result["runtime_lora_int_id"] = int(runtime_lora_int_id)
        result["runtime_lora_max_rank"] = int(runtime_lora_max_rank)
        result["runtime_lora_max_loras"] = int(runtime_lora_max_loras)
        result["runtime_lora_stage3_policy"] = str(runtime_lora_stage3_policy)
        result["runtime_lora_strict"] = bool(runtime_lora_strict)
        result["response_trace"] = response_trace
        result["stage_last_response"] = stage_last_response
        result["response_trace_count"] = len(response_trace)
        print("\n  [Stage Last Response Summary]")
        for stage_key in ("stage1", "stage2", "stage3"):
            item = stage_last_response.get(stage_key)
            if item is None:
                print(f"    {stage_key}: (no response captured)")
                continue
            print(
                f"    {stage_key}: {item.get('response_preview', '')} "
                f"(phase={item.get('phase')}, latency={float(item.get('latency_s', 0.0)):.3f}s)"
            )
        if "stage1" in result["stages"]:
            s1 = result["stages"]["stage1"]
            result["ttft"] = s1.get("ttft_e2e_from_cold_start") or (
                s1.get("cold_start_time", 0.0) + s1.get("ttft_request_only", 0.0)
            )
        if last:
            result["throughput_tok_per_sec"] = result["stages"][last].get("throughput_tok_per_sec")
        return result
    finally:
        chatbot.cleanup()


def print_comparison_table(baseline: Dict[str, Any], progressive: Dict[str, Any]):
    def pct(base: float, cur: float) -> str:
        if base == 0:
            return "N/A"
        return f"{(base - cur) / base * 100:.1f}%"

    p_stage1 = progressive.get("stages", {}).get("stage1", {})
    p_last_key = _last_stage_key(progressive.get("stages", {}))
    p_last = progressive.get("stages", {}).get(p_last_key, {}) if p_last_key else {}

    b_cold = baseline.get("cold_start_time", 0.0)
    p_cold = p_stage1.get("cold_start_time", 0.0)
    b_ttft_cold = baseline.get("ttft", 0.0)
    p_ttft_cold = p_stage1.get("ttft_e2e_from_cold_start", progressive.get("ttft", 0.0))
    b_steady_1tok = (
        baseline.get("single_token_latency_steady", {}).get("single_token_p50_s")
        if isinstance(baseline.get("single_token_latency_steady"), dict)
        else None
    )
    p_steady_1tok = (
        p_stage1.get("single_token_latency_steady", {}).get("single_token_p50_s")
        if isinstance(p_stage1.get("single_token_latency_steady"), dict)
        else None
    )
    b_tp = baseline.get("throughput_tok_per_sec", 0.0)
    p_tp = p_last.get("throughput_tok_per_sec", 0.0)
    b_mem = baseline.get("gpu_after_load", {}).get("allocated_gb", 0.0)
    p_mem = p_stage1.get("gpu_memory", {}).get("allocated_gb", 0.0)

    print("\n" + "=" * 90)
    print("COMPARISON: BASELINE vs PROGRESSIVE")
    print("=" * 90)
    print(f"{'Metric':<32} {'Baseline':<18} {'Progressive':<18} {'Improvement':<15}")
    print("-" * 90)
    print(f"{'Cold Start (s)':<32} {b_cold:<18.4f} {p_cold:<18.4f} {pct(b_cold, p_cold):<15}")
    print(
        f"{'Stage1 TTFT cold E2E (s)':<32} "
        f"{b_ttft_cold:<18.4f} {p_ttft_cold:<18.4f} {pct(b_ttft_cold, p_ttft_cold):<15}"
    )
    if b_steady_1tok is not None and p_steady_1tok is not None:
        print(
            f"{'Stage1 warm 1-token p50 (s)':<32} "
            f"{b_steady_1tok:<18.4f} {p_steady_1tok:<18.4f} {pct(b_steady_1tok, p_steady_1tok):<15}"
        )
    if b_tp > 0:
        tp_imp = f"{(p_tp - b_tp) / b_tp * 100:+.1f}%"
    else:
        tp_imp = "N/A"
    print(f"{'Throughput (tok/s)':<32} {b_tp:<18.2f} {p_tp:<18.2f} {tp_imp:<15}")
    print(f"{'GPU After Load (GB)':<32} {b_mem:<18.2f} {p_mem:<18.2f} {pct(b_mem, p_mem):<15}")

    print("\n" + "-" * 90)
    print(f"{'Stage':<12} {'Layers':<12} {'Metric':<12} {'FirstTok(s)':<16} {'Throughput':<16} {'Transition':<14}")
    print("-" * 90)
    for stage_name in sorted(progressive.get("stages", {}).keys(), key=lambda s: int(s.replace("stage", ""))):
        s = progressive["stages"][stage_name]
        layers = s.get("active_layers", 0)
        metric_name = "TTFT"
        first_token_metric = s.get("ttft_e2e_from_cold_start")
        if stage_name != "stage1":
            metric_name = "T2FT"
            first_token_metric = s.get("t2ft_e2e_from_request", s.get("ttft_e2e_from_request"))
        if first_token_metric is None:
            first_token_metric = s.get("ttft_request_only", 0.0)
        tp = s.get("throughput_tok_per_sec", 0.0)
        trans = s.get("transition_time", 0.0) if "transition_time" in s else 0.0
        trans_txt = f"{trans:.4f}" if "transition_time" in s else "-"
        print(
            f"{stage_name:<12} {layers:<12} {metric_name:<12} "
            f"{first_token_metric:<16.4f} {tp:<16.2f} {trans_txt:<14}"
        )


def save_json(path: str, data: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(data)
    payload["saved_at"] = datetime.now().isoformat()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Progressive Serving Chatbot Test (Graph + Cache Preservation)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["chat", "baseline", "progressive", "both"],
        default="chat",
        help="chat: interactive, baseline: baseline only, progressive: stage benchmark, both: compare",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(CANONICAL_MODEL_CHOICES),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--fixed-max-tokens",
        type=int,
        default=FIXED_MAX_TOKENS_DEFAULT,
        help=f"Max tokens for TTFT/Throughput measurement (default: {FIXED_MAX_TOKENS_DEFAULT})",
    )
    parser.add_argument(
        "--overlap-rounds",
        type=int,
        default=4,
        help="Max overlap inference rounds while prefetch is in progress (default: 4)",
    )
    parser.add_argument(
        "--throughput-requests",
        type=int,
        default=THROUGHPUT_REQUESTS_DEFAULT,
        help=f"Number of single-request iterations for throughput measurement (default: {THROUGHPUT_REQUESTS_DEFAULT})",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="chatbot_partial_cache_runtime_lora_metrics.json",
        help="Path for /save command (default: chatbot_partial_cache_runtime_lora_metrics.json)",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run non-interactive auto chat mode using --auto-prompt",
    )
    parser.add_argument(
        "--auto-prompt",
        type=str,
        default=AUTO_PROMPT_DEFAULT,
        help=f"Prompt used in --auto mode (default: {AUTO_PROMPT_DEFAULT})",
    )
    parser.add_argument(
        "--auto-turns",
        type=int,
        default=1,
        help="Number of auto chat turns in --auto mode (default: 1)",
    )
    parser.add_argument(
        "--pre-transition-turns",
        type=int,
        default=1,
        help="Chat turns before each stage transition to build KV cache for partial recompute (default: 1)",
    )
    parser.add_argument(
        "--pre-transition-delay-s",
        type=float,
        default=0.0,
        help=(
            "In live mode, idle delay before each transition request after starting early prefetch "
            "(default: 0.0)"
        ),
    )
    parser.add_argument(
        "--pre-transition-max-turns",
        type=int,
        default=24,
        help=(
            "In stage2 auto-promotion mode, maximum stage1 user turns to wait for prefetch-ready "
            "before forcing transition (default: 24)"
        ),
    )
    parser.add_argument(
        "--stage2-auto-promote-on-ready",
        action="store_true",
        help=(
            "In live mode, keep serving stage1 turns and promote to stage2 immediately when "
            "prefetch-ready is observed"
        ),
    )
    parser.add_argument(
        "--auto-stage-flow",
        type=str,
        choices=["none", "stage2", "stage3"],
        default="stage3",
        help="In --auto chat mode, how far to auto-transition before sending prompts (default: stage3)",
    )
    parser.add_argument(
        "--max-stage",
        type=int,
        choices=[1, 2, 3],
        default=3,
        help="In benchmark mode(progressive/both), run up to this stage (default: 3)",
    )
    parser.add_argument(
        "--transition-mode",
        type=str,
        choices=["clean", "live"],
        default="live",
        help="Transition measurement mode: clean(no overlap) or live(overlap enabled) (default: live)",
    )
    parser.add_argument(
        "--reconcile-mode",
        type=str,
        choices=["auto", "surgery", "fallback", "full_prefill", "none"],
        default="auto",
        help="Cache reconciliation mode for stage promotion (default: auto)",
    )
    parser.add_argument(
        "--forward-variant",
        type=str,
        choices=[
            "dualpath_inplace",
            "singlepath_branch",
            "skip_inactive_layers",
            "alpha_host_scalar",
        ],
        default="dualpath_inplace",
        help="Forward variant for invariant ablation (default: dualpath_inplace)",
    )
    parser.add_argument(
        "--instant-streams",
        type=int,
        default=4,
        help="Number of CUDA streams for instant activation (default: 4)",
    )
    parser.add_argument(
        "--alpha-update-mode",
        type=str,
        choices=["inplace", "rebind"],
        default="inplace",
        help="Alpha update mode in UniversalBypassLayer (default: inplace)",
    )
    parser.add_argument(
        "--transition-window-samples",
        type=int,
        default=64,
        help="Number of request latencies to summarize for promotion before/after windows (default: 64)",
    )
    parser.add_argument(
        "--consistency-probe",
        action="store_true",
        help="Run post-promotion correctness/cache-consistency probe against full-prefill reference",
    )
    parser.add_argument(
        "--consistency-probe-max-tokens",
        type=int,
        default=16,
        help="Max tokens for consistency probe generation (default: 16)",
    )
    parser.add_argument(
        "--consistency-probe-prompt",
        type=str,
        default="Summarize cache consistency in one short sentence.",
        help="Prompt used for consistency probe (default: cache consistency sentence)",
    )
    parser.add_argument(
        "--enable-runtime-lora",
        action="store_true",
        default=os.environ.get("RUNTIME_LORA_ENABLE", "0") == "1",
        help="Enable runtime adapter attach/detach flow for progressive path.",
    )
    parser.add_argument(
        "--runtime-lora-path",
        type=str,
        default=os.environ.get("RUNTIME_LORA_PATH") or None,
        help="Explicit adapter path for runtime add/remove. Defaults to model config.",
    )
    parser.add_argument(
        "--runtime-lora-name",
        type=str,
        default=os.environ.get("RUNTIME_LORA_NAME") or "stage12_runtime_lora",
        help="Adapter name passed to vLLM runtime LoRA API.",
    )
    parser.add_argument(
        "--runtime-lora-stage2-path",
        type=str,
        default=os.environ.get("RUNTIME_LORA_STAGE2_PATH") or None,
        help="Optional adapter path to hot-swap in at stage2 within the same engine.",
    )
    parser.add_argument(
        "--runtime-lora-stage2-name",
        type=str,
        default=os.environ.get("RUNTIME_LORA_STAGE2_NAME") or None,
        help="Optional adapter name to hot-swap in at stage2 (defaults to current name).",
    )
    parser.add_argument(
        "--runtime-lora-int-id",
        type=int,
        default=int(os.environ.get("RUNTIME_LORA_INT_ID", "1")),
        help="Integer LoRA adapter id for vLLM runtime add/remove (default: 1).",
    )
    parser.add_argument(
        "--runtime-lora-max-rank",
        type=int,
        default=int(os.environ.get("RUNTIME_LORA_MAX_RANK", "64")),
        help="max_lora_rank passed to vLLM when runtime LoRA is enabled (default: 64).",
    )
    parser.add_argument(
        "--runtime-lora-max-loras",
        type=int,
        default=int(os.environ.get("RUNTIME_LORA_MAX_LORAS", "1")),
        help="max_loras/max_cpu_loras passed to vLLM when runtime LoRA is enabled (default: 1).",
    )
    parser.add_argument(
        "--runtime-lora-stage3-policy",
        type=str,
        choices=["off", "remove", "keep"],
        default=os.environ.get("RUNTIME_LORA_STAGE3_POLICY", "off"),
        help=(
            "Stage3 policy for runtime LoRA: off(no request), remove(engine remove), keep(still apply). "
            "default: off"
        ),
    )
    parser.add_argument(
        "--runtime-lora-strict",
        action="store_true",
        default=os.environ.get("RUNTIME_LORA_STRICT", "0") == "1",
        help="Fail immediately if runtime LoRA add/remove cannot be applied.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Use eager execution in vLLM (ablation; disables CUDA graph replay benefits)",
    )
    parser.add_argument(
        "--strict-cold-paired",
        action="store_true",
        help=(
            "Mode=both only: enforce per-arm cold condition "
            "(drop_caches + GPU clear before baseline and progressive independently)."
        ),
    )
    parser.add_argument(
        "--strict-cold-pair-order",
        type=str,
        choices=["baseline_first", "progressive_first"],
        default="baseline_first",
        help="Mode=both + strict-cold only: arm execution order (default: baseline_first).",
    )
    args = parser.parse_args()
    if args.stage2_auto_promote_on_ready and args.transition_mode != "live":
        print("  [WARN] --stage2-auto-promote-on-ready is intended for --transition-mode live.")
    if args.stage2_auto_promote_on_ready and args.max_stage < 2:
        print("  [WARN] --stage2-auto-promote-on-ready has no effect when --max-stage < 2.")

    print("\n" + "=" * 72)
    print("Progressive Serving Chatbot Test (Graph + Cache Preservation)")
    print(f"  Mode:             {args.mode}")
    print(f"  Model:            {args.model}")
    print(f"  Fixed max tokens: {args.fixed_max_tokens}")
    print(f"  Overlap rounds:   {args.overlap_rounds}")
    print(f"  TP requests:      {args.throughput_requests}")
    print(f"  Auto mode:        {args.auto}")
    print(f"  Auto stage flow:  {args.auto_stage_flow}")
    print(f"  Pre-transition turns: {args.pre_transition_turns}")
    print(f"  Pre-transition delay (s): {args.pre_transition_delay_s}")
    print(f"  Pre-transition max turns: {args.pre_transition_max_turns}")
    print(f"  Stage2 auto promote on ready: {args.stage2_auto_promote_on_ready}")
    print(f"  Max stage:        {args.max_stage}")
    print(f"  Transition mode:  {args.transition_mode}")
    print(f"  Reconcile mode:   {args.reconcile_mode}")
    print(f"  Forward variant:  {args.forward_variant}")
    print(f"  Instant streams:  {args.instant_streams}")
    print(f"  Alpha update:     {args.alpha_update_mode}")
    print(f"  Window samples:   {args.transition_window_samples}")
    print(f"  Consistency probe:{args.consistency_probe}")
    print(f"  Runtime LoRA:     {args.enable_runtime_lora}")
    if args.enable_runtime_lora:
        resolved_lora_path = args.runtime_lora_path or MODELS[args.model].get("runtime_lora_default_path")
        resolved_stage2_lora_path = (
            args.runtime_lora_stage2_path
            or MODELS[args.model].get("runtime_lora_stage2_default_path")
        )
        print(f"  LoRA path:        {resolved_lora_path}")
        print(f"  LoRA name/id:     {args.runtime_lora_name} / {args.runtime_lora_int_id}")
        if resolved_stage2_lora_path:
            print(f"  LoRA stage2 path: {resolved_stage2_lora_path}")
            print(
                f"  LoRA stage2 name: "
                f"{args.runtime_lora_stage2_name or args.runtime_lora_name}"
            )
        print(f"  LoRA rank/slots:  {args.runtime_lora_max_rank} / {args.runtime_lora_max_loras}")
        print(f"  LoRA stage3 pol.: {args.runtime_lora_stage3_policy}")
        print(f"  LoRA strict mode: {args.runtime_lora_strict}")
    print(f"  Enforce eager:    {args.enforce_eager}")
    print(f"  Strict cold pair: {args.strict_cold_paired}")
    if args.strict_cold_paired:
        print(f"  Pair order:       {args.strict_cold_pair_order}")
    print(f"  GPU:              {torch.cuda.get_device_name(0)}")
    print("=" * 72)

    if args.mode in ("baseline", "progressive", "both"):
        experiment: Dict[str, Any] = {
            "script": "chatbot_partial_cache_runtime_lora.py",
            "mode": args.mode,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "fixed_max_tokens": args.fixed_max_tokens,
            "overlap_rounds": args.overlap_rounds,
            "throughput_requests": args.throughput_requests,
            "max_stage": args.max_stage,
            "auto_prompt": args.auto_prompt,
            "auto_turns": args.auto_turns,
            "pre_transition_turns": args.pre_transition_turns,
            "pre_transition_delay_s": args.pre_transition_delay_s,
            "pre_transition_max_turns": args.pre_transition_max_turns,
            "stage2_auto_promote_on_ready": args.stage2_auto_promote_on_ready,
            "transition_mode": args.transition_mode,
            "reconcile_mode": args.reconcile_mode,
            "forward_variant": args.forward_variant,
            "instant_streams": args.instant_streams,
            "alpha_update_mode": args.alpha_update_mode,
            "transition_window_samples": args.transition_window_samples,
            "consistency_probe": args.consistency_probe,
            "consistency_probe_max_tokens": args.consistency_probe_max_tokens,
            "consistency_probe_prompt": args.consistency_probe_prompt,
            "enable_runtime_lora": args.enable_runtime_lora,
            "runtime_lora_path": args.runtime_lora_path,
            "runtime_lora_name": args.runtime_lora_name,
            "runtime_lora_stage2_path": args.runtime_lora_stage2_path,
            "runtime_lora_stage2_name": args.runtime_lora_stage2_name,
            "runtime_lora_int_id": args.runtime_lora_int_id,
            "runtime_lora_max_rank": args.runtime_lora_max_rank,
            "runtime_lora_max_loras": args.runtime_lora_max_loras,
            "runtime_lora_stage3_policy": args.runtime_lora_stage3_policy,
            "runtime_lora_strict": args.runtime_lora_strict,
            "enforce_eager": args.enforce_eager,
            "strict_cold_paired": args.strict_cold_paired,
            "strict_cold_pair_order": args.strict_cold_pair_order,
        }

        def _run_baseline_arm():
            return measure_baseline(
                model_name=args.model,
                fixed_max_tokens=args.fixed_max_tokens,
                throughput_requests=args.throughput_requests,
                enforce_eager=args.enforce_eager,
            )

        def _run_progressive_arm():
            return measure_progressive_auto(
                model_name=args.model,
                fixed_max_tokens=args.fixed_max_tokens,
                overlap_rounds=args.overlap_rounds,
                throughput_requests=args.throughput_requests,
                max_stage=args.max_stage,
                auto_prompt=args.auto_prompt,
                auto_turns=args.auto_turns,
                pre_transition_turns=args.pre_transition_turns,
                pre_transition_delay_s=args.pre_transition_delay_s,
                pre_transition_max_turns=args.pre_transition_max_turns,
                stage2_auto_promote_on_ready=args.stage2_auto_promote_on_ready,
                transition_mode=args.transition_mode,
                reconcile_mode=args.reconcile_mode,
                forward_variant=args.forward_variant,
                instant_streams=args.instant_streams,
                alpha_update_mode=args.alpha_update_mode,
                transition_window_samples=args.transition_window_samples,
                consistency_probe=args.consistency_probe,
                consistency_probe_max_tokens=args.consistency_probe_max_tokens,
                consistency_probe_prompt=args.consistency_probe_prompt,
                enable_runtime_lora=args.enable_runtime_lora,
                runtime_lora_path=args.runtime_lora_path,
                runtime_lora_name=args.runtime_lora_name,
                runtime_lora_stage2_path=args.runtime_lora_stage2_path,
                runtime_lora_stage2_name=args.runtime_lora_stage2_name,
                runtime_lora_int_id=args.runtime_lora_int_id,
                runtime_lora_max_rank=args.runtime_lora_max_rank,
                runtime_lora_max_loras=args.runtime_lora_max_loras,
                runtime_lora_stage3_policy=args.runtime_lora_stage3_policy,
                runtime_lora_strict=args.runtime_lora_strict,
                enforce_eager=args.enforce_eager,
            )

        if args.strict_cold_paired and args.mode != "both":
            print("  [WARN] --strict-cold-paired applies only to --mode both; ignored.")

        if args.mode == "both" and args.strict_cold_paired:
            arm_order = ["baseline", "progressive"]
            if args.strict_cold_pair_order == "progressive_first":
                arm_order = ["progressive", "baseline"]
            experiment["strict_cold_arm_order_executed"] = arm_order

            for arm in arm_order:
                prepare_strict_cold_arm(arm)
                if arm == "baseline":
                    experiment["baseline"] = _run_baseline_arm()
                else:
                    experiment["progressive"] = _run_progressive_arm()
        else:
            if args.mode in ("baseline", "both"):
                experiment["baseline"] = _run_baseline_arm()

            if args.mode in ("progressive", "both"):
                experiment["progressive"] = _run_progressive_arm()

        if args.mode == "both":
            print_comparison_table(experiment["baseline"], experiment["progressive"])

        saved = save_json(args.save_path, experiment)
        print(f"\nResults saved to: {saved}")
        return

    chatbot = ProgressiveChatbotMeasured(
        model_name=args.model,
        fixed_max_tokens=args.fixed_max_tokens,
        overlap_rounds=args.overlap_rounds,
        throughput_requests=args.throughput_requests,
        transition_mode=args.transition_mode,
        reconcile_mode=args.reconcile_mode,
        forward_variant=args.forward_variant,
        instant_streams=args.instant_streams,
        alpha_update_mode=args.alpha_update_mode,
        transition_window_samples=args.transition_window_samples,
        consistency_probe=args.consistency_probe,
        consistency_probe_max_tokens=args.consistency_probe_max_tokens,
        consistency_probe_prompt=args.consistency_probe_prompt,
        enable_runtime_lora=args.enable_runtime_lora,
        runtime_lora_path=args.runtime_lora_path,
        runtime_lora_name=args.runtime_lora_name,
        runtime_lora_stage2_path=args.runtime_lora_stage2_path,
        runtime_lora_stage2_name=args.runtime_lora_stage2_name,
        runtime_lora_int_id=args.runtime_lora_int_id,
        runtime_lora_max_rank=args.runtime_lora_max_rank,
        runtime_lora_max_loras=args.runtime_lora_max_loras,
        runtime_lora_stage3_policy=args.runtime_lora_stage3_policy,
        runtime_lora_strict=args.runtime_lora_strict,
        enforce_eager=args.enforce_eager,
    )

    print(f"\n{'='*72}")
    print(f"  Ready! (Stage {chatbot.current_stage})")
    print("  Commands: /stage2, /stage3, /bench, /status, /metrics, /save, /reset, /quit")
    print("  Note: /stage2,/stage3는 준비만 시작하고, 준비 완료 시 자동 전환+측정됩니다.")
    print(f"{'='*72}\n")

    if args.auto:
        turns = max(1, args.auto_turns)
        if args.auto_stage_flow in ("stage2", "stage3"):
            ok2 = chatbot.transition_to_stage(2)
            if not ok2:
                print("  [Auto] Stage2 transition failed.")
        if args.auto_stage_flow == "stage3":
            ok3 = chatbot.transition_to_stage(3)
            if not ok3:
                print("  [Auto] Stage3 transition failed.")

        print(f"  [Auto] prompt='{args.auto_prompt}', turns={turns}")
        for i in range(turns):
            chatbot._maybe_complete_pending_transition(trigger="auto_loop_poll")
            user_input = args.auto_prompt
            print(f"You [Stage {chatbot.current_stage}] (auto {i + 1}/{turns}): {user_input}")
            t0 = time.time()
            response = chatbot.chat(user_input)
            elapsed = time.time() - t0
            print(f"Assistant [Stage {chatbot.current_stage}] ({elapsed:.1f}s): {response}\n")
        saved = chatbot.save_metrics(args.save_path)
        print(f"  [Auto] Metrics saved to: {saved}")
        chatbot.cleanup()
        return

    try:
        while True:
            chatbot._maybe_complete_pending_transition(trigger="loop_poll")
            prompt = f"You [Stage {chatbot.current_stage}]: "
            try:
                if chatbot.pending_transition is None:
                    user_input = input(prompt).strip()
                else:
                    # pending transition이 있을 때는 입력 대기 중에도 짧게 poll해서
                    # 준비 완료 즉시 자동 전환을 수행한다.
                    sys.stdout.write(prompt)
                    sys.stdout.flush()
                    while True:
                        chatbot._maybe_complete_pending_transition(trigger="stdin_poll")
                        readable, _, _ = select.select([sys.stdin], [], [], 0.2)
                        if readable:
                            line = sys.stdin.readline()
                            if line == "":
                                raise EOFError
                            user_input = line.strip()
                            break
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not user_input:
                continue

            chatbot._maybe_complete_pending_transition(trigger="before_command")

            if user_input == "/quit":
                print("Bye!")
                break
            if user_input == "/stage2":
                chatbot.advance_to_stage2()
                continue
            if user_input == "/stage3":
                chatbot.advance_to_stage3()
                continue
            if user_input == "/bench":
                chatbot.measure_current_stage()
                continue
            if user_input == "/status":
                chatbot.print_status()
                continue
            if user_input == "/metrics":
                chatbot.print_metrics_json()
                continue
            if user_input == "/save":
                saved = chatbot.save_metrics(args.save_path)
                print(f"  Metrics saved to: {saved}")
                continue
            if user_input == "/reset":
                chatbot.reset_conversation()
                continue

            t0 = time.time()
            response = chatbot.chat(user_input)
            elapsed = time.time() - t0
            print(f"Assistant [Stage {chatbot.current_stage}] ({elapsed:.1f}s): {response}\n")
    finally:
        chatbot.cleanup()


if __name__ == "__main__":
    main()
