#!/usr/bin/env python3
"""
Progressive Serving Chatbot - Partial KV Cache Recomputation
=============================================================

Interactive chatbot with progressive model serving (vLLM v0 engine).
Stage transitions on user command (/stage2, /stage3).

**핵심 차이점 (vs chatbot_full_cache.py):**
- KV Cache를 턴 사이에 유지 (재초기화 안 함)
- Stage 전환 시에만 부분적 KV Cache 재계산:
  * Boundary 이전 레이어: 완전 스킵 (가중치 불변 → KV cache 그대로 유효)
  * Boundary 이후 레이어: full forward (새로운 가중치로 재계산)
- Hidden states CPU 캐싱으로 GPU 연산 최소화
- CUDA Graph 재캡처 최소화 (prefill에서만 partial recompute)

Usage:
  python chatbot_partial_cache.py --model llama
  python chatbot_partial_cache.py --model mistral

  Commands during chat:
    /stage2  - Transition to Stage 2 (partial KV recomputation)
    /stage3  - Transition to Stage 3 (partial KV recomputation)
    /status  - Show model status
    /reset   - Reset conversation
    /quit    - Exit
"""

import os
import sys
import json
import time
import argparse

# vLLM v0 엔진 강제 사용 (모델 직접 접근 필요)
os.environ["VLLM_USE_V1"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# Progressive model
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "progressive_serve"))
from progressive_for_causal_lm import ProgressiveForCausalLM

# =========================================================
# 🔥 vLLM v0.8.0 버그 우회 (Prefix Caching 강제 활성화 패치)
# =========================================================
import vllm.config
# vLLM이 이 모델을 멀티모달로 착각하지 않도록 속임
vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)
# =========================================================

# ============================================================================
# 모델 설정 (02_universal.py 동일)
# ============================================================================

MODELS = {
    "llama": {
        "progressive_path": "/acpl-ssd30/7b_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/7b_results/pruning/checkpoints/stage3_layers_C.safetensors",
    },
    "mistral": {
        "progressive_path": "/home/devewha/entropy_routing/25_mistral_results/pruning/A",
        "stage_b_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage3_layers_C.safetensors",
    },
}


# ============================================================================
# Chatbot with Partial KV Cache Recomputation
# ============================================================================

class ProgressiveChatbotPartial:
    """
    Progressive Serving 기반 대화형 챗봇 (Partial KV Cache Recomputation)

    핵심 기능:
    - KV Cache 턴 간 유지 (vLLM KV 블록 재사용)
    - Stage 전환 시 부분 재계산:
      * Unchanged layers (< boundary): 완전 스킵 (GPU 연산 없음, KV cache 유효)
      * Changed layers (>= boundary): full forward (새 가중치로 KV 재계산)
    - Hidden states CPU 캐싱으로 GPU 연산 최소화
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = MODELS[model_name]
        self.current_stage = 1
        self.conversation = []  # [{"role": "user"/"assistant", "content": "..."}]

        # config.json에서 아키텍처 읽기 → 등록
        model_path = self.config["progressive_path"]
        with open(os.path.join(model_path, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        ModelRegistry.register_model(arch, ProgressiveForCausalLM)
        print(f"  Registered ProgressiveForCausalLM as: {arch}")

        # LLM 생성
        print(f"\n  Loading {model_name} Stage 1...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            max_model_len=2048,
            # 🔥 enforce_eager=False: CUDA graph 활성화
            # Persistent GPU buffer + index_copy_()가 CUDA graph에 캡처되어
            # decode phase에서도 hidden states가 자동 누적됨
            enforce_eager=False,
            # Prefix caching 활성화 (KV cache 턴 간 유지)
            enable_prefix_caching=True,
        )

        # v0 엔진 모델 핸들 (02_universal.py 동일)
        self.model = self._get_model_handle()

        # 🔥 Persistent GPU buffer: warmup 중 기록된 쓰레기값 제거
        # CUDA graph 캡처 후 buffer는 이미 할당되어 있음 → zero_()로 초기화만
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'clear_persistent_buffers'):
            self.model.model.clear_persistent_buffers()
            print(f"  ✅ Persistent GPU buffers cleared (warmup data removed)")

        # 토크나이저 캐시
        self.tokenizer = self.llm.get_tokenizer()

        # Sampling params
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

        print(f"  ✅ Partial KV Cache Recomputation enabled")
        print(f"  ✅ Prefix caching enabled (KV cache persists between turns)")

    def _get_model_handle(self):
        """v0 엔진에서 progressive model 객체 가져오기 (02_universal.py 동일)"""
        engine = self.llm.llm_engine
        if hasattr(engine, "engine_core"):
            raise RuntimeError(
                "V1 engine detected. This script is v0-only. "
                "Use VLLM_USE_V1=0."
            )
        try:
            return engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError as exc:
            raise RuntimeError(
                "Could not resolve v0 model handle path."
            ) from exc

    # ----------------------------------------------------------------
    # 프롬프트 빌드
    # ----------------------------------------------------------------
    def _build_prompt(self) -> str:
        """대화 기록 → 전체 프롬프트 생성 (chat template 사용)"""
        # chat_template 사용 가능하면 사용
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    self.conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception:
                pass

        # Fallback: 단순 포맷
        prompt = ""
        for msg in self.conversation:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt

    # ----------------------------------------------------------------
    # 채팅
    # ----------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """
        사용자 입력 → 응답 생성.

        Prefix caching 활성화로 이전 대화의 KV cache가 재사용됩니다.
        Stage 전환 직후 첫 턴에는 partial KV recomputation이 자동 실행됩니다.
        """
        self.conversation.append({"role": "user", "content": user_input})

        prompt = self._build_prompt()

        # 프롬프트 길이 경고
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > 1800:  # max_model_len=2048, 여유 확보
            print(f"  [Warning] Conversation length ({len(token_ids)} tokens) "
                  f"approaching limit. Consider /reset.")

        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        self.conversation.append({"role": "assistant", "content": response})
        return response

    # ----------------------------------------------------------------
    # Stage 전환 (Partial KV Recomputation)
    # ----------------------------------------------------------------
    def advance_to_stage2(self) -> bool:
        """
        Stage 1 → Stage 2 전환 (prefetch → instant transition)

        Partial KV Recomputation:
        - Stage 전환 즉시 boundary layer 설정
        - 현재 대화를 즉시 재계산하여 partial recompute 실행
        - Boundary 이전: KV-only (빠름, 캐시된 hidden states 사용)
        - Boundary 이후: full forward (정확, 새 가중치 반영)
        """
        if self.current_stage >= 2:
            print("  Already at Stage 2 or higher.")
            return False

        stage_b_path = self.config.get("stage_b_checkpoint")
        if not stage_b_path or not os.path.exists(stage_b_path):
            print(f"  Stage B checkpoint not found: {stage_b_path}")
            return False

        # 🔥 Stage 전환 전: GPU persistent buffer → CPU cache 동기화
        self._sync_cache_before_transition()

        print("  [Stage 1 -> 2] Prefetching...")
        t0 = time.time()
        self.model.prefetch_stage2(stage_b_path)

        ready = self.model.wait_for_prefetch(timeout_s=120.0)
        if not ready:
            print("  Stage 2 prefetch failed or timed out.")
            return False

        # Instant transition (partial recompute boundary 자동 설정됨)
        transitioned = self.model.advance_to_stage2_instant(wait_if_needed=False)
        if not transitioned:
            print("  Stage 2 instant transition failed.")
            return False

        self.current_stage = 2
        elapsed = time.time() - t0

        stage_info = self.model.get_stage_info()
        print(f"  ✅ Stage 2 transition complete ({elapsed:.2f}s)")
        print(f"  Active layers: {len(stage_info['active_layers'])}, "
              f"Progress: {stage_info['activation_progress']}")
        
        # =========================================================
        # 🔥 추가할 부분: vLLM이 프롬프트를 자르지 못하게 캐시 강제 초기화
        # =========================================================
        if hasattr(self.llm, "reset_prefix_cache"):
            self.llm.reset_prefix_cache()

        # 🔥 CRITICAL: Trigger partial recompute NOW with current conversation
        # This ensures cached hidden states match the current prompt length
        self._trigger_partial_recompute()

        return True

    def advance_to_stage3(self) -> bool:
        """
        Stage 2 → Stage 3 전환 (prefetch → instant transition)

        Partial KV Recomputation:
        - Stage 전환 즉시 boundary layer 설정
        - 현재 대화를 즉시 재계산하여 partial recompute 실행
        """
        if self.current_stage < 2:
            print("  Must be at Stage 2 first. Use /stage2.")
            return False
        if self.current_stage >= 3:
            print("  Already at Stage 3.")
            return False

        stage_c_path = self.config.get("stage_c_checkpoint")
        if not stage_c_path or not os.path.exists(stage_c_path):
            print(f"  Stage C checkpoint not found: {stage_c_path}")
            return False

        # 🔥 Stage 전환 전: GPU persistent buffer → CPU cache 동기화
        self._sync_cache_before_transition()

        print("  [Stage 2 -> 3] Prefetching...")
        t0 = time.time()
        self.model.prefetch_stage3(stage_c_path)

        ready = self.model.wait_for_prefetch(timeout_s=120.0)
        if not ready:
            print("  Stage 3 prefetch failed or timed out.")
            return False

        # Instant transition (partial recompute boundary 자동 설정됨)
        transitioned = self.model.advance_to_stage3_instant(wait_if_needed=False)
        if not transitioned:
            print("  Stage 3 instant transition failed.")
            return False

        self.current_stage = 3
        elapsed = time.time() - t0

        stage_info = self.model.get_stage_info()
        print(f"  ✅ Stage 3 transition complete ({elapsed:.2f}s)")
        print(f"  Active layers: {len(stage_info['active_layers'])}, "
              f"Progress: {stage_info['activation_progress']}")

        # =========================================================
        # 🔥 추가할 부분: vLLM이 프롬프트를 자르지 못하게 캐시 강제 초기화
        # =========================================================
        if hasattr(self.llm, "reset_prefix_cache"):
            self.llm.reset_prefix_cache()

        # 🔥 CRITICAL: Trigger partial recompute NOW with current conversation
        self._trigger_partial_recompute()

        return True

    # ----------------------------------------------------------------
    # Persistent Buffer → CPU Cache 동기화
    # ----------------------------------------------------------------
    def _sync_cache_before_transition(self):
        """
        Stage 전환 직전: GPU persistent buffer에 누적된 hidden states를 CPU로 동기화.

        - Persistent buffer에는 prefill + decode의 hidden states가 index_copy_()로 누적
        - 현재 대화의 전체 토큰 수를 계산하여 해당 범위만 CPU로 복사
        - 이후 partial recompute에서 CPU cache를 사용
        """
        if not hasattr(self.model, 'model'):
            return

        inner_model = self.model.model
        if not hasattr(inner_model, 'sync_persistent_cache'):
            return

        # 현재 대화의 토큰 수 계산
        prompt = self._build_prompt()
        token_ids = self.tokenizer.encode(prompt)
        seq_len = len(token_ids)

        print(f"  [Sync] GPU buffer → CPU cache ({seq_len} tokens)")
        inner_model.sync_persistent_cache(seq_len)

    # ----------------------------------------------------------------
    # Partial Recompute 트리거
    # ----------------------------------------------------------------
    def _trigger_partial_recompute(self):
        """
        Stage 전환 직후 현재 대화를 재계산하여 partial KV recompute 실행.

        sync_persistent_cache() 기반 방식 (benchmark_chatbots.py 동일):
        - _sync_cache_before_transition()에서 이미 GPU hidden states →
          _layer_output_cache 저장 완료
        - reset_prefix_cache()로 stale KV blocks 제거
        - generate()로 partial recompute 트리거:
          * Boundary 이전 레이어: _layer_output_cache의 hidden states 재사용
          * Boundary 이후 레이어: full forward (새 가중치로 KV 재계산)
        - 완료 후 새 K,V가 prefix cache에 저장됨 → 다음 generate()에서 prefill 스킵
        """
        if len(self.conversation) == 0:
            print(f"  [PartialRecompute] No conversation history, skipping")
            return

        print(f"\n  [PartialRecompute] Triggering with current conversation...")
        print(f"  Conversation turns: {len(self.conversation) // 2}")

        # boundary 확인 (advance_to_stage*_instant에서 이미 설정됨)
        inner_model = self.model.model  # ProgressiveModelDualPath
        boundary = inner_model._partial_recompute_boundary
        if boundary is None:
            print(f"  [PartialRecompute] No boundary set, skipping")
            return

        # 현재 대화로 프롬프트 생성
        prompt = self._build_prompt()
        token_ids = self.tokenizer.encode(prompt)
        print(f"  Prompt tokens: {len(token_ids)}, boundary layer: {boundary}")

        # stale KV prefix cache blocks 제거
        self.llm.reset_prefix_cache()

        # 최소 생성으로 partial recompute 트리거 (forward pass + KV cache write)
        minimal_params = SamplingParams(temperature=0.0, max_tokens=1)

        print(f"  [PartialRecompute] Running generate()...")
        t0 = time.time()
        self.llm.generate([prompt], minimal_params)
        elapsed = time.time() - t0

        print(f"  ✅ Partial recomputation complete ({elapsed:.2f}s)")
        print(f"  📌 Front layers (0~{boundary-1}): hidden states from cache (GPU-only)")
        print(f"  📌 Back  layers ({boundary}~): full forward with new weights")
        print(f"  📌 Prefix cache populated → next generate() will skip prefill\n")

    # ----------------------------------------------------------------
    # 상태 / 리셋
    # ----------------------------------------------------------------
    def reset_conversation(self):
        """
        대화 기록 초기화.

        Hidden state cache도 함께 클리어됩니다.
        """
        self.conversation = []

        # Hidden state cache + persistent buffer 클리어
        if hasattr(self.model, 'model'):
            inner = self.model.model
            if hasattr(inner, 'clear_hidden_cache'):
                inner.clear_hidden_cache()
            if hasattr(inner, 'clear_persistent_buffers'):
                inner.clear_persistent_buffers()
            print("  Conversation, hidden cache, and persistent buffers reset.")
        else:
            print("  Conversation reset.")

    def print_status(self):
        """현재 상태 출력"""
        stage_info = self.model.get_stage_info()

        # Partial recompute 상태 확인
        partial_mode = False
        if hasattr(self.model, 'model'):
            inner_model = self.model.model
            if hasattr(inner_model, '_partial_recompute_boundary'):
                boundary = inner_model._partial_recompute_boundary
                partial_mode = boundary is not None

        print(f"\n  {'='*50}")
        print(f"  Model:    {self.model_name}")
        print(f"  Stage:    {self.current_stage}")
        print(f"  Active:   {len(stage_info['active_layers'])} layers")
        print(f"  Inactive: {len(stage_info['inactive_layers'])} layers")
        print(f"  Progress: {stage_info['activation_progress']}")
        print(f"  Turns:    {len(self.conversation) // 2}")
        print(f"  GPU Mem:  {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Partial Recompute: {'Active' if partial_mode else 'Idle'}")
        print(f"  {'='*50}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Progressive Serving Chatbot (Partial KV Cache Recomputation)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        default="llama",
        help="Model to use (default: llama)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Progressive Serving Chatbot - Partial KV Recomputation")
    print(f"  Model: {args.model}")
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    chatbot = ProgressiveChatbotPartial(args.model)

    print(f"\n{'='*60}")
    print(f"  Ready! (Stage {chatbot.current_stage})")
    print(f"  Commands: /stage2, /stage3, /status, /reset, /quit")
    print(f"  🚀 KV Cache persists between turns (prefix caching)")
    print(f"  🚀 Partial recomputation on stage transitions")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input(f"You [Stage {chatbot.current_stage}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        # 명령어 처리
        if user_input == "/quit":
            print("Bye!")
            break
        elif user_input == "/stage2":
            chatbot.advance_to_stage2()
            continue
        elif user_input == "/stage3":
            chatbot.advance_to_stage3()
            continue
        elif user_input == "/status":
            chatbot.print_status()
            continue
        elif user_input == "/reset":
            chatbot.reset_conversation()
            continue

        # 채팅
        t0 = time.time()
        response = chatbot.chat(user_input)
        elapsed = time.time() - t0

        print(f"Assistant [Stage {chatbot.current_stage}] ({elapsed:.1f}s): {response}\n")


if __name__ == "__main__":
    main()
