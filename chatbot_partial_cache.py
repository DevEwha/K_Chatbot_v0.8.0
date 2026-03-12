#!/usr/bin/env python3
"""
Progressive Serving Chatbot - True KV Block Surgery
=====================================================

Interactive chatbot with progressive model serving (vLLM v0 engine).
Stage transitions on user command (/stage2, /stage3).

**핵심 동작 (True KV Block Surgery):**
- KV Cache를 턴 사이에 유지 (prefix caching)
- Stage 전환 시:
  * Lower layers (boundary 이전): KV cache 전혀 손대지 않음 (가중치 동일 → KV 동일)
  * Upper layers (boundary 이후): 동일 physical block에 새 가중치로 KV 덮어쓰기
  * vLLM prefix cache hash→block 매핑 유지 → 다음 generate()에서 prefill 완전 스킵!
- Fallback: surgery 실패 시 reset_prefix_cache + partial recompute

Usage:
  python chatbot_partial_cache.py --model llama
  python chatbot_partial_cache.py --model mistral

  Commands during chat:
    /stage2  - Transition to Stage 2 (KV block surgery)
    /stage3  - Transition to Stage 3 (KV block surgery)
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

# vLLM v0.8.0 workaround: custom 모델을 멀티모달로 잘못 판단하여
# prefix caching이 비활성화되는 버그를 우회.
import vllm.config
vllm.config.ModelConfig.is_multimodal_model = property(lambda self: False)

# ============================================================================
# 모델 설정
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
# Chatbot with True KV Block Surgery
# ============================================================================

class ProgressiveChatbotPartial:
    """
    Progressive Serving 기반 대화형 챗봇 (True KV Block Surgery)

    핵심 기능:
    - KV Cache 턴 간 유지 (vLLM prefix caching)
    - Stage 전환 시 KV block surgery:
      * Lower layers: KV 완전 보존 (연산 없음)
      * Upper layers: 동일 physical block에 새 KV 덮어쓰기
      * 다음 generate()에서 prefill 스킵 → 진짜 zero-overhead transition
    - Surgery 실패 시 fallback: reset + partial recompute
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
            enforce_eager=False,         # CUDA graph 활성화
            enable_prefix_caching=True,  # KV cache 턴 간 유지
        )

        # v0 엔진 모델 핸들
        self.model = self._get_model_handle()

        # Warmup 데이터 제거
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

        print(f"  ✅ True KV Block Surgery enabled")
        print(f"  ✅ Prefix caching enabled (KV cache persists between turns)")

    def _get_model_handle(self):
        """v0 엔진에서 progressive model 객체 가져오기"""
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
        """대화 기록 → 전체 프롬프트 생성"""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    self.conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback
        prompt = ""
        for msg in self.conversation:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            else:
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt

    def _get_current_seq_len(self) -> int:
        """현재 시퀀스 길이 반환 (vLLM 내부 저장값 우선)"""
        inner = self.model.model
        if (inner._surgery_seq_lens_tensor is not None
                and inner._surgery_seq_lens_tensor.numel() > 0):
            return int(inner._surgery_seq_lens_tensor[0].item())
        # Fallback: tokenizer로 직접 계산
        if self.conversation:
            prompt = self._build_prompt()
            return len(self.tokenizer.encode(prompt))
        return 0

    # ----------------------------------------------------------------
    # 채팅
    # ----------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """
        사용자 입력 → 응답 생성.

        Prefix caching으로 이전 대화의 KV cache가 재사용됩니다.
        """
        self.conversation.append({"role": "user", "content": user_input})

        prompt = self._build_prompt()

        # 프롬프트 길이 경고
        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > 1800:
            print(f"  [Warning] Conversation length ({len(token_ids)} tokens) "
                  f"approaching limit. Consider /reset.")

        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()

        self.conversation.append({"role": "assistant", "content": response})
        return response

    # ----------------------------------------------------------------
    # Stage 전환 (True KV Block Surgery)
    # ----------------------------------------------------------------
    def advance_to_stage2(self) -> bool:
        """
        Stage 1 → Stage 2 전환

        1. 가중치 prefetch (백그라운드)
        2. 가중치 instant activation (GPU copy)
        3. KV Block Surgery:
           - Lower layers: KV 그대로 (연산 없음)
           - Upper layers: 동일 block에 새 KV 덮어쓰기
           - prefix cache 유지 → 다음 generate() prefill 스킵
        4. Surgery 실패 시: reset_prefix_cache + partial recompute fallback
        """
        if self.current_stage >= 2:
            print("  Already at Stage 2 or higher.")
            return False

        stage_b_path = self.config.get("stage_b_checkpoint")
        if not stage_b_path or not os.path.exists(stage_b_path):
            print(f"  Stage B checkpoint not found: {stage_b_path}")
            return False

        inner_model = self.model.model  # ProgressiveModelDualPath
        b_indices = self.model._get_b_indices()
        boundary = self.model.get_recompute_boundary(b_indices)
        has_conversation = len(self.conversation) > 0

        # Surgery fallback을 위해 hidden state 캐시 미리 준비
        # (surgery 실패 시 partial recompute에서 사용)
        if has_conversation and boundary is not None:
            seq_len = self._get_current_seq_len()
            if seq_len > 0 and hasattr(inner_model, 'sync_persistent_cache'):
                print(f"  [Prep] Caching {seq_len} tokens (fallback용)")
                inner_model.sync_persistent_cache(seq_len)

        print("  [Stage 1 -> 2] Prefetching...")
        t0 = time.time()
        self.model.prefetch_stage2(stage_b_path)

        ready = self.model.wait_for_prefetch(timeout_s=120.0)
        if not ready:
            print("  Stage 2 prefetch failed or timed out.")
            return False

        # Instant transition (GPU weight copy + alpha 변경)
        transitioned = self.model.advance_to_stage2_instant(wait_if_needed=False)
        if not transitioned:
            print("  Stage 2 instant transition failed.")
            return False

        self.current_stage = 2
        elapsed = time.time() - t0

        stage_info = self.model.get_stage_info()
        print(f"  ✅ Stage 2 weight activation complete ({elapsed:.2f}s)")
        print(f"  Active layers: {len(stage_info['active_layers'])}, "
              f"Progress: {stage_info['activation_progress']}")

        # ── KV Block Surgery ──
        if has_conversation and boundary is not None:
            surgery_ok = inner_model.inject_upper_layer_kv(boundary)

            if not surgery_ok:
                # Fallback: reset prefix cache + partial recompute
                print("  [Stage2] ⚠️ Surgery 실패, partial recompute fallback 실행")
                inner_model.set_partial_recompute(boundary)
                self.llm.reset_prefix_cache()
                self._trigger_partial_recompute()
            # else: surgery 성공 → prefix cache 유지 → 다음 generate()에서 prefill 스킵

        elif not has_conversation:
            # 대화 없음: prefix cache만 초기화 (upper layer KV = zeros였음)
            self.llm.reset_prefix_cache()

        return True

    def advance_to_stage3(self) -> bool:
        """
        Stage 2 → Stage 3 전환

        동일 KV Block Surgery 방식 적용.
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

        inner_model = self.model.model
        c_indices = self.model._get_c_indices()
        boundary = self.model.get_recompute_boundary(c_indices)
        has_conversation = len(self.conversation) > 0

        # Surgery fallback용 hidden state 캐시 준비
        if has_conversation and boundary is not None:
            seq_len = self._get_current_seq_len()
            if seq_len > 0 and hasattr(inner_model, 'sync_persistent_cache'):
                print(f"  [Prep] Caching {seq_len} tokens (fallback용)")
                inner_model.sync_persistent_cache(seq_len)

        print("  [Stage 2 -> 3] Prefetching...")
        t0 = time.time()
        self.model.prefetch_stage3(stage_c_path)

        ready = self.model.wait_for_prefetch(timeout_s=120.0)
        if not ready:
            print("  Stage 3 prefetch failed or timed out.")
            return False

        transitioned = self.model.advance_to_stage3_instant(wait_if_needed=False)
        if not transitioned:
            print("  Stage 3 instant transition failed.")
            return False

        self.current_stage = 3
        elapsed = time.time() - t0

        stage_info = self.model.get_stage_info()
        print(f"  ✅ Stage 3 weight activation complete ({elapsed:.2f}s)")
        print(f"  Active layers: {len(stage_info['active_layers'])}, "
              f"Progress: {stage_info['activation_progress']}")

        # ── KV Block Surgery ──
        if has_conversation and boundary is not None:
            surgery_ok = inner_model.inject_upper_layer_kv(boundary)

            if not surgery_ok:
                print("  [Stage3] ⚠️ Surgery 실패, partial recompute fallback 실행")
                inner_model.set_partial_recompute(boundary)
                self.llm.reset_prefix_cache()
                self._trigger_partial_recompute()

        elif not has_conversation:
            self.llm.reset_prefix_cache()

        return True

    # ----------------------------------------------------------------
    # Partial Recompute (fallback)
    # ----------------------------------------------------------------
    def _trigger_partial_recompute(self):
        """
        Surgery 실패 시 fallback: 현재 대화로 partial KV recompute 실행.

        reset_prefix_cache() 이후에 호출 (새 블록 할당 필요).
        """
        if len(self.conversation) == 0:
            print(f"  [PartialRecompute] No conversation history, skipping")
            return

        inner_model = self.model.model
        boundary = inner_model._partial_recompute_boundary
        if boundary is None:
            print(f"  [PartialRecompute] No boundary set, skipping")
            return

        print(f"\n  [PartialRecompute] Running fallback partial recompute...")

        prompt = self._build_prompt()
        token_ids = self.tokenizer.encode(prompt)
        print(f"  Prompt tokens: {len(token_ids)}, boundary layer: {boundary}")

        minimal_params = SamplingParams(temperature=0.0, max_tokens=1)

        t0 = time.time()
        self.llm.generate([prompt], minimal_params)
        elapsed = time.time() - t0

        print(f"  ✅ Partial recompute complete ({elapsed:.2f}s)")
        print(f"  📌 Lower (0~{boundary-1}): KV from cache")
        print(f"  📌 Upper ({boundary}~): full forward with new weights\n")

    # ----------------------------------------------------------------
    # 상태 / 리셋
    # ----------------------------------------------------------------
    def reset_conversation(self):
        """대화 기록 초기화."""
        self.conversation = []

        if hasattr(self.model, 'model'):
            inner = self.model.model
            if hasattr(inner, 'clear_hidden_cache'):
                inner.clear_hidden_cache()
            if hasattr(inner, 'clear_persistent_buffers'):
                inner.clear_persistent_buffers()
            # _surgery_block_tables / _surgery_seq_lens_tensor는 CUDA graph input buffer의 view임
            # → 초기화 불필요: 다음 decode step에서 자동으로 새 대화의 값으로 업데이트됨
            print("  Conversation and hidden caches reset.")
        else:
            print("  Conversation reset.")

    def print_status(self):
        """현재 상태 출력"""
        stage_info = self.model.get_stage_info()
        inner_model = self.model.model

        partial_mode = inner_model._partial_recompute_boundary is not None
        # Surgery state는 CUDA graph view → has_conversation일 때만 유효
        has_valid_surgery_state = (
            inner_model._surgery_block_tables is not None
            and len(self.conversation) > 0
        )

        print(f"\n  {'='*50}")
        print(f"  Model:    {self.model_name}")
        print(f"  Stage:    {self.current_stage}")
        print(f"  Active:   {len(stage_info['active_layers'])} layers")
        print(f"  Inactive: {len(stage_info['inactive_layers'])} layers")
        print(f"  Progress: {stage_info['activation_progress']}")
        print(f"  Turns:    {len(self.conversation) // 2}")
        print(f"  GPU Mem:  {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  Surgery Ready: {'Yes' if has_valid_surgery_state else 'No (no conversation yet)'}")
        if partial_mode:
            print(f"  Partial Recompute: Active (boundary={inner_model._partial_recompute_boundary})")
        print(f"  {'='*50}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Progressive Serving Chatbot (True KV Block Surgery)"
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
    print("Progressive Serving Chatbot - True KV Block Surgery")
    print(f"  Model: {args.model}")
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    chatbot = ProgressiveChatbotPartial(args.model)

    print(f"\n{'='*60}")
    print(f"  Ready! (Stage {chatbot.current_stage})")
    print(f"  Commands: /stage2, /stage3, /status, /reset, /quit")
    print(f"  🔪 KV Block Surgery: upper layers overwritten in-place")
    print(f"  🚀 Prefix cache preserved → next generate() skips prefill")
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
