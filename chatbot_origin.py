#!/usr/bin/env python3
"""
Progressive Serving Chatbot - Origin (Full KV Recompute on Stage Transition)
=============================================================================

Interactive chatbot with progressive model serving (vLLM v0 engine).
Stage transitions on user command (/stage2, /stage3).

Uses origin_progressive_serve (partial recompute 최적화 없음).

**동작 방식:**
- 일반 대화: prefix caching으로 KV cache 턴 간 재사용 (빠름)
- Stage 전환: 맥락(conversation history) 유지 + KV cache 완전 초기화
  → 다음 generate() 호출 시 vLLM이 자동으로 전체 prefill 실행
  → 새 stage 가중치로 KV cache 자동 재구축 (별도 코드 불필요)

**chatbot_partial_cache.py 와의 차이:**
- progressive_serve → origin_progressive_serve 사용
- KV Snapshot / Partial recompute 없음
- Stage 전환 시 reset_prefix_cache()만 호출 → 다음 turn에서 full prefill

Usage:
  python chatbot_origin.py --model llama
  python chatbot_origin.py --model mistral

  Commands during chat:
    /stage2  - Transition to Stage 2 (KV cache cleared, full recompute on next turn)
    /stage3  - Transition to Stage 3 (KV cache cleared, full recompute on next turn)
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

# origin_progressive_serve: partial recompute 없는 원본 구현
_ORIGIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "origin_progressive_serve")
sys.path.insert(0, _ORIGIN_DIR)

# progressive_model_dual_path.py 내부에 hardcoded sys.path가 있어서
# model_config를 먼저 캐싱해두면 해당 경로를 무시하고 올바른 버전이 사용됨
import model_config  # noqa: F401  (origin_progressive_serve/model_config.py)
from progressive_for_causal_lm import ProgressiveForCausalLM


# ============================================================================
# 모델 설정
# ============================================================================

MODELS = {
    "llama": {
        "progressive_path":   "/acpl-ssd32/llama2-7b/base",
        "stage_b_checkpoint": "/acpl-ssd32/llama2-7b/merged/kd-lora/checkpoints/stage2_layers_B_merged.safetensors",
        "stage_c_checkpoint": "/acpl-ssd32/llama2-7b/merged/kd-lora/checkpoints/stage3_layers_C_merged.safetensors",
    },
    "falcon": {
        "progressive_path": "/acpl-ssd32/falcon-7b/base",
        "stage_b_checkpoint": "/acpl-ssd32/falcon-7b/merged/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd32/falcon-7b/merged/stage3_layers_C.safetensors",
    },
    # "falcon": {
    #     "progressive_path": "/acpl-ssd32/falcon-7b/pruned/A",
    #     "stage_b_checkpoint": "/acpl-ssd32/falcon-7b/pruned/stage2_layers_B.safetensors",
    #     "stage_c_checkpoint": "/acpl-ssd32/falcon-7b/pruned/stage3_layers_C.safetensors",
    # },
}


# ============================================================================
# Chatbot - Full KV Recompute on Stage Transition
# ============================================================================

class ProgressiveChatbotOrigin:
    """
    Origin progressive serving 기반 대화형 챗봇.

    핵심 기능:
    - 일반 대화: prefix caching으로 KV cache 재사용 (빠름)
    - Stage 전환: 맥락 유지 + KV cache 완전 초기화
      → 다음 generate() 호출 시 vLLM이 새 stage 가중치로 KV cache 자동 재구축
    - Partial recompute / KV snapshot 없음 (origin 버전)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = MODELS[model_name]
        self.current_stage = 1
        self.conversation = []  # [{"role": "user"/"assistant", "content": "..."}]

        model_path = self.config["progressive_path"]
        with open(os.path.join(model_path, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        ModelRegistry.register_model(arch, ProgressiveForCausalLM)
        print(f"  Registered ProgressiveForCausalLM as: {arch}")

        print(f"\n  Loading {model_name} Stage 1...")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.4,
            max_model_len=2048,
            enforce_eager=False,
            # 일반 대화 중 KV cache 재사용 (prefix caching)
            # Stage 전환 시 reset_prefix_cache()로 초기화
            enable_prefix_caching=True,
        )

        self.model = self._get_model_handle()
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
        )

        print(f"  ✅ Prefix caching enabled (KV cache persists between turns)")
        print(f"  ✅ Stage transitions clear KV cache → full recompute on next turn")

    def _get_model_handle(self):
        """v0 엔진에서 progressive model 객체 가져오기"""
        engine = self.llm.llm_engine
        if hasattr(engine, "engine_core"):
            raise RuntimeError(
                "V1 engine detected. This script is v0-only. Use VLLM_USE_V1=0."
            )
        try:
            return engine.model_executor.driver_worker.worker.model_runner.model
        except AttributeError as exc:
            raise RuntimeError("Could not resolve v0 model handle path.") from exc

    # ----------------------------------------------------------------
    # 프롬프트 빌드
    # ----------------------------------------------------------------
    def _build_prompt(self) -> str:
        """대화 기록 → 전체 프롬프트 (chat template 우선)"""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    self.conversation,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        # Fallback: 단순 포맷
        prompt = ""
        for msg in self.conversation:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            prompt += prefix + msg["content"] + "\n"
        return prompt + "Assistant: "

    # ----------------------------------------------------------------
    # 채팅
    # ----------------------------------------------------------------
    def chat(self, user_input: str) -> str:
        """
        사용자 입력 → 응답 생성.

        Prefix caching 활성화로 이전 대화의 KV cache가 재사용됩니다.
        Stage 전환 직후 첫 turn에는 cache miss → vLLM이 full prefill 자동 실행.
        """
        self.conversation.append({"role": "user", "content": user_input})
        prompt = self._build_prompt()

        token_ids = self.tokenizer.encode(prompt)
        if len(token_ids) > 1800:
            print(f"  [Warning] Conversation length ({len(token_ids)} tokens) "
                  f"approaching limit. Consider /reset.")

        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        self.conversation.append({"role": "assistant", "content": response})
        return response

    # ----------------------------------------------------------------
    # KV Cache 초기화
    # ----------------------------------------------------------------
    def _clear_kv_cache(self) -> None:
        """
        Stage 전환 후 stale KV cache 전체 초기화.

        Stage 전환으로 model weights가 변경되므로 기존 KV blocks는 무효.
        reset_prefix_cache()로 모든 cached blocks를 evict한 뒤,
        다음 generate() 호출 시 vLLM이 새 stage weights로 full prefill을
        자동 실행하여 KV cache를 재구축합니다.

        별도의 재계산 코드가 필요 없습니다 — 다음 turn에서 자동 처리됩니다.
        """
        try:
            success = self.llm.reset_prefix_cache()
            if success:
                print(f"  [KVCache] ✅ All prefix cache blocks evicted")
                print(f"  [KVCache]    Next turn will run full prefill "
                      f"with new Stage {self.current_stage} weights")
            else:
                print(f"  [KVCache] ⚠️  reset_prefix_cache() returned False "
                      f"(blocks may still be in use)")
        except Exception as e:
            print(f"  [KVCache] ⚠️  Could not clear prefix cache: {e}")

    # ----------------------------------------------------------------
    # Stage 전환
    # ----------------------------------------------------------------
    def advance_to_stage2(self) -> bool:
        """
        Stage 1 → Stage 2 전환.

        동작:
        1. prefetch_stage2(): checkpoint를 백그라운드에서 CPU에 로드
        2. wait_for_prefetch(): 완료 대기
        3. advance_to_stage2_instant(): GPU weight copy + alpha 변경
        4. _clear_kv_cache(): stale KV blocks 퇴출
           → 다음 turn에서 vLLM이 full prefill 자동 실행
        """
        if self.current_stage >= 2:
            print("  Already at Stage 2 or higher.")
            return False

        stage_b_path = self.config.get("stage_b_checkpoint")
        if not stage_b_path or not os.path.exists(stage_b_path):
            print(f"  Stage B checkpoint not found: {stage_b_path}")
            return False

        print("  [Stage 1 → 2] Prefetching B layers...")
        t0 = time.time()
        self.model.prefetch_stage2(stage_b_path)

        ready = self.model.wait_for_prefetch(timeout_s=120.0)
        if not ready:
            print("  Stage 2 prefetch failed or timed out.")
            return False

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

        # KV cache 초기화 (다음 turn에서 자동 full prefill)
        self._clear_kv_cache()
        return True

    def advance_to_stage3(self) -> bool:
        """
        Stage 2 → Stage 3 전환.

        동작:
        1. prefetch_stage3(): checkpoint를 백그라운드에서 CPU에 로드
        2. wait_for_prefetch(): 완료 대기
        3. advance_to_stage3_instant(): GPU weight copy + alpha 변경
        4. _clear_kv_cache(): stale KV blocks 퇴출
           → 다음 turn에서 vLLM이 full prefill 자동 실행
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

        print("  [Stage 2 → 3] Prefetching C layers...")
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
        print(f"  ✅ Stage 3 transition complete ({elapsed:.2f}s)")
        print(f"  Active layers: {len(stage_info['active_layers'])}, "
              f"Progress: {stage_info['activation_progress']}")

        # KV cache 초기화
        self._clear_kv_cache()
        return True

    # ----------------------------------------------------------------
    # 상태 / 리셋
    # ----------------------------------------------------------------
    def reset_conversation(self):
        """대화 기록 초기화"""
        self.conversation = []
        print("  Conversation reset.")

    def print_status(self):
        """현재 상태 출력"""
        stage_info = self.model.get_stage_info()
        print(f"\n  {'='*50}")
        print(f"  Model:    {self.model_name}")
        print(f"  Stage:    {self.current_stage}")
        print(f"  Active:   {len(stage_info['active_layers'])} layers")
        print(f"  Inactive: {len(stage_info['inactive_layers'])} layers")
        print(f"  Progress: {stage_info['activation_progress']}")
        print(f"  Turns:    {len(self.conversation) // 2}")
        print(f"  GPU Mem:  {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  {'='*50}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Progressive Serving Chatbot (Origin / Full KV Recompute)"
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
    print("Progressive Serving Chatbot  [Origin / Full Recompute]")
    print(f"  Model: {args.model}")
    print(f"  GPU:   {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    chatbot = ProgressiveChatbotOrigin(args.model)

    print(f"\n{'='*60}")
    print(f"  Ready! (Stage {chatbot.current_stage})")
    print(f"  Commands: /stage2, /stage3, /status, /reset, /quit")
    print(f"  ✅ KV cache reused between turns (prefix caching)")
    print(f"  🔄 Stage transition → KV cache cleared → auto full recompute next turn")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input(f"You [Stage {chatbot.current_stage}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

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

        t0 = time.time()
        response = chatbot.chat(user_input)
        elapsed = time.time() - t0

        print(f"Assistant [Stage {chatbot.current_stage}] ({elapsed:.1f}s): {response}\n")


if __name__ == "__main__":
    main()
