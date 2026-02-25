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

        # KV snapshot을 위해 vLLM이 실제 처리한 정확한 token IDs 저장
        # (prompt_token_ids + generated_token_ids)
        # _build_prompt()로 재토크나이징하면 chat template 차이로 해시 불일치 발생
        self._last_generate_token_ids = (
            list(outputs[0].prompt_token_ids) +
            list(outputs[0].outputs[0].token_ids)
        )

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
    # KV Snapshot (Stage 전환 전 GPU 캐시 직접 읽기)
    # ----------------------------------------------------------------
    def _save_kv_snapshot(self, boundary_layer_idx: int):
        """
        Stage 전환 직전, GPU KV 캐시 블록에서 layers 0~boundary-1의 K,V를
        직접 읽어 CPU에 저장.

        원리:
        - Stage N과 Stage N+1에서 layers 0~boundary-1의 weights는 동일
        - 따라서 해당 레이어들의 K,V 값도 동일
        - 전환 전에 GPU block에서 K,V를 읽으면 재계산 없이 재사용 가능
        - Full blocks (block_size=16 단위)만 저장 (hash로 block_id 조회 가능)
        - Partial last block은 QKV_write_only로 처리 (K,V 계산하지만 attention 없음)

        반환:
            (snapshot, num_full_tokens) 또는 (None, 0) on failure
        """
        try:
            from vllm.core.block.prefix_caching_block import PrefixCachingBlock
            import torch

            # 1. 토큰 ID 결정
            #    핵심: _build_prompt()로 재토크나이징하면 chat template 포맷 차이로
            #    vLLM이 캐싱한 실제 token IDs와 해시 불일치 발생.
            #    → chat()에서 저장한 실제 token IDs 사용 (prompt_ids + generated_ids)
            if hasattr(self, '_last_generate_token_ids') and self._last_generate_token_ids:
                token_ids = self._last_generate_token_ids
                print(f"  [KVSnapshot] Using actual generate token IDs "
                      f"({len(token_ids)} tokens, exact match with vLLM cache)")
            else:
                prompt = self._build_prompt()
                token_ids = self.tokenizer.encode(prompt)
                print(f"  [KVSnapshot] ⚠️  No saved token IDs, "
                      f"falling back to _build_prompt() ({len(token_ids)} tokens)")

            total_tokens = len(token_ids)

            # 2. Block 설정
            block_size = self.llm.llm_engine.cache_config.block_size
            num_full_blocks = total_tokens // block_size
            num_full_tokens = num_full_blocks * block_size

            if num_full_blocks == 0:
                print(f"  [KVSnapshot] ⚠️  No full blocks "
                      f"(total_tokens={total_tokens} < block_size={block_size})")
                return None, 0

            # 3. Block allocator에서 cached_blocks 가져오기
            #    경로: LLMEngine → Scheduler → SelfAttnBlockSpaceManager
            #          → block_allocator (CpuGpuBlockAllocator)
            #          → _allocators[Device.GPU] (PrefixCachingBlockAllocator)
            from vllm.utils import Device
            scheduler = self.llm.llm_engine.scheduler[0]
            gpu_alloc = scheduler.block_manager.block_allocator._allocators[Device.GPU]
            cached_blocks = gpu_alloc._cached_blocks  # Dict[hash, block_id]

            # 4. 토큰 → 블록 해시 계산 → block_id 조회 (순서대로)
            #    중간에 block이 없으면 abort하지 않고 그 시점까지만 사용
            block_ids = []
            prev_hash = None
            for i in range(num_full_blocks):
                chunk = token_ids[i * block_size: (i + 1) * block_size]
                bh = PrefixCachingBlock.hash_block_tokens(
                    is_first_block=(i == 0),
                    prev_block_hash=prev_hash,
                    cur_block_token_ids=chunk,
                    extra_hash=None,
                )
                bid = cached_blocks.get(bh)
                if bid is None:
                    print(f"  [KVSnapshot] ⚠️  Block {i} (tokens {i*block_size}~"
                          f"{(i+1)*block_size-1}) not found → "
                          f"using {i} blocks ({i*block_size} tokens)")
                    # abort하지 않고 찾은 블록까지만 사용
                    num_full_blocks = i
                    num_full_tokens = i * block_size
                    break
                block_ids.append(bid)
                prev_hash = bh

            if num_full_blocks == 0:
                print(f"  [KVSnapshot] ⚠️  No blocks found → fallback to QKV_write_only")
                return None, 0

            # 5. GPU KV 캐시에서 K,V 읽기 (layers 0~boundary-1)
            #    각 레이어의 Attention 객체: layer_wrapper.layer.self_attn.attn
            #    kv_cache[ve][0]: key cache [num_blocks, block_size, num_kv_heads, head_size]
            #    kv_cache[ve][1]: val cache [num_blocks, block_size, num_kv_heads, head_size]
            inner_model = self.model.model  # ProgressiveModelDualPath
            snapshot = {}

            for layer_idx in range(boundary_layer_idx):
                layer_wrapper = inner_model.layers[layer_idx]
                if not hasattr(layer_wrapper.layer, 'self_attn'):
                    continue
                attn_obj = layer_wrapper.layer.self_attn.attn  # Attention (vllm)
                kv = attn_obj.kv_cache[0]  # virtual engine 0
                # kv shape: [2, num_blocks, block_size, num_kv_heads, head_size]

                dev = kv.device
                bids_t = torch.tensor(block_ids, dtype=torch.long, device=dev)

                key_cache = kv[0]  # [num_blocks, block_size, num_kv_heads, head_size]
                val_cache = kv[1]

                # [num_full_blocks, block_size, num_kv_heads, head_size]
                k_blocks = key_cache[bids_t]
                v_blocks = val_cache[bids_t]

                # [num_full_tokens, num_kv_heads, head_size] → CPU
                k_all = k_blocks.reshape(num_full_tokens, *key_cache.shape[2:]).cpu()
                v_all = v_blocks.reshape(num_full_tokens, *val_cache.shape[2:]).cpu()

                snapshot[layer_idx] = (k_all, v_all)

            print(f"  [KVSnapshot] ✅ Snapshot saved: {len(snapshot)} layers × "
                  f"{num_full_blocks} blocks × {block_size} = {num_full_tokens} tokens  "
                  f"[GPU→CPU memcopy, 0 FLOPs]")
            return snapshot, num_full_tokens

        except Exception as e:
            print(f"  [KVSnapshot] ⚠️  Failed to save snapshot: {e}")
            import traceback
            traceback.print_exc()
            return None, 0

    # ----------------------------------------------------------------
    # Partial Recompute 트리거
    # ----------------------------------------------------------------
    def _clear_kv_prefix_cache(self) -> None:
        """
        Stage 전환 후 stale KV prefix cache blocks 퇴출.

        Stage 전환 시 weights가 변경되므로 기존에 캐싱된 KV blocks는
        잘못된 값을 가질 수 있습니다. _trigger_partial_recompute() 전에
        반드시 호출하여 stale blocks를 퇴출한 후 올바른 K,V로 다시 채웁니다.

        vLLM LLM.reset_prefix_cache() → LLMEngine → Scheduler → BlockManager 순으로
        내부적으로 PrefixCachingBlockAllocator._cached_blocks.clear()를 호출합니다.
        """
        try:
            success = self.llm.reset_prefix_cache()
            if success:
                print(f"  [KVCache] ✅ Prefix cache evicted (stale blocks removed)")
            else:
                print(f"  [KVCache] ⚠️ reset_prefix_cache() returned False "
                      f"(prefix caching may not be active or blocks still in use)")
        except Exception as e:
            print(f"  [KVCache] ⚠️ Could not clear prefix cache: {e}")

    def _trigger_partial_recompute(self):
        """
        Stage 전환 직후 현재 대화를 재계산하여 partial KV recompute 실행.

        핵심 원리 (KV Snapshot 최적화):
        ┌─────────────────────────────────────────────────────────────────┐
        │ Front layers (0~boundary-1): weights 불변 → K,V 동일           │
        │   STEP A: GPU KV 캐시에서 K,V를 직접 읽어 CPU에 저장 (snapshot) │
        │   STEP B: reset_prefix_cache() → stale blocks 퇴출             │
        │   STEP C: generate() → partial recompute:                      │
        │     - Full blocks: snapshot memcopy → KV cache (0 FLOPs)       │
        │     - Partial block: QKV proj + rope → KV cache (flash_attn 생략)│
        │   결과: front layers K,V가 재계산 없이 새 blocks에 복원됨       │
        │                                                                 │
        │ Back layers (boundary~end): weights 변경됨                      │
        │   STEP C에서 full forward → 새 K,V 계산 및 저장                 │
        └─────────────────────────────────────────────────────────────────┘

        동작 순서:
        1. _sync_cache_before_transition(): GPU hidden states → CPU cache
        2. Stage 전환 → boundary 설정 (advance_to_stage*_instant)
        3. 🔥 _save_kv_snapshot(): GPU KV 블록 직접 읽기 → CPU 저장
        4. 🔥 _clear_kv_prefix_cache(): stale blocks 퇴출 (reset_prefix_cache)
        5. model.set_kv_snapshot(): snapshot을 progressive model에 전달
        6. generate() → forward() partial recompute 실행:
           - Front: snapshot/QKV_write_only → K,V 복원 (hidden states from CPU)
           - Back: full forward → K,V 재계산 (hidden states computed)
        7. 새 K,V가 prefix cache에 저장됨 → 다음 generate()에서 prefill 스킵
        """
        if len(self.conversation) == 0:
            print(f"  [PartialRecompute] No conversation history, skipping")
            return

        print(f"\n  [PartialRecompute] Triggering with current conversation...")
        print(f"  Conversation turns: {len(self.conversation) // 2}")

        # 🔥 Step 1: boundary 확인 (set_partial_recompute()에서 이미 설정됨)
        inner_model = self.model.model  # ProgressiveModelDualPath
        boundary = inner_model._partial_recompute_boundary
        if boundary is None:
            print(f"  [PartialRecompute] No boundary set, skipping")
            return

        # 🔥 Step 2: GPU KV 캐시에서 K,V snapshot 저장 (reset 전에 해야 함!)
        print(f"  [Step 2] Saving KV snapshot from GPU cache (layers 0~{boundary-1})...")
        snapshot, num_full_tokens = self._save_kv_snapshot(boundary)

        # 🔥 Step 3: Stale KV prefix cache blocks 퇴출
        print(f"  [Step 3] Evicting stale KV prefix cache blocks...")
        self._clear_kv_prefix_cache()

        # 🔥 Step 4: Snapshot을 progressive model에 전달
        print(f"  [Step 4] Passing KV snapshot to progressive model...")
        inner_model.set_kv_snapshot(snapshot, num_full_tokens)

        # 현재 대화 기록으로 프롬프트 생성
        prompt = self._build_prompt()
        token_ids = self.tokenizer.encode(prompt)
        print(f"  Prompt tokens: {len(token_ids)} "
              f"(full-block tokens: {num_full_tokens}, "
              f"partial: {len(token_ids) - num_full_tokens})")

        # 최소 생성으로 partial recompute 트리거
        # max_tokens=1: forward pass + KV cache write만 필요
        minimal_params = SamplingParams(temperature=0.0, max_tokens=1)

        print(f"  [Step 5] Running partial recompute generate()...")
        t0 = time.time()

        # generate() 호출:
        # - prefix cache miss (reset 후) → vLLM이 prefill 실행
        # - forward()에서 partial recompute 모드 동작 (per-layer 로그 출력됨):
        #   * Front layers: KV snapshot memcopy + QKV_write_only fallback
        #   * Back layers: full forward (새 가중치)
        # - 완료 후 K,V가 prefix cache에 저장됨 → 다음 generate()에서 prefill 스킵
        self.llm.generate([prompt], minimal_params)

        elapsed = time.time() - t0
        print(f"  ✅ Partial recomputation complete ({elapsed:.2f}s)")
        print(f"  📌 Front layers: K,V REUSED from snapshot (0 FLOPs for full blocks)")
        print(f"  📌 Back  layers: K,V RECOMPUTED with new weights")
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
