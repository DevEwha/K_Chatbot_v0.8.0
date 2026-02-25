#!/usr/bin/env python3
"""
Universal Progressive Serving - Quick Start Example
===================================================

Llama, Mistral, Qwen, Phi, Gemma 등 모든 Decoder-only 모델 지원
"""

import os
import sys

# vLLM v1: 멀티프로세싱 비활성화 (모델 직접 접근 필요)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
from vllm.model_executor.models.registry import ModelRegistry

# ============================================================================
# Step 1: Progressive Model 등록
# ============================================================================

# Import universal progressive model
sys.path.insert(0, "/home/devewha/K_Chatbot/Juwon/01_universal/progressive_serve")  # 실제 경로로 변경하세요
from progressive_for_causal_lm import ProgressiveForCausalLM

# ============================================================================
# Step 2: 모델 선택
# ============================================================================

import json

MODELS = {
    "llama":   "/home/devewha/K_Chatbot_v0.8.0/models/7b_results/pruning/A",
    "mistral": "/home/devewha/entropy_routing/25_mistral_results/pruning/A",
    "qwen":    "/path/to/qwen2-7b",
    "gemma":   "/path/to/gemma-7b",
    "phi":     "/path/to/phi-3-mini",
}

# 사용할 모델 선택 (여기만 바꾸면 됨)
MODEL_NAME = "llama"
MODEL_PATH = MODELS[MODEL_NAME]

# config.json에서 아키텍처 자동 읽어서 등록
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    _arch = json.load(f)["architectures"][0]

ModelRegistry.register_model(_arch, ProgressiveForCausalLM)
print(f"✅ Registered ProgressiveForCausalLM as: {_arch}")

print(f"\n{'='*60}")
print(f"Loading Model: {MODEL_NAME}")
print(f"Path: {MODEL_PATH}")
print(f"{'='*60}")

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    gpu_memory_utilization=0.4,
    max_model_len=2048,
    enforce_eager=False,  # CUDA Graph 활성화
)

print(f"✅ {MODEL_NAME.upper()} model loaded at Stage 1")


# ============================================================================
# Step 3: Stage 1 추론
# ============================================================================

prompts = [
    "Explain Paris",
    "The future of AI is ",
    "In a galaxy far, far away",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

print("\n" + "="*60)
print(f"STAGE 1 INFERENCE ({MODEL_NAME.upper()})")
print("="*60)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"\nPrompt: {prompt}")
    print(f"Output: {generated_text}")


# ============================================================================
# Step 4: 모델 상태 확인
# ============================================================================

print("\n" + "="*60)
print("MODEL STATUS")
print("="*60)

# vLLM v1 모델 접근 경로 (VLLM_ENABLE_V1_MULTIPROCESSING=0 필요)
model = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.model

stage_info = model.get_stage_info()
print(f"Model Type: {stage_info['model_type']}")
print(f"Current Stage: {stage_info['stage']}")
print(f"Active Layers: {len(stage_info['active_layers'])}")
print(f"Inactive Layers: {len(stage_info['inactive_layers'])}")
print(f"Progress: {stage_info['activation_progress']}")


# ============================================================================
# Step 5: Stage 2로 전환 (Optional)
# ============================================================================

# 각 모델별 checkpoint 경로 설정
STAGE_B_CHECKPOINTS = {
    "llama": "/home/devewha/K_Chatbot_v0.8.0/models/7b_results/pruning/checkpoints/stage2_layers_B.safetensors",
    "mistral": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage2_layers_B.safetensors",
    "qwen": "/path/to/qwen/layer_B.safetensors",
    # ... 다른 모델들 ...
}

if MODEL_NAME in STAGE_B_CHECKPOINTS:
    STAGE_B_CHECKPOINT = STAGE_B_CHECKPOINTS[MODEL_NAME]

    # Stage 1 서빙 직후 즉시 백그라운드 prefetch 시작
    model.prefetch_stage2(STAGE_B_CHECKPOINT)

    # Stage 1 추가 추론 (이 시간 동안 백그라운드에서 디스크 I/O 완료)
    print("\n" + "="*60)
    print(f"STAGE 1 INFERENCE (prefetch 진행 중) ({MODEL_NAME.upper()})")
    print("="*60)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

    # 즉각 전환 (GPU copy + alpha만, 디스크 I/O 없음)
    print("\n" + "="*60)
    print(f"TRANSITIONING TO STAGE 2 (instant) ({MODEL_NAME.upper()})...")
    print("="*60)
    model.advance_to_stage2_instant()
    print(f"✅ Transitioned to Stage 2")

    # Stage 2 추론
    print("\n" + "="*60)
    print(f"STAGE 2 INFERENCE ({MODEL_NAME.upper()})")
    print("="*60)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated_text}")


# ============================================================================
# Step 6: Stage 3로 전환 (Optional)
# ============================================================================

STAGE_C_CHECKPOINTS = {
    "llama": "/home/devewha/K_Chatbot_v0.8.0/models/7b_results/pruning/checkpoints/stage3_layers_C.safetensors",
    "mistral": "/acpl-ssd30/25_mistral_results/pruning/bundles/stage3_layers_C.safetensors",
    "qwen": "/path/to/qwen/layer_C.safetensors",
    # ... 다른 모델들 ...
}

if MODEL_NAME in STAGE_C_CHECKPOINTS:
    STAGE_C_CHECKPOINT = STAGE_C_CHECKPOINTS[MODEL_NAME]

    # Stage 2 서빙 직후 즉시 백그라운드 prefetch 시작
    model.prefetch_stage3(STAGE_C_CHECKPOINT)

    # Stage 2 추가 추론 (이 시간 동안 백그라운드에서 디스크 I/O 완료)
    print("\n" + "="*60)
    print(f"STAGE 2 INFERENCE (prefetch 진행 중) ({MODEL_NAME.upper()})")
    print("="*60)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(f"\nPrompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")

    # 즉각 전환
    print("\n" + "="*60)
    print(f"TRANSITIONING TO STAGE 3 (instant) ({MODEL_NAME.upper()})...")
    print("="*60)
    model.advance_to_stage3_instant()
    print(f"✅ Transitioned to Stage 3 (Full Model)")

    # Stage 3 추론
    print("\n" + "="*60)
    print(f"STAGE 3 INFERENCE ({MODEL_NAME.upper()})")
    print("="*60)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated_text}")


# ============================================================================
# Step 7: 최종 상태
# ============================================================================

print("\n" + "="*60)
print(f"FINAL STATUS ({MODEL_NAME.upper()})")
print("="*60)

model.print_status()

print("\n✅ Universal Progressive Serving Complete!")
print("="*60)


# ============================================================================
# 모델별 특수 처리 예제
# ============================================================================

print("\n" + "="*60)
print("MODEL-SPECIFIC FEATURES")
print("="*60)

# Qwen2: Tokenizer special handling
if MODEL_NAME == "qwen":
    print("Qwen2 detected - Using Qwen-specific tokenizer settings")
    # Add Qwen-specific logic here

# Mistral: Sliding window attention
elif MODEL_NAME == "mistral":
    print("Mistral detected - Sliding window attention enabled")
    # Add Mistral-specific logic here

# Phi: Custom vocab size
elif MODEL_NAME == "phi":
    print("Phi detected - Custom vocabulary handling")
    # Add Phi-specific logic here

# Gemma: RMSNorm epsilon
elif MODEL_NAME == "gemma":
    print("Gemma detected - Using Gemma-specific normalization")
    # Add Gemma-specific logic here

else:
    print(f"{MODEL_NAME} - Using default settings")

print("="*60)


# ============================================================================
# Advanced: Alpha Control
# ============================================================================

print("\n" + "="*60)
print("ADVANCED: MANUAL ALPHA CONTROL")
print("="*60)

# Get current alpha values
alphas = model.get_layer_alphas()
print(f"Current alpha distribution:")
print(f"  Min: {min(alphas):.2f}")
print(f"  Max: {max(alphas):.2f}")
print(f"  Mean: {sum(alphas)/len(alphas):.2f}")

# Set specific layer alpha (advanced usage)
# model.set_layer_alpha(10, 0.5)  # Layer 10 at 50% activation

print("="*60)