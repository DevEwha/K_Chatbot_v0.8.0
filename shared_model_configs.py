#!/usr/bin/env python3
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


DEFAULT_MODEL = "llama2-7b"

# Shared canonical model paths.
MODEL_BASE_ROOTS: dict[str, str] = {
    "llama2-7b": "/acpl-ssd32/llama2-7b/base",
    "llama2-13b": "/acpl-ssd32/llama2-13b/base",
    "falcon-7b": "/acpl-ssd32/falcon-7b/base",
    "gemma-7b": "/acpl-ssd32/gemma-7b/base",
}

BASE_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "llama2-7b": {
        "baseline_path": MODEL_BASE_ROOTS["llama2-7b"],
        "progressive_path": "/acpl-ssd32/llama2-7b/final_merged/A_merged",
        "stage_b_checkpoint": "/acpl-ssd32/llama2-7b/final_merged/checkpoints/stage2_layers_B_merged.safetensors",
        "stage_c_checkpoint": "/acpl-ssd32/llama2-7b/final_merged/checkpoints/stage3_layers_C_merged.safetensors",
    },
    "llama2-13b": {
        "baseline_path": MODEL_BASE_ROOTS["llama2-13b"],
        "progressive_path": "/acpl-ssd32/llama2-13b/final_merged/A_merged",
        "stage_b_checkpoint": "/acpl-ssd32/llama2-13b/final_merged/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd32/llama2-13b/final_merged/stage3_layers_C.safetensors",
    },
    "falcon-7b": {
        "baseline_path": MODEL_BASE_ROOTS["falcon-7b"],
        "progressive_path": "/acpl-ssd32/falcon-7b/final_merged/A_merged",
        "stage_b_checkpoint": "/acpl-ssd32/falcon-7b/final_merged/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd32/falcon-7b/final_merged/stage3_layers_C.safetensors",
    },
    "gemma-7b": {
        "baseline_path": MODEL_BASE_ROOTS["gemma-7b"],
        "progressive_path": "/acpl-ssd32/gemma-7b/final_merged/A_merged",
        "stage_b_checkpoint": "/acpl-ssd32/gemma-7b/final_merged/stage2_layers_B.safetensors",
        "stage_c_checkpoint": "/acpl-ssd32/gemma-7b/final_merged/stage3_layers_C.safetensors",
    },
}

CANONICAL_MODEL_CHOICES = tuple(BASE_MODEL_CONFIGS.keys())

MODEL_ALIASES = {
    "llama": "llama2-7b",
    "llama-7b": "llama2-7b",
    "llama2-7b": "llama2-7b",
    "llama-13b": "llama2-13b",
    "llama2-13b": "llama2-13b",
    "falcon": "falcon-7b",
    "falcon-7b": "falcon-7b",
    "gemma": "gemma-7b",
    "gemma7b": "gemma-7b",
    "gemma-7b": "gemma-7b",
}

MODEL_CHOICES = tuple(sorted(MODEL_ALIASES.keys()))


def canonical_model_name(model_name: str) -> str:
    key = str(model_name).strip().lower()
    if key not in MODEL_ALIASES:
        raise KeyError(f"Unknown model alias: {model_name}")
    return MODEL_ALIASES[key]


def merge_model_configs(
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    merged = {name: deepcopy(config) for name, config in BASE_MODEL_CONFIGS.items()}
    if overrides is None:
        return merged

    for model_name, override in overrides.items():
        if model_name not in merged:
            raise KeyError(f"Unknown model config override target: {model_name}")
        merged[model_name].update(deepcopy(dict(override)))
    return merged


# Paper-wide runtime policy for ASPLOS experiments.
# Keep the experimental conditions aligned across scripts and models.
# Only gpu_memory_utilization remains model-specific for capacity/stability.
PAPER_MODEL_OVERRIDES: dict[str, dict[str, Any]] = {
    "llama2-7b": {
        "trust_remote_code": False,
        "enable_prefix_caching": True,
        "disable_sliding_window": False,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "tensor_parallel_size": 1,
        "default_token_counts": [200, 500, 1000, 1500],
    },
    "llama2-13b": {
        "trust_remote_code": False,
        "enable_prefix_caching": True,
        "disable_sliding_window": False,
        "gpu_memory_utilization": 0.60,
        "max_model_len": 2048,
        "tensor_parallel_size": 1,
        "default_token_counts": [200, 500, 1000, 1500],
    },
    "falcon-7b": {
        "trust_remote_code": False,
        "enable_prefix_caching": True,
        "disable_sliding_window": False,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 2048,
        "tensor_parallel_size": 1,
        "default_token_counts": [200, 500, 1000, 1500],
    },
    "gemma-7b": {
        "enable_prefix_caching": True,
        "trust_remote_code": False,
        "disable_sliding_window": False,
        "gpu_memory_utilization": 0.55,
        "max_model_len": 2048,
        "tensor_parallel_size": 1,
        "default_token_counts": [200, 500, 1000, 1500],
    },
}

PAPER_MODEL_CONFIGS = merge_model_configs(PAPER_MODEL_OVERRIDES)

# Backward-compatible exported names for the existing scripts.
BENCHMARK_MODELS = PAPER_MODEL_CONFIGS
CHATBOT_MODELS = PAPER_MODEL_CONFIGS
EVAL_PPL_MODEL_CONFIGS = PAPER_MODEL_CONFIGS


def model_default_token_counts(
    model_name: str,
    max_model_len: int,
    model_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[int]:
    configs = PAPER_MODEL_CONFIGS if model_configs is None else model_configs
    canonical = canonical_model_name(model_name)
    counts = list(configs[canonical].get("default_token_counts", []))
    if not counts:
        counts = [200, 500, 1000, 1500, 2000]
    filtered = [token_count for token_count in counts if token_count + 80 <= max_model_len]
    return filtered or counts
