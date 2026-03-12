"""
모델별 가중치 이름 패턴 및 레이어 클래스 정보를 정의.

모델마다 QKV 가중치 이름이 다르다 (Llama: q_proj, GPT-2: c_attn 등).
WeightNamingPattern으로 패턴을 표준화하여 단일 코드로 다양한 모델을 지원.
"""

from typing import Dict, List, Optional, Any 
from dataclasses import dataclass 

@dataclass 
class WeightNamingPattern:
    """모델별 가중치 이름 패턴"""
    qkv_weights: List[str] # Q, K, V projection weights 
    qkv_fused_name: Optional[str] = None # Fused QKV name (if exists)
    mlp_gate_up: List[str]=None #Gate, Up projection weights 
    mlp_fused_name: Optional[str]=None #Fused Gate0Up name (if exists)
    output_proj: str="o_proj" # Output projection 
    mlp_down: str= "down_proj" # MLP down projection


# Model-specific weight naming patterns
WEIGHT_NAMING_PATTERNS: Dict[str, WeightNamingPattern] = {
    "llama": WeightNamingPattern(
        qkv_weights=["q_proj", "k_proj", "v_proj"],
        qkv_fused_name="qkv_proj",
        mlp_gate_up=["gate_proj", "up_proj"],
        mlp_fused_name="gate_up_proj",
        output_proj="o_proj",
        mlp_down="down_proj",
    ),
    "mistral": WeightNamingPattern(
        qkv_weights=["q_proj", "k_proj", "v_proj"],
        qkv_fused_name="qkv_proj",
        mlp_gate_up=["gate_proj", "up_proj"],
        mlp_fused_name="gate_up_proj",
        output_proj="o_proj",
        mlp_down="down_proj",
    ),
    "qwen2": WeightNamingPattern(
        qkv_weights=["q_proj", "k_proj", "v_proj"],
        qkv_fused_name="qkv_proj",
        mlp_gate_up=["gate_proj", "up_proj"],
        mlp_fused_name="gate_up_proj",
        output_proj="o_proj",
        mlp_down="down_proj",
    ),
    "gemma": WeightNamingPattern(
        qkv_weights=["q_proj", "k_proj", "v_proj"],
        qkv_fused_name="qkv_proj",
        mlp_gate_up=["gate_proj", "up_proj"],
        mlp_fused_name="gate_up_proj",
        output_proj="o_proj",
        mlp_down="down_proj",
    ),
    "phi": WeightNamingPattern(
        qkv_weights=["q_proj", "k_proj", "v_proj"],
        qkv_fused_name="qkv_proj",
        mlp_gate_up=["fc1", "fc2"],
        mlp_fused_name="gate_up_proj",
        output_proj="dense",
        mlp_down="fc2",
    ),
    "gpt2": WeightNamingPattern(
        qkv_weights=["c_attn"],  # GPT-2 has pre-fused QKV
        qkv_fused_name="c_attn",
        mlp_gate_up=["c_fc"],
        mlp_fused_name=None,
        output_proj="c_proj",
        mlp_down="c_proj",
    ),
    "falcon": WeightNamingPattern(
        qkv_weights=["query_key_value"],  # Falcon has pre-fused QKV
        qkv_fused_name="query_key_value",
        mlp_gate_up=["dense_h_to_4h"],
        mlp_fused_name=None,
        output_proj="dense",
        mlp_down="dense_4h_to_h",
    ),
}


# 모델 타입별 vLLM 레이어 클래스 위치 (module path → class name)
LAYER_CLASS_MAPPING: Dict[str, Dict[str, str]] = {
    "llama": {
        "module": "vllm.model_executor.models.llama",
        "v1_module": "vllm.v1.model_executor.models.llama",
        "layer_class": "LlamaDecoderLayer",
    },
    "mistral": {
        "module": "vllm.model_executor.models.llama",
        "v1_module": "vllm.v1.model_executor.models.llama",
        "layer_class": "LlamaDecoderLayer",
    },
    "qwen2": {
        "module": "vllm.model_executor.models.qwen2",
        "v1_module": "vllm.v1.model_executor.models.qwen2",
        "layer_class": "Qwen2DecoderLayer",
    },
    "gemma": {
        "module": "vllm.model_executor.models.gemma",
        "v1_module": "vllm.v1.model_executor.models.gemma",
        "layer_class": "GemmaDecoderLayer",
    },
    "gemma2": {
        "module": "vllm.model_executor.models.gemma2",
        "v1_module": "vllm.v1.model_executor.models.gemma2",
        "layer_class": "Gemma2DecoderLayer",
    },
    "phi": {
        "module": "vllm.model_executor.models.phi",
        "v1_module": "vllm.v1.model_executor.models.phi",
        "layer_class": "PhiDecoderLayer",
    },
    "phi3": {
        "module": "vllm.model_executor.models.phi3",
        "v1_module": "vllm.v1.model_executor.models.phi3",
        "layer_class": "Phi3DecoderLayer",
    },
    "gpt2": {
        "module": "vllm.model_executor.models.gpt2",
        "v1_module": "vllm.v1.model_executor.models.gpt2",
        "layer_class": "GPT2Block",
    },
    "falcon": {
        "module": "vllm.model_executor.models.falcon",
        "v1_module": "vllm.v1.model_executor.models.falcon",
        "layer_class": "FalconDecoderLayer",
    },
}

# Model aliases (different names for same architecture)
MODEL_ALIASES: Dict[str, str] = {
    "llama2": "llama",
    "llama3": "llama",
    "codellama": "llama",
    "vicuna": "llama",
    "alpaca": "llama",
    "yi": "llama",
    "deepseek": "llama",  # DeepSeek v1/v2 uses Llama architecture
    "qwen": "qwen2",
    "phi-2": "phi",
    "phi-3": "phi3",
}


def get_model_type(config: Any) -> str:
    """
    모델 타입 정규화
    
    Args:
        config: HuggingFace config object
        
    Returns:
        Normalized model type
    """
    model_type = getattr(config, 'model_type', 'llama')
    
    # Apply aliases
    model_type = MODEL_ALIASES.get(model_type, model_type)
    
    return model_type

def get_weight_pattern(model_type: str) -> WeightNamingPattern:
    """
    모델 타입에 따른 가중치 패턴 반환
    
    Args:
        model_type: Model type (e.g., "llama", "mistral")
        
    Returns:
        WeightNamingPattern for the model
    """
    # Default to llama pattern if not found
    return WEIGHT_NAMING_PATTERNS.get(model_type, WEIGHT_NAMING_PATTERNS["llama"])


def get_layer_class_info(model_type: str) -> Dict[str, str]:
    """
    모델 타입에 따른 레이어 클래스 정보 반환
    
    Args:
        model_type: Model type (e.g., "llama", "mistral")
        
    Returns:
        Dict with module path and class name
    """
    # Default to llama if not found
    return LAYER_CLASS_MAPPING.get(model_type, LAYER_CLASS_MAPPING["llama"])


def is_qkv_fused(model_type: str) -> bool:
    """모델이 QKV fusion을 사용하는지 확인"""
    pattern = get_weight_pattern(model_type)
    return pattern.qkv_fused_name is not None

def is_mlp_fused(model_type: str) -> bool:
    """모델이 MLP fusion을 사용하는지 확인"""
    pattern = get_weight_pattern(model_type)
    return pattern.mlp_fused_name is not None
