""" 
모델별 레이어 클래스, 가중치 이름 패턴 등을 정의

Llama는 q_proj, GPT-2는 c_attn처럼 모델마다 가중치 이름이 다름

WeightNamingPattern 클래스를 통해 "이 모델은 QKV 가중치 이름이 뭐야?"라고 물으면 정확한 이름을 알려줌
-> 하나의 코드로 Llama, Mistral, Qwen, Phi 등을 모두 지원(Universal)
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


# vLLM 내부에서 해당 모델의 레이어를 구현한 파이썬 클래스가 어디에 있는지 알려줌
# Layer class mapping (module path → class name)
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


# HuggingFace 설정 파일(config.json)을 읽어 모델 타입을 알아냄
# 별칭 사전을 통해 표준 이름(예: vicuna -> llama)으로 변환
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

# 모델 타입에 맞는 WeightNamingPattern 객체를 반환 
# 모르는 모델이면 기본값으로 Llama 패턴을 줌
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


# 해당 모델의 레이어 클래스 경로 정보를 반환
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


# 이 모델이 가중치를 합쳐서(Fused) 사용하는지 확인하는 "질문 함수"
def is_qkv_fused(model_type: str) -> bool:
    """모델이 QKV fusion을 사용하는지 확인"""
    pattern = get_weight_pattern(model_type)
    return pattern.qkv_fused_name is not None

# 이 모델이 가중치를 합쳐서(Fused) 사용하는지 확인하는 "질문 함수"
def is_mlp_fused(model_type: str) -> bool:
    """모델이 MLP fusion을 사용하는지 확인"""
    pattern = get_weight_pattern(model_type)
    return pattern.mlp_fused_name is not None
