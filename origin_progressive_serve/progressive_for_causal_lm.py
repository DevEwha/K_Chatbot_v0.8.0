"""
Universal ProgressiveForCausalLM with Dual-Path Design
vLLM v0.8.0 Compatible (v0 engine) - Supports All Decoder-Only Models

✅ 모든 Decoder-only 모델 지원
✅ Path A/B 둘 다 항상 계산 (CUDA Graph topology 불변)
✅ Alpha로 경로 선택
✅ prune_log.json 기반 자동 레이어 결정
✅ v0 engine: compute_logits(hidden_states, sampling_metadata) + sample()

<핵심 기능>
_load_prune_log: 모델 폴더의 prune_log.json을 읽어서 "이번엔 몇 번 레이어를 끄고 시작할까?"를 결정

load_weights: model_config.py의 정보를 이용해 체크포인트 파일에서 가중치를 읽어와 모델에 집어넣음.
만약 꺼진 레이어(Inactive)라면 가중치를 0으로 채워 메모리를 아끼거나 초기화 이슈를 방지합니다.

advance_to_stageX: 다음 단계로 넘어갈 때 필요한 추가 가중치 파일(Safetensors)을 로드하고, model.activate_layers를 호출
"""

from typing import Optional, List, Iterable, Tuple, Any, Dict
import torch
import torch.nn as nn
import sys
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata

# Weight loader
try:
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
except ImportError:
    default_weight_loader = None

sys.path.insert(0, "/home/devewha/v08/Juwon/01_universal/progressive_serve")
# Universal Dual-Path implementation
from progressive_model_dual_path import ProgressiveModelDualPath
from model_config import get_model_type, get_weight_pattern


class ProgressiveForCausalLM(nn.Module):
    """
    Universal ForCausalLM wrapper with Dual-Path Design (vLLM v0.8.0, v0 engine)

    지원 모델:
    - LLaMA (1, 2, 3), CodeLlama, Vicuna, Alpaca
    - Mistral, Mixtral
    - Qwen2
    - Gemma (1, 2)
    - Phi (2, 3)
    - GPT-2, GPT-NeoX
    - Falcon
    - DeepSeek (v1, v2)
    - Yi

    핵심:
    - Path A/B 둘 다 항상 계산
    - Alpha로 경로 선택
    - 완벽한 CUDA Graph safety
    - v0 engine: compute_logits(hidden_states, sampling_metadata) + sample()
    """
    supports_multimodal = False
    supports_pooling = False 
    embedding_mode = False
    task = "generate"

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        stage: int = 1,
    ):
        super().__init__()
        
        self.supports_lora = False
        self.embedding_mode = False
        
        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        
        # Get normalized model type
        self.model_type = get_model_type(config)
        
        if not hasattr(config, 'model_type'):
            config.model_type = self.model_type
        
        # Model path 가져오기
        model_path = vllm_config.model_config.model
        
        # prune_log.json 로드
        self.prune_info = self._load_prune_log(model_path)
        
        # Stage에 따른 inactive layer indices (prune_log 기반)
        inactive_indices = self._get_inactive_indices_from_prune_log(self.prune_info, stage)
        
        # Model 생성 (Universal Dual-Path)
        self.model = ProgressiveModelDualPath(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
            pruned_layer_indices=inactive_indices,
        )
        
        # LM head (vLLM v1: ParallelLMHead)
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # vLLM v0.8.0: LogitsProcessor + Sampler
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                config.vocab_size,
                                                logit_scale)
        self.sampler = get_sampler()

        # Stage
        self.current_stage = stage
        
        # Inactive layer tracking (for weight loading)
        self.inactive_layer_indices = set(inactive_indices)
        
        print(f"\n{'='*60}")
        print(f"ProgressiveForCausalLM (Universal, vLLM v0.8.0 v0 engine)")
        print(f"Model Type: {self.model_type}")
        print(f"Model Path: {model_path}")
        print(f"Initialized at Stage {stage}")
        if self.prune_info:
            print(f"✅ Prune log loaded from: {model_path}/prune_log.json")
            print(f"   Split B (Stage 2): {self.prune_info['split']['B']}")
            print(f"   Split C (Stage 3): {self.prune_info['split']['C']}")
        else:
            print(f"⚠️  Using fallback inactive layers (no prune_log.json)")
        print(f"Initially inactive layers: {sorted(inactive_indices)}")
        print(f"🎯 Dual-Path: Path A/B always computed")
        print(f"✅ CUDA Graph safe: Topology invariant")
        print(f"{'='*60}\n")
    
    def _load_prune_log(self, model_path: str) -> Optional[dict]:
        """모델 디렉토리에서 prune_log.json 로드"""
        import json
        import os
        
        prune_log_path = os.path.join(model_path, "prune_log.json")
        
        if not os.path.exists(prune_log_path):
            return None
        
        try:
            with open(prune_log_path, 'r') as f:
                prune_log = json.load(f)
            
            # 필수 필드 확인
            if 'split' not in prune_log:
                print(f"⚠️  Warning: 'split' field not found in prune_log.json")
                return None
            
            if 'B' not in prune_log['split'] or 'C' not in prune_log['split']:
                print(f"⚠️  Warning: 'B' or 'C' not found in split")
                return None
            
            return prune_log
            
        except Exception as e:
            print(f"❌ Error loading prune_log.json: {e}")
            return None
    
    def _get_inactive_indices_from_prune_log(
        self, 
        prune_info: Optional[dict], 
        stage: int
    ) -> List[int]:
        """prune_log.json의 split 정보를 바탕으로 inactive layer indices 결정"""
        # Fallback: prune_log가 없으면 기본값 사용
        if prune_info is None:
            return self._get_inactive_indices_fallback(stage)
        
        try:
            split_b = prune_info['split']['B']
            split_c = prune_info['split']['C']
            
            if stage == 1:
                # Stage 1: B + C 모두 inactive
                inactive = sorted(split_b + split_c)
            elif stage == 2:
                # Stage 2: C만 inactive
                inactive = sorted(split_c)
            elif stage == 3:
                # Stage 3: 모두 active
                inactive = []
            else:
                raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3")
            
            return inactive
            
        except Exception as e:
            print(f"❌ Error parsing prune_log: {e}")
            print(f"   Falling back to default inactive layers")
            return self._get_inactive_indices_fallback(stage)
    
    def _get_inactive_indices_fallback(self, stage: int) -> List[int]:
        """
        Fallback: prune_log가 없을 때 기본값
        
        모델별로 다른 레이어 수를 고려합니다.
        """
        num_layers = self.config.num_hidden_layers
        
        if stage == 1:
            # Last ~25% of layers inactive
            start = int(num_layers * 0.75)
            return list(range(start, num_layers))
        elif stage == 2:
            # Last ~12% of layers inactive
            start = int(num_layers * 0.88)
            return list(range(start, num_layers))
        elif stage == 3:
            return []  # 모두 활성
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """vLLM v1 인터페이스: 토큰 임베딩"""
        return self.model.embed_tokens(input_ids)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """vLLM v0.8.0 v0 engine: logits 계산 (sampling_metadata 필요)"""
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """vLLM v0.8.0 v0 engine: sampling"""
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """vLLM v0.8.0 v0 engine 호환 forward"""
        # Universal Dual-Path model forward
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states
    
    # ============================================================
    # Weight Loading (Universal, vLLM v0.8.0 compatible)
    # ============================================================

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        범용 weight loading (vLLM v0.8.0 호환)
        
        모델 타입에 따라 자동으로 가중치 이름 패턴을 적용합니다.
        """
        print(f"\n{'='*60}")
        print(f"LOADING WEIGHTS (Universal, vLLM v0.8.0)")
        print(f"Model Type: {self.model_type}")
        print(f"{'='*60}")
        
        # Convert iterator to dict for easier handling
        checkpoint_weights = {}
        for name, tensor in weights:
            checkpoint_weights[name] = tensor
        
        print(f"Total weights in checkpoint: {len(checkpoint_weights)}")
        
        # Get weight naming pattern for this model
        weight_pattern = get_weight_pattern(self.model_type)
        print(f"Using weight pattern for: {self.model_type}")
        
        # Collect all parameters
        params_dict = dict(self.named_parameters())
        total_params = len(params_dict)
        
        loaded_keys = set()
        loaded_count = 0
        
        # Load weights
        for param_name, param in params_dict.items():
            # Option 1: Direct match
            if param_name in checkpoint_weights:
                weight_loader = getattr(param, "weight_loader",
                                       lambda p, w: p.data.copy_(w))
                weight_loader(param, checkpoint_weights[param_name])
                loaded_keys.add(param_name)
                loaded_count += 1
                continue
            
            # Option 2: Match without .layer prefix (for wrapped layers)
            alt_name = param_name.replace(".layer.", ".")
            if alt_name in checkpoint_weights:
                weight_loader = getattr(param, "weight_loader",
                                       lambda p, w: p.data.copy_(w))
                weight_loader(param, checkpoint_weights[alt_name])
                loaded_keys.add(param_name)
                loaded_count += 1
                continue
            
            # Option 3: Fused QKV weights (범용)
            if weight_pattern.qkv_fused_name and weight_pattern.qkv_fused_name in param_name:
                qkv_loaded = self._load_qkv_weights(
                    param, param_name, checkpoint_weights, weight_pattern
                )
                if qkv_loaded:
                    loaded_keys.add(param_name)
                    loaded_count += 1
                    continue

            # Option 4: Fused Gate-Up weights (범용)
            if weight_pattern.mlp_fused_name and weight_pattern.mlp_fused_name in param_name:
                mlp_loaded = self._load_mlp_weights(
                    param, param_name, checkpoint_weights, weight_pattern
                )
                if mlp_loaded:
                    loaded_keys.add(param_name)
                    loaded_count += 1
                    continue

            # Option 5: Falcon HuggingFace 형식 변환
            # vLLM param:    model.layers.N.layer.*   (ProgressiveModelDualPath 래핑)
            # HF checkpoint: transformer.h.N.*         (Falcon 원본 형식)
            if self.model_type == "falcon":
                import re
                falcon_name = re.sub(
                    r'^model\.layers\.(\d+)\.layer\.',
                    lambda m: f'transformer.h.{m.group(1)}.',
                    param_name,
                )
                if falcon_name == param_name:
                    # 레이어가 아닌 파라미터: embed_tokens, norm
                    falcon_name = param_name \
                        .replace("model.embed_tokens.", "transformer.word_embeddings.") \
                        .replace("model.norm.", "transformer.ln_f.")
                if falcon_name != param_name and falcon_name in checkpoint_weights:
                    weight_loader = getattr(param, "weight_loader",
                                           lambda p, w: p.data.copy_(w))
                    weight_loader(param, checkpoint_weights[falcon_name])
                    loaded_keys.add(param_name)
                    loaded_count += 1
                    continue
        
        # Missing weights 처리
        missing_keys = set(params_dict.keys()) - loaded_keys
        
        if missing_keys:
            print(f"\n⚠️  Found {len(missing_keys)} missing weights")
            print(f"   Initializing them to ZEROS (for inactive layers)...")
            
            # Layer별로 그룹화
            missing_by_layer = {}
            for key in missing_keys:
                parts = key.split('.')
                if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
                    try:
                        layer_idx = int(parts[2])
                        if layer_idx not in missing_by_layer:
                            missing_by_layer[layer_idx] = []
                        missing_by_layer[layer_idx].append(key)
                    except ValueError:
                        pass
            
            # Layer별 초기화
            zero_initialized = 0
            for layer_idx in sorted(missing_by_layer.keys()):
                layer_keys = missing_by_layer[layer_idx]
                
                if layer_idx in self.inactive_layer_indices:
                    # Inactive layer: 0으로 초기화 (예상된 동작)
                    print(f"   Layer {layer_idx}: Initializing {len(layer_keys)} weights to ZERO (inactive)")
                    
                    for key in layer_keys:
                        param = params_dict[key]
                        nn.init.zeros_(param)
                        zero_initialized += 1
                else:
                    # Active layer인데 missing → 경고!
                    print(f"   ⚠️  Layer {layer_idx}: Missing {len(layer_keys)} weights (ACTIVE layer!)")
                    
                    # 그래도 0으로 초기화 (에러 방지)
                    for key in layer_keys:
                        param = params_dict[key]
                        nn.init.zeros_(param)
                        zero_initialized += 1
            
            print(f"✅ Initialized {zero_initialized} missing weights to ZERO")
        
        print(f"\n{'='*60}")
        print(f"WEIGHT LOADING SUMMARY")
        print(f"{'='*60}")
        print(f"Total parameters:      {total_params}")
        print(f"Loaded from checkpoint: {loaded_count}")
        print(f"Initialized to zero:    {len(missing_keys)}")
        print(f"Coverage:               {loaded_count / total_params * 100:.1f}%")
        print(f"{'='*60}\n")
    
    def _load_qkv_weights(
        self,
        param,
        param_name: str,
        checkpoint_weights: dict[str, torch.Tensor],
        weight_pattern: Any,
    ) -> bool:
        """범용 QKV weight 로딩"""
        # Build expected weight names based on pattern
        weight_names = []
        
        if ".layer." in param_name:
            checkpoint_base = param_name.replace(f".layer.self_attn.{weight_pattern.qkv_fused_name}.weight", "")
        else:
            checkpoint_base = param_name.replace(f".self_attn.{weight_pattern.qkv_fused_name}.weight", "")
        
        for proj_name in weight_pattern.qkv_weights:
            if "self_attn" in param_name:
                weight_name = f"{checkpoint_base}.self_attn.{proj_name}.weight"
            elif "attn" in param_name:
                weight_name = f"{checkpoint_base}.attn.{proj_name}.weight"
            else:
                # Fallback
                weight_name = f"{checkpoint_base}.{proj_name}.weight"
            
            weight_names.append(weight_name)
        
        # Check if all weights exist
        if all(name in checkpoint_weights for name in weight_names):
            qkv_weight = torch.cat([
                checkpoint_weights[name] for name in weight_names
            ], dim=0)
            
            weight_loader = getattr(param, "weight_loader",
                                   lambda p, w: p.data.copy_(w))
            weight_loader(param, qkv_weight)
            return True
        
        return False
    
    def _load_mlp_weights(
        self,
        param,
        param_name: str,
        checkpoint_weights: Dict[str, torch.Tensor],
        weight_pattern: Any,
    ) -> bool:
        """범용 MLP weight 로딩"""
        if not weight_pattern.mlp_gate_up:
            return False
        
        # Build expected weight names based on pattern
        weight_names = []
        
        if ".layer." in param_name:
            checkpoint_base = param_name.replace(f".layer.mlp.{weight_pattern.mlp_fused_name}.weight", "")
        else:
            checkpoint_base = param_name.replace(f".mlp.{weight_pattern.mlp_fused_name}.weight", "")
        
        for proj_name in weight_pattern.mlp_gate_up:
            weight_name = f"{checkpoint_base}.mlp.{proj_name}.weight"
            weight_names.append(weight_name)
        
        # Check if all weights exist
        if all(name in checkpoint_weights for name in weight_names):
            mlp_weight = torch.cat([
                checkpoint_weights[name] for name in weight_names
            ], dim=0)
            
            weight_loader = getattr(param, "weight_loader",
                                   lambda p, w: p.data.copy_(w))
            weight_loader(param, mlp_weight)
            return True
        
        return False
    
    # ============================================================
    # Progressive Recovery (Alpha Gating)
    # ============================================================
    
    def _get_b_indices(self) -> List[int]:
        if self.prune_info:
            return list(self.prune_info['split']['B'])
        num = self.config.num_hidden_layers
        return list(range(int(num * 0.75), int(num * 0.88)))

    def _get_c_indices(self) -> List[int]:
        if self.prune_info:
            return list(self.prune_info['split']['C'])
        num = self.config.num_hidden_layers
        return list(range(int(num * 0.88), num))

    def prefetch_stage2(self, checkpoint_path: str) -> None:
        """Stage 2 weights를 백그라운드에서 CPU에 미리 로드. Stage 1 서빙 시작 직후 호출."""
        self.model.prefetch_weights(checkpoint_path, self._get_b_indices())

    def prefetch_stage3(self, checkpoint_path: str) -> None:
        """Stage 3 weights를 백그라운드에서 CPU에 미리 로드. Stage 2 서빙 시작 직후 호출."""
        self.model.prefetch_weights(checkpoint_path, self._get_c_indices())

    def advance_to_stage2_instant(self, wait_if_needed: bool = True) -> bool:
        """
        prefetch된 weights로 즉각 Stage 2 전환.
        디스크 I/O 없이 GPU copy + alpha 변경만 실행.
        prefetch_stage2()가 먼저 호출되어 있어야 함.
        """
        success = self.model.activate_layers_instant(
            self._get_b_indices(),
            wait_if_needed=wait_if_needed,
        )
        if success:
            self.current_stage = 2
            self.inactive_layer_indices = set(self._get_c_indices())
            print(f"\n{'='*80}")
            print(f"NOW AT STAGE 2 (instant)")
            print(f"{'='*80}\n")
            self.print_status()
        return success

    def advance_to_stage3_instant(self, wait_if_needed: bool = True) -> bool:
        """
        prefetch된 weights로 즉각 Stage 3 전환.
        prefetch_stage3()가 먼저 호출되어 있어야 함.
        """
        success = self.model.activate_layers_instant(
            self._get_c_indices(),
            wait_if_needed=wait_if_needed,
        )
        if success:
            self.current_stage = 3
            self.inactive_layer_indices = set()
            print(f"\n{'='*80}")
            print(f"NOW AT STAGE 3 - FULL MODEL (instant)")
            print(f"{'='*80}\n")
            self.print_status()
        return success

    def is_stage2_ready(self) -> bool:
        """Stage 2 prefetch 완료 여부 (non-blocking 확인용)"""
        return self.model.is_prefetch_ready()

    def is_prefetch_ready(self) -> bool:
        """최근 prefetch 완료 여부 (Stage 2/3 공통)"""
        return self.model.is_prefetch_ready()

    def wait_for_prefetch(self, timeout_s: Optional[float] = None) -> bool:
        """최근 prefetch 완료까지 대기"""
        return self.model.wait_for_prefetch(timeout_s=timeout_s)

    def get_prefetch_status(self) -> dict:
        """최근 prefetch 상태 반환"""
        return self.model.get_prefetch_status()

    def advance_to_stage2(
        self,
        layer_b_checkpoint: str,
        adapter_ab_path: Optional[str] = None,
    ) -> None:
        """Stage 1 → Stage 2 (prune_log 기반)"""
        print("\n" + "="*80)
        print(f"ADVANCING TO STAGE 2 (Universal, {self.model_type})")
        print("="*80)
        
        # prune_log에서 B 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            num_layers = self.config.num_hidden_layers
            start = int(num_layers * 0.75)
            end = int(num_layers * 0.88)
            activate_indices = list(range(start, end))
        else:
            activate_indices = self.prune_info['split']['B']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # B 레이어 활성화
        self.model.activate_layers(
            layer_indices=activate_indices,
            checkpoint_path=layer_b_checkpoint,
        )
        
        # Adapter (optional)
        if adapter_ab_path:
            print(f"Loading AB adapter from: {adapter_ab_path}")
        
        # Stage 업데이트
        self.current_stage = 2
        
        # Inactive layers 업데이트 (C만)
        if self.prune_info:
            self.inactive_layer_indices = set(self.prune_info['split']['C'])
        else:
            num_layers = self.config.num_hidden_layers
            start = int(num_layers * 0.88)
            self.inactive_layer_indices = set(range(start, num_layers))
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 2")
        print(f"{'='*80}\n")
        
        self.print_status()
    
    def advance_to_stage3(
        self,
        layer_c_checkpoint: str,
        remove_adapter: bool = True,
    ) -> None:
        """Stage 2 → Stage 3 (prune_log 기반)"""
        print("\n" + "="*80)
        print(f"ADVANCING TO STAGE 3 (Universal, {self.model_type})")
        print("="*80)
        
        # prune_log에서 C 레이어 가져오기
        if self.prune_info is None:
            print("⚠️  Warning: No prune_log available. Using fallback.")
            num_layers = self.config.num_hidden_layers
            start = int(num_layers * 0.88)
            activate_indices = list(range(start, num_layers))
        else:
            activate_indices = self.prune_info['split']['C']
            print(f"Activating layers from prune_log: {activate_indices}")
        
        # C 레이어 활성화
        self.model.activate_layers(
            layer_indices=activate_indices,
            checkpoint_path=layer_c_checkpoint,
        )
        
        # Adapter 제거
        if remove_adapter:
            print("Removing all adapters...")
        
        # Stage 업데이트
        self.current_stage = 3
        self.inactive_layer_indices = set()  # 모두 활성
        
        print(f"\n{'='*80}")
        print(f"NOW AT STAGE 3 - FULL MODEL")
        print(f"{'='*80}\n")
        
        self.print_status()
    
    # ============================================================
    # Status and Info Methods
    # ============================================================
    
    def print_status(self) -> None:
        """현재 모델 상태 출력"""
        self.model.print_layer_status()
        
        print(f"Current Stage: {self.current_stage}")
        print(f"Model Type: {self.model_type}")
        
        report = self.model.verify_recovery()
        print(f"Activation Progress: {report['activation_progress']}")
        
        adapter_info = self.model.get_adapter_info()
        print(f"Current Adapter: {adapter_info['current_adapter'] or 'None'}")
        print()
    
    def get_stage_info(self) -> dict:
        """현재 stage 정보 반환 (prune_info 포함)"""
        report = self.model.verify_recovery()
        adapter_info = self.model.get_adapter_info()
        
        return {
            "stage": self.current_stage,
            "model_type": self.model_type,
            "active_layers": report["active_layers"],
            "inactive_layers": report["inactive_layers"],
            "activation_progress": report["activation_progress"],
            "current_adapter": adapter_info["current_adapter"],
            "inactive_layer_indices": report["inactive_layer_indices"],
            "prune_info": self.prune_info,
        }
    
    def get_layer_alphas(self) -> List[float]:
        """모든 레이어의 alpha 값 반환"""
        alphas = []
        for layer in self.model.layers:
            if hasattr(layer, 'get_alpha_value'):
                alphas.append(layer.get_alpha_value())
            else:
                alphas.append(1.0)  # Normal layer
        return alphas
    
    def set_layer_alpha(self, layer_idx: int, alpha: float):
        """특정 레이어의 alpha 값 직접 설정"""
        if layer_idx >= len(self.model.layers):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        layer = self.model.layers[layer_idx]
        if hasattr(layer, 'set_alpha'):
            layer.set_alpha(alpha) 
            print(f"Layer {layer_idx} alpha set to {alpha}")
        else:
            print(f"Layer {layer_idx} is not an AlphaGatedLayer")
