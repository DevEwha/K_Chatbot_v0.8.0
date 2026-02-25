"""
vLLM v1(0.15.1)을 위한 코드 
* 모든 Decoder-only 모델 지원(Llama, Mistral, QWen, Phi, Gemma, GPT-2, Falcon등)
레이어 항상 실행해 topology 불변
Path A(레이어 통과)+Path B(직접 연결) 둘 다 계산
Alpha로 어느 경로를 다음 레이어로 전달할지 선택
"""


from typing import Optional, List, Dict, Any
import importlib
import threading
import inspect
import os
import torch
import torch.nn as nn
import sys

from vllm.config import VllmConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.layers.layernorm import RMSNorm



from safetensors.torch import load_file 

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_config import (
    get_model_type,
    get_layer_class_info,
    get_weight_pattern,
)

# Universal bypass layer
from universal_bypass_layer import UniversalBypassLayer 

class ProgressiveModelDualPath(nn.Module):
    """
    Universal Progressive Model with Dual-Path Design

    지원 모델:
    - LLaMA (1, 2, 3)
    - Mistral
    - Qwen2
    - Gemma (1, 2)
    - Phi (2, 3)
    - GPT-2
    - Falcon
    - 기타 Decoder-only 모델

    핵심 아이디어:
    - 레이어는 항상 실행 (CUDA Graph topology 불변)
    - 두 경로를 모두 계산:
      * Path A: 레이어를 통과한 값
      * Path B: 레이어 간 직접 연결 (bypass)
    - Alpha로 어느 경로를 사용할지 선택:
      * alpha=1: Path A (레이어 통과)
      * alpha=0: Path B (직접 연결)
      * 0<alpha<1: blend

    CUDA Graph 안전성:
    - 레이어 항상 실행 → kernel sequence 불변
    - Path A/B 둘 다 항상 계산 → topology 불변
    - Alpha blending 항상 수행 → topology 불변
    - Alpha 값만 변경 (scalar buffer) → CUDA Graph safe
    - NO .item() calls in forward → capture safe!

    Partial KV Recomputation:
    - Stage 전환 시 boundary layer 기준 KV cache 부분 재계산
    - Boundary 이전 layer: KV-only (norm + qkv_proj + rotary + cache write)
    - Boundary 이후 layer: full forward
    - Prefill(eager mode)에서만 동작 → CUDA Graph 재캡처 없음
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        pruned_layer_indices: Optional[List[int]] = None,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config

        # Get normalized model type
        self.model_type = get_model_type(config)

        self.initially_inactive = set(pruned_layer_indices or [])

        # Embedding
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Decoder layers
        self.layers = nn.ModuleList()
        self._init_layers(prefix)
        self._layer_forward_mode = self._resolve_layer_forward_mode()

        # Final norm
        self.norm = RMSNorm(
            config.hidden_size,
            eps=getattr(config, 'rms_norm_eps', 1e-6),
        )

        self.current_adapter = None

        # ── Partial KV Recomputation ──
        # layer_idx → {"output": (hidden_states_cpu, residual_cpu)}
        self._layer_output_cache: Dict[int, Any] = {}
        # None이면 일반 forward, 정수면 해당 layer부터 full forward
        self._partial_recompute_boundary: Optional[int] = None
        # 캐싱할 최대 레이어 인덱스 (다음 stage의 boundary-1)
        self._max_cacheable_layer: Optional[int] = None

        # ── GPU-resident Partial Recompute (Method A) ──
        # CPU 복사 없이 GPU persistent buffer에서 직접 boundary hidden states 사용.
        # Front layers KV cache는 그대로 유지, back layers만 재계산.
        self._recompute_from_boundary_gpu: Optional[int] = None

        # ── Persistent GPU Buffers (CUDA graph safe) ──
        # index_copy_는 in-place 연산 → CUDA graph에 캡처됨
        # Prefill (eager): 직접 실행, Decode (graph replay): 자동 실행
        # 따라서 prefill + decode 모두에서 hidden states가 자동 누적됨
        self._persistent_h_buffers: List[torch.Tensor] = []
        self._persistent_r_buffers: List[torch.Tensor] = []
        self._persistent_buffers_initialized = False

        print(f"✅ Initialized ProgressiveModelDualPath for: {self.model_type}")
        print(f"✅ Layer forward mode: {self._layer_forward_mode}")
    
    def _get_layer_class(self, model_type: str):
        """
        모델 타입에 따른 레이어 클래스 동적 로드
        
        Args:
            model_type: Normalized model type (e.g., "llama", "mistral")
            
        Returns:
            Layer class (e.g., LlamaDecoderLayer)
        """
        layer_info = get_layer_class_info(model_type)
        
        # Try v1 module first
        try:
            module = importlib.import_module(layer_info["v1_module"])
            layer_class = getattr(module, layer_info["layer_class"])
            print(f"  ✅ Loaded {layer_info['layer_class']} from v1 module")
            return layer_class
        except (ImportError, AttributeError):
            pass
        
        # Fallback to v0 module
        try:
            module = importlib.import_module(layer_info["module"])
            layer_class = getattr(module, layer_info["layer_class"])
            print(f"  ✅ Loaded {layer_info['layer_class']} from v0 module")
            return layer_class
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load layer class for model type '{model_type}'. "
                f"Tried: {layer_info['v1_module']}.{layer_info['layer_class']}, "
                f"{layer_info['module']}.{layer_info['layer_class']}. "
                f"Error: {e}"
            )
    
    def _init_layers(self, prefix: str):
        """모든 레이어를 UniversalBypassLayer로 감싸기"""
        
        # Get layer class for this model type
        LayerClass = self._get_layer_class(self.model_type)
        
        num_layers = self.config.num_hidden_layers
        
        for layer_idx in range(num_layers):
            # Base layer 생성 - Try multiple initialization styles
            base_layer = self._create_base_layer(LayerClass, layer_idx, prefix)
            
            # UniversalBypassLayer로 감싸기
            if layer_idx in self.initially_inactive:
                print(f"[Init] Layer {layer_idx:2d}: DualPath (alpha=0, Path B)")
                
                # Weight를 0으로 초기화
                # alpha=0일 때 Path A는 zero-output이므로 GPU 최적화됨
                self._initialize_weights_to_zero(base_layer)
                
                wrapped = UniversalBypassLayer(
                    base_layer=base_layer,
                    initial_alpha=0.0,
                    layer_idx=layer_idx,
                )
                self.layers.append(wrapped)
            else:
                print(f"[Init] Layer {layer_idx:2d}: DualPath (alpha=1, Path A)")
                
                wrapped = UniversalBypassLayer(
                    base_layer=base_layer,
                    initial_alpha=1.0,
                    layer_idx=layer_idx,
                )
                self.layers.append(wrapped)
    
    def _create_base_layer(self, LayerClass, layer_idx: int, prefix: str):
        """
        범용적인 레이어 초기화
        
        다양한 초기화 시그니처를 시도합니다:
        1. v1 style: vllm_config only
        2. v0 style: config + cache_config + quant_config
        3. Minimal: layer_idx + config
        """
        layer_prefix = f"{prefix}.layers.{layer_idx}"
        
        # Try v1 style first (vllm_config만 사용)
        try:
            return LayerClass(
                vllm_config=self.vllm_config,
                prefix=layer_prefix,
            )
        except TypeError:
            pass
        
        # Try v0 style with full config
        try:
            return LayerClass(
                config=self.config,
                cache_config=self.vllm_config.cache_config,
                quant_config=self.vllm_config.quant_config,
                prefix=layer_prefix,
            )
        except TypeError:
            pass
        
        # Try with layer_idx
        try:
            return LayerClass(
                layer_idx=layer_idx,
                config=self.config,
                prefix=layer_prefix,
            )
        except TypeError:
            pass
        
        # Minimal fallback
        try:
            return LayerClass(
                config=self.config,
                prefix=layer_prefix,
            )
        except TypeError as e:
            raise TypeError(
                f"Failed to initialize {LayerClass.__name__} with any known signature. "
                f"Last error: {e}"
            )
    
    def _initialize_weights_to_zero(self, layer: nn.Module):
        """Weight를 0으로 초기화"""
        for param in layer.parameters():
            param.data.zero_()

    def _resolve_layer_forward_mode(self) -> str:
        """
        런타임 try/except 디스패치를 없애기 위해, 초기화 시 1회만
        layer forward 시그니처를 분석해 고정 모드를 선택한다.
        """
        if len(self.layers) == 0:
            return "kwargs_v1"

        layer = self.layers[0].layer
        try:
            sig = inspect.signature(layer.forward)
            param_names = {
                p.name for p in sig.parameters.values()
                if p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
        except (TypeError, ValueError):
            param_names = set()

        if {"positions", "hidden_states", "residual"}.issubset(param_names):
            return "kwargs_v1"
        if {"positions", "hidden_states"}.issubset(param_names):
            return "kwargs_no_residual"
        return "positional"
    
    # ================================================================
    # Persistent GPU Buffers (CUDA graph safe caching)
    # ================================================================

    def _init_persistent_buffers(self, device, dtype):
        """
        Persistent GPU buffer 사전 할당 (최초 forward 시 1회 호출)

        CUDA graph 캡처 전에 호출되어야 함 (memory profiling 단계에서 자동 호출)
        - index_copy_()가 CUDA graph에 캡처되려면 buffer가 먼저 존재해야 함
        - vLLM flow: model init → weight load → memory profile(forward) → graph capture(forward)
        - memory profile 시 최초 forward → 여기서 buffer 할당
        """
        if self._persistent_buffers_initialized:
            return

        max_seq_len = self.vllm_config.model_config.max_model_len
        hidden_dim = self.config.hidden_size
        num_layers = len(self.layers)

        for _ in range(num_layers):
            self._persistent_h_buffers.append(
                torch.zeros(max_seq_len, hidden_dim, dtype=dtype, device=device)
            )
            self._persistent_r_buffers.append(
                torch.zeros(max_seq_len, hidden_dim, dtype=dtype, device=device)
            )

        self._persistent_buffers_initialized = True
        mem_mb = num_layers * max_seq_len * hidden_dim * 2 * 2 / (1024**2)
        print(f"✅ Persistent GPU buffers: {num_layers} layers × {max_seq_len} seq = {mem_mb:.0f} MB")

    def sync_persistent_cache(self, seq_len: int):
        """
        GPU persistent buffer → CPU _layer_output_cache

        Stage 전환 직전에 chatbot에서 호출.
        GPU buffer의 [0:seq_len] 구간을 CPU로 복사하여 partial recompute에 사용.
        """
        if not self._persistent_buffers_initialized:
            print(f"[Cache] ⚠️ Persistent buffers not initialized")
            return

        max_layer = self._max_cacheable_layer if self._max_cacheable_layer is not None else len(self.layers) - 1

        self._layer_output_cache.clear()
        for layer_idx in range(max_layer + 1):
            h = self._persistent_h_buffers[layer_idx][:seq_len].cpu()
            r = self._persistent_r_buffers[layer_idx][:seq_len].cpu()
            self._layer_output_cache[layer_idx] = {"output": (h, r)}

        print(f"[Cache] Synced {max_layer + 1} layers × {seq_len} tokens (GPU → CPU)")

    def clear_persistent_buffers(self):
        """Persistent buffer 초기화 (warmup 데이터 제거)"""
        with torch.inference_mode():
            for buf in self._persistent_h_buffers:
                buf.zero_()
            for buf in self._persistent_r_buffers:
                buf.zero_()

    # ================================================================
    # Forward: Dual-Path Design (Universal for all decoder models)
    # ================================================================

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[Any] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with Dual-Path Design (Universal)

        핵심:
        1. 레이어 항상 실행 (topology 불변)
        2. Path A/B 둘 다 계산
        3. Alpha로 선택

        Partial KV Recomputation:
        - _partial_recompute_boundary가 설정되면, boundary 이전 레이어는
          KV-only forward (norm+qkv+rotary+cache_write만), boundary 이후는 full forward
        - 캐시된 hidden states를 사용해 boundary 이전 레이어를 빠르게 처리
        - Prefill(eager mode)에서만 동작 → CUDA Graph 재캡처 없음

        CUDA Graph Safety:
        - get_alpha() returns tensor (not float!)
        - No .item() calls anywhere in forward
        - All operations on GPU tensors
        """

        # Embedding
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None

        # ── GPU-resident Partial Recompute (Method A) ──
        # Front layers KV cache는 가중치 변경 없음 → 그대로 유효.
        # _persistent_h_buffers[gpu_boundary-1]에서 boundary hidden states 직접 읽어
        # back layers만 재계산. CPU 복사, KV-only pass 완전 제거.
        gpu_boundary = self._recompute_from_boundary_gpu
        if gpu_boundary is not None:
            seq_len = hidden_states.shape[0]
            if seq_len > 1 and self._persistent_buffers_initialized:
                # boundary-1 레이어의 GPU 저장 hidden states 읽기
                # positions: [seq_len] 텐서, buffer[positions] → [seq_len, hidden]
                boundary_h = self._persistent_h_buffers[gpu_boundary - 1][positions]
                boundary_r = self._persistent_r_buffers[gpu_boundary - 1][positions]

                self._recompute_from_boundary_gpu = None  # 1회성

                print(f"\n[GPURecompute] 🚀 GPU-resident partial recompute")
                print(f"  Boundary layer : {gpu_boundary}")
                print(f"  Front layers   : 0-{gpu_boundary-1} → skipped (KV cache already valid)")
                print(f"  Back layers    : {gpu_boundary}-{len(self.layers)-1} → full forward")
                print(f"  Tokens         : {seq_len}")

                # Front layers 완전 스킵: 해당 attention의 write_kv_to_cache 호출 안 됨
                # → front layer KV cache slots 그대로 유지
                hidden_states = boundary_h
                residual = boundary_r

                # Back layers만 실행 (dual-path 그대로 유지)
                for layer_idx in range(gpu_boundary, len(self.layers)):
                    layer_wrapper = self.layers[layer_idx]

                    alpha = layer_wrapper.get_alpha()  # tensor, CUDA Graph safe

                    # Path A: Layer 통과 (attention이 내부적으로 write_kv_to_cache 호출)
                    hidden_a, residual_a = self._call_layer_forward_fast(
                        layer_wrapper.layer,
                        positions=positions,
                        hidden_states=hidden_states,
                        residual=residual,
                    )

                    # Path B: bypass
                    hidden_b = hidden_states
                    residual_b = residual

                    # Alpha blending
                    hidden_states = alpha * hidden_a + (1.0 - alpha) * hidden_b
                    if residual_a is not None and residual_b is not None:
                        residual = alpha * residual_a + (1.0 - alpha) * residual_b
                    elif residual_a is not None:
                        residual = alpha * residual_a
                    else:
                        residual = residual_b

                    # Persistent buffer 업데이트 (back layers용)
                    if self._max_cacheable_layer is None or layer_idx <= self._max_cacheable_layer:
                        self._persistent_h_buffers[layer_idx].index_copy_(
                            0, positions, hidden_states)
                        if residual is not None:
                            self._persistent_r_buffers[layer_idx].index_copy_(
                                0, positions, residual)

                    if layer_idx == gpu_boundary or layer_idx == len(self.layers) - 1:
                        print(f"  Layer {layer_idx:2d}: ↻ full forward (GPU-resident recompute)")

                print(f"[GPURecompute] ✅ Back layers recomputed, front KV cache preserved\n")

                # Final residual + norm
                if residual is not None:
                    hidden_states = hidden_states + residual
                hidden_states = self.norm(hidden_states)
                return hidden_states

            else:
                # seq_len=1 (decode phase) 이거나 버퍼 미초기화 → 모드 클리어 후 일반 forward
                self._recompute_from_boundary_gpu = None

        # ── Partial KV Recompute Mode ──
        boundary = self._partial_recompute_boundary
        use_partial = (
            boundary is not None
            and len(self._layer_output_cache) > 0
            and self._is_cache_compatible(hidden_states)
        )

        # 디버그: Partial recompute 시작
        if use_partial:
            print(f"\n[PartialRecompute] 🚀 Starting partial KV recomputation")
            print(f"  Boundary: {boundary}")
            print(f"  Cached layers: {len(self._layer_output_cache)}")
            kv_only_count = 0
            full_forward_count = 0

        for layer_idx, layer_wrapper in enumerate(self.layers):

            if use_partial and layer_idx < boundary:
                # ── KV-only path: 캐시된 hidden states로 KV만 기록 ──

                # 입력 결정: Layer 0은 현재 embedding, 나머지는 이전 레이어 출력
                if layer_idx == 0:
                    kv_input_h = hidden_states
                    kv_input_r = residual
                else:
                    prev_cached = self._layer_output_cache.get(layer_idx - 1)
                    if prev_cached is not None:
                        kv_input_h = prev_cached["output"][0].to(hidden_states.device)
                        kv_input_r = prev_cached["output"][1].to(hidden_states.device) if prev_cached["output"][1] is not None else None
                    else:
                        # Fallback: 현재 hidden states 사용
                        kv_input_h = hidden_states
                        kv_input_r = residual

                # KV-only: norm → qkv_proj → rotary → cache_write
                self._kv_only_forward_layer(
                    layer_wrapper.layer,
                    positions=positions,
                    hidden_states=kv_input_h,
                    residual=kv_input_r,
                )

                # 출력: 현재 레이어 캐시에서
                cached = self._layer_output_cache.get(layer_idx)
                if cached is not None:
                    hidden_states = cached["output"][0].to(hidden_states.device)
                    residual = cached["output"][1].to(hidden_states.device) if cached["output"][1] is not None else None

                    # 디버그: KV-only 카운트
                    if layer_idx == 0 or layer_idx % 5 == 0 or layer_idx == boundary - 1:
                        print(f"  Layer {layer_idx:2d}: ✓ KV-only (cached)")
                    kv_only_count += 1
                    continue

            # ── Normal dual-path forward ──
            # Alpha 값 (tensor, CUDA Graph safe!)
            alpha = layer_wrapper.get_alpha()  # ← Returns tensor!

            # Path A: Layer 통과
            hidden_a, residual_a = self._call_layer_forward_fast(
                layer_wrapper.layer,
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

            # Path B: 레이어 간 직접 연결 (bypass)
            hidden_b = hidden_states  # 이전 값 그대로
            residual_b = residual if residual is not None else None

            # Alpha로 경로 선택
            # Hidden states blending (tensor operations, CUDA Graph safe!)
            hidden_states = alpha * hidden_a + (1.0 - alpha) * hidden_b

            # Residual blending
            if residual_a is not None and residual_b is not None:
                residual = alpha * residual_a + (1.0 - alpha) * residual_b
            elif residual_a is not None:
                residual = alpha * residual_a
            else:
                residual = residual_b

            # ── Persistent GPU buffer에 hidden states 기록 ──
            # index_copy_()는 in-place 연산 → CUDA graph에 캡처됨
            # Prefill(eager): 직접 실행, Decode(graph replay): 자동 실행
            if self._max_cacheable_layer is None or layer_idx <= self._max_cacheable_layer:
                self._init_persistent_buffers(hidden_states.device, hidden_states.dtype)
                self._persistent_h_buffers[layer_idx].index_copy_(0, positions, hidden_states)
                if residual is not None:
                    self._persistent_r_buffers[layer_idx].index_copy_(0, positions, residual)

            # 디버그: Full forward 카운트
            if use_partial and layer_idx >= boundary:
                if layer_idx == boundary or layer_idx % 5 == 0 or layer_idx == len(self.layers) - 1:
                    print(f"  Layer {layer_idx:2d}: ↻ Full forward (recompute)")
                full_forward_count += 1

        # 디버그: Partial recompute 완료 통계
        if use_partial:
            print(f"\n[PartialRecompute] ✅ Completed")
            print(f"  KV-only:      {kv_only_count} layers (skipped attention+MLP)")
            print(f"  Full forward: {full_forward_count} layers (recomputed)")
            savings = (kv_only_count / len(self.layers)) * 100
            print(f"  Savings:      ~{savings:.1f}% of layers optimized\n")

        # Partial recompute는 1회성 (성공 여부 무관, 다음 forward부터 일반 모드)
        if boundary is not None:
            self._partial_recompute_boundary = None

        # Final residual add
        if residual is not None:
            hidden_states = hidden_states + residual

        # Final norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    # ================================================================
    # Partial KV Recomputation Helpers
    # ================================================================

    def _is_cache_compatible(self, current_hidden: torch.Tensor) -> bool:
        """
        캐시된 hidden states가 현재 입력과 호환되는지 확인

        Causal attention + 동일 가중치 → 동일 입력이면 동일 hidden states
        따라서 길이 일치만 확인하면 충분 (값 비교 불필요, CPU-GPU 전송 회피)
        """
        current_len = current_hidden.shape[0]

        # 🔥 Decode phase (seq_len=1)는 partial recompute 불필요 → 즉시 False
        if current_len == 1:
            return False

        if 0 not in self._layer_output_cache:
            print(f"[CacheCheck] ❌ No cached layer 0")
            return False

        cached_len = self._layer_output_cache[0]["output"][0].shape[0]

        compatible = (cached_len == current_len)
        print(f"[CacheCheck] Cached: {cached_len} tokens, Current: {current_len} tokens → "
              f"{'✅ Compatible' if compatible else '❌ Incompatible'}")
        return compatible

    def _kv_only_forward_layer(
        self,
        layer: nn.Module,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> None:
        """
        KV-only forward: norm → qkv_proj → rotary → write_kv_to_cache
        Attention 연산(softmax + o_proj) 및 MLP 실행 안 함.

        지원: Llama/Mistral (self_attn + input_layernorm),
              Falcon (self_attention + ln_attn + query_key_value)
        """
        # Falcon 감지: self_attention + ln_attn 조합
        is_falcon = hasattr(layer, 'self_attention') and hasattr(layer, 'ln_attn')

        # 1. Input layernorm
        if is_falcon:
            # Falcon: parallel attn/mlp 구조, ln_attn만 attention path에 적용
            normed = layer.ln_attn(hidden_states)
        elif hasattr(layer, 'input_layernorm'):
            # Llama/Mistral: fused RMSNorm (hidden_states, residual) or plain call
            if residual is None:
                normed = layer.input_layernorm(hidden_states)
            else:
                try:
                    normed, _ = layer.input_layernorm(hidden_states, residual)
                except TypeError:
                    normed = layer.input_layernorm(hidden_states)
        else:
            normed = hidden_states

        # 2. Attention 모듈 선택
        if is_falcon:
            attn = layer.self_attention
        else:
            attn = getattr(layer, 'self_attn', None)
        if attn is None:
            return

        # 3. QKV projection
        if is_falcon:
            qkv_proj = getattr(attn, 'query_key_value', None)
        else:
            qkv_proj = getattr(attn, 'qkv_proj', None)
        if qkv_proj is None:
            return

        qkv, _ = qkv_proj(normed)

        # Split Q, K, V
        q_size = getattr(attn, 'q_size', None)
        kv_size = getattr(attn, 'kv_size', None)
        if q_size is None or kv_size is None:
            return

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        # Rotary embedding
        rotary_emb = getattr(attn, 'rotary_emb', None)
        if rotary_emb is not None:
            q, k = rotary_emb(positions, q, k)

        # Write to KV cache (skip attention computation)
        attn_module = getattr(attn, 'attn', None)
        if attn_module is not None and hasattr(attn_module, 'write_kv_to_cache'):
            attn_module.write_kv_to_cache(k, v)

    def set_partial_recompute(self, boundary_layer_idx: int) -> None:
        """
        Stage 전환 후 partial KV recomputation 모드 설정.
        boundary_layer_idx 이전 layer는 KV-only, 이후는 full forward.
        """
        if boundary_layer_idx <= 0 or boundary_layer_idx >= len(self.layers):
            print(f"[PartialRecompute] Invalid boundary {boundary_layer_idx}, "
                  f"falling back to full recompute")
            self._partial_recompute_boundary = None
            return

        if len(self._layer_output_cache) == 0:
            print(f"[PartialRecompute] No cached hidden states, "
                  f"falling back to full recompute")
            self._partial_recompute_boundary = None
            return

        self._partial_recompute_boundary = boundary_layer_idx
        print(f"[PartialRecompute] Boundary set at layer {boundary_layer_idx}")
        print(f"  Layers 0-{boundary_layer_idx-1}: KV-only (cached hidden states)")
        print(f"  Layers {boundary_layer_idx}-{len(self.layers)-1}: full forward")

    def set_recompute_from_boundary_gpu(self, boundary_layer_idx: int) -> bool:
        """
        GPU-resident partial recompute 모드 설정 (Method A).

        Stage 전환 후 호출. Front layers KV cache는 가중치 변경 없으므로 유효.
        _persistent_h_buffers[boundary-1]에서 boundary hidden states를 직접 읽어
        back layers만 재계산. CPU 복사 없음, KV-only pass 없음.

        Returns:
            True: 모드 설정 성공
            False: 버퍼 미초기화 등으로 설정 불가 (일반 forward로 진행됨)
        """
        if boundary_layer_idx <= 0 or boundary_layer_idx >= len(self.layers):
            print(f"[GPURecompute] Invalid boundary {boundary_layer_idx} "
                  f"(layers: {len(self.layers)}), GPU mode not set")
            return False

        if not self._persistent_buffers_initialized:
            print(f"[GPURecompute] Persistent buffers not initialized yet, "
                  f"GPU mode not available")
            return False

        self._recompute_from_boundary_gpu = boundary_layer_idx
        print(f"[GPURecompute] ✅ GPU-resident mode set: boundary={boundary_layer_idx}")
        print(f"  Front layers 0-{boundary_layer_idx-1}: skipped (KV cache already valid)")
        print(f"  Back layers {boundary_layer_idx}-{len(self.layers)-1}: full forward")
        return True

    def clear_hidden_cache(self) -> None:
        """Hidden state 캐시 초기화"""
        self._layer_output_cache.clear()
        self._partial_recompute_boundary = None
        self._recompute_from_boundary_gpu = None
    
    def _call_layer_forward_fast(
        self,
        layer,
        positions,
        hidden_states,
        residual,
    ):
        """
        초기화 시 선택된 고정 모드로 레이어 forward를 호출.
        (per-token try/except 디스패치 제거)
        """
        mode = self._layer_forward_mode

        if mode == "kwargs_v1":
            output = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        elif mode == "kwargs_no_residual":
            output = layer(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            output = layer(positions, hidden_states, residual)

        if isinstance(output, tuple):
            return output
        return output, None
    
    # ================================================================
    # Layer Activation (Weight Loading) - Universal
    # ================================================================
    
    def activate_layers(
        self,
        layer_indices: List[int],
        checkpoint_path: str,
    ) -> None:
        """
        레이어 활성화: alpha 0→1 + weight 로드 (범용)
        
        CUDA Graph 호환:
        - .copy_()로 in-place weight 로드
        - alpha.fill_()로 in-place alpha 업데이트
        - Topology 불변 (레이어는 계속 실행됨)
        """
        print(f"\n{'='*60}")
        print(f"ACTIVATING LAYERS: {layer_indices}")
        print(f"Model Type: {self.model_type}")
        print(f"{'='*60}")
        
        # Checkpoint 로드
        print(f"Loading checkpoint from: {checkpoint_path}")
        state_dict = load_file(checkpoint_path)
        
        device = next(self.parameters()).device
        
        # Get weight naming pattern for this model
        weight_pattern = get_weight_pattern(self.model_type)
        
        for layer_idx in layer_indices:
            print(f"\n📂 Activating layer {layer_idx}...")
            
            layer_wrapper = self.layers[layer_idx]
            
            # 이미 활성화된 레이어
            if layer_wrapper.is_active():
                print(f"  ℹ️  Layer {layer_idx} is already active")
                continue
            
            # 1. Weight 추출
            print(f"  🔥 Loading weights...")
            layer_prefix = f"model.layers.{layer_idx}."
            layer_weights = {
                k.replace(layer_prefix, ""): v
                for k, v in state_dict.items()
                if k.startswith(layer_prefix)
            }
            
            if not layer_weights:
                print(f"  ⚠️  No weights found for layer {layer_idx}")
                continue
            
            # 2. In-place weight 로드 (범용, CUDA Graph 호환!)
            loaded_count = self._load_layer_weights(
                layer_wrapper.layer,
                layer_weights,
                weight_pattern,
                device,
            )
            
            print(f"  ✅ Loaded {loaded_count} weight tensors")
            
            # 3. Alpha 활성화 (0 → 1)
            layer_wrapper.activate()
            
            # 4. initially_inactive에서 제거
            self.initially_inactive.discard(layer_idx)
            
            print(f"  ✅ Layer {layer_idx} activated!")
        
        print(f"\n{'='*60}")
        print(f"LAYER ACTIVATION COMPLETE")
        print(f"Inactive layers: {self.count_inactive_layers()}")
        print(f"ℹ️  Topology는 고정되지만, vLLM 런타임에서 graph 재캡처가 발생할 수 있음")
        print(f"{'='*60}\n")
    
    def prefetch_weights(self, checkpoint_path: str, layer_indices: List[int]) -> None:
        """
        백그라운드 스레드에서 checkpoint를 CPU 메모리에 미리 로드.
        서빙 중 디스크 I/O를 미리 처리 → 전환 시 GPU copy만 남음.

        안전장치:
        - 이미 동일 indices로 완료된 prefetch는 skip
        - 진행 중인 prefetch가 있으면 완료 대기 후 새로 시작
        - worker 예외 발생 시에도 event는 반드시 set (blocking 방지)
        """
        # 이미 동일 indices로 완료된 경우 skip
        if (hasattr(self, '_prefetch_event')
                and self._prefetch_event.is_set()
                and hasattr(self, '_prefetch_indices')
                and self._prefetch_indices == list(layer_indices)):
            print("[Prefetch] Already completed for these layers, skipping")
            return

        # 진행 중인 prefetch가 있으면 완료 대기
        if hasattr(self, '_prefetch_event') and not self._prefetch_event.is_set():
            print("[Prefetch] Waiting for previous prefetch to finish...")
            self._prefetch_event.wait()

        self._prefetch_buffer = None
        self._prefetch_indices = list(layer_indices)
        self._prefetch_path = checkpoint_path
        self._prefetch_event = threading.Event()

        def _worker():
            try:
                print(f"[Prefetch] Loading {checkpoint_path} in background...")
                state_dict = load_file(checkpoint_path)
                state_dict = {k: v.pin_memory() for k, v in state_dict.items()}
                self._prefetch_buffer = state_dict
                print(f"[Prefetch] ✅ {len(state_dict)} tensors ready in CPU pinned memory")
            except Exception as e:
                print(f"[Prefetch] ❌ Failed: {e}")
                self._prefetch_buffer = None
            finally:
                self._prefetch_event.set()  # 예외가 나도 반드시 set

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

    def activate_layers_instant(
        self,
        layer_indices: List[int],
        wait_if_needed: bool = True,
    ) -> bool:
        """
        prefetch_weights()로 CPU에 올려둔 버퍼에서 즉각 활성화.
        디스크 I/O 없이 GPU copy + alpha 변경만 실행.

        Returns:
            True: 성공
            False: prefetch 미완료 (wait_if_needed=False)
        """
        if not hasattr(self, '_prefetch_event'):
            raise RuntimeError("prefetch_weights()를 먼저 호출하세요.")

        if not self._prefetch_event.is_set():
            if wait_if_needed:
                print("[Prefetch] Waiting for background load to finish...")
                self._prefetch_event.wait()
            else:
                print("[Prefetch] Not ready yet.")
                return False

        if self._prefetch_buffer is None:
            raise RuntimeError("[Prefetch] 버퍼가 비어 있습니다. prefetch가 실패했습니다.")

        # indices 검증
        if set(layer_indices) != set(self._prefetch_indices):
            raise ValueError(
                f"Layer indices mismatch: prefetch={self._prefetch_indices}, "
                f"requested={layer_indices}"
            )

        state_dict = self._prefetch_buffer
        device = next(self.parameters()).device
        weight_pattern = get_weight_pattern(self.model_type)

        print(f"\n{'='*60}")
        print(f"INSTANT ACTIVATION: {layer_indices}")
        print(f"{'='*60}")

        try:
            for layer_idx in layer_indices:
                layer_wrapper = self.layers[layer_idx]

                if layer_wrapper.is_active():
                    print(f"  Layer {layer_idx}: already active")
                    continue

                layer_prefix = f"model.layers.{layer_idx}."
                layer_weights = {
                    k.replace(layer_prefix, ""): v
                    for k, v in state_dict.items()
                    if k.startswith(layer_prefix)
                }

                if not layer_weights:
                    print(f"  ⚠️ No weights for layer {layer_idx}")
                    continue

                loaded = self._load_layer_weights(
                    layer_wrapper.layer, layer_weights, weight_pattern, device
                )
                print(f"  ✅ Layer {layer_idx}: {loaded} tensors → GPU")

                layer_wrapper.activate()
                self.initially_inactive.discard(layer_idx)
                print(f"  ✅ Layer {layer_idx} activated (alpha 0→1)")

            print(f"\n✅ Instant activation complete")
            print(f"ℹ️  Topology는 고정되지만, vLLM 런타임에서 graph 재캡처가 발생할 수 있음\n")
            return True

        finally:
            # 성공/실패 관계없이 전체 prefetch 상태 정리
            self._prefetch_buffer = None
            if hasattr(self, '_prefetch_event'):
                del self._prefetch_event
            if hasattr(self, '_prefetch_path'):
                del self._prefetch_path
            if hasattr(self, '_prefetch_indices'):
                del self._prefetch_indices

    def is_prefetch_ready(self) -> bool:
        """prefetch 완료 여부 확인 (non-blocking)"""
        return (
            hasattr(self, '_prefetch_event')
            and self._prefetch_event.is_set()
            and self._prefetch_buffer is not None
        )

    def wait_for_prefetch(self, timeout_s: Optional[float] = None) -> bool:
        """
        prefetch 완료까지 대기.

        Returns:
            True: prefetch 완료 + 버퍼 준비됨
            False: 아직 미완료/실패/미시작
        """
        if not hasattr(self, '_prefetch_event'):
            return False

        if timeout_s is None:
            finished = self._prefetch_event.wait()
        else:
            finished = self._prefetch_event.wait(timeout=timeout_s)

        if not finished:
            return False
        return self._prefetch_buffer is not None

    def get_prefetch_status(self) -> Dict[str, Any]:
        """prefetch 상태 스냅샷 반환"""
        has_event = hasattr(self, '_prefetch_event')
        ready = self.is_prefetch_ready()
        in_progress = has_event and (not getattr(self, '_prefetch_event').is_set())

        return {
            "started": has_event,
            "ready": ready,
            "in_progress": in_progress,
            "checkpoint_path": getattr(self, '_prefetch_path', None),
            "layer_indices": list(getattr(self, '_prefetch_indices', [])),
        }

    def _load_layer_weights(
        self,
        layer: nn.Module,
        layer_weights: Dict[str, torch.Tensor],
        weight_pattern: Any,
        device: torch.device,
    ) -> int:
        """
        범용 가중치 로딩 로직
        
        모델별 가중치 이름 패턴에 따라 자동으로 처리합니다.
        """
        loaded_count = 0
        
        for name, param in layer.named_parameters():
            # QKV fusion 처리
            if weight_pattern.qkv_fused_name and weight_pattern.qkv_fused_name in name:
                qkv_loaded = self._load_qkv_fused(
                    param, name, layer_weights, weight_pattern, device
                )
                if qkv_loaded:
                    loaded_count += 1
                    continue
            
            # MLP Gate-Up fusion 처리
            if weight_pattern.mlp_fused_name and weight_pattern.mlp_fused_name in name:
                mlp_loaded = self._load_mlp_fused(
                    param, name, layer_weights, weight_pattern, device
                )
                if mlp_loaded:
                    loaded_count += 1
                    continue
            
            # 일반 weights (direct match)
            if name in layer_weights:
                param.data.copy_(layer_weights[name], non_blocking=True)
                loaded_count += 1
        
        return loaded_count
    
    def _load_qkv_fused(
        self,
        param,
        param_name: str,
        layer_weights: Dict[str, torch.Tensor],
        weight_pattern: Any,
        device: torch.device,
    ) -> bool:
        """QKV fusion weight 로드"""
        # Build expected weight names
        weight_names = []
        for proj_name in weight_pattern.qkv_weights:
            # Extract base path from param_name
            base_path = param_name.replace(f".{weight_pattern.qkv_fused_name}.weight", "")
            weight_name = f"{base_path}.{proj_name}.weight"
            weight_name = weight_name.lstrip('.')  # Remove leading dot
            weight_names.append(weight_name)
        
        # Check if all weights exist
        if all(name in layer_weights for name in weight_names):
            offset = 0
            for name in weight_names:
                t = layer_weights[name]
                n = t.shape[0]
                param.data[offset : offset + n].copy_(t, non_blocking=True)
                offset += n
            print(f"  ✅ Loaded fused QKV ({len(weight_names)} weights → {offset} rows)")
            return True
        
        return False
    
    def _load_mlp_fused(
        self,
        param,
        param_name: str,
        layer_weights: Dict[str, torch.Tensor],
        weight_pattern: Any,
        device: torch.device,
    ) -> bool:
        """MLP Gate-Up fusion weight 로드"""
        if not weight_pattern.mlp_gate_up:
            return False
        
        # Build expected weight names
        weight_names = []
        for proj_name in weight_pattern.mlp_gate_up:
            base_path = param_name.replace(f".{weight_pattern.mlp_fused_name}.weight", "")
            weight_name = f"{base_path}.{proj_name}.weight"
            weight_name = weight_name.lstrip('.')
            weight_names.append(weight_name)
        
        # Check if all weights exist
        if all(name in layer_weights for name in weight_names):
            offset = 0
            for name in weight_names:
                t = layer_weights[name]
                n = t.shape[0]
                param.data[offset : offset + n].copy_(t, non_blocking=True)
                offset += n
            print(f"  ✅ Loaded fused MLP ({len(weight_names)} weights → {offset} rows)")
            return True
        
        return False
    
    # ================================================================
    # Status Methods (CUDA Graph safe!)
    # ================================================================
    
    def get_layer_status(self) -> Dict[int, Dict]:
        """레이어 상태 확인"""
        status = {}
        for i, layer in enumerate(self.layers):
            alpha_value = layer.get_alpha_value()
            
            status[i] = {
                "type": "DualPath",
                "active": layer.is_active(),
                "alpha": alpha_value,
                "path": "A" if alpha_value > 0.5 else "B"
            }
        return status
    
    def count_inactive_layers(self) -> int:
        """비활성 레이어 개수"""
        count = 0
        for layer in self.layers:
            if not layer.is_active():
                count += 1
        return count
    
    def print_layer_status(self) -> None:
        """레이어 상태 출력"""
        status = self.get_layer_status()
        
        print("\n" + "="*60)
        print(f"LAYER STATUS (Dual-Path, {self.model_type.upper()})")
        print("="*60)
        
        for i in range(0, len(status), 10):
            print(f"\nLayers {i:2d}-{min(i+9, len(status)-1):2d}:")
            for j in range(i, min(i+10, len(status))):
                info = status[j]
                alpha = info['alpha']
                path = info['path']
                symbol = "◉" if alpha > 0.5 else "⊗"
                print(f"  L{j:2d}: {symbol} alpha={alpha:.1f} (Path {path})")
        
        print(f"\nTotal layers: {len(status)}")
        print(f"Path A (active):   {len(status) - self.count_inactive_layers()}")
        print(f"Path B (bypass):   {self.count_inactive_layers()}")
        print("="*60 + "\n")
    
    # ================================================================
    # Additional Status Methods
    # ================================================================
    
    def verify_recovery(self) -> Dict[str, Any]:
        """Progressive recovery 상태 확인"""
        active = []
        inactive = []
        
        for i, layer in enumerate(self.layers):
            if layer.is_active():
                active.append(i)
            else:
                inactive.append(i)
        
        return {
            "active_layers": active,
            "inactive_layers": inactive,
            "inactive_layer_indices": inactive,
            "activation_progress": f"{len(active)}/{len(self.layers)}",
            "model_type": self.model_type,
        }
    
    def get_adapter_info(self) -> Dict[str, Any]:
        """Adapter 정보"""
        return {
            "current_adapter": self.current_adapter,
            "adapter_enabled": self.current_adapter is not None
        }
