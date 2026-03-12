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
import time
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

        # ── Partial KV Recomputation (fallback) ──
        # layer_idx → {"output": (hidden_states_gpu, residual_gpu)}
        self._layer_output_cache: Dict[int, Any] = {}
        # None이면 일반 forward, 정수면 해당 layer부터 full forward
        self._partial_recompute_boundary: Optional[int] = None
        # 캐싱할 최대 레이어 인덱스 (다음 stage의 boundary-1)
        self._max_cacheable_layer: Optional[int] = None

        # ── KV Block Surgery ──
        # Decode 단계에서 저장: full sequence의 physical block mapping
        self._surgery_block_tables: Optional[torch.Tensor] = None  # [1, max_blocks]
        self._surgery_seq_lens_tensor: Optional[torch.Tensor] = None  # [1]

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

    # ----------------------------------------------------------------
    # Persistent Buffer → GPU Cache 동기화 (CPU 전송 제거)
    # ----------------------------------------------------------------
    def sync_persistent_cache(self, seq_len: int):
        """
        GPU persistent buffer의 현재 상태를 스냅샷하여 _layer_output_cache에 저장.
        
        [최적화 내용]
        - 기존의 .cpu() 호출을 제거하여 악명 높은 PCIe 대역폭 병목(D2H)을 없앴습니다.
        - 대신 GPU 내에서 .clone()을 사용하여 빠르게 복사합니다.
        - clone()을 사용하는 이유는 partial recompute 도중 버퍼에 in-place 기록이 
          발생하여 참조가 꼬이는 것을 방지하기 위함입니다 (VRAM to VRAM 복사는 1ms 이하로 매우 빠름).
        """
        if not self._persistent_buffers_initialized:
            print(f"[Cache] ⚠️ Persistent buffers not initialized")
            return

        max_layer = self._max_cacheable_layer if self._max_cacheable_layer is not None else len(self.layers) - 1

        self._layer_output_cache.clear()
        
        # GPU 내에서 바로 슬라이싱 및 복제 수행
        for layer_idx in range(max_layer + 1):
            h = self._persistent_h_buffers[layer_idx][:seq_len].clone()
            r = self._persistent_r_buffers[layer_idx][:seq_len].clone()
            self._layer_output_cache[layer_idx] = {"output": (h, r)}

        print(f"[Cache] Synced {max_layer + 1} layers × {seq_len} tokens (GPU-only, No CPU transfer)")
    # def sync_persistent_cache(self, seq_len: int):
    #     """
    #     GPU persistent buffer → CPU _layer_output_cache

    #     Stage 전환 직전에 chatbot에서 호출.
    #     GPU buffer의 [0:seq_len] 구간을 CPU로 복사하여 partial recompute에 사용.
    #     """
    #     if not self._persistent_buffers_initialized:
    #         print(f"[Cache] ⚠️ Persistent buffers not initialized")
    #         return

    #     max_layer = self._max_cacheable_layer if self._max_cacheable_layer is not None else len(self.layers) - 1

    #     self._layer_output_cache.clear()
    #     for layer_idx in range(max_layer + 1):
    #         h = self._persistent_h_buffers[layer_idx][:seq_len].cpu()
    #         r = self._persistent_r_buffers[layer_idx][:seq_len].cpu()
    #         self._layer_output_cache[layer_idx] = {"output": (h, r)}

    #     print(f"[Cache] Synced {max_layer + 1} layers × {seq_len} tokens (GPU → CPU)")

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

        # ── KV Surgery: block_tables 추적 ──
        # CUDA graph 호환: clone() 대신 view 저장
        #   - CUDA graph 캡처 시: Python forward() 실행 → view 저장
        #   - CUDA graph replay 시: prepare_graph_input_buffers()가 원본 텐서 in-place 업데이트
        #     → view가 자동으로 최신 값 반영 (라이브 포인터)
        #   - Eager 모드 시: 매 step마다 forward() 실행 → view 갱신
        # 주의: clone() 사용 시 CUDA graph 캡처 시점의 dummy 값이 고정됨 → 버그!
        try:
            from vllm.forward_context import get_forward_context as _get_fwd_ctx
            _fwd_meta = _get_fwd_ctx().attn_metadata
            if (_fwd_meta is not None
                    and getattr(_fwd_meta, 'num_decode_tokens', 0) > 0
                    and getattr(_fwd_meta, 'block_tables', None) is not None
                    and _fwd_meta.block_tables.numel() > 0
                    and getattr(_fwd_meta, 'seq_lens_tensor', None) is not None
                    and _fwd_meta.seq_lens_tensor.numel() > 0):
                # view (not clone): 원본 텐서와 동일한 메모리 공유
                # → CUDA graph replay 전 prepare_graph_input_buffers()가 원본 업데이트 시 자동 반영
                self._surgery_block_tables = _fwd_meta.block_tables[:1]
                self._surgery_seq_lens_tensor = _fwd_meta.seq_lens_tensor[:1]
        except Exception:
            pass

        # Embedding
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_tokens(input_ids)

        residual = None

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

        지원: Llama, Mistral, Qwen2, Gemma 등 self_attn 패턴 모델
        """
        # 1. Input layernorm
        if hasattr(layer, 'input_layernorm'):
            if residual is None:
                normed = layer.input_layernorm(hidden_states)
            else:
                normed, _ = layer.input_layernorm(hidden_states, residual)
        else:
            normed = hidden_states

        # 2. QKV projection + rotary + cache write
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            return

        # qkv_proj
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

    def clear_hidden_cache(self) -> None:
        """Hidden state 캐시 초기화"""
        self._layer_output_cache.clear()
        self._partial_recompute_boundary = None

    # ================================================================
    # True KV Block Surgery
    # ================================================================

    def inject_upper_layer_kv(
        self,
        boundary: int,
        seq_len: Optional[int] = None,
    ) -> bool:
        """
        Stage 전환 시 upper layer KV만 업데이트하는 진짜 KV block surgery.

        원리:
        - Lower layers (0..boundary-1): 가중치 동일 → KV 불변 → 손대지 않음
        - Upper layers (boundary..N-1): 새 가중치로 forward → 동일 physical block에 덮어씀
        - vLLM prefix cache hash→block 매핑 유지 → 다음 generate()에서 full prefix hit!
        - 결과: 다음 generate()에서 prefill 완전 스킵

        Returns:
            True: surgery 성공
            False: 실패 (호출자가 reset_prefix_cache + partial recompute로 fallback)
        """
        if not self._persistent_buffers_initialized:
            print("[Surgery] ❌ Persistent buffers not initialized")
            return False

        if self._surgery_block_tables is None:
            print("[Surgery] ❌ No block tables saved (no decode step happened yet)")
            return False

        # seq_len 결정
        if seq_len is None:
            if self._surgery_seq_lens_tensor is None:
                print("[Surgery] ❌ No seq_lens_tensor saved")
                return False
            seq_len = int(self._surgery_seq_lens_tensor[0].item())

        if seq_len < 1:
            print(f"[Surgery] ❌ Invalid seq_len={seq_len}")
            return False

        if boundary <= 0 or boundary >= len(self.layers):
            print(f"[Surgery] ❌ Invalid boundary {boundary} for {len(self.layers)} layers")
            return False

        device = self._persistent_h_buffers[0].device
        block_size = self.vllm_config.cache_config.block_size
        num_blocks_needed = (seq_len + block_size - 1) // block_size

        if self._surgery_block_tables.shape[1] < num_blocks_needed:
            print(f"[Surgery] ❌ Not enough blocks: have {self._surgery_block_tables.shape[1]}, "
                  f"need {num_blocks_needed} for seq_len={seq_len}")
            return False

        print(f"\n[Surgery] 🔪 KV block surgery starting")
        print(f"  Lower layers 0~{boundary-1}: KV 보존 (가중치 동일, 연산 없음)")
        print(f"  Upper layers {boundary}~{len(self.layers)-1}: 새 가중치로 KV 재계산")
        print(f"  Seq len: {seq_len} tokens | Blocks: {num_blocks_needed}")

        # block_tables에서 slot_mapping 재구성
        try:
            slot_mapping = self._reconstruct_slot_mapping(
                self._surgery_block_tables[:1, :num_blocks_needed],
                seq_len, block_size, device,
            )
        except Exception as e:
            print(f"[Surgery] ❌ slot_mapping 재구성 실패: {e}")
            return False

        # boundary-1 레이어의 출력 = boundary 레이어의 입력
        h = self._persistent_h_buffers[boundary - 1][:seq_len].clone()
        r = self._persistent_r_buffers[boundary - 1][:seq_len].clone()

        positions = torch.arange(seq_len, device=device, dtype=torch.long)

        # Surgery용 FlashAttentionMetadata 구성
        # context_lens_tensor=zeros → 전체 seq_len 토큰을 fresh prefill로 처리
        # → K,V cache write + full causal attention 계산
        try:
            from vllm.attention.backends.flash_attn import FlashAttentionMetadata

            surgery_meta = FlashAttentionMetadata(
                num_prefills=1,
                num_prefill_tokens=seq_len,
                num_decode_tokens=0,
                slot_mapping=slot_mapping,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=False,
                seq_lens=[seq_len],
                seq_lens_tensor=torch.tensor(
                    [seq_len], device=device, dtype=torch.int32),
                max_prefill_seq_len=seq_len,
                max_decode_seq_len=0,
                context_lens_tensor=torch.zeros(
                    1, dtype=torch.int32, device=device),
                block_tables=self._surgery_block_tables[:1, :num_blocks_needed],
                use_cuda_graph=False,
                max_query_len=seq_len,
                query_start_loc=torch.tensor(
                    [0, seq_len], device=device, dtype=torch.int32),
                seq_start_loc=torch.tensor(
                    [0, seq_len], device=device, dtype=torch.int32),
                max_decode_query_len=0,
            )
        except Exception as e:
            print(f"[Surgery] ❌ Metadata 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Upper layers를 surgery ForwardContext 안에서 직접 실행
        t0 = time.perf_counter()
        try:
            from vllm.forward_context import set_forward_context
            with torch.inference_mode():
                with set_forward_context(
                    surgery_meta, self.vllm_config, virtual_engine=0
                ):
                    for layer_idx in range(boundary, len(self.layers)):
                        layer_wrapper = self.layers[layer_idx]
                        alpha = layer_wrapper.get_alpha()

                        # Full layer forward: attention이 slot_mapping으로 KV 쓰기 수행
                        hidden_a, residual_a = self._call_layer_forward_fast(
                            layer_wrapper.layer,
                            positions=positions,
                            hidden_states=h,
                            residual=r,
                        )

                        hidden_b = h
                        residual_b = r

                        h = alpha * hidden_a + (1.0 - alpha) * hidden_b

                        if residual_a is not None and residual_b is not None:
                            r = alpha * residual_a + (1.0 - alpha) * residual_b
                        elif residual_a is not None:
                            r = alpha * residual_a
                        else:
                            r = residual_b

        except Exception as e:
            print(f"[Surgery] ❌ Surgery forward 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"  ✅ KV surgery 완료 ({elapsed_ms:.1f} ms)")
        print(f"  📌 Lower (0~{boundary-1}): KV 그대로 (재계산 없음)")
        print(f"  📌 Upper ({boundary}~{len(self.layers)-1}): KV 업데이트됨")
        print(f"  📌 Prefix cache 유지 → 다음 generate()에서 prefill 스킵\n")

        return True

    def _reconstruct_slot_mapping(
        self,
        block_tables: torch.Tensor,  # [1, num_blocks]
        seq_len: int,
        block_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """block_tables + seq_len → flat slot_mapping [seq_len] 재구성."""
        token_indices = torch.arange(seq_len, dtype=torch.long, device=device)
        logical_blocks = token_indices // block_size
        offsets = token_indices % block_size
        physical_blocks = block_tables[0, logical_blocks]
        return physical_blocks * block_size + offsets

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
        
        # non_blocking=True copy가 모두 GPU에서 완료될 때까지 대기
        torch.cuda.synchronize()
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
                self._prefetch_buffer = state_dict
                print(f"[Prefetch] ✅ {len(state_dict)} tensors ready in CPU memory")
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

            # non_blocking=True copy가 모두 GPU에서 완료될 때까지 대기
            # surgery / forward가 이 weights를 즉시 사용하므로 필수
            torch.cuda.synchronize()
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
                w = layer_weights[name]
                param.data[offset:offset + w.shape[0]].copy_(w, non_blocking=True)
                offset += w.shape[0]
            print(f"  ✅ Loaded fused QKV ({len(weight_names)} weights)")
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
                w = layer_weights[name]
                param.data[offset:offset + w.shape[0]].copy_(w, non_blocking=True)
                offset += w.shape[0]
            print(f"  ✅ Loaded fused MLP ({len(weight_names)} weights)")
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
