"""
Universal Bypass Layer - Simple Wrapper (CUDA Graph Safe)
progressive_serve/universal_bypass_layer.py

✅ 단순 wrapper (alpha 관리만)
✅ Forward는 base layer 그대로 호출
✅ Two-path blending은 모델 forward에서 처리
✅ CUDA Graph compatible: No .item() calls in forward!
"""

import torch
import torch.nn as nn
import os
from typing import Optional


class UniversalBypassLayer(nn.Module):
    """
    Universal Bypass Layer - Simple Wrapper
    
    역할:
    - Base layer를 감싸기
    - Alpha 값 관리
    - Forward는 단순히 base layer 호출만
    
    Two-path blending은 모델의 forward 루프에서 처리
    
    CUDA Graph Safety:
    - get_alpha() returns tensor (not float!)
    - No .item() calls during forward pass
    - Alpha updates via .fill_() (in-place, graph-safe)
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        initial_alpha: float = 0.0,
        layer_idx: Optional[int] = None,
    ):
        """
        Args:
            base_layer: 원본 vLLM layer
            initial_alpha: 초기 alpha 값
            layer_idx: Layer 인덱스 (로깅용)
        """
        super().__init__()
        
        # Base layer
        self.layer = base_layer
        self.layer_idx = layer_idx
        
        # Alpha buffer (CUDA Graph safe)
        # register_buffer ensures it's on GPU and tracked by module
        self.register_buffer('alpha', torch.tensor(initial_alpha))
        self._alpha_update_mode = os.environ.get("P2_ALPHA_UPDATE_MODE", "inplace").strip().lower()
        if self._alpha_update_mode not in ("inplace", "rebind"):
            self._alpha_update_mode = "inplace"
        
        # State tracking (Python bool, not tensor - safe for conditional logic)
        self._is_active = initial_alpha > 0.5
    
    def forward(self, *args, **kwargs):
        """
        단순히 base layer 호출만
        
        Two-path blending은 모델 forward에서 처리
        """
        return self.layer(*args, **kwargs)
    
    # ================================================================
    # Alpha 관리
    # ================================================================
    
    def activate(self):
        """레이어 활성화 (alpha = 1.0)"""
        if self._alpha_update_mode == "rebind":
            self.alpha = torch.tensor(1.0, device=self.alpha.device, dtype=self.alpha.dtype)
        else:
            self.alpha.fill_(1.0)
        self._is_active = True
        if self.layer_idx is not None:
            # ✅ SAFE: .item() only during non-forward operations
            print(f"✅ Layer {self.layer_idx} activated (alpha={self.alpha.item():.1f} → LAYER path)")
    
    def deactivate(self):
        """레이어 비활성화 (alpha = 0.0)"""
        if self._alpha_update_mode == "rebind":
            self.alpha = torch.tensor(0.0, device=self.alpha.device, dtype=self.alpha.dtype)
        else:
            self.alpha.fill_(0.0)
        self._is_active = False
        if self.layer_idx is not None:
            print(f"⊗ Layer {self.layer_idx} deactivated (alpha={self.alpha.item():.1f} → BYPASS path)")
    
    def set_alpha(self, value: float):
        """Alpha 값 직접 설정"""
        if self._alpha_update_mode == "rebind":
            self.alpha = torch.tensor(float(value), device=self.alpha.device, dtype=self.alpha.dtype)
        else:
            self.alpha.fill_(value)
        self._is_active = value > 0.5
    
    def is_active(self) -> bool:
        """활성화 여부 (Python bool, CUDA Graph safe)"""
        return self._is_active
    
    def get_alpha(self) -> torch.Tensor:
        """
        현재 alpha 값 (tensor 반환)
        
        CUDA Graph Compatibility:
        - Returns tensor, not float!
        - No .item() call during forward pass
        - Use this in forward loops
        
        Returns:
            torch.Tensor: Alpha value as 0-d tensor (scalar)
        """
        return self.alpha
    
    def get_alpha_value(self) -> float:
        """
        현재 alpha 값 (float 반환)
        
        WARNING: Only use outside of CUDA Graph capture!
        - For logging, debugging, status printing
        - NOT for forward pass computations
        
        Returns:
            float: Alpha value as Python float
        """
        return self.alpha.item()
    
    # ================================================================
    # Properties
    # ================================================================
    
    @property
    def is_alpha_gated(self) -> bool:
        """AlphaGatedLayer 호환"""
        return True
    
    @property
    def is_universal_bypass(self) -> bool:
        """UniversalBypassLayer 식별자"""
        return True


# Structural mutation variant: injects a deterministic no-op op to force
# a different CUDA graph while preserving outputs.
class StructuralMutationBypassLayer(UniversalBypassLayer):
    def forward(self, *args, **kwargs):
        # Try to locate positions/hidden_states/residual in args/kwargs.
        use_kwargs = False
        positions = None
        hidden_states = None
        residual = None

        if "hidden_states" in kwargs:
            use_kwargs = True
            hidden_states = kwargs.get("hidden_states", None)
            positions = kwargs.get("positions", None)
            residual = kwargs.get("residual", None)
        elif len(args) >= 2 and torch.is_tensor(args[1]):
            positions = args[0] if (len(args) > 0 and torch.is_tensor(args[0])) else None
            hidden_states = args[1]
            residual = args[2] if (len(args) > 2 and torch.is_tensor(args[2])) else None

        if hidden_states is None or not torch.is_tensor(hidden_states):
            return self.layer(*args, **kwargs)

        # Default: deterministic no-op on hidden_states (same shape).
        if os.environ.get("STRUCTURAL_MUTATION_REAL_SHAPE", "0").lower() not in ("1", "true", "yes", "on"):
            doubled = torch.cat((hidden_states, hidden_states), dim=-1)
            hidden_states = doubled[..., :hidden_states.shape[-1]]
            if use_kwargs:
                kwargs["hidden_states"] = hidden_states
            else:
                args = (positions, hidden_states, residual) if len(args) >= 2 else args
            return self.layer(*args, **kwargs)

        # Aggressive: append a dummy token (seq+1), run layer, then slice back.
        # This changes the captured shape while preserving outputs for original tokens
        # under causal masking.
        try:
            if positions is None or positions.numel() == 0:
                return self.layer(*args, **kwargs)

            # Expect token-major [T, H] or [T, ...]
            if hidden_states.dim() < 2 or hidden_states.shape[0] != positions.shape[0]:
                return self.layer(*args, **kwargs)

            pad_h = torch.zeros_like(hidden_states[:1])
            hs_pad = torch.cat([hidden_states, pad_h], dim=0)

            pos_last = positions[-1:]
            pos_pad = torch.cat([positions, pos_last + 1], dim=0)

            if residual is not None and torch.is_tensor(residual) and residual.shape == hidden_states.shape:
                res_pad = torch.cat([residual, torch.zeros_like(residual[:1])], dim=0)
            else:
                res_pad = residual

            if use_kwargs:
                out = self.layer(positions=pos_pad, hidden_states=hs_pad, residual=res_pad)
            else:
                if len(args) >= 3:
                    out = self.layer(pos_pad, hs_pad, res_pad)
                else:
                    out = self.layer(pos_pad, hs_pad)

            def _slice_out(x):
                if torch.is_tensor(x) and x.shape[0] == hs_pad.shape[0]:
                    return x[:hidden_states.shape[0]]
                return x

            if isinstance(out, tuple):
                return tuple(_slice_out(o) for o in out)
            return _slice_out(out)
        except Exception:
            # Fallback to original behavior if anything goes wrong.
            return self.layer(*args, **kwargs)


# Backward compatibility
AlphaGatedLayer = UniversalBypassLayer
