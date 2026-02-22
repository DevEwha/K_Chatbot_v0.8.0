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
        self.alpha.fill_(1.0)
        self._is_active = True
        if self.layer_idx is not None:
            # ✅ SAFE: .item() only during non-forward operations
            print(f"✅ Layer {self.layer_idx} activated (alpha={self.alpha.item():.1f} → LAYER path)")
    
    def deactivate(self):
        """레이어 비활성화 (alpha = 0.0)"""
        self.alpha.fill_(0.0)
        self._is_active = False
        if self.layer_idx is not None:
            print(f"⊗ Layer {self.layer_idx} deactivated (alpha={self.alpha.item():.1f} → BYPASS path)")
    
    def set_alpha(self, value: float):
        """Alpha 값 직접 설정"""
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


# Backward compatibility
AlphaGatedLayer = UniversalBypassLayer
