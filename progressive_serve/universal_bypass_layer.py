"""
Universal Bypass Layer - base layer를 alpha 값과 함께 감싸는 wrapper.

forward는 base layer를 그대로 호출하고, two-path blending
(alpha * Path_A + (1-alpha) * Path_B)은 모델의 forward 루프에서 처리.
CUDA Graph 호환: forward에서 .item() 호출 없음, alpha는 in-place fill_()로만 업데이트.
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
        """alpha 값을 tensor로 반환. forward 루프에서 사용 (CUDA Graph safe)."""
        return self.alpha

    def get_alpha_value(self) -> float:
        """
        alpha 값을 float으로 반환.
        WARNING: CUDA Graph 캡처 중 사용 금지. 로깅/상태 출력에만 사용.
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
