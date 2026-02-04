"""Flash Attention optimization utilities."""

from typing import Optional
import torch
import torch.nn as nn


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal_mask: bool = True,
    block_size: int = 128,
    dropout: Optional[nn.Module] = None
) -> torch.Tensor:
    """Flash Attention implementation (simplified block-wise version).
    
    For production use, consider using the flash-attn library which provides
    optimized CUDA kernels.
    
    Args:
        Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
        K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
        V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
        causal_mask: Whether to apply causal masking
        block_size: Block size for block-wise computation
        dropout: Optional dropout layer
        
    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
    """
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    # Initialize output and normalization
    O = torch.zeros_like(Q)
    L = torch.zeros(batch_size, num_heads, seq_len, 1, device=Q.device)
    m = torch.full((batch_size, num_heads, seq_len, 1), float('-inf'), device=Q.device)
    
    # Process in blocks
    for i in range(0, seq_len, block_size):
        i_end = min(i + block_size, seq_len)
        Q_block = Q[:, :, i:i_end, :]
        
        max_k_idx = i_end if causal_mask else seq_len
        
        for j in range(0, max_k_idx, block_size):
            j_end = min(j + block_size, max_k_idx)
            K_block = K[:, :, j:j_end, :]
            V_block = V[:, :, j:j_end, :]
            
            # Compute scores for this block
            scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / (head_dim ** 0.5)
            
            # Apply causal mask
            if causal_mask:
                for q_idx in range(i, i_end):
                    for k_idx in range(j, j_end):
                        if k_idx > q_idx:
                            rel_q = q_idx - i
                            rel_k = k_idx - j
                            scores[:, :, rel_q, rel_k] = float('-inf')
            
            # Update output using online softmax
            m_new = torch.maximum(m[:, :, i:i_end], scores.max(dim=-1, keepdim=True)[0])
            alpha = torch.exp(m[:, :, i:i_end] - m_new)
            O[:, :, i:i_end, :] = O[:, :, i:i_end, :] * alpha + \
                torch.matmul(torch.exp(scores - m_new), V_block)
            L[:, :, i:i_end] = L[:, :, i:i_end] * alpha + \
                torch.exp(scores - m_new).sum(dim=-1, keepdim=True)
            m[:, :, i:i_end] = m_new
    
    # Normalize
    O = O / (L + 1e-8)
    
    # Apply dropout if provided
    if dropout is not None:
        O = dropout(O)
    
    return O


class FlashAttention(nn.Module):
    """Flash Attention module wrapper.
    
    This is a wrapper that can use either the custom implementation
    or the optimized flash-attn library if available.
    
    Args:
        use_flash_attn_lib: Whether to use flash-attn library if available
        block_size: Block size for custom implementation
    """
    
    def __init__(
        self,
        use_flash_attn_lib: bool = True,
        block_size: int = 128
    ):
        super().__init__()
        self.use_flash_attn_lib = use_flash_attn_lib
        self.block_size = block_size
        self.flash_attn_func = None
        
        # Try to import flash-attn library
        if use_flash_attn_lib:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("Warning: flash-attn library not available. Using custom implementation.")
                self.use_flash_attn_lib = False
    
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal: bool = True,
        dropout_p: float = 0.0
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            causal: Whether to apply causal masking
            dropout_p: Dropout probability
            
        Returns:
            Output tensor
        """
        if self.use_flash_attn_lib and self.flash_attn_func is not None:
            # Use optimized flash-attn library
            return self.flash_attn_func(
                Q, K, V,
                dropout_p=dropout_p,
                causal=causal
            )
        else:
            # Use custom implementation
            dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
            return flash_attention(
                Q, K, V,
                causal_mask=causal,
                block_size=self.block_size,
                dropout=dropout
            )

