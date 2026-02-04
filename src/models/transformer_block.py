"""Transformer block implementation with pre/post normalization and residual scaling."""

from typing import Optional, Union
import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention
from .feedforward import SwiGLU, GeGLU, MixtureOfExperts


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).
    
    RMSNorm is an alternative to LayerNorm that doesn't center the activations.
    Used in models like LLaMA.
    
    Args:
        dim: Dimension to normalize
        eps: Epsilon for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (..., dim)
            
        Returns:
            Normalized tensor
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    """Transformer block with configurable normalization and residual connections.
    
    Supports both pre-norm (GPT style) and post-norm (original Transformer) architectures.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim_mult: Multiplier for feedforward dimension (typically 4)
        dropout_prob: Dropout probability
        pre_norm: Whether to use pre-norm (True) or post-norm (False)
        use_rope: Whether to use Rotary Position Embedding
        use_alibi: Whether to use ALiBi positional encoding
        use_gqa: Whether to use Grouped Query Attention
        num_kv_heads: Number of KV heads for GQA (if use_gqa=True)
        max_seq_len: Maximum sequence length
        use_flash_attn: Whether to use Flash Attention
        ffn_type: Feedforward type ('swiglu', 'geglu', or 'moe')
        use_moe: Whether to use Mixture of Experts
        num_experts: Number of experts for MoE
        moe_top_k: Top-k experts for MoE
        norm_type: Normalization type ('layernorm' or 'rmsnorm')
        residual_scale: Scaling factor for residuals
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim_mult: int = 4,
        dropout_prob: float = 0.1,
        pre_norm: bool = True,
        use_rope: bool = True,
        use_alibi: bool = False,
        use_gqa: bool = False,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 1024,
        use_flash_attn: bool = False,
        ffn_type: str = 'swiglu',
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        norm_type: str = 'layernorm',
        residual_scale: float = 1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.residual_scale = residual_scale
        self.use_moe = use_moe
        
        # Attention
        if use_gqa:
            if num_kv_heads is None:
                num_kv_heads = num_heads // 2  # Default: half the query heads
            self.attn = GroupedQueryAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                attn_dropout_prob=dropout_prob,
                resid_dropout_prob=dropout_prob,
                use_rope=use_rope,
                max_seq_len=max_seq_len
            )
        else:
            self.attn = MultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout_prob=dropout_prob,
                resid_dropout_prob=dropout_prob,
                use_rope=use_rope,
                use_alibi=use_alibi,
                max_seq_len=max_seq_len,
                use_flash_attn=use_flash_attn
            )
        
        # Normalization layers
        if norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(embed_dim)
            self.norm2 = RMSNorm(embed_dim)
        else:
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        ffn_dim = ffn_dim_mult * embed_dim
        
        if use_moe:
            self.ffn = MixtureOfExperts(
                dim=embed_dim,
                hidden_dim=ffn_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
                activation=ffn_type.replace('glu', ''),
                dropout_prob=dropout_prob
            )
        elif ffn_type == 'swiglu':
            self.ffn = SwiGLU(embed_dim, ffn_dim, dropout_prob=dropout_prob)
        elif ffn_type == 'geglu':
            self.ffn = GeGLU(embed_dim, ffn_dim, dropout_prob=dropout_prob)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_causal: bool = True
    ) -> Union[torch.Tensor, tuple]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            use_cache: Whether to use KV caching
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor, or tuple (output, load_balance_loss) if using MoE
        """
        if self.pre_norm:
            # Pre-norm architecture (GPT style)
            # Attention
            attn_output = self.attn(
                self.norm1(x),
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_causal=is_causal
            )
            x = x + self.residual_scale * attn_output
            
            # Feedforward
            if self.use_moe:
                ffn_output, load_balance_loss = self.ffn(self.norm2(x))
                x = x + self.residual_scale * ffn_output
                return x, load_balance_loss
            else:
                ffn_output = self.ffn(self.norm2(x))
                x = x + self.residual_scale * ffn_output
        else:
            # Post-norm architecture (original Transformer style)
            # Attention
            attn_output = self.attn(
                x,
                attention_mask=attention_mask,
                use_cache=use_cache,
                is_causal=is_causal
            )
            x = self.norm1(x + self.residual_scale * attn_output)
            
            # Feedforward
            if self.use_moe:
                ffn_output, load_balance_loss = self.ffn(x)
                x = self.norm2(x + self.residual_scale * ffn_output)
                return x, load_balance_loss
            else:
                ffn_output = self.ffn(x)
                x = self.norm2(x + self.residual_scale * ffn_output)
        
        return x

