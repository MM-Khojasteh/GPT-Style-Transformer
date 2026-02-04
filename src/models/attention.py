"""Attention mechanisms for GPT transformer including multi-head attention, flash attention, and GQA."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .embeddings import RotaryPositionEmbedding, ALiBiPositionalEncoding


def apply_causal_mask(
    scores: torch.Tensor,
    mask_value: Optional[float] = None
) -> torch.Tensor:
    """Apply causal mask to attention scores.
    
    Args:
        scores: Attention scores of shape (..., seq_len_q, seq_len_k)
        mask_value: Value to use for masked positions (auto-detected based on dtype)
        
    Returns:
        Masked attention scores
    """
    seq_len_q, seq_len_k = scores.shape[-2], scores.shape[-1]
    
    # Auto-detect appropriate mask value based on dtype
    if mask_value is None:
        if scores.dtype == torch.float16:
            mask_value = -1e4  # Smaller value for float16 to avoid overflow
        elif scores.dtype == torch.bfloat16:
            mask_value = -1e4
        else:
            mask_value = -1e9
    
    # Create causal mask (lower triangular for seq_len_q x seq_len_k)
    # Only mask positions where k_idx > q_idx
    mask = torch.triu(torch.ones(seq_len_q, seq_len_k, device=scores.device, dtype=scores.dtype), diagonal=1)
    mask = mask.bool()
    scores = scores.masked_fill(mask, mask_value)
    return scores


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with support for RoPE, ALiBi, Flash Attention, and KV caching.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        head_dim: Dimension of each head (default: embed_dim // num_heads)
        attn_dropout_prob: Attention dropout probability
        resid_dropout_prob: Residual dropout probability
        use_rope: Whether to use Rotary Position Embedding
        use_alibi: Whether to use ALiBi positional encoding
        max_seq_len: Maximum sequence length for RoPE/ALiBi
        use_flash_attn: Whether to use Flash Attention (if available)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout_prob: float = 0.1,
        resid_dropout_prob: float = 0.1,
        use_rope: bool = True,
        use_alibi: bool = False,
        max_seq_len: int = 1024,
        use_flash_attn: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        self.use_flash_attn = use_flash_attn
        
        assert embed_dim == self.num_heads * self.head_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        # Linear projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.resid_dropout = nn.Dropout(resid_dropout_prob)
        
        # Positional encodings
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len
        ) if use_rope else None
        
        self.alibi = ALiBiPositionalEncoding(
            num_heads=num_heads,
            max_seq_len=max_seq_len
        ) if use_alibi else None
        
        # KV cache for inference
        self.kv_cache: Optional[Dict[str, torch.Tensor]] = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using normal distribution."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split embedding dimension into multiple heads.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge multiple heads back into embedding dimension.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, head_dim)
            
        Returns:
            Tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2)  # (batch_size, seq_len, num_heads, head_dim)
        return x.contiguous().view(batch_size, seq_len, num_heads * head_dim)
    
    def _compute_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal_mask: bool = True,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute standard scaled dot-product attention.
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            causal_mask: Whether to apply causal masking
            attention_mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor and attention weights
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply ALiBi bias if used
        if self.alibi is not None:
            scores = self.alibi(scores)
        
        # Apply causal mask
        if causal_mask:
            # Use appropriate mask value for the dtype
            mask_value = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = apply_causal_mask(scores, mask_value=mask_value)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def _flash_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        causal_mask: bool = True,
        block_size: int = 128
    ) -> torch.Tensor:
        """Flash Attention implementation (simplified version).
        
        This is a simplified block-wise Flash Attention. For production use,
        consider using the flash-attn library.
        
        Args:
            Q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            K: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            V: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            causal_mask: Whether to apply causal masking
            block_size: Block size for block-wise computation
            
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
                scores = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(head_dim)
                
                # Apply ALiBi if used
                if self.alibi is not None:
                    # Extract relevant bias
                    bias = self.alibi.get_bias(seq_len, Q.device)
                    bias_block = bias[:, i:i_end, j:j_end]
                    scores = scores + bias_block.unsqueeze(0)
                
                # Apply causal mask
                if causal_mask:
                    # Only mask positions where j > i
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
        
        return O
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        is_causal: bool = True
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            use_cache: Whether to use KV caching
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Apply RoPE if used
        if self.rope is not None:
            Q, K = self.rope(Q, K, seq_len=seq_len)
        
        # Handle KV caching for inference
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = {'keys': K, 'values': V, 'seq_len': seq_len}
            else:
                # Concatenate with cached K, V
                self.kv_cache['keys'] = torch.cat([self.kv_cache['keys'], K], dim=2)
                self.kv_cache['values'] = torch.cat([self.kv_cache['values'], V], dim=2)
                self.kv_cache['seq_len'] += seq_len
                K = self.kv_cache['keys']
                V = self.kv_cache['values']
        
        # Compute attention
        if self.use_flash_attn and not use_cache:
            # Use Flash Attention
            output = self._flash_attention(Q, K, V, causal_mask=is_causal)
            attn_weights = None
        else:
            # Use standard attention
            output, attn_weights = self._compute_attention(
                Q, K, V, causal_mask=is_causal, attention_mask=attention_mask
            )
        
        # Merge heads
        output = self._merge_heads(output)
        
        # Output projection
        output = self.W_o(output)
        output = self.resid_dropout(output)
        
        return output
    
    def clear_cache(self):
        """Clear KV cache."""
        self.kv_cache = None


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention (GQA) - uses fewer KV heads than query heads.
    
    This is more efficient than standard multi-head attention while maintaining
    similar performance. Used in models like LLaMA-2.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of query heads
        num_kv_heads: Number of key-value heads (should divide num_heads)
        head_dim: Dimension of each head
        attn_dropout_prob: Attention dropout probability
        resid_dropout_prob: Residual dropout probability
        use_rope: Whether to use Rotary Position Embedding
        max_seq_len: Maximum sequence length for RoPE
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        attn_dropout_prob: float = 0.1,
        resid_dropout_prob: float = 0.1,
        use_rope: bool = True,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads
        
        assert embed_dim == self.num_heads * self.head_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        # Query projection (full number of heads)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        
        # Key and Value projections (fewer heads)
        kv_dim = self.num_kv_heads * self.head_dim
        self.W_k = nn.Linear(embed_dim, kv_dim)
        self.W_v = nn.Linear(embed_dim, kv_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.resid_dropout = nn.Dropout(resid_dropout_prob)
        
        # RoPE
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len
        ) if use_rope else None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Split into heads."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads."""
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, seq_len, num_heads * head_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Split into heads
        Q = self._split_heads(Q, self.num_heads)  # (batch_size, num_heads, seq_len, head_dim)
        K = self._split_heads(K, self.num_kv_heads)  # (batch_size, num_kv_heads, seq_len, head_dim)
        V = self._split_heads(V, self.num_kv_heads)  # (batch_size, num_kv_heads, seq_len, head_dim)
        
        # Apply RoPE if used
        if self.rope is not None:
            Q, K = self.rope(Q, K, seq_len=seq_len)
        
        # Repeat K and V to match number of query heads
        # Each KV head is shared by num_heads // num_kv_heads query heads
        repeat_factor = self.num_heads // self.num_kv_heads
        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if is_causal:
            mask_value = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
            scores = apply_causal_mask(scores, mask_value=mask_value)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        output = torch.matmul(attn_weights, V)
        
        # Merge heads
        output = self._merge_heads(output)
        
        # Output projection
        output = self.W_o(output)
        output = self.resid_dropout(output)
        
        return output

