"""Embedding modules for GPT transformer including token and positional embeddings."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """Token embedding with scaling and dropout.
    
    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        dropout_prob: Dropout probability
        scale: Whether to scale embeddings by sqrt(embed_dim)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dropout_prob: float = 0.1,
        scale: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_factor = math.sqrt(embed_dim) if scale else 1.0
        
        # Initialize embedding weights
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize with normal distribution (similar to glorot_uniform)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len) or (seq_len,)
            
        Returns:
            Embeddings of shape (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
        """
        # Get embeddings
        emb = self.embedding(x)
        
        # Scale if needed
        if self.scale_factor != 1.0:
            emb = emb * self.scale_factor
        
        # Apply dropout
        return self.dropout(emb)


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings with support for extending context length.
    
    Args:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Initialize positional embeddings
        self.embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize with normal distribution
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Forward pass.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional embeddings of shape (seq_len, embed_dim)
        """
        if seq_len <= self.max_seq_len:
            pos_emb = self.embedding[:seq_len]
        else:
            # Extend beyond max_seq_len by extrapolating
            orig_emb = self.embedding
            extra_positions = seq_len - self.max_seq_len
            last_pos_diff = orig_emb[-1] - orig_emb[-2]
            
            # Create extrapolated embeddings
            extra_embs = []
            for i in range(1, extra_positions + 1):
                extra_emb = orig_emb[-1] + i * last_pos_diff
                extra_embs.append(extra_emb)
            
            pos_emb = torch.cat([orig_emb, torch.stack(extra_embs)], dim=0)
        
        return self.dropout(pos_emb)


def sinusoidal_position_embedding(seq_len: int, embed_dim: int) -> torch.Tensor:
    """Create sinusoidal positional embeddings.
    
    Args:
        seq_len: Sequence length
        embed_dim: Embedding dimension
        
    Returns:
        Positional embeddings of shape (seq_len, embed_dim)
    """
    pe = torch.zeros(seq_len, embed_dim)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2).float() * 
        (-math.log(10000.0) / embed_dim)
    )
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings with support for extending context length.
    
    Args:
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Register as buffer (non-trainable)
        pe = sinusoidal_position_embedding(max_seq_len, embed_dim)
        self.register_buffer('embedding', pe)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, seq_len: int) -> torch.Tensor:
        """Forward pass.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Positional embeddings of shape (seq_len, embed_dim)
        """
        if seq_len <= self.max_seq_len:
            pos_emb = self.embedding[:seq_len]
        else:
            # Generate extended embeddings
            pos_emb = sinusoidal_position_embedding(seq_len, self.embed_dim)
            if self.embedding.is_cuda:
                pos_emb = pos_emb.to(self.embedding.device)
        
        return self.dropout(pos_emb)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as used in LLaMA and other modern models.
    
    Args:
        dim: Dimension of each head
        max_seq_len: Maximum sequence length
        base: Base frequency for RoPE
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 1024,
        base: float = 10000.0
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for all positions
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            seq_len: Sequence length (if None, uses q.shape[-2])
            
        Returns:
            Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # Get cached cos and sin
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply rotary embedding
        q_rot = self._rotate_half(q) * sin + q * cos
        k_rot = self._rotate_half(k) * sin + k * cos
        
        return q_rot, k_rot
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def extend_context(self, new_seq_len: int) -> 'RotaryPositionEmbedding':
        """Extend context length by recomputing frequencies.
        
        Args:
            new_seq_len: New maximum sequence length
            
        Returns:
            New RotaryPositionEmbedding with extended context
        """
        if new_seq_len <= self.max_seq_len:
            return self
        
        # Create new instance with extended context
        return RotaryPositionEmbedding(
            dim=self.dim,
            max_seq_len=new_seq_len,
            base=self.base
        )


class ALiBiPositionalEncoding(nn.Module):
    """Attention with Linear Biases (ALiBi) positional encoding.
    
    ALiBi adds a linear bias to attention scores based on distance,
    eliminating the need for positional embeddings.
    
    Args:
        num_heads: Number of attention heads
        max_seq_len: Maximum sequence length
    """
    
    def __init__(
        self,
        num_heads: int,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute slopes for each head
        # Slopes follow the pattern: 2^(-8/n), 2^(-8*2/n), ..., 2^(-8*n/n)
        slopes = []
        for i in range(1, num_heads + 1):
            slope = 2 ** (-8 * i / num_heads)
            slopes.append(slope)
        
        slopes = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer('slopes', slopes)
    
    def get_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get ALiBi bias matrix for given sequence length.
        
        Args:
            seq_len: Sequence length
            device: Device to create bias on
            
        Returns:
            Bias matrix of shape (num_heads, seq_len, seq_len)
        """
        # Create distance matrix
        positions = torch.arange(seq_len, device=device).float()
        distance = positions.unsqueeze(0) - positions.unsqueeze(1)
        distance = torch.abs(distance)
        
        # Apply slopes: bias = -distance * slope for each head
        # Shape: (num_heads, seq_len, seq_len)
        bias = -distance.unsqueeze(0) * self.slopes.unsqueeze(1).unsqueeze(2)
        
        return bias
    
    def forward(
        self,
        attention_scores: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """Add ALiBi bias to attention scores.
        
        Args:
            attention_scores: Attention scores of shape 
                (batch_size, num_heads, seq_len, seq_len)
            seq_len: Sequence length (if None, uses attention_scores.shape[-1])
            
        Returns:
            Attention scores with ALiBi bias added
        """
        if seq_len is None:
            seq_len = attention_scores.shape[-1]
        
        bias = self.get_bias(seq_len, attention_scores.device)
        
        # Add bias: (batch_size, num_heads, seq_len, seq_len)
        return attention_scores + bias.unsqueeze(0)

