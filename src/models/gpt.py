"""Main GPT model implementation with all advanced features."""

from typing import Optional, Dict, Any, Union, Tuple, List
import torch
import torch.nn as nn

from .embeddings import (
    TokenEmbedding,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding
)
from .transformer_block import TransformerBlock, RMSNorm


class TiedLinear(nn.Module):
    """Tied linear layer that shares weights with embedding layer.
    
    This is used for weight tying between input embeddings and output projection.
    
    Args:
        embedding: Embedding layer to tie weights with
    """
    
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size)
        """
        # Use embedding weights for output projection
        return torch.matmul(x, self.embedding.weight.t())


class AdvancedGPT(nn.Module):
    """Advanced GPT model with all modern features.
    
    This implementation includes:
    - Multiple positional encoding options (learned, sinusoidal, RoPE, ALiBi)
    - Configurable transformer blocks with pre/post normalization
    - Weight tying between embeddings and output head
    - Support for MoE, GQA, Flash Attention, and more
    
    Args:
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dropout_prob: Dropout probability
        pos_emb_type: Positional embedding type ('learned', 'sinusoidal', or None)
        pre_norm: Whether to use pre-norm (True) or post-norm (False)
        use_rope: Whether to use Rotary Position Embedding
        use_alibi: Whether to use ALiBi positional encoding
        use_gqa: Whether to use Grouped Query Attention
        num_kv_heads: Number of KV heads for GQA
        use_flash_attn: Whether to use Flash Attention
        ffn_dim_mult: Multiplier for feedforward dimension
        ffn_type: Feedforward type ('swiglu' or 'geglu')
        use_moe: Whether to use Mixture of Experts
        num_experts: Number of experts for MoE
        moe_top_k: Top-k experts for MoE
        norm_type: Normalization type ('layernorm' or 'rmsnorm')
        weight_tying: Whether to tie input embeddings and output head weights
        residual_scale: Scaling factor for residuals
    """
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout_prob: float = 0.1,
        pos_emb_type: str = 'learned',
        pre_norm: bool = True,
        use_rope: bool = True,
        use_alibi: bool = False,
        use_gqa: bool = False,
        num_kv_heads: Optional[int] = None,
        use_flash_attn: bool = False,
        ffn_dim_mult: int = 4,
        ffn_type: str = 'swiglu',
        use_moe: bool = False,
        num_experts: int = 8,
        moe_top_k: int = 2,
        norm_type: str = 'layernorm',
        weight_tying: bool = True,
        residual_scale: float = 1.0
    ):
        super().__init__()
        
        # Store configuration
        self.config = {
            'vocab_size': vocab_size,
            'max_seq_len': max_seq_len,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_layers': num_layers,
            'dropout_prob': dropout_prob,
            'pos_emb_type': pos_emb_type,
            'pre_norm': pre_norm,
            'use_rope': use_rope,
            'use_alibi': use_alibi,
            'use_gqa': use_gqa,
            'num_kv_heads': num_kv_heads,
            'use_flash_attn': use_flash_attn,
            'ffn_dim_mult': ffn_dim_mult,
            'ffn_type': ffn_type,
            'use_moe': use_moe,
            'num_experts': num_experts,
            'moe_top_k': moe_top_k,
            'norm_type': norm_type,
            'weight_tying': weight_tying,
            'residual_scale': residual_scale
        }
        
        # Token embeddings
        self.token_emb = TokenEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout_prob=0.0  # Dropout applied after adding positional embeddings
        )
        
        # Positional embeddings
        if pos_emb_type == 'learned':
            self.pos_emb = LearnedPositionalEmbedding(
                max_seq_len=max_seq_len,
                embed_dim=embed_dim,
                dropout_prob=0.0
            )
        elif pos_emb_type == 'sinusoidal':
            self.pos_emb = SinusoidalPositionalEmbedding(
                max_seq_len=max_seq_len,
                embed_dim=embed_dim,
                dropout_prob=0.0
            )
        else:
            self.pos_emb = None
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout_prob)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim_mult=ffn_dim_mult,
                dropout_prob=dropout_prob,
                pre_norm=pre_norm,
                use_rope=use_rope,
                use_alibi=use_alibi,
                use_gqa=use_gqa,
                num_kv_heads=num_kv_heads,
                max_seq_len=max_seq_len,
                use_flash_attn=use_flash_attn,
                ffn_type=ffn_type,
                use_moe=use_moe,
                num_experts=num_experts,
                moe_top_k=moe_top_k,
                norm_type=norm_type,
                residual_scale=residual_scale
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        if norm_type == 'rmsnorm':
            self.ln_f = RMSNorm(embed_dim)
        else:
            self.ln_f = nn.LayerNorm(embed_dim)
        
        # Output head
        if weight_tying:
            # Tie weights with token embeddings
            self.head = TiedLinear(self.token_emb.embedding)
        else:
            self.head = nn.Linear(embed_dim, vocab_size)
            # Initialize output head
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
            if self.head.bias is not None:
                nn.init.zeros_(self.head.bias)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for different module types."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
        use_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass.
        
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len) or (seq_len,)
            attention_mask: Optional attention mask of shape (batch_size, seq_len)
            return_all_layers: Whether to return activations from all layers
            use_cache: Whether to use KV caching
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size) or (seq_len, vocab_size)
            If return_all_layers=True, also returns list of layer activations
        """
        # Handle single sequence input
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        tok_emb = self.token_emb(input_ids)  # (batch_size, seq_len, embed_dim)
        
        # Get positional embeddings
        if self.pos_emb is not None:
            pos_emb = self.pos_emb(seq_len)  # (seq_len, embed_dim)
            # Expand to batch dimension
            pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)
            # Add embeddings
            h = tok_emb + pos_emb
        else:
            h = tok_emb
        
        # Apply embedding dropout
        h = self.emb_dropout(h)
        
        # Store activations if requested
        activations = [h] if return_all_layers else None
        
        # Pass through transformer blocks
        load_balance_losses = []
        for block in self.blocks:
            if self.config['use_moe']:
                h, load_balance_loss = block(
                    h,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    is_causal=True
                )
                load_balance_losses.append(load_balance_loss)
            else:
                h = block(
                    h,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    is_causal=True
                )
            
            if return_all_layers:
                activations.append(h)
        
        # Final layer norm
        h = self.ln_f(h)
        
        # Output projection
        logits = self.head(h)  # (batch_size, seq_len, vocab_size)
        
        if return_all_layers:
            return logits, activations
        
        return logits
    
    def clear_cache(self):
        """Clear KV caches in all attention layers."""
        for block in self.blocks:
            if hasattr(block.attn, 'clear_cache'):
                block.attn.clear_cache()
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters in the model.
        
        Args:
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def extend_context(self, new_seq_len: int):
        """Extend context length for RoPE if used.
        
        Args:
            new_seq_len: New maximum sequence length
        """
        if self.config['use_rope']:
            for block in self.blocks:
                if hasattr(block.attn, 'rope'):
                    block.attn.rope = block.attn.rope.extend_context(new_seq_len)
        
        self.config['max_seq_len'] = new_seq_len

