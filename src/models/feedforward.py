"""Feedforward network modules including SwiGLU, GeGLU, and Mixture of Experts."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit (SwiGLU) activation.
    
    SwiGLU(x) = Swish(W_gate * x) ⊗ (W_up * x)
    where Swish(x) = x * sigmoid(x)
    
    Used in models like PaLM and LLaMA.
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim)
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.W_up = nn.Linear(dim, hidden_dim)
        self.W_gate = nn.Linear(dim, hidden_dim)
        self.W_down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.W_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_down.weight, mean=0.0, std=0.02)
        if self.W_up.bias is not None:
            nn.init.zeros_(self.W_up.bias)
        if self.W_gate.bias is not None:
            nn.init.zeros_(self.W_gate.bias)
        if self.W_down.bias is not None:
            nn.init.zeros_(self.W_down.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        up = self.W_up(x)
        gate = self.W_gate(x)
        
        # Swish activation: x * sigmoid(x)
        swish_gate = gate * torch.sigmoid(gate)
        
        # Element-wise multiplication
        hidden = swish_gate * up
        
        # Down projection with dropout
        return self.dropout(self.W_down(hidden))


class GeGLU(nn.Module):
    """GELU-Gated Linear Unit (GeGLU) activation.
    
    GeGLU(x) = GELU(W_gate * x) ⊗ (W_up * x)
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim)
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout_prob: float = 0.1
    ):
        super().__init__()
        self.W_up = nn.Linear(dim, hidden_dim)
        self.W_gate = nn.Linear(dim, hidden_dim)
        self.W_down = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.W_up.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_gate.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_down.weight, mean=0.0, std=0.02)
        if self.W_up.bias is not None:
            nn.init.zeros_(self.W_up.bias)
        if self.W_gate.bias is not None:
            nn.init.zeros_(self.W_gate.bias)
        if self.W_down.bias is not None:
            nn.init.zeros_(self.W_down.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        up = self.W_up(x)
        gate = self.W_gate(x)
        
        # GELU activation
        gelu_gate = F.gelu(gate)
        
        # Element-wise multiplication
        hidden = gelu_gate * up
        
        # Down projection with dropout
        return self.dropout(self.W_down(hidden))


class Expert(nn.Module):
    """Single expert in Mixture of Experts.
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension
        activation: Activation function ('swiglu' or 'gelu')
        dropout_prob: Dropout probability
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str = 'swiglu',
        dropout_prob: float = 0.1
    ):
        super().__init__()
        if activation == 'swiglu':
            self.ffn = SwiGLU(dim, hidden_dim, dropout_prob)
        elif activation == 'gelu':
            self.ffn = GeGLU(dim, hidden_dim, dropout_prob)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.ffn(x)


class Router(nn.Module):
    """Router for Mixture of Experts.
    
    Routes tokens to top-k experts.
    
    Args:
        dim: Input dimension
        num_experts: Number of experts
    """
    
    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(dim, num_experts)
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)
    
    def forward(self, x: torch.Tensor, top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            top_k: Number of top experts to select
            
        Returns:
            expert_weights: Expert weights of shape (batch_size, seq_len, top_k)
            expert_indices: Expert indices of shape (batch_size, seq_len, top_k)
        """
        # Compute routing logits
        logits = self.gate(x)  # (batch_size, seq_len, num_experts)
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        
        # Apply softmax to get weights
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        return expert_weights, top_k_indices


class MixtureOfExperts(nn.Module):
    """Mixture of Experts (MoE) with GShard routing.
    
    Implements a sparse MoE layer where each token is routed to top-k experts.
    Includes load balancing loss to ensure experts are used evenly.
    
    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension for each expert
        num_experts: Number of experts
        top_k: Number of experts to route each token to
        activation: Activation function ('swiglu' or 'gelu')
        dropout_prob: Dropout probability
        load_balance_loss_weight: Weight for load balancing loss
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        activation: str = 'swiglu',
        dropout_prob: float = 0.1,
        load_balance_loss_weight: float = 0.01
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_loss_weight = load_balance_loss_weight
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(dim, hidden_dim, activation, dropout_prob)
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = Router(dim, num_experts)
        
        # Register buffer for load balancing loss
        self.register_buffer('expert_usage', torch.zeros(num_experts))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            output: Output tensor of shape (batch_size, seq_len, dim)
            load_balance_loss: Load balancing loss (scalar)
        """
        batch_size, seq_len, _ = x.shape
        num_tokens = batch_size * seq_len

        # Route tokens to experts
        expert_weights, expert_indices = self.router(x, top_k=self.top_k)
        # expert_weights: (batch_size, seq_len, top_k)
        # expert_indices: (batch_size, seq_len, top_k)

        # Flatten for processing
        x_flat = x.view(num_tokens, self.dim)  # (num_tokens, dim)
        expert_weights_flat = expert_weights.view(num_tokens, self.top_k)  # (num_tokens, top_k)
        expert_indices_flat = expert_indices.view(num_tokens, self.top_k)  # (num_tokens, top_k)

        # Vectorized MoE computation using scatter/gather
        output = self._compute_expert_outputs(x_flat, expert_weights_flat, expert_indices_flat, num_tokens)

        # Reshape back
        output = output.view(batch_size, seq_len, self.dim)

        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(expert_weights_flat, expert_indices_flat)

        return output, load_balance_loss

    def _compute_expert_outputs(
        self,
        x: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """Compute expert outputs using vectorized scatter/gather operations.

        Args:
            x: Input tensor of shape (num_tokens, dim)
            expert_weights: Expert weights of shape (num_tokens, top_k)
            expert_indices: Expert indices of shape (num_tokens, top_k)
            num_tokens: Number of tokens

        Returns:
            Output tensor of shape (num_tokens, dim)
        """
        # Create token indices repeated for each top-k position
        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        # token_indices: (num_tokens, top_k)

        # Flatten everything for scatter operations
        token_indices_flat = token_indices.reshape(-1)  # (num_tokens * top_k,)
        expert_indices_flat = expert_indices.reshape(-1)  # (num_tokens * top_k,)
        weights_flat = expert_weights.reshape(-1)  # (num_tokens * top_k,)

        # Initialize output accumulator
        output = torch.zeros(num_tokens, self.dim, device=x.device, dtype=x.dtype)

        # For each expert, gather tokens, process, and scatter results back
        # This still loops over experts but processes all tokens for each expert
        # in a single batch (no per-token Python loops)
        for expert_idx in range(self.num_experts):
            # Find all (token_idx, k_idx) positions where this expert is selected
            mask = expert_indices_flat == expert_idx  # (num_tokens * top_k,)

            if mask.any():
                # Get token indices and weights for this expert
                expert_token_indices = token_indices_flat[mask]  # (num_tokens_for_expert,)
                expert_weights_for_tokens = weights_flat[mask].unsqueeze(1)  # (num_tokens_for_expert, 1)

                # Gather tokens for this expert (vectorized)
                expert_tokens = x[expert_token_indices]  # (num_tokens_for_expert, dim)

                # Process all tokens through this expert in one batch
                expert_output = self.experts[expert_idx](expert_tokens)  # (num_tokens_for_expert, dim)

                # Weight the output
                weighted_output = expert_weights_for_tokens * expert_output  # (num_tokens_for_expert, dim)

                # Scatter-add back to output (vectorized)
                output.index_add_(0, expert_token_indices, weighted_output)

        return output
    
    def _compute_load_balance_loss(
        self,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage.

        Uses vectorized operations for efficiency.

        Args:
            expert_weights: Expert weights of shape (num_tokens, top_k)
            expert_indices: Expert indices of shape (num_tokens, top_k)

        Returns:
            Load balancing loss (scalar)
        """
        num_tokens = expert_weights.shape[0]

        # Vectorized computation of expert usage
        # Create one-hot encoding: (num_tokens, top_k, num_experts)
        expert_one_hot = F.one_hot(expert_indices, num_classes=self.num_experts).to(expert_weights.dtype)

        # Weight the one-hot encoding by expert weights
        # expert_weights: (num_tokens, top_k) -> (num_tokens, top_k, 1)
        weighted_one_hot = expert_one_hot * expert_weights.unsqueeze(-1)

        # Sum over top_k dimension to get total weight per expert per token
        # (num_tokens, num_experts)
        token_expert_weights = weighted_one_hot.sum(dim=1)

        # Sum over tokens and normalize to get fraction
        # (num_experts,)
        expert_fraction = token_expert_weights.sum(dim=0) / num_tokens

        # Compute coefficient of variation
        mean_fraction = expert_fraction.mean()
        if mean_fraction > 0:
            cv = expert_fraction.std() / mean_fraction
        else:
            cv = torch.tensor(0.0, device=expert_weights.device)

        # Load balancing loss is the coefficient of variation
        return self.load_balance_loss_weight * cv

