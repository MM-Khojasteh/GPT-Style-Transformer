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
        
        # Route tokens to experts
        expert_weights, expert_indices = self.router(x, top_k=self.top_k)
        # expert_weights: (batch_size, seq_len, top_k)
        # expert_indices: (batch_size, seq_len, top_k)
        
        # Flatten for processing
        x_flat = x.view(-1, self.dim)  # (batch_size * seq_len, dim)
        expert_weights_flat = expert_weights.view(-1, self.top_k)  # (batch_size * seq_len, top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)  # (batch_size * seq_len, top_k)
        
        # Process through experts
        output = torch.zeros_like(x_flat)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (expert_indices_flat == expert_idx).any(dim=1)  # (batch_size * seq_len,)
            
            if mask.any():
                # Get tokens for this expert
                expert_tokens = x_flat[mask]  # (num_tokens, dim)
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_tokens)  # (num_tokens, dim)
                
                # Get weights for this expert
                token_indices = torch.where(mask)[0]
                for token_idx, global_idx in enumerate(token_indices):
                    # Find which position in top_k this expert is
                    expert_pos = (expert_indices_flat[global_idx] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(expert_pos) > 0:
                        weight = expert_weights_flat[global_idx, expert_pos[0]]
                        output[global_idx] += weight * expert_output[token_idx]
        
        # Reshape back
        output = output.view(batch_size, seq_len, self.dim)
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(expert_weights_flat, expert_indices_flat)
        
        return output, load_balance_loss
    
    def _compute_load_balance_loss(
        self,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage.
        
        Args:
            expert_weights: Expert weights of shape (num_tokens, top_k)
            expert_indices: Expert indices of shape (num_tokens, top_k)
            
        Returns:
            Load balancing loss (scalar)
        """
        num_tokens = expert_weights.shape[0]
        
        # Compute fraction of tokens routed to each expert
        expert_fraction = torch.zeros(self.num_experts, device=expert_weights.device)
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx).any(dim=1)
            if mask.any():
                # Average weight for this expert
                expert_weights_for_expert = expert_weights[mask]
                expert_indices_for_expert = expert_indices[mask]
                
                # Get weights for this expert
                weights = []
                for i in range(expert_weights_for_expert.shape[0]):
                    pos = (expert_indices_for_expert[i] == expert_idx).nonzero(as_tuple=True)[0]
                    if len(pos) > 0:
                        weights.append(expert_weights_for_expert[i, pos[0]])
                
                if weights:
                    expert_fraction[expert_idx] = sum(weights) / num_tokens
        
        # Compute coefficient of variation
        mean_fraction = expert_fraction.mean()
        if mean_fraction > 0:
            cv = expert_fraction.std() / mean_fraction
        else:
            cv = torch.tensor(0.0, device=expert_weights.device)
        
        # Load balancing loss is the coefficient of variation
        return self.load_balance_loss_weight * cv

