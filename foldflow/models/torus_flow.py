"""
Torus Flow Matching for dihedral angles in protein structure generation.
Implements flow matching on the torus manifold for modeling periodic dihedral angles.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time conditioning."""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] time steps
        Returns:
            [B, dim] time embeddings
        """
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=t.device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
        return embeddings


class TorusAwareLayer(nn.Module):
    """Neural network layer that respects the torus geometry."""
    
    def __init__(self, input_dim: int, hidden_dim: int, time_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
        # Main processing layers
        self.linear1 = nn.Linear(input_dim + time_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        
        # Residual connection
        if input_dim == hidden_dim:
            self.residual_proj = nn.Identity()
        else:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, input_dim] input features on torus
            t_embed: [B, time_dim] time embedding
        Returns:
            [B, N, hidden_dim] processed features
        """
        residual = self.residual_proj(x)
        
        # Normalize input
        x = self.norm(x)
        
        # Expand time embedding to match sequence dimension
        B, N, _ = x.shape
        t_expanded = t_embed.unsqueeze(1).expand(-1, N, -1)  # [B, N, time_dim]
        
        # Concatenate input and time
        x_t = torch.cat([x, t_expanded], dim=-1)  # [B, N, input_dim + time_dim]
        
        # Forward pass
        h = self.activation(self.linear1(x_t))
        h = self.linear2(h)
        
        # Residual connection
        return h + residual


class TorusFlowMatcher(nn.Module):
    """Flow matching model for torus-valued data (dihedral angles)."""
    
    def __init__(self, 
                 input_dim: int = 2,  # (cos θ, sin θ) 
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Torus-aware layers
        self.layers = nn.ModuleList([
            TorusAwareLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                time_dim=hidden_dim
            ) for _ in range(num_layers)
        ])
        
        # Self-attention layers for sequence modeling
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection to tangent space
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Layer normalization for attention
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, input_dim] torus coordinates (cos θ, sin θ) for each dihedral
            t: [B] time steps
            mask: [B, N] optional sequence mask
        Returns:
            v: [B, N, input_dim] tangent vectors (velocity field)
        """
        # Ensure input is on unit circle (project to torus)
        x = self.project_to_torus(x)
        
        # Time embedding
        t_embed = self.time_embedding(t)  # [B, hidden_dim]
        
        # Input projection
        h = self.input_proj(x)  # [B, N, hidden_dim]
        
        # Process through layers
        for layer, attn, attn_norm in zip(self.layers, self.attention_layers, self.attn_norms):
            # Torus-aware processing
            h_processed = layer(h, t_embed)
            
            # Self-attention for sequence modeling
            h_norm = attn_norm(h_processed)
            attn_out, _ = attn(h_norm, h_norm, h_norm, key_padding_mask=~mask if mask is not None else None)
            h = h_processed + attn_out
        
        # Output projection
        v = self.output_proj(h)  # [B, N, input_dim]
        
        # Project to tangent space (ensure tangent vectors are perpendicular to position)
        v = self.project_to_tangent_space(v, x)
        
        return v
    
    def project_to_torus(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to torus (unit circle for each angle)."""
        if x.shape[-1] == 2:
            # Input is (cos θ, sin θ), normalize to unit circle
            norm = torch.norm(x, dim=-1, keepdim=True)
            return x / (norm + 1e-8)
        else:
            # Input is raw angles, convert to (cos θ, sin θ)
            return torch.stack([torch.cos(x), torch.sin(x)], dim=-1)
    
    def project_to_tangent_space(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project velocity to tangent space of torus."""
        # For unit circle, tangent vectors should be perpendicular to position vector
        # Remove component parallel to x
        dot_product = torch.sum(v * x, dim=-1, keepdim=True)
        v_tangent = v - dot_product * x
        return v_tangent


class EuclideanFlowMatcher(nn.Module):
    """Flow matching model for Euclidean-valued data (bond angles, lengths)."""
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 256, 
                 num_layers: int = 4,
                 num_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbedding(hidden_dim)
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Standard transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, input_dim] Euclidean values (bond angles/lengths)
            t: [B] time steps
            mask: [B, N] optional sequence mask
        Returns:
            v: [B, N, input_dim] velocity field
        """
        # Time embedding
        t_embed = self.time_embedding(t)  # [B, hidden_dim]
        
        # Input projection
        h = self.input_proj(x)  # [B, N, hidden_dim]
        
        # Add time embedding to each position
        B, N, _ = h.shape
        t_expanded = t_embed.unsqueeze(1).expand(-1, N, -1)  # [B, N, hidden_dim]
        h = h + t_expanded
        
        # Process through transformer layers
        src_key_padding_mask = ~mask if mask is not None else None
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=src_key_padding_mask)
        
        # Output projection
        v = self.output_proj(h)  # [B, N, input_dim]
        
        return v


class MixedFlowMatcher(nn.Module):
    """
    Mixed flow matching model that combines torus flow for dihedral angles
    with Euclidean flow for bond angles and lengths.
    """
    
    def __init__(self,
                 torus_hidden_dim: int = 256,
                 torus_layers: int = 6,
                 euclidean_hidden_dim: int = 256,
                 euclidean_layers: int = 4,
                 num_heads: int = 8):
        super().__init__()
        
        # Torus flow for dihedral angles (phi, psi, omega)
        self.torus_flow = TorusFlowMatcher(
            input_dim=2,  # (cos θ, sin θ)
            hidden_dim=torus_hidden_dim,
            num_layers=torus_layers,
            num_heads=num_heads
        )
        
        # Euclidean flow for bond angles
        self.bond_angle_flow = EuclideanFlowMatcher(
            input_dim=1,  # scalar bond angle
            hidden_dim=euclidean_hidden_dim,
            num_layers=euclidean_layers,
            num_heads=num_heads
        )
        
        # Euclidean flow for bond lengths (optional)
        self.bond_length_flow = EuclideanFlowMatcher(
            input_dim=1,  # scalar bond length
            hidden_dim=euclidean_hidden_dim,
            num_layers=euclidean_layers,
            num_heads=num_heads
        )
        
        # Cross-modal attention to allow interaction between different geometric quantities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=max(torus_hidden_dim, euclidean_hidden_dim),
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, batch: Dict[str, torch.Tensor], t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict containing:
                'torus_coords': [B, N, 3, 2] torus coordinates for (phi, psi, omega)
                'bond_angles': [B, N, 3] bond angles (N-CA-C, CA-C-N, C-N-CA)  
                'bond_lengths': [B, N, 3] bond lengths (N-CA, CA-C, C-N)
                'sequence_mask': [B, N] sequence mask
            t: [B] time steps
        Returns:
            dict containing velocity fields for each quantity
        """
        torus_coords = batch['torus_coords']  # [B, N, 3, 2]
        bond_angles = batch['bond_angles']    # [B, N, 3]
        bond_lengths = batch['bond_lengths']  # [B, N, 3]
        mask = batch.get('sequence_mask', None)  # [B, N]
        
        B, N, num_dihedrals, _ = torus_coords.shape
        
        # Process each type of dihedral angle separately
        dihedral_velocities = []
        for i in range(num_dihedrals):  # phi, psi, omega
            dih_coords = torus_coords[:, :, i, :]  # [B, N, 2]
            dih_velocity = self.torus_flow(dih_coords, t, mask)  # [B, N, 2]
            dihedral_velocities.append(dih_velocity)
        
        torus_velocity = torch.stack(dihedral_velocities, dim=2)  # [B, N, 3, 2]
        
        # Process bond angles
        bond_angle_velocities = []
        for i in range(3):  # Three types of bond angles
            angles = bond_angles[:, :, i:i+1]  # [B, N, 1]
            velocity = self.bond_angle_flow(angles, t, mask)  # [B, N, 1]
            bond_angle_velocities.append(velocity)
        
        bond_angle_velocity = torch.cat(bond_angle_velocities, dim=-1)  # [B, N, 3]
        
        # Process bond lengths
        bond_length_velocities = []
        for i in range(3):  # Three types of bond lengths
            lengths = bond_lengths[:, :, i:i+1]  # [B, N, 1]
            velocity = self.bond_length_flow(lengths, t, mask)  # [B, N, 1]
            bond_length_velocities.append(velocity)
        
        bond_length_velocity = torch.cat(bond_length_velocities, dim=-1)  # [B, N, 3]
        
        return {
            'dihedral_velocity': torus_velocity,      # [B, N, 3, 2]
            'bond_angle_velocity': bond_angle_velocity,  # [B, N, 3]
            'bond_length_velocity': bond_length_velocity, # [B, N, 3]
        }


class TorsionFlowLoss(nn.Module):
    """Loss function for mixed torus and Euclidean flow matching."""
    
    def __init__(self, 
                 torus_weight: float = 1.0,
                 bond_angle_weight: float = 1.0,
                 bond_length_weight: float = 0.1):
        super().__init__()
        self.torus_weight = torus_weight
        self.bond_angle_weight = bond_angle_weight  
        self.bond_length_weight = bond_length_weight
    
    def forward(self, 
                pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: predicted velocities
            target: target velocities
            mask: [B, N] sequence mask
        Returns:
            dict with individual and total losses
        """
        # Torus loss (for dihedral angles)
        torus_loss = self.torus_flow_loss(
            pred['dihedral_velocity'],
            target['dihedral_velocity'],
            mask
        )
        
        # Euclidean losses
        bond_angle_loss = self.euclidean_loss(
            pred['bond_angle_velocity'],
            target['bond_angle_velocity'], 
            mask
        )
        
        bond_length_loss = self.euclidean_loss(
            pred['bond_length_velocity'],
            target['bond_length_velocity'],
            mask
        )
        
        # Total weighted loss
        total_loss = (self.torus_weight * torus_loss +
                     self.bond_angle_weight * bond_angle_loss +
                     self.bond_length_weight * bond_length_loss)
        
        return {
            'total_loss': total_loss,
            'torus_loss': torus_loss,
            'bond_angle_loss': bond_angle_loss,
            'bond_length_loss': bond_length_loss
        }
    
    def torus_flow_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Loss for torus flow matching (in tangent space)."""
        mse_loss = F.mse_loss(pred, target, reduction='none')  # [B, N, 3, 2]
        
        if mask is not None:
            # Apply mask: [B, N] -> [B, N, 1, 1]
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
            mse_loss = mse_loss * mask_expanded
            return mse_loss.sum() / (mask_expanded.sum() + 1e-8)
        else:
            return mse_loss.mean()
    
    def euclidean_loss(self, pred: torch.Tensor, target: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Loss for Euclidean flow matching."""
        mse_loss = F.mse_loss(pred, target, reduction='none')  # [B, N, D]
        
        if mask is not None:
            # Apply mask: [B, N] -> [B, N, 1]
            mask_expanded = mask.unsqueeze(-1)
            mse_loss = mse_loss * mask_expanded
            return mse_loss.sum() / (mask_expanded.sum() + 1e-8)
        else:
            return mse_loss.mean()


def sample_torus_noise(shape: Tuple[int, ...], device: torch.device, sigma: float = 0.1) -> torch.Tensor:
    """Sample noise on torus manifold."""
    # Sample angles uniformly and convert to torus coordinates
    angles = torch.rand(*shape[:-1], device=device) * 2 * math.pi  # [B, N, 3]
    
    # Add Gaussian noise to angles
    angles = angles + torch.randn_like(angles) * sigma
    
    # Convert to torus coordinates
    torus_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # [B, N, 3, 2]
    
    return torus_coords


def sample_euclidean_noise(shape: Tuple[int, ...], device: torch.device, sigma: float = 0.1) -> torch.Tensor:
    """Sample Gaussian noise for Euclidean quantities."""
    return torch.randn(*shape, device=device) * sigma
