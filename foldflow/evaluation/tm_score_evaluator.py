"""
TM-Score evaluation and protein backbone chirality visualization for Torus Flow.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

def tm_score(pred_coords: torch.Tensor, 
             true_coords: torch.Tensor, 
             mask: Optional[torch.Tensor] = None) -> float:
    """
    Calculate TM-Score between predicted and true coordinates.
    
    Args:
        pred_coords: [N, 3] predicted CA coordinates
        true_coords: [N, 3] true CA coordinates  
        mask: [N] optional mask for valid residues
        
    Returns:
        TM-Score value (0-1, higher is better)
    """
    if mask is not None:
        pred_coords = pred_coords[mask.bool()]
        true_coords = true_coords[mask.bool()]
    
    N = len(pred_coords)
    if N < 3:
        return 0.0
    
    # Kabsch alignment
    pred_centered = pred_coords - pred_coords.mean(dim=0)
    true_centered = true_coords - true_coords.mean(dim=0)
    
    # Compute rotation matrix using SVD
    H = pred_centered.T @ true_centered
    U, S, Vt = torch.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if torch.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply rotation
    pred_aligned = pred_centered @ R
    
    # Calculate distances after alignment
    distances = torch.norm(pred_aligned - true_centered, dim=1)
    
    # TM-Score calculation
    d0 = 1.24 * (N - 15)**(1/3) - 1.8 if N > 21 else 0.5
    tm_score_val = torch.sum(1 / (1 + (distances / d0)**2)) / N
    
    return tm_score_val.item()


def batch_tm_score(pred_coords: torch.Tensor,
                   true_coords: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate TM-Score for a batch of structures.
    
    Args:
        pred_coords: [B, N, 3] batch of predicted coordinates
        true_coords: [B, N, 3] batch of true coordinates
        mask: [B, N] optional mask for valid residues
        
    Returns:
        [B] TM-Score values for each structure in batch
    """
    batch_size = pred_coords.shape[0]
    tm_scores = []
    
    for i in range(batch_size):
        pred_i = pred_coords[i]
        true_i = true_coords[i]
        mask_i = mask[i] if mask is not None else None
        
        score = tm_score(pred_i, true_i, mask_i)
        tm_scores.append(score)
    
    return torch.tensor(tm_scores)


def extract_ca_coordinates(coords: torch.Tensor, 
                          atom_types: str = "backbone") -> torch.Tensor:
    """
    Extract CA coordinates from full atom coordinates.
    
    Args:
        coords: [N, 3*K, 3] where K is atoms per residue (typically 3 for N,CA,C)
        atom_types: Type of atoms ("backbone" assumes N,CA,C order)
        
    Returns:
        [N, 3] CA coordinates
    """
    if atom_types == "backbone":
        # Assume order is N, CA, C for each residue
        ca_coords = coords[:, 1::3, :]  # Take every 3rd atom starting from index 1
        return ca_coords.squeeze()
    else:
        raise NotImplementedError(f"Atom type {atom_types} not supported")


def compute_backbone_chirality(coords: torch.Tensor) -> torch.Tensor:
    """
    Compute backbone chirality (dihedral angles) from coordinates.
    
    Args:
        coords: [N, 9, 3] backbone coordinates (N, CA, C for each residue)
        
    Returns:
        [N-1, 3] dihedral angles (phi, psi, omega) in radians
    """
    def dihedral_angle(p1, p2, p3, p4):
        """Calculate dihedral angle between four points."""
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        
        # Normalize
        n1 = n1 / torch.norm(n1, dim=-1, keepdim=True)
        n2 = n2 / torch.norm(n2, dim=-1, keepdim=True)
        
        # Calculate angle
        cos_angle = torch.sum(n1 * n2, dim=-1)
        sin_angle = torch.sum(torch.cross(n1, n2) * v2 / torch.norm(v2, dim=-1, keepdim=True), dim=-1)
        
        angle = torch.atan2(sin_angle, cos_angle)
        return angle
    
    # Reshape to get individual atoms
    coords_flat = coords.view(-1, 3)  # [N*3, 3]
    
    dihedrals = []
    n_residues = coords.shape[0]
    
    for i in range(n_residues - 1):
        # Get atoms for dihedral calculation
        # phi: C(i-1) - N(i) - CA(i) - C(i)
        # psi: N(i) - CA(i) - C(i) - N(i+1)
        # omega: CA(i) - C(i) - N(i+1) - CA(i+1)
        
        if i > 0:  # phi angle
            c_prev = coords_flat[i*3 - 1]  # C of previous residue
            n_curr = coords_flat[i*3]      # N of current residue
            ca_curr = coords_flat[i*3 + 1] # CA of current residue
            c_curr = coords_flat[i*3 + 2]  # C of current residue
            phi = dihedral_angle(c_prev, n_curr, ca_curr, c_curr)
        else:
            phi = torch.tensor(0.0)  # Undefined for first residue
        
        # psi angle
        n_curr = coords_flat[i*3]      # N of current residue
        ca_curr = coords_flat[i*3 + 1] # CA of current residue
        c_curr = coords_flat[i*3 + 2]  # C of current residue
        n_next = coords_flat[(i+1)*3]  # N of next residue
        psi = dihedral_angle(n_curr, ca_curr, c_curr, n_next)
        
        # omega angle
        ca_curr = coords_flat[i*3 + 1] # CA of current residue
        c_curr = coords_flat[i*3 + 2]  # C of current residue
        n_next = coords_flat[(i+1)*3]  # N of next residue
        ca_next = coords_flat[(i+1)*3 + 1] # CA of next residue
        omega = dihedral_angle(ca_curr, c_curr, n_next, ca_next)
        
        dihedrals.append([phi, psi, omega])
    
    return torch.stack([torch.stack(d) for d in dihedrals])


def plot_ramachandran(phi_angles: np.ndarray, 
                     psi_angles: np.ndarray,
                     title: str = "Ramachandran Plot",
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Ramachandran diagram showing backbone chirality.
    
    Args:
        phi_angles: Array of phi angles in radians
        psi_angles: Array of psi angles in radians
        title: Plot title
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    # Convert to degrees
    phi_deg = np.degrees(phi_angles)
    psi_deg = np.degrees(psi_angles)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(phi_deg, psi_deg, bins=36, 
                                         range=[[-180, 180], [-180, 180]])
    
    # Plot heatmap
    im = ax.imshow(hist.T, origin='lower', extent=[-180, 180, -180, 180],
                   cmap='Blues', alpha=0.7)
    
    # Add allowed regions (approximate)
    # Beta sheet region
    beta_sheet = patches.Rectangle((-180, 100), 120, 80, 
                                  linewidth=2, edgecolor='green', 
                                  facecolor='none', alpha=0.8, 
                                  label='β-sheet region')
    ax.add_patch(beta_sheet)
    
    # Alpha helix region  
    alpha_helix = patches.Ellipse((-60, -45), 60, 60,
                                 linewidth=2, edgecolor='red',
                                 facecolor='none', alpha=0.8,
                                 label='α-helix region')
    ax.add_patch(alpha_helix)
    
    # Left-handed alpha helix region
    left_alpha = patches.Ellipse((60, 45), 40, 40,
                                linewidth=2, edgecolor='orange',
                                facecolor='none', alpha=0.8,
                                label='Left-handed α-helix')
    ax.add_patch(left_alpha)
    
    ax.set_xlabel('φ (degrees)', fontsize=12)
    ax.set_ylabel('ψ (degrees)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', fontsize=12)
    
    # Set axis limits and ticks
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-180, 181, 60))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_3d_backbone(coords: torch.Tensor,
                    title: str = "Protein Backbone Structure",
                    save_path: Optional[str] = None,
                    show_chirality: bool = True) -> plt.Figure:
    """
    Plot 3D backbone structure with chirality visualization.
    
    Args:
        coords: [N, 9, 3] backbone coordinates
        title: Plot title
        save_path: Optional path to save the plot
        show_chirality: Whether to color by local chirality
        
    Returns:
        matplotlib Figure object
    """
    # Extract CA coordinates for main chain
    ca_coords = coords[:, 1, :].detach().cpu().numpy()  # CA atoms
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if show_chirality and len(coords) > 3:
        # Compute local chirality measures
        dihedrals = compute_backbone_chirality(coords)
        phi_angles = dihedrals[:, 0].detach().cpu().numpy()
        
        # Color by phi angle (proxy for chirality)
        colors = plt.cm.RdYlBu((phi_angles + np.pi) / (2 * np.pi))
        
        # Plot backbone trace with chirality coloring
        for i in range(len(ca_coords) - 1):
            if i < len(colors):
                ax.plot(ca_coords[i:i+2, 0], ca_coords[i:i+2, 1], ca_coords[i:i+2, 2],
                       color=colors[i], linewidth=3, alpha=0.8)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, 
                                  norm=plt.Normalize(vmin=-np.pi, vmax=np.pi))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('φ angle (radians)', fontsize=12)
        
    else:
        # Simple backbone trace
        ax.plot(ca_coords[:, 0], ca_coords[:, 1], ca_coords[:, 2],
               'b-', linewidth=2, alpha=0.8, label='Backbone')
    
    # Mark N and C termini
    ax.scatter(ca_coords[0, 0], ca_coords[0, 1], ca_coords[0, 2],
              c='green', s=100, marker='o', label='N-terminus')
    ax.scatter(ca_coords[-1, 0], ca_coords[-1, 1], ca_coords[-1, 2],
              c='red', s=100, marker='s', label='C-terminus')
    
    ax.set_xlabel('X (Å)', fontsize=12)
    ax.set_ylabel('Y (Å)', fontsize=12)
    ax.set_zlabel('Z (Å)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([ca_coords[:, 0].max() - ca_coords[:, 0].min(),
                         ca_coords[:, 1].max() - ca_coords[:, 1].min(),
                         ca_coords[:, 2].max() - ca_coords[:, 2].min()]).max() / 2.0
    mid_x = (ca_coords[:, 0].max() + ca_coords[:, 0].min()) * 0.5
    mid_y = (ca_coords[:, 1].max() + ca_coords[:, 1].min()) * 0.5
    mid_z = (ca_coords[:, 2].max() + ca_coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_metrics(tm_scores: List[float],
                         loss_history: List[float],
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training progress with TM-Score and loss.
    
    Args:
        tm_scores: List of TM-Score values over training
        loss_history: List of loss values over training
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    epochs = range(len(tm_scores))
    
    # TM-Score plot
    ax1.plot(epochs, tm_scores, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_ylabel('TM-Score', fontsize=12)
    ax1.set_title('Training Progress: TM-Score', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Loss plot
    ax2.plot(epochs, loss_history, 'r-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Progress: Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


class TorsionFlowEvaluator:
    """Comprehensive evaluator for Torus Flow model."""
    
    def __init__(self, model, device, output_dir="./evaluation_results"):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_epoch(self, test_loader, epoch: int, 
                      reconstruction_model=None) -> Dict[str, float]:
        """
        Evaluate model on test set for one epoch.
        
        Args:
            test_loader: DataLoader for test set
            epoch: Current epoch number
            reconstruction_model: NERF model for coordinate reconstruction
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_tm_scores = []
        all_losses = []
        all_phi_angles = []
        all_psi_angles = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Model prediction
                t = torch.zeros(batch['batch_size'], device=self.device)
                pred_velocities = self.model(batch, t)
                
                # Compute loss
                loss_fn = TorsionFlowLoss()
                loss_dict = loss_fn(pred_velocities, batch, t)
                all_losses.append(loss_dict['total_loss'].item())
                
                # Reconstruct coordinates if model available
                if reconstruction_model is not None:
                    # Extract dihedral angles from velocities (simplified)
                    dihedral_coords = batch['torus_coords']  # [B, N, 3, 2]
                    phi = torch.atan2(dihedral_coords[:, :, 0, 1], dihedral_coords[:, :, 0, 0])
                    psi = torch.atan2(dihedral_coords[:, :, 1, 1], dihedral_coords[:, :, 1, 0])
                    omega = torch.atan2(dihedral_coords[:, :, 2, 1], dihedral_coords[:, :, 2, 0])
                    
                    # Reconstruct coordinates
                    pred_coords = reconstruction_model(phi, psi, omega)
                    
                    # Compute TM-Score for each structure in batch
                    batch_tm = batch_tm_score(pred_coords, batch.get('true_coords', pred_coords),
                                            batch.get('sequence_mask'))
                    all_tm_scores.extend(batch_tm.tolist())
                    
                    # Collect angles for Ramachandran plot
                    all_phi_angles.extend(phi.flatten().cpu().numpy())
                    all_psi_angles.extend(psi.flatten().cpu().numpy())
        
        # Compute average metrics
        metrics = {
            'avg_loss': np.mean(all_losses),
            'avg_tm_score': np.mean(all_tm_scores) if all_tm_scores else 0.0,
            'std_tm_score': np.std(all_tm_scores) if all_tm_scores else 0.0
        }
        
        # Generate plots
        if all_phi_angles and all_psi_angles:
            # Ramachandran plot
            rama_fig = plot_ramachandran(
                np.array(all_phi_angles), np.array(all_psi_angles),
                title=f"Ramachandran Plot - Epoch {epoch}",
                save_path=f"{self.output_dir}/ramachandran_epoch_{epoch}.png"
            )
            plt.close(rama_fig)
        
        self.logger.info(f"Epoch {epoch} evaluation: "
                        f"Loss={metrics['avg_loss']:.4f}, "
                        f"TM-Score={metrics['avg_tm_score']:.4f}±{metrics['std_tm_score']:.4f}")
        
        return metrics
    
    def plot_sample_structures(self, test_batch, reconstruction_model,
                              num_samples: int = 4, epoch: int = 0):
        """Plot sample generated structures."""
        self.model.eval()
        
        with torch.no_grad():
            # Generate samples
            t = torch.zeros(test_batch['batch_size'], device=self.device)
            pred_velocities = self.model(test_batch, t)
            
            # Reconstruct coordinates
            dihedral_coords = test_batch['torus_coords']
            phi = torch.atan2(dihedral_coords[:, :, 0, 1], dihedral_coords[:, :, 0, 0])
            psi = torch.atan2(dihedral_coords[:, :, 1, 1], dihedral_coords[:, :, 1, 0])
            omega = torch.atan2(dihedral_coords[:, :, 2, 1], dihedral_coords[:, :, 2, 0])
            
            pred_coords = reconstruction_model(phi, psi, omega)
            
            # Plot first few samples
            for i in range(min(num_samples, pred_coords.shape[0])):
                coords_i = pred_coords[i]  # [N, 3]
                
                # Reshape to backbone format [N//3, 3, 3] assuming N,CA,C order
                n_residues = coords_i.shape[0] // 3
                backbone_coords = coords_i[:n_residues*3].view(n_residues, 3, 3)
                
                # Plot 3D structure
                struct_fig = plot_3d_backbone(
                    backbone_coords,
                    title=f"Generated Structure {i+1} - Epoch {epoch}",
                    save_path=f"{self.output_dir}/structure_sample_{i+1}_epoch_{epoch}.png",
                    show_chirality=True
                )
                plt.close(struct_fig)


def main():
    """Example usage of evaluation functions."""
    # Create dummy data for testing
    n_residues = 50
    coords = torch.randn(n_residues, 3, 3) * 5  # [N, 3, 3] for N,CA,C
    
    # Test chirality computation
    dihedrals = compute_backbone_chirality(coords)
    print(f"Computed dihedrals shape: {dihedrals.shape}")
    
    # Test TM-Score
    pred_ca = coords[:, 1, :]  # CA coordinates
    true_ca = pred_ca + torch.randn_like(pred_ca) * 0.5  # Add some noise
    tm = tm_score(pred_ca, true_ca)
    print(f"TM-Score: {tm:.4f}")
    
    # Test visualization
    rama_fig = plot_ramachandran(dihedrals[:, 0].numpy(), dihedrals[:, 1].numpy())
    struct_fig = plot_3d_backbone(coords)
    
    plt.show()


if __name__ == "__main__":
    main()
