"""
NERF (Natural Extension Reference Frame) implementation for FoldFlow.
Reconstructs protein backbone coordinates from torsion angles and bond angles.
Adapted from foldingdiff/nerf.py for integration with FoldFlow.
"""
import os
from functools import cached_property
from typing import Union, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# Standard bond lengths (Angstroms)
N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.54  
C_N_LENGTH = 1.34

# Standard bond angles (radians)
BOND_ANGLE_N_CA = 121.0 / 180.0 * np.pi  # N-CA-C angle
BOND_ANGLE_CA_C = 109.0 / 180.0 * np.pi  # CA-C-N angle (tau)
BOND_ANGLE_C_N = 115.0 / 180.0 * np.pi   # C-N-CA angle

# Initial coordinates from 1CRN (THR residue)
N_INIT = np.array([17.047, 14.099, 3.625])
CA_INIT = np.array([16.967, 12.784, 4.338])
C_INIT = np.array([15.685, 12.755, 5.133])


class NERFBuilder:
    """
    Builder for NERF reconstruction from torsion angles.
    """

    def __init__(
        self,
        phi_dihedrals: Union[np.ndarray, torch.Tensor],
        psi_dihedrals: Union[np.ndarray, torch.Tensor],
        omega_dihedrals: Union[np.ndarray, torch.Tensor],
        bond_len_n_ca: Union[float, np.ndarray, torch.Tensor] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray, torch.Tensor] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray, torch.Tensor] = C_N_LENGTH,
        bond_angle_n_ca: Union[float, np.ndarray, torch.Tensor] = BOND_ANGLE_N_CA,
        bond_angle_ca_c: Union[float, np.ndarray, torch.Tensor] = BOND_ANGLE_CA_C,
        bond_angle_c_n: Union[float, np.ndarray, torch.Tensor] = BOND_ANGLE_C_N,
        init_coords: Union[np.ndarray, torch.Tensor] = None,
    ) -> None:
        self.use_torch = False
        if any([isinstance(v, torch.Tensor) for v in [phi_dihedrals, psi_dihedrals, omega_dihedrals]]):
            self.use_torch = True

        self.phi = phi_dihedrals.squeeze() if hasattr(phi_dihedrals, 'squeeze') else phi_dihedrals
        self.psi = psi_dihedrals.squeeze() if hasattr(psi_dihedrals, 'squeeze') else psi_dihedrals
        self.omega = omega_dihedrals.squeeze() if hasattr(omega_dihedrals, 'squeeze') else omega_dihedrals

        # Bond lengths and angles
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
        }
        
        # Initial coordinates
        if init_coords is None:
            init_coords = [N_INIT, CA_INIT, C_INIT]
        self.init_coords = [c.squeeze() if hasattr(c, 'squeeze') else c for c in init_coords]
        
        assert len(self.init_coords) == 3, f"Requires 3 initial coords for N-CA-C but got {len(self.init_coords)}"

    @cached_property
    def cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Build out the molecule using NERF."""
        retval = self.init_coords.copy()
        if self.use_torch:
            device = self.phi.device if hasattr(self.phi, 'device') else 'cpu'
            retval = [torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device) for x in retval]

        # Handle batch dimension
        if self.use_torch and self.phi.dim() > 1:
            return self._build_batch()
        
        # Single sequence case
        # First phi at N terminus is not defined
        # Last psi and omega at C terminus are not defined
        phi = self.phi[1:] if len(self.phi.shape) > 0 and self.phi.shape[0] > 1 else self.phi
        psi = self.psi[:-1] if len(self.psi.shape) > 0 and self.psi.shape[0] > 1 else self.psi
        omega = self.omega[:-1] if len(self.omega.shape) > 0 and self.omega.shape[0] > 1 else self.omega
        
        # Stack dihedral angles
        if self.use_torch:
            if phi.dim() == 0:
                phi = phi.unsqueeze(0)
            if psi.dim() == 0:
                psi = psi.unsqueeze(0)
            if omega.dim() == 0:
                omega = omega.unsqueeze(0)
            dih_angles = torch.stack([psi, omega, phi], dim=-1)
        else:
            dih_angles = np.stack([psi, omega, phi], axis=-1)

        for i in range(dih_angles.shape[0]):
            dih = dih_angles[i]
            # Place N-CA-C atoms for each residue
            for j, bond in enumerate(self.bond_lengths.keys()):
                coords = place_dihedral(
                    retval[-3],
                    retval[-2],
                    retval[-1],
                    bond_angle=self._get_bond_angle(bond, i),
                    bond_length=self._get_bond_length(bond, i),
                    torsion_angle=dih[j],
                    use_torch=self.use_torch,
                )
                retval.append(coords)

        if self.use_torch:
            return torch.stack(retval)
        return np.array(retval)

    def _build_batch(self) -> torch.Tensor:
        """Build batch of structures."""
        batch_size = self.phi.shape[0]
        seq_len = self.phi.shape[1]
        device = self.phi.device
        
        # Initialize with first residue
        coords = torch.tensor(np.array([N_INIT, CA_INIT, C_INIT]), 
                             dtype=torch.float32, device=device)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]
        
        # Build remaining residues
        phi = self.phi[:, 1:]    # [B, seq_len-1]
        psi = self.psi[:, :-1]   # [B, seq_len-1] 
        omega = self.omega[:, :-1]  # [B, seq_len-1]
        
        for i in range(phi.shape[1]):
            # Place C-N bond
            n_coord = place_dihedral(
                coords[:, -3, :],
                coords[:, -2, :],
                coords[:, -1, :],
                bond_angle=self._get_bond_angle(("C", "N"), i, batch_idx=slice(None)),
                bond_length=self._get_bond_length(("C", "N"), i, batch_idx=slice(None)),
                torsion_angle=psi[:, i],
                use_torch=True
            )
            
            # Place N-CA bond
            ca_coord = place_dihedral(
                coords[:, -2, :],
                coords[:, -1, :],
                n_coord,
                bond_angle=self._get_bond_angle(("N", "CA"), i, batch_idx=slice(None)),
                bond_length=self._get_bond_length(("N", "CA"), i, batch_idx=slice(None)),
                torsion_angle=omega[:, i],
                use_torch=True
            )
            
            # Place CA-C bond
            c_coord = place_dihedral(
                coords[:, -1, :],
                n_coord,
                ca_coord,
                bond_angle=self._get_bond_angle(("CA", "C"), i, batch_idx=slice(None)),
                bond_length=self._get_bond_length(("CA", "C"), i, batch_idx=slice(None)),
                torsion_angle=phi[:, i],
                use_torch=True
            )
            
            # Concatenate new coordinates
            new_coords = torch.stack([n_coord, ca_coord, c_coord], dim=1)  # [B, 3, 3]
            coords = torch.cat([coords, new_coords], dim=1)  # [B, current_atoms, 3]
        
        return coords

    @cached_property
    def centered_cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Returns the centered coordinates."""
        coords = self.cartesian_coords
        if self.use_torch:
            means = coords.mean(dim=-2, keepdim=True)
        else:
            means = coords.mean(axis=-2, keepdims=True)
        return coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int, batch_idx=None):
        """Get the ith bond distance."""
        v = self.bond_lengths[bond]
        if isinstance(v, (float, int)):
            return v
        if batch_idx is not None:
            return v[batch_idx, idx] if v.dim() > 1 else v[batch_idx]
        return v[idx] if hasattr(v, '__getitem__') else v

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int, batch_idx=None):
        """Get the ith bond angle."""
        v = self.bond_angles[bond]
        if isinstance(v, (float, int)):
            return v
        if batch_idx is not None:
            return v[batch_idx, idx] if v.dim() > 1 else v[batch_idx]
        return v[idx] if hasattr(v, '__getitem__') else v


def place_dihedral(
    a: Union[np.ndarray, torch.Tensor],
    b: Union[np.ndarray, torch.Tensor],
    c: Union[np.ndarray, torch.Tensor],
    bond_angle: Union[float, np.ndarray, torch.Tensor],
    bond_length: Union[float, np.ndarray, torch.Tensor],
    torsion_angle: Union[float, np.ndarray, torch.Tensor],
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Place point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.shape == b.shape == c.shape
    assert a.shape[-1] == 3

    if not use_torch:
        unit_vec = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
        cross = lambda x, y: np.cross(x, y, axis=-1)
    else:
        # Ensure all inputs are tensors on the same device
        device = a.device if hasattr(a, 'device') else 'cpu'
        ensure_tensor = lambda x: (torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True) 
                                  if not isinstance(x, torch.Tensor) else x.to(device))
        
        a, b, c = [ensure_tensor(x) for x in (a, b, c)]
        bond_angle = ensure_tensor(bond_angle)
        bond_length = ensure_tensor(bond_length)
        torsion_angle = ensure_tensor(torsion_angle)
        
        unit_vec = lambda x: x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
        cross = lambda x, y: torch.cross(x, y, dim=-1)

    ab = b - a
    bc = unit_vec(c - b)
    
    # Handle degenerate case where ab and bc are collinear
    ab_norm = unit_vec(ab)
    cross_ab_bc = cross(ab, bc)
    cross_norm = torch.norm(cross_ab_bc, dim=-1, keepdim=True) if use_torch else np.linalg.norm(cross_ab_bc, axis=-1, keepdims=True)
    
    # If vectors are nearly collinear, use a different approach
    if use_torch:
        is_collinear = cross_norm < 1e-6
        if torch.any(is_collinear):
            # Use a perpendicular vector
            perp = torch.zeros_like(bc)
            perp[..., 0] = 1.0  # Try x direction first
            cross_test = cross(bc, perp)
            cross_test_norm = torch.norm(cross_test, dim=-1, keepdim=True)
            # If still too small, try y direction
            small_norm = (cross_test_norm < 1e-6).squeeze(-1)  # Remove last dimension
            if torch.any(small_norm):
                perp[small_norm, 0] = 0.0
                perp[small_norm, 1] = 1.0
            n = unit_vec(cross(bc, perp))
        else:
            n = unit_vec(cross_ab_bc)
    else:
        if np.any(cross_norm < 1e-6):
            # Use a perpendicular vector
            perp = np.zeros_like(bc)
            perp[..., 0] = 1.0
            cross_test = cross(bc, perp)
            cross_test_norm = np.linalg.norm(cross_test, axis=-1, keepdims=True)
            if np.any(cross_test_norm < 1e-6):
                perp[cross_test_norm.squeeze(-1) < 1e-6, 0] = 0.0
                perp[cross_test_norm.squeeze(-1) < 1e-6, 1] = 1.0
            n = unit_vec(cross(bc, perp))
        else:
            n = unit_vec(cross_ab_bc)
    
    nbc = cross(n, bc)

    if not use_torch:
        m = np.stack([bc, nbc, n], axis=-1)
        d_local = np.stack([
            -bond_length * np.cos(bond_angle),
            bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
            bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
        ], axis=-1)
        d = np.matmul(m, d_local[..., None]).squeeze(-1)
    else:
        m = torch.stack([bc, nbc, n], dim=-1)
        
        # Handle scalar vs tensor bond_angle, bond_length, torsion_angle
        if bond_angle.dim() == 0:
            bond_angle = bond_angle.expand(a.shape[:-1]) if a.dim() > 1 else bond_angle
        if bond_length.dim() == 0:
            bond_length = bond_length.expand(a.shape[:-1]) if a.dim() > 1 else bond_length
        if torsion_angle.dim() == 0:
            torsion_angle = torsion_angle.expand(a.shape[:-1]) if a.dim() > 1 else torsion_angle
        
        d_local = torch.stack([
            -bond_length * torch.cos(bond_angle),
            bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
            bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle),
        ], dim=-1)
        
        d = torch.matmul(m, d_local.unsqueeze(-1)).squeeze(-1)

    return d + c


def nerf_build_batch(
    phi: torch.Tensor,
    psi: torch.Tensor,
    omega: torch.Tensor,
    bond_angle_n_ca_c: torch.Tensor,
    bond_angle_ca_c_n: torch.Tensor,
    bond_angle_c_n_ca: torch.Tensor,
    bond_len_n_ca: Union[float, torch.Tensor] = N_CA_LENGTH,
    bond_len_ca_c: Union[float, torch.Tensor] = CA_C_LENGTH,
    bond_len_c_n: Union[float, torch.Tensor] = C_N_LENGTH,
) -> torch.Tensor:
    """
    Build out a batch of phi, psi, omega values. Returns 3D coordinates
    in Cartesian space with shape (batch, length * 3, 3).
    """
    assert phi.ndim == psi.ndim == omega.ndim == 2  # batch, seq
    assert phi.shape == psi.shape == omega.shape
    batch_size = phi.shape[0]
    seq_len = phi.shape[1]
    device = phi.device

    # Initialize with first residue
    init_coords = torch.tensor(np.array([N_INIT, CA_INIT, C_INIT]), 
                              dtype=torch.float32, device=device)
    coords = init_coords.unsqueeze(0).expand(batch_size, 3, 3).clone()  # [batch, 3, 3]

    # Broadcast bond lengths
    ensure_tensor = lambda x: (torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True).expand(phi.shape) 
                              if isinstance(x, (float, int)) else x)
    bond_len_n_ca = ensure_tensor(bond_len_n_ca)
    bond_len_ca_c = ensure_tensor(bond_len_ca_c)
    bond_len_c_n = ensure_tensor(bond_len_c_n)

    # For NERF, we need to handle the indexing properly
    # phi[0] is undefined (N-terminus), psi[-1] and omega[-1] are undefined (C-terminus)
    # So we use phi[1:], psi[:-1], omega[:-1] for the main loop
    
    # Determine how many residues we can actually build
    build_length = min(seq_len - 1, seq_len - 1, seq_len - 1)  # Based on available angles
    
    for i in range(build_length):
        # Use angles starting from appropriate indices
        phi_i = phi[:, i + 1] if i + 1 < seq_len else phi[:, -1]  # phi starts from residue 1
        psi_i = psi[:, i] if i < seq_len else psi[:, -1]          # psi goes until residue n-1
        omega_i = omega[:, i] if i < seq_len else omega[:, -1]    # omega goes until residue n-1
        
        # Bond parameters for this step
        bond_len_c_n_i = bond_len_c_n[:, i] if bond_len_c_n.dim() > 1 else bond_len_c_n
        bond_len_n_ca_i = bond_len_n_ca[:, i] if bond_len_n_ca.dim() > 1 else bond_len_n_ca
        bond_len_ca_c_i = bond_len_ca_c[:, i] if bond_len_ca_c.dim() > 1 else bond_len_ca_c
        
        bond_angle_ca_c_n_i = bond_angle_ca_c_n[:, i] if bond_angle_ca_c_n.dim() > 1 else bond_angle_ca_c_n
        bond_angle_c_n_ca_i = bond_angle_c_n_ca[:, i] if bond_angle_c_n_ca.dim() > 1 else bond_angle_c_n_ca
        bond_angle_n_ca_c_i = bond_angle_n_ca_c[:, i] if bond_angle_n_ca_c.dim() > 1 else bond_angle_n_ca_c

        # Place C-N bond
        n_coord = place_dihedral(
            coords[:, -3, :],
            coords[:, -2, :],
            coords[:, -1, :],
            bond_angle=bond_angle_ca_c_n_i,
            bond_length=bond_len_c_n_i,
            torsion_angle=psi_i,
            use_torch=True,
        )

        # Place N-CA bond
        ca_coord = place_dihedral(
            coords[:, -2, :],
            coords[:, -1, :],
            n_coord,
            bond_angle=bond_angle_c_n_ca_i,
            bond_length=bond_len_n_ca_i,
            torsion_angle=omega_i,
            use_torch=True,
        )

        # Place CA-C bond
        c_coord = place_dihedral(
            coords[:, -1, :],
            n_coord,
            ca_coord,
            bond_angle=bond_angle_n_ca_c_i,
            bond_length=bond_len_ca_c_i,
            torsion_angle=phi_i,
            use_torch=True,
        )

        # Add new coordinates
        new_coords = torch.stack([n_coord, ca_coord, c_coord], dim=1)  # [batch, 3, 3]
        coords = torch.cat([coords, new_coords], dim=1)

    return coords


class DifferentiableNERF(nn.Module):
    """
    Differentiable NERF module for use in neural networks.
    """
    
    def __init__(self, 
                 use_standard_lengths: bool = True,
                 use_standard_angles: bool = True):
        super().__init__()
        self.use_standard_lengths = use_standard_lengths
        self.use_standard_angles = use_standard_angles
        
        if use_standard_lengths:
            self.register_buffer('bond_len_n_ca', torch.tensor(N_CA_LENGTH))
            self.register_buffer('bond_len_ca_c', torch.tensor(CA_C_LENGTH))
            self.register_buffer('bond_len_c_n', torch.tensor(C_N_LENGTH))
        
        if use_standard_angles:
            self.register_buffer('bond_angle_n_ca', torch.tensor(BOND_ANGLE_N_CA))
            self.register_buffer('bond_angle_ca_c', torch.tensor(BOND_ANGLE_CA_C))
            self.register_buffer('bond_angle_c_n', torch.tensor(BOND_ANGLE_C_N))
    
    def forward(self, 
                phi: torch.Tensor, 
                psi: torch.Tensor, 
                omega: torch.Tensor,
                bond_lengths: Optional[torch.Tensor] = None,
                bond_angles: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of differentiable NERF.
        
        Args:
            phi: [B, L] phi dihedral angles
            psi: [B, L] psi dihedral angles  
            omega: [B, L] omega dihedral angles
            bond_lengths: [B, L, 3] optional bond lengths (N-CA, CA-C, C-N)
            bond_angles: [B, L, 3] optional bond angles (N-CA-C, CA-C-N, C-N-CA)
            
        Returns:
            coords: [B, (L+1)*3, 3] Cartesian coordinates
        """
        batch_size, seq_len = phi.shape
        device = phi.device
        
        # Use provided or default bond parameters
        if bond_lengths is not None:
            bond_len_n_ca = bond_lengths[:, :, 0]
            bond_len_ca_c = bond_lengths[:, :, 1] 
            bond_len_c_n = bond_lengths[:, :, 2]
        else:
            bond_len_n_ca = self.bond_len_n_ca.expand(batch_size, seq_len).clone()
            bond_len_ca_c = self.bond_len_ca_c.expand(batch_size, seq_len).clone()
            bond_len_c_n = self.bond_len_c_n.expand(batch_size, seq_len).clone()
        
        if bond_angles is not None:
            bond_angle_n_ca_c = bond_angles[:, :, 0]
            bond_angle_ca_c_n = bond_angles[:, :, 1]
            bond_angle_c_n_ca = bond_angles[:, :, 2]
        else:
            bond_angle_n_ca_c = self.bond_angle_n_ca.expand(batch_size, seq_len).clone()
            bond_angle_ca_c_n = self.bond_angle_ca_c.expand(batch_size, seq_len).clone()
            bond_angle_c_n_ca = self.bond_angle_c_n.expand(batch_size, seq_len).clone()
        
        # Build coordinates using batch NERF function
        coords = nerf_build_batch(
            phi=phi,
            psi=psi, 
            omega=omega,
            bond_angle_n_ca_c=bond_angle_n_ca_c,
            bond_angle_ca_c_n=bond_angle_ca_c_n,
            bond_angle_c_n_ca=bond_angle_c_n_ca,
            bond_len_n_ca=bond_len_n_ca,
            bond_len_ca_c=bond_len_ca_c,
            bond_len_c_n=bond_len_c_n
        )
        
        return coords


def main():
    """Test function."""
    import torch
    
    # Test data
    batch_size = 2
    seq_len = 10
    
    phi = torch.randn(batch_size, seq_len) * np.pi
    psi = torch.randn(batch_size, seq_len) * np.pi  
    omega = torch.randn(batch_size, seq_len) * np.pi
    
    # Test differentiable NERF
    nerf = DifferentiableNERF()
    coords = nerf(phi, psi, omega)
    print(f"Output shape: {coords.shape}")
    print(f"Expected shape: ({batch_size}, {(seq_len+1)*3}, 3)")
    
    # Test gradient flow
    loss = coords.sum()
    loss.backward()
    print("Gradient test passed!")


if __name__ == "__main__":
    main()
