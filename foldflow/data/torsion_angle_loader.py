"""
Torsion angle and bond angle data loader for FoldFlow.
Converts PDB structures to torsion angles (dihedral) and bond angles representation.
"""
import functools as fn
import logging
import math
import os
import pickle
import random
import time
from functools import partial
from multiprocessing import get_context
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Optional, Dict, List, Tuple

import lmdb
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tree
from torch.utils import data
from tqdm import tqdm

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
import warnings

from foldflow.data import utils as du
from foldflow.utils.rigid_helpers import assemble_rigid_mat, extract_trans_rots_mat
from openfold.data import data_transforms
from openfold.np import residue_constants
from openfold.utils import rigid_utils

warnings.simplefilter(action="ignore", category=FutureWarning)

# Standard bond lengths and angles (from nerf.py and angles_and_coords.py)
N_CA_LENGTH = 1.46
CA_C_LENGTH = 1.54
C_N_LENGTH = 1.34

# Standard bond angles in radians
BOND_ANGLE_N_CA = 121 / 180 * np.pi  # N-CA-C angle
BOND_ANGLE_CA_C = 109 / 180 * np.pi  # CA-C-N angle (tau)
BOND_ANGLE_C_N = 115 / 180 * np.pi   # C-N-CA angle

# Exhaustive and minimal angle sets
EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]
MINIMAL_ANGLES = ["phi", "psi", "omega"]
MINIMAL_DISTS = []


def xyz_to_torsion_bond_angles(coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Convert Cartesian coordinates to torsion angles and bond angles.
    
    Args:
        coords: [B, N, 3] or [N, 3] Cartesian coordinates for backbone atoms (N, CA, C pattern)
        
    Returns:
        dict containing:
            'dihedral_angles': [B, N//3-1, 3] phi, psi, omega angles
            'bond_angles': [B, N//3-1, 3] bond angles (N-CA-C, CA-C-N, C-N-CA)
            'bond_lengths': [B, N//3-1, 3] bond lengths (N-CA, CA-C, C-N)
            'torus_coords': [B, N//3-1, 3, 2] torus coordinates (cos, sin) for dihedrals
    """
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)  # Add batch dimension
    
    B, N, _ = coords.shape
    assert N % 3 == 0, f"Expected coordinates for N-CA-C pattern, got {N} atoms"
    n_residues = N // 3
    
    # Reshape to [B, n_residues, 3, 3] where last dim is (N, CA, C)
    coords_reshaped = coords.view(B, n_residues, 3, 3)
    
    # Extract N, CA, C coordinates
    N_coords = coords_reshaped[:, :, 0, :]  # [B, n_residues, 3]
    CA_coords = coords_reshaped[:, :, 1, :] # [B, n_residues, 3]  
    C_coords = coords_reshaped[:, :, 2, :]  # [B, n_residues, 3]
    
    # Compute dihedral angles (phi, psi, omega)
    phi_angles = []
    psi_angles = []
    omega_angles = []
    
    for i in range(1, n_residues):
        # phi angle: C(i-1) - N(i) - CA(i) - C(i)
        if i > 0:
            phi = compute_dihedral_angle(
                C_coords[:, i-1, :],  # C(i-1)
                N_coords[:, i, :],    # N(i)
                CA_coords[:, i, :],   # CA(i)
                C_coords[:, i, :]     # C(i)
            )
            phi_angles.append(phi)
        
        # psi angle: N(i) - CA(i) - C(i) - N(i+1)
        if i < n_residues - 1:
            psi = compute_dihedral_angle(
                N_coords[:, i, :],    # N(i)
                CA_coords[:, i, :],   # CA(i)
                C_coords[:, i, :],    # C(i)
                N_coords[:, i+1, :]   # N(i+1)
            )
            psi_angles.append(psi)
        
        # omega angle: CA(i-1) - C(i-1) - N(i) - CA(i)
        if i > 0:
            omega = compute_dihedral_angle(
                CA_coords[:, i-1, :], # CA(i-1)
                C_coords[:, i-1, :],  # C(i-1)
                N_coords[:, i, :],    # N(i)
                CA_coords[:, i, :]    # CA(i)
            )
            omega_angles.append(omega)
    
    # Stack angles and pad appropriately
    max_len = n_residues - 1
    dihedral_angles = torch.zeros(B, max_len, 3)  # [phi, psi, omega]
    
    # Fill phi angles (starts from residue 1)
    if phi_angles:
        phi_tensor = torch.stack(phi_angles, dim=1)  # [B, n_residues-1]
        dihedral_angles[:, :phi_tensor.shape[1], 0] = phi_tensor
    
    # Fill psi angles (ends at residue n_residues-2)
    if psi_angles:
        psi_tensor = torch.stack(psi_angles, dim=1)  # [B, n_residues-2]
        dihedral_angles[:, :psi_tensor.shape[1], 1] = psi_tensor
    
    # Fill omega angles (starts from residue 1)
    if omega_angles:
        omega_tensor = torch.stack(omega_angles, dim=1)  # [B, n_residues-1]
        dihedral_angles[:, :omega_tensor.shape[1], 2] = omega_tensor
    
    # Compute bond angles
    bond_angles = torch.zeros(B, max_len, 3)  # [N-CA-C, CA-C-N, C-N-CA]
    
    for i in range(n_residues - 1):
        # N-CA-C angle (tau)
        if i < n_residues:
            angle_nca_c = compute_bond_angle(
                N_coords[:, i, :],
                CA_coords[:, i, :],
                C_coords[:, i, :]
            )
            bond_angles[:, i, 0] = angle_nca_c
        
        # CA-C-N angle
        if i < n_residues - 1:
            angle_cac_n = compute_bond_angle(
                CA_coords[:, i, :],
                C_coords[:, i, :],
                N_coords[:, i+1, :]
            )
            bond_angles[:, i, 1] = angle_cac_n
        
        # C-N-CA angle
        if i < n_residues - 1:
            angle_cn_ca = compute_bond_angle(
                C_coords[:, i, :],
                N_coords[:, i+1, :],
                CA_coords[:, i+1, :]
            )
            bond_angles[:, i, 2] = angle_cn_ca
    
    # Compute bond lengths
    bond_lengths = torch.zeros(B, max_len, 3)  # [N-CA, CA-C, C-N]
    
    for i in range(n_residues - 1):
        # N-CA distance
        if i < n_residues:
            dist_n_ca = torch.norm(CA_coords[:, i, :] - N_coords[:, i, :], dim=-1)
            bond_lengths[:, i, 0] = dist_n_ca
        
        # CA-C distance
        if i < n_residues:
            dist_ca_c = torch.norm(C_coords[:, i, :] - CA_coords[:, i, :], dim=-1)
            bond_lengths[:, i, 1] = dist_ca_c
        
        # C-N distance (to next residue)
        if i < n_residues - 1:
            dist_c_n = torch.norm(N_coords[:, i+1, :] - C_coords[:, i, :], dim=-1)
            bond_lengths[:, i, 2] = dist_c_n
    
    # Convert dihedral angles to torus coordinates (cos, sin)
    torus_coords = torch.stack([
        torch.cos(dihedral_angles),
        torch.sin(dihedral_angles)
    ], dim=-1)  # [B, max_len, 3, 2]
    
    return {
        'dihedral_angles': dihedral_angles,  # [B, max_len, 3]
        'bond_angles': bond_angles,          # [B, max_len, 3]
        'bond_lengths': bond_lengths,        # [B, max_len, 3]
        'torus_coords': torus_coords,        # [B, max_len, 3, 2]
        'n_residues': n_residues
    }


def compute_dihedral_angle(p1: torch.Tensor, p2: torch.Tensor, 
                          p3: torch.Tensor, p4: torch.Tensor) -> torch.Tensor:
    """
    Compute dihedral angle defined by four points.
    
    Args:
        p1, p2, p3, p4: [B, 3] coordinates of four points
        
    Returns:
        [B] dihedral angles in radians
    """
    # Vectors between consecutive points
    v1 = p2 - p1  # [B, 3]
    v2 = p3 - p2  # [B, 3]
    v3 = p4 - p3  # [B, 3]
    
    # Normal vectors to planes
    n1 = torch.cross(v1, v2, dim=-1)  # [B, 3]
    n2 = torch.cross(v2, v3, dim=-1)  # [B, 3]
    
    # Normalize normal vectors
    n1 = n1 / (torch.norm(n1, dim=-1, keepdim=True) + 1e-8)
    n2 = n2 / (torch.norm(n2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute dihedral angle
    cos_angle = torch.sum(n1 * n2, dim=-1)  # [B]
    sin_angle = torch.sum(torch.cross(n1, n2, dim=-1) * (v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)), dim=-1)
    
    dihedral = torch.atan2(sin_angle, cos_angle)
    return dihedral


def compute_bond_angle(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """
    Compute bond angle defined by three points (angle at p2).
    
    Args:
        p1, p2, p3: [B, 3] coordinates of three points
        
    Returns:
        [B] bond angles in radians
    """
    v1 = p1 - p2  # [B, 3]
    v2 = p3 - p2  # [B, 3]
    
    # Normalize vectors
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + 1e-8)
    v2_norm = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute angle
    cos_angle = torch.sum(v1_norm * v2_norm, dim=-1)  # [B]
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)  # Numerical stability
    angle = torch.acos(cos_angle)
    
    return angle


def torsion_to_cartesian(dihedral_angles: torch.Tensor, 
                        bond_angles: torch.Tensor,
                        bond_lengths: torch.Tensor = None,
                        init_coords: torch.Tensor = None) -> torch.Tensor:
    """
    Convert torsion angles and bond angles back to Cartesian coordinates.
    Uses NERF (Natural Extension Reference Frame) algorithm.
    
    Args:
        dihedral_angles: [B, N, 3] phi, psi, omega angles
        bond_angles: [B, N, 3] bond angles
        bond_lengths: [B, N, 3] bond lengths (optional, uses standard if None)
        init_coords: [B, 3, 3] initial N, CA, C coordinates (optional)
        
    Returns:
        coords: [B, (N+1)*3, 3] Cartesian coordinates
    """
    B, N, _ = dihedral_angles.shape
    device = dihedral_angles.device
    
    # Use standard bond lengths if not provided
    if bond_lengths is None:
        bond_lengths = torch.tensor([N_CA_LENGTH, CA_C_LENGTH, C_N_LENGTH], 
                                   device=device).expand(B, N, 3)
    
    # Use standard initial coordinates if not provided
    if init_coords is None:
        # Initial coordinates from 1CRN
        N_INIT = torch.tensor([17.047, 14.099, 3.625], device=device)
        CA_INIT = torch.tensor([16.967, 12.784, 4.338], device=device)
        C_INIT = torch.tensor([15.685, 12.755, 5.133], device=device)
        init_coords = torch.stack([N_INIT, CA_INIT, C_INIT]).expand(B, 3, 3)
    
    # Initialize coordinate list with first residue
    coords_list = [init_coords[:, i, :] for i in range(3)]  # N, CA, C of first residue
    
    # Build subsequent residues using NERF
    for i in range(N):
        # Extract angles for current residue
        phi = dihedral_angles[:, i, 0]    # phi angle
        psi = dihedral_angles[:, i, 1]    # psi angle  
        omega = dihedral_angles[:, i, 2]  # omega angle
        
        # Extract bond angles for current residue
        bond_angle_n_ca_c = bond_angles[:, i, 0]  # N-CA-C angle (tau)
        bond_angle_ca_c_n = bond_angles[:, i, 1]  # CA-C-N angle
        bond_angle_c_n_ca = bond_angles[:, i, 2]  # C-N-CA angle
        
        # Extract bond lengths for current residue
        len_n_ca = bond_lengths[:, i, 0]  # N-CA length
        len_ca_c = bond_lengths[:, i, 1]  # CA-C length
        len_c_n = bond_lengths[:, i, 2]   # C-N length
        
        # Place next N atom (using psi dihedral)
        n_coord = place_dihedral(
            coords_list[-3],  # Previous CA
            coords_list[-2],  # Previous C
            coords_list[-1],  # Current N would be here, but we use last C
            bond_angle=bond_angle_ca_c_n,
            bond_length=len_c_n,
            torsion_angle=psi
        )
        coords_list.append(n_coord)
        
        # Place next CA atom (using omega dihedral)
        ca_coord = place_dihedral(
            coords_list[-3],  # Previous C
            coords_list[-2],  # Current N
            coords_list[-1],  # Would be CA position
            bond_angle=bond_angle_c_n_ca,
            bond_length=len_n_ca,
            torsion_angle=omega
        )
        coords_list.append(ca_coord)
        
        # Place next C atom (using phi dihedral)
        c_coord = place_dihedral(
            coords_list[-3],  # Current N
            coords_list[-2],  # Current CA
            coords_list[-1],  # Would be C position
            bond_angle=bond_angle_n_ca_c,
            bond_length=len_ca_c,
            torsion_angle=phi
        )
        coords_list.append(c_coord)
    
    # Stack all coordinates
    coords = torch.stack(coords_list, dim=1)  # [B, total_atoms, 3]
    return coords


def place_dihedral(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
                  bond_angle: torch.Tensor, bond_length: torch.Tensor,
                  torsion_angle: torch.Tensor) -> torch.Tensor:
    """
    Place point d such that bond angle, length, and torsion angle are satisfied.
    
    Args:
        a, b, c: [B, 3] coordinates of three existing points
        bond_angle: [B] bond angle at point c in radians
        bond_length: [B] bond length from c to d
        torsion_angle: [B] torsion angle a-b-c-d in radians
        
    Returns:
        d: [B, 3] coordinates of new point
    """
    assert a.shape == b.shape == c.shape
    assert a.shape[-1] == 3
    
    def unit_vec(x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)
    
    def cross(x, y):
        return torch.cross(x, y, dim=-1)
    
    ab = b - a
    bc = unit_vec(c - b)
    n = unit_vec(cross(ab, bc))
    nbc = cross(n, bc)
    
    # Build rotation matrix
    m = torch.stack([bc, nbc, n], dim=-1)  # [B, 3, 3]
    
    # Compute d in local coordinate system
    d_local = torch.stack([
        -bond_length * torch.cos(bond_angle),
        bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
        bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle)
    ], dim=-1)  # [B, 3]
    
    # Transform to global coordinates
    d = torch.matmul(m, d_local.unsqueeze(-1)).squeeze(-1) + c
    
    return d


class TorsionAngleDataset(data.Dataset):
    """
    Dataset that loads PDB structures and converts them to torsion angle representation.
    """
    
    def __init__(self, data_conf, is_training=True):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self._data_conf = data_conf
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize metadata from CSV file."""
        csv_path = self._data_conf.csv_path
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self._csv = pd.read_csv(csv_path)
        self._log.info(f"Loaded CSV with {len(self._csv)} entries")
        
        # Apply length filtering
        if hasattr(self._data_conf, 'filtering'):
            min_len = getattr(self._data_conf.filtering, 'min_len', 0)
            max_len = getattr(self._data_conf.filtering, 'max_len', float('inf'))
            
            if 'length' in self._csv.columns:
                length_mask = (self._csv['length'] >= min_len) & (self._csv['length'] <= max_len)
                self._csv = self._csv[length_mask].reset_index(drop=True)
                self._log.info(f"After length filtering ({min_len}-{max_len}): {len(self._csv)} entries")
    
    def __len__(self):
        return len(self._csv)
    
    def __getitem__(self, idx):
        """Get a single example converted to torsion angle representation."""
        try:
            # Get the original processed features
            csv_row = self._csv.iloc[idx]
            processed_file_path = csv_row["processed_path"]
            
            # Load and process the chain features
            chain_feats = self._process_csv_row(processed_file_path)
            
            # Extract backbone coordinates (N, CA, C atoms)
            backbone_coords = self._extract_backbone_coords(chain_feats)
            
            # Convert to torsion angle representation
            torsion_data = xyz_to_torsion_bond_angles(backbone_coords)
            
            # Add metadata
            torsion_data.update({
                'aatype': chain_feats['aatype'],
                'seq_idx': chain_feats['seq_idx'],
                'chain_idx': chain_feats['chain_idx'],
                'res_mask': chain_feats['res_mask'],
                'pdb_name': csv_row.get('pdb_name', csv_row.get('chain_name', f'idx_{idx}')),
                'length': torsion_data['n_residues']
            })
            
            return torsion_data
            
        except Exception as e:
            self._log.warning(f"Error processing example {idx}: {e}")
            # Return a dummy example or re-raise based on your preference
            raise e
    
    def _process_csv_row(self, processed_file_path):
        """Process a single CSV row to get chain features."""
        processed_feats = du.read_pkl(processed_file_path)
        processed_feats = du.parse_chain_feats(processed_feats)

        # Only take modeled residues
        modeled_idx = processed_feats["modeled_idx"]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats["modeled_idx"]
        processed_feats = tree.map_structure(lambda x: x[min_idx : (max_idx + 1)], processed_feats)

        # Run through OpenFold data transforms
        chain_feats = {
            "aatype": torch.tensor(processed_feats["aatype"]).long(),
            "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
            "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
        }
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        chain_feats = data_transforms.make_atom14_masks(chain_feats)
        chain_feats = data_transforms.make_atom14_positions(chain_feats)

        # Re-number residue indices
        chain_idx = processed_feats["chain_index"]
        res_idx = processed_feats["residue_index"]
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = np.array(random.sample(all_chain_idx, len(all_chain_idx))) - np.min(all_chain_idx) + 1
        
        for i, chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # Return necessary features
        final_feats = {
            "aatype": chain_feats["aatype"],
            "seq_idx": new_res_idx,
            "chain_idx": new_chain_idx,
            "res_mask": processed_feats["bb_mask"],
            "atom37_pos": chain_feats["all_atom_positions"],
            "atom37_mask": chain_feats["all_atom_mask"],
        }

        return final_feats
    
    def _extract_backbone_coords(self, chain_feats):
        """Extract backbone N, CA, C coordinates from chain features."""
        atom37_pos = chain_feats["atom37_pos"]  # [N_res, 37, 3]
        atom37_mask = chain_feats["atom37_mask"]  # [N_res, 37]
        
        # Indices for N, CA, C atoms in atom37 representation
        N_idx = residue_constants.atom_order['N']    # 0
        CA_idx = residue_constants.atom_order['CA']  # 1  
        C_idx = residue_constants.atom_order['C']    # 2
        
        # Extract coordinates
        N_coords = atom37_pos[:, N_idx, :]   # [N_res, 3]
        CA_coords = atom37_pos[:, CA_idx, :] # [N_res, 3]
        C_coords = atom37_pos[:, C_idx, :]   # [N_res, 3]
        
        # Check masks
        N_mask = atom37_mask[:, N_idx]   # [N_res]
        CA_mask = atom37_mask[:, CA_idx] # [N_res]
        C_mask = atom37_mask[:, C_idx]   # [N_res]
        
        backbone_mask = N_mask * CA_mask * C_mask  # [N_res]
        valid_residues = backbone_mask.bool()
        
        if not valid_residues.any():
            raise ValueError("No valid backbone atoms found")
        
        # Stack coordinates in N-CA-C pattern and filter valid residues
        backbone_coords = torch.stack([N_coords, CA_coords, C_coords], dim=1)  # [N_res, 3, 3]
        backbone_coords = backbone_coords[valid_residues]  # [N_valid, 3, 3]
        backbone_coords = backbone_coords.view(-1, 3)  # [N_valid*3, 3]
        
        return backbone_coords.unsqueeze(0)  # Add batch dimension: [1, N_valid*3, 3]


def collate_torsion_angles(batch):
    """
    Collate function for TorsionAngleDataset.
    """
    # Find maximum sequence length in batch
    max_len = max([item['dihedral_angles'].shape[1] for item in batch])
    batch_size = len(batch)
    
    # Initialize padded tensors
    dihedral_angles = torch.zeros(batch_size, max_len, 3)
    bond_angles = torch.zeros(batch_size, max_len, 3)
    bond_lengths = torch.zeros(batch_size, max_len, 3)
    torus_coords = torch.zeros(batch_size, max_len, 3, 2)
    
    # Masks and metadata
    sequence_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    aatypes = []
    seq_indices = []
    pdb_names = []
    lengths = []
    
    for i, item in enumerate(batch):
        seq_len = item['dihedral_angles'].shape[1]
        
        # Copy data with padding
        dihedral_angles[i, :seq_len] = item['dihedral_angles'].squeeze(0)
        bond_angles[i, :seq_len] = item['bond_angles'].squeeze(0)
        bond_lengths[i, :seq_len] = item['bond_lengths'].squeeze(0)
        torus_coords[i, :seq_len] = item['torus_coords'].squeeze(0)
        
        # Set mask
        sequence_mask[i, :seq_len] = True
        
        # Collect metadata
        aatypes.append(item['aatype'])
        seq_indices.append(item['seq_idx'])
        pdb_names.append(item['pdb_name'])
        lengths.append(item['length'])
    
    return {
        'dihedral_angles': dihedral_angles,
        'bond_angles': bond_angles,
        'bond_lengths': bond_lengths,
        'torus_coords': torus_coords,
        'sequence_mask': sequence_mask,
        'aatype': aatypes,
        'seq_idx': seq_indices,
        'pdb_names': pdb_names,
        'lengths': lengths,
        'batch_size': batch_size,
        'max_len': max_len
    }
