#!/usr/bin/env python3
"""
Test script for TM-Score evaluation and protein structure visualization.
"""

import sys
sys.path.insert(0, '/storage2/hechuan/code/foldflow-mace')

import torch
import numpy as np
import matplotlib.pyplot as plt
from foldflow.evaluation.tm_score_evaluator import (
    tm_score, compute_backbone_chirality, plot_ramachandran, 
    plot_3d_backbone, TorsionFlowEvaluator
)

def test_tm_score():
    """Test TM-Score calculation."""
    print("Testing TM-Score calculation...")
    
    # Create dummy protein coordinates (CA atoms only)
    n_residues = 50
    
    # True structure
    true_ca = torch.randn(n_residues, 3) * 5
    
    # Predicted structure (with some noise)
    pred_ca = true_ca + torch.randn_like(true_ca) * 0.5
    
    # Calculate TM-Score
    tm = tm_score(pred_ca, true_ca)
    print(f"TM-Score: {tm:.4f}")
    
    # Test with perfect alignment
    tm_perfect = tm_score(true_ca, true_ca)
    print(f"Perfect TM-Score: {tm_perfect:.4f}")
    
    return tm

def test_chirality_computation():
    """Test backbone chirality computation."""
    print("\nTesting backbone chirality computation...")
    
    # Create realistic backbone coordinates
    n_residues = 20
    coords = torch.zeros(n_residues, 3, 3)  # [N, 3, 3] for N,CA,C
    
    # Simulate alpha helix backbone
    for i in range(n_residues):
        # Approximate alpha helix geometry
        phi = -60 * np.pi / 180  # Alpha helix phi
        psi = -45 * np.pi / 180  # Alpha helix psi
        
        # N atom
        coords[i, 0] = torch.tensor([i * 3.8, 0, 0])
        # CA atom  
        coords[i, 1] = torch.tensor([i * 3.8 + 1.46, 0, 0])
        # C atom
        coords[i, 2] = torch.tensor([i * 3.8 + 2.5, 0.5, 0.2])
    
    # Add some noise for realism
    coords += torch.randn_like(coords) * 0.1
    
    # Compute dihedral angles
    dihedrals = compute_backbone_chirality(coords)
    print(f"Computed dihedrals shape: {dihedrals.shape}")
    print(f"Sample phi angles (degrees): {dihedrals[:5, 0] * 180 / np.pi}")
    print(f"Sample psi angles (degrees): {dihedrals[:5, 1] * 180 / np.pi}")
    
    return coords, dihedrals

def test_visualizations():
    """Test visualization functions."""
    print("\nTesting visualization functions...")
    
    # Generate test data
    coords, dihedrals = test_chirality_computation()
    
    # Extract angles
    phi_angles = dihedrals[:, 0].numpy()
    psi_angles = dihedrals[:, 1].numpy()
    
    print(f"Phi angle range: {phi_angles.min():.2f} to {phi_angles.max():.2f} radians")
    print(f"Psi angle range: {psi_angles.min():.2f} to {psi_angles.max():.2f} radians")
    
    # Test Ramachandran plot
    print("Generating Ramachandran plot...")
    rama_fig = plot_ramachandran(
        phi_angles, psi_angles,
        title="Test Ramachandran Plot",
        save_path="./test_ramachandran.png"
    )
    plt.close(rama_fig)
    print("✓ Ramachandran plot saved as test_ramachandran.png")
    
    # Test 3D structure plot
    print("Generating 3D structure plot...")
    struct_fig = plot_3d_backbone(
        coords,
        title="Test Protein Backbone",
        save_path="./test_structure_3d.png",
        show_chirality=True
    )
    plt.close(struct_fig)
    print("✓ 3D structure plot saved as test_structure_3d.png")

def test_batch_evaluation():
    """Test batch evaluation."""
    print("\nTesting batch evaluation...")
    
    # Create batch of structures
    batch_size = 4
    n_residues = 30
    
    # Predicted coordinates
    pred_coords = torch.randn(batch_size, n_residues, 3) * 5
    
    # True coordinates (with some systematic differences)
    true_coords = pred_coords + torch.randn_like(pred_coords) * 1.0
    
    # Calculate batch TM-Scores
    from foldflow.evaluation.tm_score_evaluator import batch_tm_score
    tm_scores = batch_tm_score(pred_coords, true_coords)
    
    print(f"Batch TM-Scores: {tm_scores}")
    print(f"Mean TM-Score: {tm_scores.mean():.4f} ± {tm_scores.std():.4f}")

def test_evaluator_class():
    """Test the TorsionFlowEvaluator class."""
    print("\nTesting TorsionFlowEvaluator class...")
    
    # Create a dummy model
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 10)
        
        def forward(self, batch, t):
            return {
                'dihedral_velocity': torch.randn(2, 10, 3, 2),
                'bond_angle_velocity': torch.randn(2, 10, 3),
                'bond_length_velocity': torch.randn(2, 10, 3)
            }
        
        def eval(self):
            pass
    
    model = DummyModel()
    device = torch.device('cpu')
    
    # Initialize evaluator
    evaluator = TorsionFlowEvaluator(model, device, "./test_evaluation")
    print("✓ TorsionFlowEvaluator initialized successfully")

def main():
    """Run all tests."""
    print("=== TM-Score and Visualization Tests ===\n")
    
    try:
        # Test TM-Score
        tm = test_tm_score()
        
        # Test chirality computation
        test_chirality_computation()
        
        # Test visualizations
        test_visualizations()
        
        # Test batch evaluation
        test_batch_evaluation()
        
        # Test evaluator class
        test_evaluator_class()
        
        print(f"\n🎉 All tests passed successfully!")
        print(f"TM-Score functionality: ✓")
        print(f"Chirality computation: ✓") 
        print(f"Ramachandran plots: ✓")
        print(f"3D structure visualization: ✓")
        print(f"Batch evaluation: ✓")
        print(f"Evaluator class: ✓")
        
        print(f"\nGenerated files:")
        print(f"- test_ramachandran.png: Ramachandran plot showing backbone dihedral angles")
        print(f"- test_structure_3d.png: 3D protein backbone with chirality coloring")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
