"""
Modified training script for torsion angle flow matching.
Integrates the new torsion angle data representation with mixed flow matching.
"""

import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"

import copy
import logging
import sys
import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional

# Add project root to Python path - using absolute path
sys.path.insert(0, '/storage2/hechuan/code/foldflow-mace')

import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tree
import wandb
import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

# FoldFlow imports
import foldflow.utils.experiments_utils as eu
from foldflow.data.torsion_angle_loader import TorsionAngleDataset, collate_torsion_angles
from foldflow.models.torus_flow import MixedFlowMatcher, TorsionFlowLoss, sample_torus_noise, sample_euclidean_noise
from foldflow.models.nerf_reconstruction import DifferentiableNERF
from foldflow.evaluation.tm_score_evaluator import TorsionFlowEvaluator, plot_training_metrics
from foldflow.data import utils as du


class TorsionFlowExperiment:
    """Experiment class for torsion angle flow matching."""
    
    def __init__(self, conf: DictConfig):
        """Initialize experiment with torsion angle configuration."""
        self._log = logging.getLogger(__name__)
        self._conf = conf
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize datasets
        self._init_datasets()
        
        # Initialize model
        self._init_model()
        
        # Initialize optimizer and scheduler
        self._init_optimizer()
        
        # Initialize loss function
        self._init_loss()
        
        # Initialize NERF reconstruction
        self._nerf_reconstructor = DifferentiableNERF().to(self._device)
        
        # Initialize evaluator
        exp_conf = self._conf.experiment
        output_dir = exp_conf.get('output_dir', './evaluation_results')
        self.evaluator = TorsionFlowEvaluator(
            model=self.model,
            device=self._device,
            output_dir=output_dir
        )
        
        # Training history for plotting
        self.tm_score_history = []
        self.loss_history = []
        
        # Experiment tracking
        self._step = 0
        self._epoch = 0
        
    def _init_datasets(self):
        """Initialize torsion angle datasets."""
        data_conf = self._conf.data
        
        # Training dataset
        self._train_dataset = TorsionAngleDataset(
            data_conf=data_conf,
            is_training=True
        )
        
        # Validation dataset (if different)
        self._valid_dataset = TorsionAngleDataset(
            data_conf=data_conf,
            is_training=False
        )
        
        self._log.info(f"Train dataset size: {len(self._train_dataset)}")
        self._log.info(f"Valid dataset size: {len(self._valid_dataset)}")
        
        # Data loaders
        exp_conf = self._conf.experiment
        self._train_loader = DataLoader(
            self._train_dataset,
            batch_size=exp_conf.batch_size,
            shuffle=True,
            num_workers=exp_conf.get('num_loader_workers', 0),
            collate_fn=collate_torsion_angles,
            pin_memory=True
        )
        
        self._valid_loader = DataLoader(
            self._valid_dataset,
            batch_size=exp_conf.get('eval_batch_size', exp_conf.batch_size),
            shuffle=False,
            num_workers=exp_conf.get('num_loader_workers', 0),
            collate_fn=collate_torsion_angles,
            pin_memory=True
        )
    
    def _init_model(self):
        """Initialize the mixed flow matching model."""
        model_conf = self._conf.model
        
        self._model = MixedFlowMatcher(
            torus_hidden_dim=model_conf.get('torus_hidden_dim', 256),
            torus_layers=model_conf.get('torus_layers', 6),
            euclidean_hidden_dim=model_conf.get('euclidean_hidden_dim', 256),
            euclidean_layers=model_conf.get('euclidean_layers', 4),
            num_heads=model_conf.get('num_heads', 8)
        ).to(self._device)
        
        self._log.info(f"Model parameters: {sum(p.numel() for p in self._model.parameters()):,}")
    
    def _init_optimizer(self):
        """Initialize optimizer and learning rate scheduler."""
        exp_conf = self._conf.experiment
        
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=exp_conf.learning_rate,
            weight_decay=exp_conf.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=exp_conf.num_epoch,
            eta_min=exp_conf.learning_rate * 0.01
        )
    
    def _init_loss(self):
        """Initialize loss function."""
        loss_conf = self._conf.get('loss', {})
        
        self._loss_fn = TorsionFlowLoss(
            torus_weight=loss_conf.get('torus_weight', 1.0),
            bond_angle_weight=loss_conf.get('bond_angle_weight', 1.0),
            bond_length_weight=loss_conf.get('bond_length_weight', 0.1)
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self._model.train()
        
        # Move batch to device
        batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Sample time steps
        B = batch['batch_size']
        t = torch.rand(B, device=self._device)  # [0, 1]
        
        # Sample noise for flow matching
        flow_conf = self._conf.get('flow_matcher', {})
        torus_sigma = flow_conf.get('torus_sigma', 0.1)
        euclidean_sigma = flow_conf.get('euclidean_sigma', 0.2)
        
        # Get clean data
        x0_torus = batch['torus_coords']      # [B, N, 3, 2]
        x0_bond_angles = batch['bond_angles'] # [B, N, 3]
        x0_bond_lengths = batch['bond_lengths'] # [B, N, 3]
        
        # Sample noise
        x1_torus = sample_torus_noise(x0_torus.shape, self._device, torus_sigma)
        x1_bond_angles = sample_euclidean_noise(x0_bond_angles.shape, self._device, euclidean_sigma)
        x1_bond_lengths = sample_euclidean_noise(x0_bond_lengths.shape, self._device, euclidean_sigma)
        
        # Interpolate between noise and data
        # xt = (1 - t) * x1 + t * x0
        t_expanded_torus = t.view(B, 1, 1, 1).expand_as(x0_torus)
        t_expanded_euclidean = t.view(B, 1, 1).expand_as(x0_bond_angles)
        
        xt_torus = (1 - t_expanded_torus) * x1_torus + t_expanded_torus * x0_torus
        xt_bond_angles = (1 - t_expanded_euclidean) * x1_bond_angles + t_expanded_euclidean * x0_bond_angles
        xt_bond_lengths = (1 - t_expanded_euclidean) * x1_bond_lengths + t_expanded_euclidean * x0_bond_lengths
        
        # Target velocities (flow field)
        target_torus = x0_torus - x1_torus
        target_bond_angles = x0_bond_angles - x1_bond_angles
        target_bond_lengths = x0_bond_lengths - x1_bond_lengths
        
        # Prepare input batch for model
        model_input = {
            'torus_coords': xt_torus,
            'bond_angles': xt_bond_angles,
            'bond_lengths': xt_bond_lengths,
            'sequence_mask': batch['sequence_mask']
        }
        
        # Forward pass
        pred_velocities = self._model(model_input, t)
        
        # Target velocities
        target_velocities = {
            'dihedral_velocity': target_torus,
            'bond_angle_velocity': target_bond_angles,
            'bond_length_velocity': target_bond_lengths
        }
        
        # Compute loss
        loss_dict = self._loss_fn(pred_velocities, target_velocities, batch['sequence_mask'])
        
        # Backward pass
        self._optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
        
        self._optimizer.step()
        
        # Convert losses to float for logging
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        
        return loss_dict
    
    @torch.no_grad()
    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step."""
        self._model.eval()
        
        # Move batch to device
        batch = {k: v.to(self._device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Sample time steps
        B = batch['batch_size']
        t = torch.rand(B, device=self._device)
        
        # Sample noise
        flow_conf = self._conf.get('flow_matcher', {})
        torus_sigma = flow_conf.get('torus_sigma', 0.1)
        euclidean_sigma = flow_conf.get('euclidean_sigma', 0.2)
        
        x0_torus = batch['torus_coords']
        x0_bond_angles = batch['bond_angles']
        x0_bond_lengths = batch['bond_lengths']
        
        x1_torus = sample_torus_noise(x0_torus.shape, self._device, torus_sigma)
        x1_bond_angles = sample_euclidean_noise(x0_bond_angles.shape, self._device, euclidean_sigma)
        x1_bond_lengths = sample_euclidean_noise(x0_bond_lengths.shape, self._device, euclidean_sigma)
        
        # Interpolate
        t_expanded_torus = t.view(B, 1, 1, 1).expand_as(x0_torus)
        t_expanded_euclidean = t.view(B, 1, 1).expand_as(x0_bond_angles)
        
        xt_torus = (1 - t_expanded_torus) * x1_torus + t_expanded_torus * x0_torus
        xt_bond_angles = (1 - t_expanded_euclidean) * x1_bond_angles + t_expanded_euclidean * x0_bond_angles
        xt_bond_lengths = (1 - t_expanded_euclidean) * x1_bond_lengths + t_expanded_euclidean * x0_bond_lengths
        
        # Target velocities
        target_velocities = {
            'dihedral_velocity': x0_torus - x1_torus,
            'bond_angle_velocity': x0_bond_angles - x1_bond_angles,
            'bond_length_velocity': x0_bond_lengths - x1_bond_lengths
        }
        
        # Forward pass
        model_input = {
            'torus_coords': xt_torus,
            'bond_angles': xt_bond_angles,
            'bond_lengths': xt_bond_lengths,
            'sequence_mask': batch['sequence_mask']
        }
        
        pred_velocities = self._model(model_input, t)
        
        # Compute loss
        loss_dict = self._loss_fn(pred_velocities, target_velocities, batch['sequence_mask'])
        
        # Convert to float
        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        
        return loss_dict
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self._model.train()
        
        epoch_losses = defaultdict(list)
        num_batches = len(self._train_loader)
        
        for batch_idx, batch in enumerate(self._train_loader):
            try:
                # Training step
                loss_dict = self.train_step(batch)
                
                # Accumulate losses
                for k, v in loss_dict.items():
                    epoch_losses[k].append(v)
                
                # Logging
                if batch_idx % self._conf.experiment.get('log_freq', 50) == 0:
                    self._log.info(
                        f"Epoch {self._epoch}, Batch {batch_idx}/{num_batches}, "
                        f"Loss: {loss_dict['total_loss']:.4f}"
                    )
                    
                    if hasattr(self, '_wandb_run'):
                        wandb.log({
                            'train/step_loss': loss_dict['total_loss'],
                            'train/step': self._step
                        })
                
                self._step += 1
                
            except Exception as e:
                self._log.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Average losses over epoch
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self._model.eval()
        
        epoch_losses = defaultdict(list)
        
        for batch_idx, batch in enumerate(self._valid_loader):
            try:
                loss_dict = self.validation_step(batch)
                
                for k, v in loss_dict.items():
                    epoch_losses[k].append(v)
                    
            except Exception as e:
                self._log.error(f"Error in validation batch {batch_idx}: {e}")
                continue
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    @torch.no_grad()
    def sample_structures(self, num_samples: int = 4, num_steps: int = 100) -> Dict[str, torch.Tensor]:
        """Sample new protein structures using the trained model."""
        self._model.eval()
        
        # Get a reference batch for dimensions
        ref_batch = next(iter(self._valid_loader))
        max_len = ref_batch['max_len']
        
        # Sample initial noise
        flow_conf = self._conf.get('flow_matcher', {})
        torus_sigma = flow_conf.get('torus_sigma', 0.1)
        euclidean_sigma = flow_conf.get('euclidean_sigma', 0.2)
        
        x_torus = sample_torus_noise((num_samples, max_len, 3, 2), self._device, torus_sigma)
        x_bond_angles = sample_euclidean_noise((num_samples, max_len, 3), self._device, euclidean_sigma)
        x_bond_lengths = sample_euclidean_noise((num_samples, max_len, 3), self._device, euclidean_sigma)
        
        # Create sequence mask (assume all positions are valid for now)
        seq_mask = torch.ones(num_samples, max_len, dtype=torch.bool, device=self._device)
        
        # Euler integration
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t = torch.ones(num_samples, device=self._device) * (1.0 - step * dt)
            
            model_input = {
                'torus_coords': x_torus,
                'bond_angles': x_bond_angles,
                'bond_lengths': x_bond_lengths,
                'sequence_mask': seq_mask
            }
            
            velocities = self._model(model_input, t)
            
            # Update positions
            x_torus = x_torus + dt * velocities['dihedral_velocity']
            x_bond_angles = x_bond_angles + dt * velocities['bond_angle_velocity'] 
            x_bond_lengths = x_bond_lengths + dt * velocities['bond_length_velocity']
            
            # Project torus coordinates back to manifold
            x_torus = x_torus / (torch.norm(x_torus, dim=-1, keepdim=True) + 1e-8)
        
        # Convert torus coordinates back to angles
        phi = torch.atan2(x_torus[:, :, 0, 1], x_torus[:, :, 0, 0])  # [B, N]
        psi = torch.atan2(x_torus[:, :, 1, 1], x_torus[:, :, 1, 0])  # [B, N]
        omega = torch.atan2(x_torus[:, :, 2, 1], x_torus[:, :, 2, 0])  # [B, N]
        
        # Reconstruct Cartesian coordinates using NERF
        coords = self._nerf_reconstructor(
            phi=phi,
            psi=psi,
            omega=omega,
            bond_lengths=x_bond_lengths,
            bond_angles=x_bond_angles
        )
        
        return {
            'coordinates': coords,
            'phi': phi,
            'psi': psi,
            'omega': omega,
            'bond_angles': x_bond_angles,
            'bond_lengths': x_bond_lengths
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
            'epoch': self._epoch,
            'step': self._step,
            'config': OmegaConf.to_yaml(self._conf)
        }
        torch.save(checkpoint, filepath)
        self._log.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self._device)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self._epoch = checkpoint['epoch']
        self._step = checkpoint['step']
        self._log.info(f"Checkpoint loaded from {filepath}")
    
    def start_training(self):
        """Start the training loop."""
        exp_conf = self._conf.experiment
        
        # Initialize wandb if specified
        if exp_conf.get('use_wandb', True):
            wandb.init(
                project=exp_conf.get('wandb_project', 'torsion-flow'),
                name=exp_conf.get('name', 'torsion_flow_experiment'),
                config=OmegaConf.to_container(self._conf, resolve=True)
            )
            self._wandb_run = wandb.run
        
        best_loss = float('inf')
        patience_counter = 0
        patience = exp_conf.get('patience', 10)
        
        for epoch in range(self._epoch, exp_conf.num_epoch):
            self._epoch = epoch
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate_epoch()
            
            # TM-Score evaluation (every few epochs to save time)
            eval_metrics = {}
            if epoch % exp_conf.get('eval_freq', 5) == 0:
                eval_metrics = self.evaluator.evaluate_epoch(
                    self._val_loader, epoch, self._nerf_reconstructor
                )
                
                # Plot sample structures occasionally
                if epoch % (exp_conf.get('eval_freq', 5) * 2) == 0:
                    # Get a sample batch for structure visualization
                    sample_batch = next(iter(self._val_loader))
                    sample_batch = {k: v.to(self._device) if torch.is_tensor(v) else v 
                                  for k, v in sample_batch.items()}
                    self.evaluator.plot_sample_structures(
                        sample_batch, self._nerf_reconstructor, 
                        num_samples=2, epoch=epoch
                    )
            
            # Store history
            self.loss_history.append(val_losses['total_loss'])
            self.tm_score_history.append(eval_metrics.get('avg_tm_score', 0.0))
            
            # Learning rate scheduling
            self._scheduler.step()
            
            # Logging
            log_msg = (f"Epoch {epoch}: Train Loss = {train_losses['total_loss']:.4f}, "
                      f"Val Loss = {val_losses['total_loss']:.4f}")
            if eval_metrics:
                log_msg += f", TM-Score = {eval_metrics.get('avg_tm_score', 0.0):.4f}"
            self._log.info(log_msg)
            
            if hasattr(self, '_wandb_run'):
                log_dict = {}
                for k, v in train_losses.items():
                    log_dict[f'train/{k}'] = v
                for k, v in val_losses.items():
                    log_dict[f'val/{k}'] = v
                if eval_metrics:
                    for k, v in eval_metrics.items():
                        log_dict[f'eval/{k}'] = v
                log_dict['epoch'] = epoch
                log_dict['lr'] = self._scheduler.get_last_lr()[0]
                wandb.log(log_dict)
            
            # Early stopping and checkpointing
            if val_losses['total_loss'] < best_loss:
                best_loss = val_losses['total_loss']
                patience_counter = 0
                
                # Save best checkpoint
                ckpt_dir = exp_conf.get('ckpt_dir', './checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                best_path = os.path.join(ckpt_dir, 'best_model.pt')
                self.save_checkpoint(best_path)
            else:
                patience_counter += 1
            
            # Save periodic checkpoint
            if epoch % exp_conf.get('ckpt_freq', 10) == 0:
                ckpt_dir = exp_conf.get('ckpt_dir', './checkpoints')
                os.makedirs(ckpt_dir, exist_ok=True)
                epoch_path = os.path.join(ckpt_dir, f'model_epoch_{epoch}.pt')
                self.save_checkpoint(epoch_path)
            
            # Early stopping
            if patience_counter >= patience:
                self._log.info(f"Early stopping at epoch {epoch}")
                break
        
        self._log.info("Training completed!")
        
        # Generate training progress plots
        if len(self.tm_score_history) > 1:
            self._log.info("Generating training progress plots...")
            progress_fig = plot_training_metrics(
                self.tm_score_history,
                self.loss_history,
                save_path=f"{self.evaluator.output_dir}/training_progress.png"
            )
            
            if hasattr(self, '_wandb_run'):
                wandb.log({"training_progress": wandb.Image(progress_fig)})
            plt.close(progress_fig)
        
        # Generate some sample structures
        if exp_conf.get('generate_samples', True):
            self._log.info("Generating sample structures...")
            samples = self.sample_structures(num_samples=4)
            
            if hasattr(self, '_wandb_run'):
                # Log sample information
                wandb.log({
                    'samples/num_generated': samples['coordinates'].shape[0],
                    'samples/avg_length': samples['coordinates'].shape[1] // 3
                })


@hydra.main(version_base=None, config_path="../runner/config", config_name="torsion_flow")
def main(conf: DictConfig):
    """Main training function."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set random seeds
    torch.manual_seed(conf.get('seed', 42))
    np.random.seed(conf.get('seed', 42))
    
    # Set torch settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create and run experiment
    experiment = TorsionFlowExperiment(conf)
    experiment.start_training()


if __name__ == "__main__":
    main()
