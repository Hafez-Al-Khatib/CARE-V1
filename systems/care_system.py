"""
CARE Lightning System

PyTorch Lightning module for training CARE networks.
Handles training loop, metrics logging (accuracy, sparsity, synops), and optimization.

Author: CARE Research Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import pytorch_lightning as pl
except ImportError:
    import lightning as pl  # lightning >= 2.0

from models.care_module import CareBlock


class CARESystem(pl.LightningModule):
    """
    PyTorch Lightning Module for CARE (Continuously Adapting and Reorganizing Ecosystems).
    
    Handles:
        - Forward pass over time steps (vectorized where possible)
        - Loss computation via snntorch.functional
        - Metric logging: accuracy, sparsity_rate, synaptic_operations
        - Optimizer configuration with optional LR scheduling
        - STDP update application after gradient steps
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Number of output classes
        num_steps: Number of simulation time steps
        reservoir_size: Size of reservoir layer
        learning_rate: Optimizer learning rate
        weight_decay: L2 regularization
        beta: LIF neuron membrane decay
        threshold: LIF spike threshold
        surrogate_slope: Surrogate gradient slope
        sparsity_threshold: Activation threshold for sparsity masking
        loss_type: Loss function type ('mse_count', 'ce_rate', 'ce_count')
        correct_rate: Target rate for correct class (mse_count)
        incorrect_rate: Target rate for incorrect classes (mse_count)
        eta_stdp: STDP learning rate
        enable_stdp: Whether to apply STDP updates
        stdp_blend: How much STDP to blend (0-1)
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        num_steps: int = 25,
        reservoir_size: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
        sparsity_threshold: float = 0.1,
        loss_type: str = "mse_count",
        correct_rate: float = 0.8,
        incorrect_rate: float = 0.2,
        eta_stdp: float = 0.001,
        enable_stdp: bool = True,
        stdp_blend: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Build network
        self.network = CareBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            reservoir_size=reservoir_size,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
            sparsity_threshold=sparsity_threshold,
            eta_stdp=eta_stdp,
        )
        
        # Store hyperparams
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.correct_rate = correct_rate
        self.incorrect_rate = incorrect_rate
        self.enable_stdp = enable_stdp
        self.stdp_blend = stdp_blend
        
        # Metrics state
        self._total_synops = 0
    
    def forward(
        self,
        x: Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through CARE network.
        
        Args:
            x: Input tensor. Shape depends on input type:
                - Static: [batch, features] (will be rate-encoded)
                - Temporal: [batch, time_steps, features]
            num_steps: Override default time steps
            
        Returns:
            Tuple of:
                - Output spike counts [batch, output_dim]
                - Metrics dict with 'sparsity', 'synops', 'spike_record'
        """
        steps = num_steps or self.num_steps
        return self.network(x, num_steps=steps)
    
    def _compute_loss(
        self,
        spike_counts: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Compute loss using snntorch functional losses.
        
        Args:
            spike_counts: Spike counts per class [batch, output_dim]
            targets: Class labels [batch]
            
        Returns:
            Scalar loss tensor
        """
        # Normalize spike counts to rates
        spike_rates = spike_counts / self.num_steps
        
        if self.loss_type == "mse_count":
            # MSE count loss: target rate for correct vs incorrect classes
            # Create one-hot targets with correct/incorrect rates
            num_classes = spike_counts.shape[-1]
            targets_onehot = F.one_hot(targets, num_classes).float()
            target_rates = (
                targets_onehot * self.correct_rate + 
                (1 - targets_onehot) * self.incorrect_rate
            )
            # MSE between actual rates and target rates
            loss = F.mse_loss(spike_rates, target_rates)
            
        elif self.loss_type == "ce_rate":
            # Cross-entropy on spike rates (softmax applied internally)
            loss = F.cross_entropy(spike_rates, targets)
            
        elif self.loss_type == "ce_count":
            # Cross-entropy on spike counts
            loss = F.cross_entropy(spike_counts, targets)
            
        else:
            # Fallback: standard cross-entropy on rates
            loss = F.cross_entropy(spike_rates, targets)
        
        return loss
    
    def _compute_metrics(
        self,
        spike_counts: Tensor,
        targets: Tensor,
        metrics: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute training/validation metrics.
        
        Args:
            spike_counts: Output spike counts [batch, output_dim]
            targets: Ground truth labels [batch]
            metrics: Dict from forward pass with 'sparsity', 'synops'
            
        Returns:
            Dict with all computed metrics
        """
        # Accuracy: argmax of spike counts
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        # Sparsity from network
        sparsity = metrics.get('sparsity', torch.tensor(0.0))
        
        # Synaptic operations
        synops = metrics.get('synops', torch.tensor(0))
        
        return {
            'accuracy': acc,
            'sparsity_rate': sparsity,
            'synaptic_operations': synops,
        }
    
    def training_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Tensor:
        """
        Execute single training step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        inputs, targets = batch
        
        # Forward pass
        spike_counts, forward_metrics = self(inputs)
        
        # Compute loss
        loss = self._compute_loss(spike_counts, targets)
        
        # Compute metrics
        metrics = self._compute_metrics(spike_counts, targets, forward_metrics)
        
        # Log everything
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', metrics['accuracy'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/sparsity_rate', metrics['sparsity_rate'], on_step=True, on_epoch=True)
        self.log('train/synops', metrics['synaptic_operations'], on_step=True, on_epoch=True)
        
        # Accumulate total synops
        self._total_synops += metrics['synaptic_operations'].item()
        
        return loss
    
    def on_after_backward(self) -> None:
        """Apply STDP updates after backward pass if enabled."""
        if self.enable_stdp:
            self.network.apply_stdp_updates(blend_factor=self.stdp_blend)
    
    def validation_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """
        Execute single validation step.
        
        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index
            
        Returns:
            Dict with metrics
        """
        inputs, targets = batch
        
        # Forward pass
        spike_counts, forward_metrics = self(inputs)
        
        # Compute loss and metrics
        loss = self._compute_loss(spike_counts, targets)
        metrics = self._compute_metrics(spike_counts, targets, forward_metrics)
        
        # Log validation metrics
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log('val/sparsity_rate', metrics['sparsity_rate'], on_epoch=True)
        self.log('val/synops', metrics['synaptic_operations'], on_epoch=True)
        
        return {'val_loss': loss, **metrics}
    
    def test_step(
        self,
        batch: Tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> Dict[str, Tensor]:
        """Execute single test step (same as validation)."""
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configure optimizer and optional LR scheduler.
        
        Returns:
            Dict with 'optimizer' and optionally 'lr_scheduler'
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        config: Dict[str, Any] = {'optimizer': optimizer}
        
        # Optional: Cosine annealing scheduler
        if hasattr(self.hparams, 'scheduler') and self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=1e-6,
            )
            config['lr_scheduler'] = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        
        return config
    
    def on_train_epoch_end(self) -> None:
        """Log epoch-level metrics."""
        self.log('train/total_synops', self._total_synops)
        self._total_synops = 0  # Reset for next epoch


# =============================================================================
# Factory function for Hydra
# =============================================================================

def build_care_system(cfg: Any) -> CARESystem:
    """
    Factory function for building CARESystem from Hydra config.
    
    Args:
        cfg: Hydra DictConfig with model, training sections
        
    Returns:
        Configured CARESystem instance
    """
    return CARESystem(
        # Model architecture
        input_dim=cfg.model.architecture.input_dim,
        hidden_dim=cfg.model.architecture.hidden_dim,
        output_dim=cfg.model.architecture.output_dim,
        num_steps=cfg.model.architecture.num_steps,
        reservoir_size=cfg.model.architecture.reservoir_size,
        sparsity_threshold=cfg.model.architecture.sparsity_threshold,
        
        # Neuron params
        beta=cfg.model.neuron.beta,
        threshold=cfg.model.neuron.threshold,
        surrogate_slope=cfg.model.neuron.slope,
        
        # Plasticity
        eta_stdp=cfg.model.plasticity.eta_stdp,
        enable_stdp=cfg.model.plasticity.enabled,
        
        # Training
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        loss_type=cfg.training.loss.type,
        correct_rate=cfg.training.loss.correct_rate,
        incorrect_rate=cfg.training.loss.incorrect_rate,
    )
