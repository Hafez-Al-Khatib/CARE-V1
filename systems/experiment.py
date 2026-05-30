"""
Dead Neuron Experiment Module - Consolidated Edition

Components:
    - DeadNeuronCallback: Enhanced with variance logging across layers
    - PhdGradeNeuronTracker: PhD-grade monitoring callback
    - DeadNeuronExperiment: LightningModule with scalable architecture

All model components (CareLIFConv, SEWResNetBlock, MSResNetBlock, CareResNet)
are imported from models.components.neuron to avoid code duplication.

Author: Hafez Al Khatib
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import Callback

# =============================================================================
# Import all model components from the SINGLE SOURCE OF TRUTH
# =============================================================================
from models.components.neuron import (
    CareLIFConv,
    SEWResNetBlock,
    MSResNetBlock,
    CareResNet,
    sabotage_init,
    normal_init,
)


# =============================================================================
# Enhanced Dead Neuron Callback
# =============================================================================

class DeadNeuronCallback(Callback):
    """
    Enhanced callback with variance logging for depth scaling experiments.
    
    Logs:
        - dead_neuron_ratio: Aggregate dead neurons
        - dead_ratio_variance: Variance across layers (high = Signal Propagation failure)
        - Per-layer dead ratios
    """
    
    def __init__(self, log_per_layer: bool = True):
        super().__init__()
        self.log_per_layer = log_per_layer
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "DeadNeuronExperiment",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute and log dead neuron metrics."""
        
        spike_records = pl_module.get_spike_records()
        
        if not spike_records:
            return
        
        total_neurons = 0
        total_dead = 0
        layer_ratios: List[float] = []
        
        for layer_name, spikes in spike_records.items():
            if spikes.numel() == 0:
                continue
                
            # Sum over time and batch to get total spikes per neuron
            if spikes.dim() == 5:
                # Convolutional: [T, B, C, H, W] -> sum over T, B, H, W
                spikes_per_neuron = spikes.sum(dim=(0, 1, 3, 4))  # [C]
            elif spikes.dim() == 3:
                # Linear: [T, B, N] -> sum over T, B
                spikes_per_neuron = spikes.sum(dim=(0, 1))  # [N]
            else:
                continue
            
            num_neurons = spikes_per_neuron.numel()
            num_dead = (spikes_per_neuron == 0).sum().item()
            dead_ratio = num_dead / num_neurons if num_neurons > 0 else 0.0
            
            total_neurons += num_neurons
            total_dead += num_dead
            layer_ratios.append(dead_ratio)
            
            # Log per-layer
            if self.log_per_layer and trainer.logger:
                pl_module.log(
                    f"dead_neurons/{layer_name}",
                    dead_ratio,
                    on_step=False,
                    on_epoch=True,
                )
        
        # Aggregate metrics
        if total_neurons > 0:
            aggregate_ratio = total_dead / total_neurons
            pl_module.log(
                "dead_neuron_ratio",
                aggregate_ratio,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        
        # Variance across layers (KEY METRIC for depth scaling)
        if len(layer_ratios) > 1:
            ratio_tensor = torch.tensor(layer_ratios)
            variance = ratio_tensor.var().item()
            pl_module.log(
                "dead_ratio_variance",
                variance,
                on_step=True,
                on_epoch=True,
            )


# =============================================================================
# PhD-Grade Monitoring Callback
# =============================================================================

class PhdGradeNeuronTracker(Callback):
    """
    PhD-Grade Monitoring Callback (Phase 8).
    
    Tracks:
    1. Dead Neurons (True Silence over full epoch)
    2. Gradient Health (Dead vs Alive Gradient Norms)
    3. Revival Quality (Loss Impact of Revivals)
    4. Weight Distribution (Sparsity/Kurtosis)
    """
    
    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.activity_masks = {} 
        self.prev_total_dead = None
        self.prev_val_loss = None
        
    def _calculate_gini(self, x: Tensor) -> float:
        if x.numel() == 0: return 0.0
        x = x.float().flatten().abs() + 1e-8
        x = torch.sort(x)[0]
        n = x.numel()
        index = torch.arange(1, n + 1, device=x.device, dtype=torch.float)
        return ((2 * index - n - 1) * x).sum() / (n * x.sum())

    def _get_weight_and_grad(self, pl_module, key):
        """Map spike record key (e.g., 'layer1_b0_conv1') to weight parameter."""
        net = pl_module.network
        if key == 'stem':
            return net.stem.conv.weight, net.stem.conv.weight.grad
        
        parts = key.split('_') # layer1, b0, conv1
        if len(parts) < 3: return None, None
        
        layer = getattr(net, parts[0], None)
        if layer is None: return None, None
        
        try:
            block_idx = int(parts[1][1:])
            block = layer[block_idx]
            conv_module = getattr(block, parts[2], None)
            if conv_module:
                 return conv_module.conv.weight, conv_module.conv.weight.grad
        except (IndexError, AttributeError, ValueError):
            pass
            
        return None, None
    
    def on_train_epoch_start(self, trainer, pl_module):
        self.activity_masks = {}
        self.grad_sum = {}
        self.grad_count = 0
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 1. Activity Masks
        spike_records = pl_module.get_spike_records()
        for name, spikes in spike_records.items():
            if spikes.numel() == 0: continue
            
            if spikes.dim() == 5:
                activity = (spikes.sum(dim=(0, 1, 3, 4)) > 0)
            elif spikes.dim() == 3:
                activity = (spikes.sum(dim=(0, 1)) > 0)
            else:
                continue
                
            if name not in self.activity_masks:
                self.activity_masks[name] = torch.zeros_like(activity, dtype=torch.bool)
            
            self.activity_masks[name] |= activity
            
            # 2. Accumulate Gradient Norms
            weight, grad = self._get_weight_and_grad(pl_module, name)
            if grad is not None:
                # Norm per output channel/neuron
                g_norm = grad.view(grad.size(0), -1).norm(dim=1).detach()
                
                if name not in self.grad_sum:
                    self.grad_sum[name] = g_norm
                else:
                    self.grad_sum[name] += g_norm
        
        # KEY FIX: Reset spike records to prevent memory leak
        if hasattr(pl_module.network, 'reset_spike_records'):
            pl_module.network.reset_spike_records()
        
        self.grad_count += 1
    
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = {'epoch': epoch}
        
        total_dead = 0
        total_neurons = 0
        
        # 1. Dead Neurons & Gradient Health
        for name, mask in self.activity_masks.items():
            num_neurons = mask.numel()
            num_dead = (~mask).sum().item()
            
            metrics[f'{name}_dead_ratio'] = num_dead / num_neurons
            total_dead += num_dead
            total_neurons += num_neurons
            
            # Gradient Analysis
            if name in self.grad_sum and self.grad_count > 0:
                mean_grad = self.grad_sum[name] / self.grad_count
                
                alive_grads = mean_grad[mask]
                dead_grads = mean_grad[~mask]
                
                if alive_grads.numel() > 0:
                    metrics[f'{name}_grad_alive_mean'] = alive_grads.mean().item()
                if dead_grads.numel() > 0:
                    metrics[f'{name}_grad_dead_mean'] = dead_grads.mean().item()
            
            # Weight Kurtosis
            weight, _ = self._get_weight_and_grad(pl_module, name)
            if weight is not None:
                w_flat = weight.data.flatten()
                w_std = w_flat.std()
                if w_std > 1e-9:
                    w_mean = w_flat.mean()
                    kurtosis = torch.mean(((w_flat - w_mean) / w_std) ** 4) - 3
                    metrics[f'{name}_w_kurtosis'] = kurtosis.item()

        if total_neurons > 0:
            metrics['global_dead_ratio'] = total_dead / total_neurons
            
        # 2. Revival Impact (Correlation with Loss)
        curr_val_loss = trainer.callback_metrics.get('val/loss', None)
        if isinstance(curr_val_loss, Tensor): curr_val_loss = curr_val_loss.item()
        
        if self.prev_total_dead is not None and self.prev_val_loss is not None and curr_val_loss is not None:
            revived_count = self.prev_total_dead - total_dead
            loss_delta = curr_val_loss - self.prev_val_loss
            
            if revived_count > 5:
                metrics['revival_event_count'] = revived_count
                metrics['revival_loss_impact'] = loss_delta 
        
        self.prev_total_dead = total_dead
        self.prev_val_loss = curr_val_loss

        # Validation Accuracy
        if 'val/accuracy' in trainer.callback_metrics:
            metrics['val_accuracy'] = trainer.callback_metrics['val/accuracy'].item()
            
        self.metrics_history.append(metrics)
        self.save_metrics()
        
    def save_metrics(self):
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.save_dir / 'neuron_metrics.csv', index=False)
        
    def get_summary(self) -> Dict:
        if not self.metrics_history: return {}
        df = pd.DataFrame(self.metrics_history)
        return {
            'final_dead_ratio': df['global_dead_ratio'].iloc[-1] if 'global_dead_ratio' in df else 0,
            'best_accuracy': df['val_accuracy'].max() if 'val_accuracy' in df else 0
        }


# =============================================================================
# Dead Neuron Experiment Lightning Module
# =============================================================================

class DeadNeuronExperiment(pl.LightningModule):
    """
    LightningModule for Dead Neuron Experiment with scalable depth.
    
    Uses CareResNet from models.components.neuron (single source of truth).
    Supports SEW and MS block types via block_type argument.
    """
    
    def __init__(
        self,
        depth: int = 18,
        in_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 64,
        num_steps: int = 25,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_plasticity: bool = True,
        init_method: str = "sabotage",
        init_std: float = 0.01,
        eta_stdp: float = 0.005,
        target_rate: float = 0.02,
        block_type: str = 'sew',
        input_size: int = 32,
        network: Optional[nn.Module] = None,
        homeo_target: str = 'gamma',
        snr_enabled: bool = True,
        snr_threshold: float = 2.0,
        snr_steepness: float = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['network'])
        
        self.depth = depth
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_plasticity = use_plasticity
        self.eta_stdp = eta_stdp
        self.target_rate = target_rate
        
        # Use provided network or build a new one
        if network is not None:
            self.network = network
        else:
            self.network = CareResNet(
                depth=depth,
                in_channels=in_channels,
                num_classes=num_classes,
                base_channels=base_channels,
                num_steps=num_steps,
                beta=beta,
                threshold=threshold,
                slope=slope,
                block_type=block_type,
                input_size=input_size,
                homeo_target=homeo_target,
                snr_enabled=snr_enabled,
                snr_threshold=snr_threshold,
                snr_steepness=snr_steepness,
            )
        
        # Apply initialization
        if init_method == "sabotage":
            self.apply(lambda m: sabotage_init(m, std=init_std))
            print(f"[SABOTAGE INIT] Applied with std={init_std} to {block_type.upper()}-ResNet{depth} (base_ch={base_channels})")
        else:
            self.apply(normal_init)
            print(f"[NORMAL INIT] Applied to {block_type.upper()}-ResNet{depth} (base_ch={base_channels})")
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        """Get spike records from network."""
        return self.network.get_spike_records()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        return self.network(x)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        inputs, targets = batch
        
        spike_counts, mem_out = self(inputs)
        # Use final membrane potential as logits (continuous, well-scaled for softmax)
        loss = F.cross_entropy(mem_out, targets)
        
        preds = mem_out.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('depth', float(self.depth), on_step=False, on_epoch=True)
        
        return loss
    
    def on_after_backward(self) -> None:
        """Apply plasticity updates after backward pass if enabled."""
        if self.use_plasticity:
            self.network.apply_homeostatic_updates(
                target_rate=self.target_rate,
                learning_rate=self.eta_stdp,
            )
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Validation step."""
        inputs, targets = batch
        
        spike_counts, mem_out = self(inputs)
        # Use final membrane potential as logits
        loss = F.cross_entropy(mem_out, targets)
        preds = mem_out.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', acc, on_epoch=True, prog_bar=True)
        
        # PREVENT OOM IN VALIDATION
        if hasattr(self.network, 'reset_spike_records'):
            self.network.reset_spike_records()
            
        return {'val_loss': loss, 'val_accuracy': acc}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 30,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }
