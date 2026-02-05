"""
Quick CARE vs Control comparison for progress meeting.
Runs ResNet-18 on Fashion-MNIST with SEW (Control) and MS (CARE) blocks.
Includes comprehensive monitoring of dead neurons, revivals, and gradients.

Expected runtime: ~2-4 hours on GPU
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any

# Your project imports
from models.components.neuron import CareResNet
from systems.experiment import DeadNeuronExperiment


class ComprehensiveNeuronMonitor(Callback):
    """
    Enhanced callback that tracks:
    - Dead neuron ratios per layer
    - Revival events (dead -> active)
    - Death events (active -> dead)
    - Weight statistics
    - Gradient statistics
    - Firing rates
    """
    
    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.prev_dead_masks = {}  # Track previous dead state for revival detection
        self.epoch_spikes = defaultdict(list)
        
    def on_train_epoch_start(self, trainer, pl_module):
        """Reset spike tracking at epoch start."""
        self.epoch_spikes.clear()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collect spike data during training."""
        # Get spike records from network layers
        if hasattr(pl_module, 'network'):
            network = pl_module.network
            if hasattr(network, 'get_spike_records'):
                records = network.get_spike_records()
                for name, spikes in records.items():
                    if isinstance(spikes, torch.Tensor):
                        # Store firing rate per neuron
                        firing_rate = spikes.float().mean(dim=0)  # Average over batch
                        self.epoch_spikes[name].append(firing_rate.detach().cpu())
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Compute comprehensive metrics at epoch end."""
        epoch = trainer.current_epoch
        metrics = {'epoch': epoch}
        
        total_neurons = 0
        total_dead = 0
        total_near_dead = 0
        total_revived = 0
        total_died = 0
        
        # Analyze spike records
        for name, spike_list in self.epoch_spikes.items():
            if len(spike_list) == 0:
                continue
                
            # Average firing rate across all batches
            avg_firing = torch.stack(spike_list).mean(dim=0)
            while avg_firing.dim() > 1:
                avg_firing = avg_firing.mean(dim=-1)
            
            num_neurons = avg_firing.numel()
            dead_mask = (avg_firing == 0)
            near_dead_mask = (avg_firing < 0.01) & ~dead_mask
            
            num_dead = dead_mask.sum().item()
            num_near_dead = near_dead_mask.sum().item()
            
            # Track revivals and deaths
            if name in self.prev_dead_masks:
                prev_dead = self.prev_dead_masks[name]
                if prev_dead.shape == dead_mask.shape:
                    revived = (prev_dead & ~dead_mask).sum().item()
                    died = (~prev_dead & dead_mask).sum().item()
                else:
                    revived, died = 0, 0
            else:
                revived, died = 0, 0
            
            self.prev_dead_masks[name] = dead_mask.clone()
            
            # Store per-layer metrics
            metrics[f'{name}_dead_ratio'] = num_dead / num_neurons if num_neurons > 0 else 0
            metrics[f'{name}_near_dead_ratio'] = num_near_dead / num_neurons if num_neurons > 0 else 0
            metrics[f'{name}_firing_rate_mean'] = avg_firing.mean().item()
            metrics[f'{name}_firing_rate_std'] = avg_firing.std().item()
            metrics[f'{name}_revived'] = revived
            metrics[f'{name}_died'] = died
            
            total_neurons += num_neurons
            total_dead += num_dead
            total_near_dead += num_near_dead
            total_revived += revived
            total_died += died
        
        # Global dead neuron metrics
        metrics['global_dead_ratio'] = total_dead / total_neurons if total_neurons > 0 else 0
        metrics['global_near_dead_ratio'] = total_near_dead / total_neurons if total_neurons > 0 else 0
        metrics['global_revived'] = total_revived
        metrics['global_died'] = total_died
        metrics['total_neurons'] = total_neurons
        
        # Weight statistics
        weight_norms = []
        for name, param in pl_module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                short_name = name.replace('.', '_')[:25]
                metrics[f'{short_name}_w_mean'] = param.data.mean().item()
                metrics[f'{short_name}_w_std'] = param.data.std().item()
                metrics[f'{short_name}_w_norm'] = param.data.norm().item()
                weight_norms.append(param.data.norm().item())
                
                # Gradient stats if available
                if param.grad is not None:
                    metrics[f'{short_name}_grad_norm'] = param.grad.norm().item()
                    metrics[f'{short_name}_grad_mean'] = param.grad.mean().item()
        
        metrics['avg_weight_norm'] = np.mean(weight_norms) if weight_norms else 0
        
        # Training metrics from trainer
        for key in ['train/loss', 'train/accuracy', 'val/loss', 'val/accuracy']:
            if key in trainer.callback_metrics:
                val = trainer.callback_metrics[key]
                metrics[key.replace('/', '_')] = val.item() if hasattr(val, 'item') else val
        
        self.metrics_history.append(metrics)
        self._save_metrics()
        
        # Print summary
        print(f"\n[Epoch {epoch}] Dead: {metrics['global_dead_ratio']:.1%}, "
              f"Revived: {total_revived}, Died: {total_died}, "
              f"Val Acc: {metrics.get('val_accuracy', 0):.2%}")
    
    def _save_metrics(self):
        """Save metrics to CSV."""
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.save_dir / 'neuron_metrics.csv', index=False)
    
    def get_summary(self) -> Dict:
        """Get experiment summary."""
        if not self.metrics_history:
            return {}
        df = pd.DataFrame(self.metrics_history)
        return {
            'final_dead_ratio': df['global_dead_ratio'].iloc[-1],
            'min_dead_ratio': df['global_dead_ratio'].min(),
            'max_dead_ratio': df['global_dead_ratio'].max(),
            'total_revived': df['global_revived'].sum(),
            'total_died': df['global_died'].sum(),
            'best_val_accuracy': df['val_accuracy'].max() if 'val_accuracy' in df else 0,
            'final_val_accuracy': df['val_accuracy'].iloc[-1] if 'val_accuracy' in df else 0,
        }


# Configuration
CONFIG = {
    'seed': 42,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 20,
    'time_steps': 25,
    'beta': 0.95,
    'dataset': 'fashion_mnist',
    'depth': 18,
}


def get_dataloaders():
    """Get Fashion-MNIST dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_ds = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=CONFIG['batch_size'], 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader


def run_experiment(block_type: str, save_dir: Path) -> Dict:
    """Run single experiment with given block type."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: ResNet-{CONFIG['depth']} with {block_type.upper()} blocks")
    print(f"{'='*60}")
    
    pl.seed_everything(CONFIG['seed'])
    
    train_loader, test_loader = get_dataloaders()
    
    # Create model
    network = CareResNet(
        depth=CONFIG['depth'],
        in_channels=1,
        num_classes=10,
        num_steps=CONFIG['time_steps'],
        beta=CONFIG['beta'],
        block_type=block_type,
    )
    
    model = DeadNeuronExperiment(
        network=network,
        num_classes=10,
        num_steps=CONFIG['time_steps'],
        learning_rate=CONFIG['learning_rate'],
    )
    
    # Callbacks
    monitor = ComprehensiveNeuronMonitor(save_dir)
    checkpoint = ModelCheckpoint(
        dirpath=str(save_dir / 'checkpoints'),
        filename='best-{epoch:02d}-{val/accuracy:.4f}',
        monitor='val/accuracy',
        mode='max',
        save_top_k=1,
    )
    
    # Logger
    logger = CSVLogger(save_dir=str(save_dir), name='logs')
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[monitor, checkpoint],
        logger=logger,
        enable_progress_bar=True,
        deterministic=True,
    )
    
    # Train
    start_time = datetime.now()
    trainer.fit(model, train_loader, test_loader)
    duration = datetime.now() - start_time
    
    # Get summary
    summary = monitor.get_summary()
    summary['block_type'] = block_type
    summary['duration_seconds'] = duration.total_seconds()
    summary['config'] = CONFIG.copy()
    
    # Save summary
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{block_type.upper()} Results:")
    print(f"  Best Val Accuracy: {summary['best_val_accuracy']:.2%}")
    print(f"  Final Dead Ratio: {summary['final_dead_ratio']:.2%}")
    print(f"  Total Revived: {summary['total_revived']}")
    print(f"  Duration: {duration}")
    
    return summary


def main():
    print("="*60)
    print("OVERNIGHT CARE vs CONTROL COMPARISON")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    
    results_dir = Path('results/overnight_comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    results = {}
    
    # Run both experiments
    for block_type, name in [('sew', 'Control (SEW)'), ('ms', 'CARE (MS)')]:
        save_dir = results_dir / block_type
        save_dir.mkdir(exist_ok=True)
        
        summary = run_experiment(block_type, save_dir)
        results[name] = summary
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    comparison = []
    for name, summary in results.items():
        print(f"\n{name}:")
        print(f"  Best Accuracy:    {summary['best_val_accuracy']:.2%}")
        print(f"  Final Dead Ratio: {summary['final_dead_ratio']:.2%}")
        print(f"  Total Revived:    {summary['total_revived']}")
        print(f"  Total Died:       {summary['total_died']}")
        comparison.append({
            'Model': name,
            'Best Accuracy': f"{summary['best_val_accuracy']:.2%}",
            'Final Dead Ratio': f"{summary['final_dead_ratio']:.2%}",
            'Revived': summary['total_revived'],
            'Died': summary['total_died'],
        })
    
    # Save comparison
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(results_dir / 'comparison.csv', index=False)
    
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {results_dir}")
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*60)


if __name__ == '__main__':
    main()
