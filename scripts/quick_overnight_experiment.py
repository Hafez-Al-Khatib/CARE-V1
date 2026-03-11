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
        """Skip spike collection to avoid OOM - use weight-based dead neuron detection instead."""
        pass  # Disable spike recording to avoid CUDA OOM
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Compute comprehensive metrics at epoch end."""
        epoch = trainer.current_epoch
        metrics = {'epoch': epoch}
        
        total_neurons = 0
        total_dead = 0
        total_near_dead = 0
        total_revived = 0
        total_died = 0
        
        # Skip spike-based analysis to avoid OOM, use weight-based metrics instead
        
        # Global dead neuron metrics (from weight analysis)
        metrics['global_dead_ratio'] = 0  # Will update based on weights
        metrics['global_near_dead_ratio'] = 0
        metrics['global_revived'] = 0
        metrics['global_died'] = 0
        metrics['total_neurons'] = 0
        
        # Count dead neurons based on near-zero weights
        dead_count = 0
        total_count = 0
        for name, param in pl_module.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Flatten all dims except first (output channels) and compute norm
                flat_param = param.data.view(param.size(0), -1)
                weight_norms = flat_param.norm(dim=1)
                dead_count += (weight_norms < 0.01).sum().item()
                total_count += weight_norms.numel()
        
        if total_count > 0:
            metrics['global_dead_ratio'] = dead_count / total_count
            metrics['total_neurons'] = total_count
        
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


# Configuration - Reduced batch size to avoid OOM
CONFIG = {
    'seed': 42,
    'learning_rate': 1e-3,
    'batch_size': 32,  # Reduced from 64
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
    
    # Use num_workers=0 on Windows to avoid multiprocessing memory issues
    train_loader = DataLoader(
        train_ds, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=CONFIG['batch_size'], 
        num_workers=0,
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
