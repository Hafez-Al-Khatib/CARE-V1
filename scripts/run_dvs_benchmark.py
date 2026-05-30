"""
Neuromorphic Dataset Benchmark Script.
Runs CARE experiments on DVS datasets: N-MNIST, DVS128 Gesture, CIFAR10-DVS.
Uses Tonic library for native event-based data loading.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path BEFORE any local imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

try:
    import tonic
    import tonic.transforms as tonic_transforms
    from tonic import DiskCachedDataset
except ImportError:
    print("ERROR: Tonic library not installed. Run: pip install tonic")
    sys.exit(1)

from torch.utils.data import DataLoader
from systems.modern_experiment import ModernArchExperiment
from systems.experiment import DeadNeuronCallback

# ============================================================
# DVS EXPERIMENT WRAPPER
# ============================================================

class DVSExperiment(ModernArchExperiment):
    """
    Wrapper for ModernArchExperiment that handles 5D DVS temporal data.
    DVS data comes as [B, T, C, H, W] where T is pre-binned time frames.
    This wrapper iterates through time bins and accumulates spike counts.
    """
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass for DVS data with shape [B, T, C, H, W].
        Iterates through time bins and accumulates output spikes.
        """
        # x shape: [B, T, C, H, W]
        # Convert to float if needed (tonic returns int16)
        x = x.float()
        
        batch_size, num_time_bins, channels, height, width = x.shape
        
        # Initialize spike accumulator
        device = x.device
        accumulated_spikes = None
        
        # Process each time bin through the network
        for t in range(num_time_bins):
            # Get frame at time t: [B, C, H, W]
            frame = x[:, t, :, :, :]
            
            # Pass through network (returns spike_counts, membrane_potentials)
            spike_counts, _ = self.network(frame)
            
            if accumulated_spikes is None:
                accumulated_spikes = spike_counts
            else:
                accumulated_spikes = accumulated_spikes + spike_counts
                
        return accumulated_spikes, None
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        
        # Forward through DVS wrapper
        spike_counts, _ = self(inputs)
        
        # Compute loss (spike_counts are already accumulated)
        spike_rates = spike_counts / (inputs.shape[1] * self.num_steps)  # T bins * internal steps
        loss = torch.nn.functional.cross_entropy(spike_rates, targets)
        
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        
        spike_counts, _ = self(inputs)
        spike_rates = spike_counts / (inputs.shape[1] * self.num_steps)
        loss = torch.nn.functional.cross_entropy(spike_rates, targets)
        
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

# ============================================================
# DATASET CONFIGURATIONS
# ============================================================

DATASET_CONFIGS = {
    'nmnist': {
        'class': tonic.datasets.NMNIST,
        'sensor_size': tonic.datasets.NMNIST.sensor_size,
        'num_classes': 10,
        'n_time_bins': 25,
        'description': 'Neuromorphic MNIST (event-based handwritten digits)',
    },
    'dvs_gesture': {
        'class': tonic.datasets.DVSGesture,
        'sensor_size': tonic.datasets.DVSGesture.sensor_size,
        'num_classes': 11,
        'n_time_bins': 25,
        'description': 'DVS128 Gesture (11 hand gestures)',
    },
    'cifar10_dvs': {
        'class': tonic.datasets.CIFAR10DVS,
        'sensor_size': tonic.datasets.CIFAR10DVS.sensor_size,
        'num_classes': 10,
        'n_time_bins': 25,
        'description': 'CIFAR10-DVS (event-stream object recognition)',
    },
}

# ============================================================
# DATA LOADING
# ============================================================

def get_dvs_loaders(dataset_name: str, batch_size: int = 64, data_root: str = './data'):
    """Get DataLoaders for a DVS dataset using Tonic."""
    
    config = DATASET_CONFIGS[dataset_name]
    sensor_size = config['sensor_size']
    n_time_bins = config['n_time_bins']
    
    # Transform: Denoise -> Convert to frames with time bins
    transform = tonic_transforms.Compose([
        tonic_transforms.Denoise(filter_time=10000),
        tonic_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins),
    ])
    
    # Load datasets
    DatasetClass = config['class']
    
    # Handle train/test split differences between datasets
    if dataset_name == 'cifar10_dvs':
        # CIFAR10-DVS doesn't have a train/test split, we manually split
        full_dataset = DatasetClass(save_to=data_root, transform=transform)
        train_size = int(0.9 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        train_dataset = DatasetClass(save_to=data_root, train=True, transform=transform)
        test_dataset = DatasetClass(save_to=data_root, train=False, transform=transform)
    
    # Cache to disk for faster loading
    cache_path = Path(data_root) / f'{dataset_name}_cache'
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Note: DiskCachedDataset requires picklable transforms
    # We'll use it if caching is needed for large datasets
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    
    return train_loader, test_loader


def collate_fn(batch):
    """Custom collate to handle variable-length event sequences."""
    # Tonic ToFrame already produces fixed-size tensors
    # batch is list of (frames, label) tuples
    # frames shape: (T, C, H, W) where T=n_time_bins, C=2 (polarity)
    
    frames = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    return frames, labels


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiment(
    dataset_name: str,
    plasticity: bool,
    arch_type: str = 'resnet',
    depth: int = 18,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    data_root: str = './data',
):
    """Run a single DVS benchmark experiment."""
    
    config = DATASET_CONFIGS[dataset_name]
    mode = 'hybrid' if plasticity else 'control'
    exp_name = f'{dataset_name}_{arch_type}{depth}_{mode}'
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Dataset: {config['description']}")
    print(f"Plasticity: {'ON (CARE)' if plasticity else 'OFF (baseline)'}")
    print(f"{'='*60}\n")
    
    # Get data loaders
    print(f"Loading {dataset_name} dataset...")
    train_loader, test_loader = get_dvs_loaders(
        dataset_name, 
        batch_size=batch_size,
        data_root=data_root
    )
    
    # Determine input size from first batch
    sample_batch = next(iter(train_loader))
    frames, _ = sample_batch
    # frames shape: (B, T, C, H, W)
    print(f"Input shape: {frames.shape}")
    
    # For DVS data: C=2 (ON/OFF polarity), we treat T as num_steps
    in_channels = frames.shape[2]  # 2 for polarity
    num_time_bins = frames.shape[1]    # n_time_bins from ToFrame transform
    
    # Create model with DVS wrapper that handles 5D temporal data
    model = DVSExperiment(
        arch_type=arch_type,
        depth=depth,
        in_channels=in_channels,
        num_classes=config['num_classes'],
        num_steps=1,  # Internal network steps per frame (accumulate across T bins in DVSExperiment)
        beta=0.9,
        threshold=1.0,
        slope=25.0,
        learning_rate=lr,
        weight_decay=1e-4,
        use_plasticity=plasticity,
        target_rate=0.1,
    )
    
    # Callbacks
    callbacks = [
        DeadNeuronCallback(log_per_layer=True),
        ModelCheckpoint(
            monitor='val/accuracy',
            mode='max',
            save_top_k=1,
            filename=f'{exp_name}-{{epoch:02d}}-{{val_accuracy:.4f}}',
        ),
        EarlyStopping(
            monitor='val/accuracy',
            patience=10,
            mode='max',
        ),
    ]
    
    # Logger
    logger = CSVLogger('logs', name=exp_name)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )
    
    # Train
    trainer.fit(model, train_loader, test_loader)
    
    # Report results
    best_acc = trainer.checkpoint_callback.best_model_score
    print(f"\n{'='*60}")
    print(f"COMPLETED: {exp_name}")
    print(f"Best Accuracy: {best_acc:.4f}" if best_acc else "No best accuracy recorded")
    print(f"{'='*60}\n")
    
    return {
        'experiment': exp_name,
        'dataset': dataset_name,
        'plasticity': plasticity,
        'best_acc': float(best_acc) if best_acc else None,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DVS Neuromorphic Benchmark')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['nmnist', 'dvs_gesture', 'cifar10_dvs', 'all'],
                        help='Dataset to benchmark')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['control', 'hybrid', 'both'],
                        help='Experiment mode')
    parser.add_argument('--arch', type=str, default='resnet',
                        choices=['resnet', 'vgg'],
                        help='Architecture type')
    parser.add_argument('--depth', type=int, default=18,
                        help='Network depth')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data directory')
    
    args = parser.parse_args()
    
    # Determine datasets to run
    if args.dataset == 'all':
        datasets = ['nmnist', 'dvs_gesture', 'cifar10_dvs']
    else:
        datasets = [args.dataset]
    
    # Determine modes to run
    if args.mode == 'both':
        modes = [False, True]  # Control first, then Hybrid
    elif args.mode == 'control':
        modes = [False]
    else:
        modes = [True]
    
    # Run experiments
    results = []
    
    for dataset_name in datasets:
        for plasticity in modes:
            try:
                result = run_experiment(
                    dataset_name=dataset_name,
                    plasticity=plasticity,
                    arch_type=args.arch,
                    depth=args.depth,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    data_root=args.data_root,
                )
                results.append(result)
            except Exception as e:
                print(f"ERROR in {dataset_name} ({'hybrid' if plasticity else 'control'}): {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    for r in results:
        mode = 'CARE' if r['plasticity'] else 'Control'
        acc = f"{r['best_acc']*100:.2f}%" if r['best_acc'] else 'N/A'
        print(f"{r['dataset']:15} | {mode:8} | Accuracy: {acc}")
    print("="*60)


if __name__ == '__main__':
    main()
