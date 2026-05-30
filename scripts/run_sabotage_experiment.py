"""
Immediate ResNet-18 Sabotage Experiment (Control vs CARE).
Running with init_std=0.01 to test "Sabotage" hypothesis.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import numpy as np

# Project imports
from models.components.neuron import CareResNet
from systems.experiment import DeadNeuronExperiment, PhdGradeNeuronTracker

# Configuration
CONFIG = {
    'seed': 42,
    'learning_rate': 1e-3,
    'batch_size': 64, # Optimized for speed
    'epochs': 15,     
    'time_steps': 8,  
    'beta': 0.95,
    'dataset': 'fashion_mnist',
    'depth': 18,
    'init_std': 0.01, 
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
        num_workers=4, # Enable workers
        pin_memory=True # Identify bottleneck
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=CONFIG['batch_size'], 
        num_workers=4,
        pin_memory=True 
    )
    
    return train_loader, test_loader

def run_experiment(block_type: str, use_plasticity: bool, save_dir: Path) -> dict:
    mode_label = 'CARE' if use_plasticity else 'Control'
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: ResNet-{CONFIG['depth']} ({block_type}) {mode_label} | Sabotage Init (std={CONFIG['init_std']})")
    print(f"{'='*60}")
    
    pl.seed_everything(CONFIG['seed'])
    train_loader, test_loader = get_dataloaders()
    
    torch.cuda.empty_cache() # Clear GPU memory before training
    
    # Create network
    network = CareResNet(
        depth=CONFIG['depth'],
        in_channels=1,
        num_classes=10,
        num_steps=CONFIG['time_steps'],
        beta=CONFIG['beta'],
        block_type=block_type,
    )
    
    # Create model with sabotage init
    model = DeadNeuronExperiment(
        network=network,
        num_classes=10,
        num_steps=CONFIG['time_steps'],
        learning_rate=CONFIG['learning_rate'],
        init_method='sabotage',
        init_std=CONFIG['init_std'],
        use_plasticity=use_plasticity,
    )
    
    # Callbacks
    monitor = PhdGradeNeuronTracker(save_dir)
    checkpoint = ModelCheckpoint(
        dirpath=str(save_dir / 'checkpoints'),
        filename='best-{epoch:02d}-{val/accuracy:.4f}',
        monitor='val/accuracy',
        mode='max',
        save_top_k=1,
    )
    
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[monitor, checkpoint],
        logger=CSVLogger(save_dir=str(save_dir), name='logs'),
        enable_progress_bar=True,
    )
    
    start_time = datetime.now()
    trainer.fit(model, train_loader, test_loader)
    duration = datetime.now() - start_time
    
    summary = monitor.get_summary()
    summary['duration'] = str(duration)
    return summary

def main():
    results_dir = Path('results/sabotage_bio') # Phase 9b: Bio-Plausible Rescue
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Run Control (SEW, no plasticity)
    summary_control = run_experiment('sew', use_plasticity=False, save_dir=results_dir / 'control')
    results['Control'] = summary_control
    
    # Run CARE (SEW, with plasticity) — same architecture, only plasticity differs
    summary_care = run_experiment('sew', use_plasticity=True, save_dir=results_dir / 'care')
    results['CARE'] = summary_care
    
    # print results
    print("\n" + "="*60)
    print("SABOTAGE RESULTS (init_std=0.01)")
    print(f"Control Dead Ratio: {summary_control['final_dead_ratio']:.2%}, Acc: {summary_control['best_accuracy']:.2%}")
    print(f"CARE Dead Ratio:    {summary_care['final_dead_ratio']:.2%}, Acc: {summary_care['best_accuracy']:.2%}")
    print("="*60)

if __name__ == '__main__':
    main()
