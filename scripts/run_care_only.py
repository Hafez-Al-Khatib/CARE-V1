"""
Rapid CARE Sabotage Check.
Runs ResNet-18 (CARE/MS) with Sabotage Init (std=0.01) for 1 epoch (200 batches).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
from datetime import datetime

# Project imports
from models.components.neuron import CareResNet
from systems.experiment import DeadNeuronExperiment, PhdGradeNeuronTracker

# Configuration
CONFIG = {
    'seed': 42,
    'learning_rate': 1e-3,
    'batch_size': 32,
    'epochs': 1,
    'time_steps': 25,
    'beta': 0.95,
    'dataset': 'fashion_mnist',
    'depth': 18,
    'init_std': 0.01,  # SABOTAGE
    'limit_batches': 200,
}

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_ds = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], num_workers=0)
    
    return train_loader, test_loader

def main():
    print(f"\n{'='*60}")
    print(f"RAPID CARE CHECK: ResNet-{CONFIG['depth']} (MS) with Sabotage Init (std={CONFIG['init_std']})")
    print(f"{'='*60}")
    
    pl.seed_everything(CONFIG['seed'])
    train_loader, test_loader = get_dataloaders()
    
    network = CareResNet(
        depth=CONFIG['depth'],
        in_channels=1,
        num_classes=10,
        num_steps=CONFIG['time_steps'],
        beta=CONFIG['beta'],
        block_type='ms',  # CARE
    )
    
    model = DeadNeuronExperiment(
        network=network,
        num_classes=10,
        num_steps=CONFIG['time_steps'],
        learning_rate=CONFIG['learning_rate'],
        init_method='sabotage',
        init_std=CONFIG['init_std']
    )
    
    save_dir = Path('results/sabotage_comparison/care_rapid')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    monitor = PhdGradeNeuronTracker(save_dir)
    logger = CSVLogger(save_dir=str(save_dir), name='logs')
    
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[monitor],
        logger=logger,
        enable_progress_bar=True,
        limit_train_batches=CONFIG['limit_batches'],
        limit_val_batches=50,
    )
    
    trainer.fit(model, train_loader, test_loader)
    
    summary = monitor.get_summary()
    print("\n" + "="*60)
    print("CARE RAPID RESULTS")
    print(f"Final Dead Ratio: {summary['final_dead_ratio']:.2%}")
    print(f"Best Accuracy:    {summary['best_accuracy']:.2%}")
    print("="*60)

if __name__ == '__main__':
    main()
