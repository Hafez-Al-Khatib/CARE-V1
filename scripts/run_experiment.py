"""
Dead Neuron Experiment Runner

Entry point for running the dead neuron experiment on Fashion-MNIST.

Usage:
    # Experimental Group (Hybrid with plasticity)
    py -3.12 scripts/run_experiment.py
    
    # Control Group (Backprop only)
    py -3.12 scripts/run_experiment.py use_plasticity=False
    
    # Quick smoke test
    py -3.12 scripts/run_experiment.py training.fast_dev_run=True training.epochs=1

Author: CARE Research Team
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from pytorch_lightning.loggers import WandbLogger, CSVLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from lightning.pytorch.loggers import WandbLogger, CSVLogger

from systems.experiment import DeadNeuronExperiment, DeadNeuronCallback, build_experiment


def get_fashion_mnist_loaders(cfg: DictConfig):
    """
    Get Fashion-MNIST train/val DataLoaders.
    
    Args:
        cfg: Hydra config with data section
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Standard normalization for Fashion-MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST stats
    ])
    
    # Download and load datasets
    train_dataset = datasets.FashionMNIST(
        root=cfg.data.root,
        train=True,
        download=True,
        transform=transform,
    )
    
    val_dataset = datasets.FashionMNIST(
        root=cfg.data.root,
        train=False,
        download=True,
        transform=transform,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.num_workers > 0,
    )
    
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    """Main experiment entry point."""
    
    # Print configuration
    print("=" * 60)
    print("DEAD NEURON EXPERIMENT")
    print("=" * 60)
    print(f"Mode: {'HYBRID (Plasticity ON)' if cfg.use_plasticity else 'CONTROL (Backprop Only)'}")
    print(f"Initialization: {cfg.initialization.method} (std={cfg.initialization.std})")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    # Set precision for torch.compile compatibility
    torch.set_float32_matmul_precision('medium')
    
    # Get data loaders
    print("\n[INFO] Loading Fashion-MNIST dataset...")
    train_loader, val_loader = get_fashion_mnist_loaders(cfg)
    print(f"[INFO] Train samples: {len(train_loader.dataset)}")
    print(f"[INFO] Val samples: {len(val_loader.dataset)}")
    
    # Build model
    print("\n[INFO] Building model...")
    model = build_experiment(cfg)
    
    # Callbacks
    callbacks = [
        # THE KEY CALLBACK - tracks dead neuron ratio
        DeadNeuronCallback(log_per_layer=True),
        
        # Standard callbacks
        ModelCheckpoint(
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            filename='experiment-{epoch:02d}-{val_accuracy:.4f}',
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=10,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Logger - Always use CSV for offline analysis
    mode_name = 'hybrid' if cfg.use_plasticity else 'control'
    depth = cfg.model.architecture.depth
    run_name = f"depth{depth}_{mode_name}"
    
    csv_logger = CSVLogger(
        save_dir="logs",
        name=run_name,
        version=None,  # Auto-increment
    )
    
    loggers = [csv_logger]
    
    # Optionally add WandB
    if cfg.wandb.entity is not None:
        tags = list(cfg.wandb.tags) if cfg.wandb.tags else []
        tags.append(mode_name)
        tags.append(f'depth_{depth}')
        tags.append(f'init_{cfg.initialization.method}')
        
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=run_name,
            log_model=cfg.wandb.log_model,
            tags=tags,
        )
        wandb_logger.watch(model, log='all', log_freq=cfg.wandb.log_freq)
        loggers.append(wandb_logger)
    else:
        print(f"\n[INFO] Logging to CSV: logs/{run_name}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=cfg.training.gradient_clip,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
        fast_dev_run=cfg.training.fast_dev_run,
    )
    
    # Train!
    print("\n[INFO] Starting training...")
    print(f"[INFO] Plasticity: {'ENABLED' if cfg.use_plasticity else 'DISABLED'}")
    trainer.fit(model, train_loader, val_loader)
    
    # Results
    if trainer.is_global_zero and not cfg.training.fast_dev_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Mode: {'HYBRID' if cfg.use_plasticity else 'CONTROL'}")
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
