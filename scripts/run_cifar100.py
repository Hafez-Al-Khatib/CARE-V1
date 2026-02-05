"""
CIFAR-100 Experiment Runner for CARE

Tests the CARE architecture on the more challenging CIFAR-100 dataset.
Uses ResNet-18 as the default backbone.

Usage:
    # ResNet-18 Control
    py -3.12 scripts/run_cifar100.py arch_type=resnet depth=18 use_plasticity=False

    # ResNet-18 Hybrid (CARE)
    py -3.12 scripts/run_cifar100.py arch_type=resnet depth=18 use_plasticity=True
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hydra
from omegaconf import DictConfig

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
    from pytorch_lightning.loggers import CSVLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    from lightning.pytorch.loggers import CSVLogger

from systems.experiment import DeadNeuronCallback
from systems.modern_experiment import ModernArchExperiment


def get_cifar100_loaders(cfg: DictConfig):
    """Get CIFAR-100 DataLoaders with appropriate normalization and augmentation."""
    # CIFAR-100 Mean and Std
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Ensure root exists
    data_root = Path(cfg.data.root)
    
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    
    val_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=val_transform,
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


@hydra.main(version_base=None, config_path="../configs", config_name="modern_experiment")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    
    # Overrides for CIFAR-100
    arch_type = cfg.arch_type  # Should be 'resnet' usually for this
    depth = cfg.depth
    mode_name = 'hybrid' if cfg.use_plasticity else 'control'
    
    dataset_name = "cifar100"
    
    print("=" * 60)
    print(f"CIFAR-100 EXPERIMENT START")
    print("=" * 60)
    print(f"Architecture: {arch_type.upper()}-{depth}")
    print(f"Dataset: CIFAR-100 (3 channels, 100 classes)")
    print(f"Mode: {'HYBRID (CARE)' if cfg.use_plasticity else 'CONTROL (Backprop)'}")
    print("=" * 60)
    
    torch.set_float32_matmul_precision('medium')
    
    train_loader, val_loader = get_cifar100_loaders(cfg)
    print(f"[INFO] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Build model with CIFAR-100 specs
    model = ModernArchExperiment(
        arch_type=arch_type,
        depth=depth,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        in_channels=3,      # RGB
        num_classes=100,    # CIFAR-100
        num_steps=cfg.num_steps,
        beta=cfg.beta,
        threshold=cfg.threshold,
        slope=cfg.slope,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        use_plasticity=cfg.use_plasticity,
        init_method=cfg.init_method,
        init_std=cfg.init_std,
        eta_stdp=cfg.eta_stdp,
        target_rate=cfg.target_rate,
    )
    
    # Callbacks
    callbacks = [
        DeadNeuronCallback(log_per_layer=True),
        ModelCheckpoint(
            monitor='val/accuracy',
            mode='max',
            save_top_k=1,
            filename=f'{dataset_name}_{arch_type}{depth}_{mode_name}-{{epoch:02d}}-{{val_accuracy:.4f}}',
        ),
        EarlyStopping(monitor='val/loss', patience=15, mode='min'), # Increased patience for harder task
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Logger
    # e.g. cifar100_resnet18_control
    run_name = f"{dataset_name}_{arch_type}{depth}_{mode_name}"
    csv_logger = CSVLogger(save_dir="logs", name=run_name, version=None)
    
    print(f"\n[INFO] Logging to CSV: logs/{run_name}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs, # Should probably increase for CIFAR-100, can override in CLI
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=csv_logger,
        gradient_clip_val=cfg.gradient_clip,
        log_every_n_steps=10,
        enable_progress_bar=True,
        fast_dev_run=cfg.fast_dev_run,
    )
    
    print(f"\n[INFO] Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    if trainer.is_global_zero and not cfg.fast_dev_run:
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"Run: {run_name}")
        if trainer.checkpoint_callback.best_model_path:
            print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
            print(f"Best Acc: {trainer.checkpoint_callback.best_model_score}")
        print("=" * 60)


if __name__ == "__main__":
    main()
