"""
Modern Architecture Experiment Runner

Run VGG-Spiking experiments.

Usage:
    # VGG-11 Control
    py -3.12 scripts/run_modern.py arch_type=vgg depth=11 use_plasticity=False
    
    # VGG-11 Hybrid
    py -3.12 scripts/run_modern.py arch_type=vgg depth=11 use_plasticity=True

Author: CARE Research Team
"""

import sys
from pathlib import Path

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


def get_fashion_mnist_loaders(cfg: DictConfig):
    """Get Fashion-MNIST DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    
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


@hydra.main(version_base=None, config_path="../configs", config_name="modern_experiment")
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    
    arch_type = cfg.arch_type
    depth = cfg.depth
    mode_name = 'hybrid' if cfg.use_plasticity else 'control'
    
    print("=" * 60)
    print(f"MODERN ARCHITECTURE EXPERIMENT")
    print("=" * 60)
    print(f"Architecture: {arch_type.upper()}")
    print(f"Depth/Blocks: {depth}")
    print(f"Mode: {'HYBRID (Plasticity ON)' if cfg.use_plasticity else 'CONTROL (Backprop Only)'}")
    print("=" * 60)
    
    torch.set_float32_matmul_precision('medium')
    
    train_loader, val_loader = get_fashion_mnist_loaders(cfg)
    print(f"[INFO] Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Build model
    model = ModernArchExperiment(
        arch_type=arch_type,
        depth=depth,
        in_channels=1,
        num_classes=10,
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
            save_top_k=1,  # Keep only best to save disk space
            filename=f'{arch_type}-{{epoch:02d}}-{{val_accuracy:.4f}}',
        ),
        EarlyStopping(monitor='val/loss', patience=10, mode='min'),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Logger
    run_name = f"{arch_type}{depth}_{mode_name}"
    csv_logger = CSVLogger(save_dir="logs", name=run_name, version=None)
    
    print(f"\n[INFO] Logging to CSV: logs/{run_name}")
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
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
        print(f"Architecture: {arch_type.upper()}-{depth}")
        print(f"Mode: {'HYBRID' if cfg.use_plasticity else 'CONTROL'}")
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
