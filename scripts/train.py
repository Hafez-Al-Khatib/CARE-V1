"""
CARE Training Script

Entry point for training CARE networks with Hydra configuration.

Run:
    python scripts/train.py
    python scripts/train.py model.neuron.beta=0.95 training.learning_rate=5e-4
    python scripts/train.py --multirun training.learning_rate=1e-3,5e-4,1e-4
    python scripts/train.py data.dataset=nmnist  # Train on N-MNIST

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

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger

from systems.care_system import CARESystem, build_care_system

# NOTE: data.datamodule does not exist in this repo.
# For neuromorphic datasets, install tonic and implement a datamodule,
# or use torchvision datasets via run_flexible_experiment.py instead.
# from data.datamodule import NeuromorphicDataModule, build_datamodule


def create_dummy_dataloader(cfg: DictConfig):
    """
    Create dummy DataLoader for testing.
    
    Used when tonic is not available or for quick smoke tests.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Dummy data matching config dimensions
    batch_size = cfg.data.batch_size
    input_dim = cfg.model.architecture.input_dim
    output_dim = cfg.model.architecture.output_dim
    num_steps = cfg.model.architecture.num_steps
    
    # Random temporal data for smoke test
    n_samples = batch_size * 10
    # [B, T, F] format matching DataModule output
    x = torch.randn(n_samples, num_steps, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    
    dataset = TensorDataset(x, y)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=cfg.data.pin_memory,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set precision for torch.compile compatibility
    torch.set_float32_matmul_precision('medium')
    
    # Try to use real neuromorphic data, fall back to dummy
    use_real_data = True
    try:
        import tonic
        print(f"\n[INFO] Using real neuromorphic dataset: {cfg.data.dataset}")
    except ImportError:
        print("\n[WARNING] tonic not installed. Using dummy data.")
        print("Install with: pip install tonic")
        use_real_data = False
    
    if use_real_data:
        # Build DataModule
        datamodule = build_datamodule(cfg)
        datamodule.prepare_data()
        datamodule.setup("fit")
        
        # Update input_dim based on actual data
        actual_input_dim = datamodule.input_dim
        if actual_input_dim != cfg.model.architecture.input_dim:
            print(f"[INFO] Updating input_dim: {cfg.model.architecture.input_dim} -> {actual_input_dim}")
            cfg.model.architecture.input_dim = actual_input_dim
        
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
    else:
        # Fall back to dummy data
        train_loader = create_dummy_dataloader(cfg)
        val_loader = create_dummy_dataloader(cfg)
    
    # Build model (after potential input_dim update)
    model = build_care_system(cfg)
    
    # Optional: torch.compile for PyTorch 2.0+ (Linux only - Triton not available on Windows)
    import platform
    if (hasattr(torch, 'compile') and 
        torch.__version__ >= '2.0' and 
        platform.system() != 'Windows'):
        print("Compiling model with torch.compile...")
        model.network = torch.compile(model.network, mode='reduce-overhead')
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            filename='care-{epoch:02d}-{val_accuracy:.4f}',
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Try to add RichProgressBar (may not be available)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    # Logger
    logger = None
    if cfg.wandb.entity is not None:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            log_model=cfg.wandb.log_model,
        )
        logger.watch(model, log='all', log_freq=cfg.wandb.log_freq)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test (optional)
    if trainer.is_global_zero:
        print("\nTraining complete!")
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
"""
CARE Training Script

Entry point for training CARE networks with Hydra configuration.

Run:
    python scripts/train.py
    python scripts/train.py model.neuron.beta=0.95 training.learning_rate=5e-4
    python scripts/train.py --multirun training.learning_rate=1e-3,5e-4,1e-4

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

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
        RichProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger

from systems.care_system import CARESystem, build_care_system


def create_dummy_dataloader(cfg: DictConfig):
    """
    Create dummy DataLoader for testing.
    
    Replace with actual neuromorphic data loading (tonic, etc.)
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Dummy data matching config dimensions
    batch_size = cfg.data.batch_size
    input_dim = cfg.model.architecture.input_dim
    output_dim = cfg.model.architecture.output_dim
    
    # Random data for smoke test
    n_samples = batch_size * 10
    x = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    
    dataset = TensorDataset(x, y)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to cfg.data.num_workers for real training
        pin_memory=cfg.data.pin_memory,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training entry point."""
    
    # Print config
    print(OmegaConf.to_yaml(cfg))
    
    # Set precision for torch.compile compatibility
    torch.set_float32_matmul_precision('medium')
    
    # Build model
    model = build_care_system(cfg)
    
    # Optional: torch.compile for PyTorch 2.0+
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0':
        print("Compiling model with torch.compile...")
        model.network = torch.compile(model.network, mode='reduce-overhead')
    
    # Create dataloaders (replace with real data)
    train_loader = create_dummy_dataloader(cfg)
    val_loader = create_dummy_dataloader(cfg)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val/accuracy',
            mode='max',
            save_top_k=3,
            filename='care-{epoch:02d}-{val_accuracy:.4f}',
        ),
        EarlyStopping(
            monitor='val/loss',
            patience=20,
            mode='min',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Try to add RichProgressBar (may not be available)
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    # Logger
    logger = None
    if cfg.wandb.entity is not None:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            log_model=cfg.wandb.log_model,
        )
        logger.watch(model, log='all', log_freq=cfg.wandb.log_freq)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='auto',
        devices='auto',
        precision='16-mixed' if torch.cuda.is_available() else 32,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=cfg.training.gradient_clip,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test (optional)
    if trainer.is_global_zero:
        print("\nTraining complete!")
        print(f"Best model: {trainer.checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
