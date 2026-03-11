"""
Flexible Experiment Runner for CARE SNN Research.

Supports parameterization via CLI arguments for dataset, depth,
initialization method, block type, and training configuration.

Features:
  - Configurable base_channels, seed, learning rate
  - Dataset support: fashion_mnist, cifar10, tiny_imagenet, imagenet
  - File-based logging: all stdout/stderr written to run.log
"""

import sys
import os
import logging
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

# Project imports — single source of truth
from systems.experiment import DeadNeuronExperiment, PhdGradeNeuronTracker


def setup_logging(save_dir: Path):
    """Set up file + console logging."""
    log_file = save_dir / 'run.log'
    
    # Create a formatter
    fmt = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(str(log_file), mode='w')
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)
    
    return logging.getLogger('CARE')


def parse_args():
    parser = argparse.ArgumentParser(description='Flexible Experiment Runner')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        choices=['fashion_mnist', 'cifar10', 'tiny_imagenet', 'imagenet'])
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--init', type=str, default='sabotage', choices=['normal', 'sabotage'])
    parser.add_argument('--block', type=str, default='sew', choices=['sew', 'ms'])
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--time_steps', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--no_plasticity', action='store_true', help='Disable homeostatic plasticity')
    parser.add_argument('--eta_stdp', type=float, default=0.001, help='Learning rate for plasticity')
    parser.add_argument('--output_dir', type=str, default='results/final_v2', help='Base output directory')
    parser.add_argument('--base_channels', type=int, default=64, help='Base channel width (use 32 for CIFAR)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--data_dir', type=str, default='data', help='Root directory for datasets')
    return parser.parse_args()


def get_dataloaders(dataset_name, batch_size, num_workers=2, data_dir='data'):
    """Get Dataloaders for the specified dataset."""
    data_path = Path(data_dir)
    
    if dataset_name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = datasets.FashionMNIST(str(data_path), train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(str(data_path), train=False, download=True, transform=transform)
        in_channels = 1
        num_classes = 10
        
    elif dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(str(data_path), train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(str(data_path), train=False, download=True, transform=transform_test)
        in_channels = 3
        num_classes = 10
    
    elif dataset_name == 'tiny_imagenet':
        # Tiny-ImageNet: 64x64, 200 classes
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
        tiny_path = data_path / 'tiny-imagenet-200'
        train_ds = datasets.ImageFolder(str(tiny_path / 'train'), transform=transform_train)
        test_ds = datasets.ImageFolder(str(tiny_path / 'val'), transform=transform_test)
        in_channels = 3
        num_classes = 200
    
    elif dataset_name == 'imagenet':
        # Full ImageNet: 224x224, 1000 classes
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        imagenet_path = data_path / 'imagenet'
        train_ds = datasets.ImageFolder(str(imagenet_path / 'train'), transform=transform_train)
        test_ds = datasets.ImageFolder(str(imagenet_path / 'val'), transform=transform_test)
        in_channels = 3
        num_classes = 1000
        
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    return train_loader, test_loader, in_channels, num_classes


def run_experiment(args, save_dir: Path) -> dict:
    logger = setup_logging(save_dir)
    
    logger.info(f"{'='*60}")
    logger.info(f"EXPERIMENT: {args.dataset} ResNet-{args.depth} ({args.block}) "
                f"Init={args.init} Plasticity={not args.no_plasticity} "
                f"base_ch={args.base_channels} seed={args.seed}")
    logger.info(f"{'='*60}")
    
    pl.seed_everything(args.seed)
    train_loader, test_loader, in_channels, num_classes = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers, args.data_dir
    )
    
    logger.info(f"Dataset={args.dataset}, InChannels={in_channels}, NumClasses={num_classes}, "
                f"TrainSamples={len(train_loader.dataset)}, TestSamples={len(test_loader.dataset)}")
    
    torch.cuda.empty_cache() 
    
    # Init Std 
    init_std = 0.01 if args.init == 'sabotage' else 0.05
    
    model = DeadNeuronExperiment(
        depth=args.depth,
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=args.base_channels,
        num_steps=args.time_steps,
        beta=0.95,
        learning_rate=args.lr,
        init_method=args.init,
        init_std=init_std,
        block_type=args.block,
        use_plasticity=not args.no_plasticity,
        eta_stdp=args.eta_stdp
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Params: {total_params:,} total, {trainable_params:,} trainable")
    
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
        max_epochs=args.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[monitor, checkpoint],
        logger=CSVLogger(save_dir=str(save_dir), name='logs'),
        enable_progress_bar=True,
    )
    
    logger.info("Starting training...")
    trainer.fit(model, train_loader, test_loader)
    
    summary = monitor.get_summary() if hasattr(monitor, 'get_summary') else {}
    logger.info(f"Training complete. Summary: {summary}")
    
    return summary


def main():
    args = parse_args()
    results_dir = Path(args.output_dir) / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    run_experiment(args, results_dir)


if __name__ == "__main__":
    main()
