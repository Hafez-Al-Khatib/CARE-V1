"""
VGG-Spiking and Spiking Attention Experiment Module

Modern architectures to prove STDP plasticity generalizes beyond ResNets.

Key insight: VGG has NO skip connections, making dead neurons catastrophic.
This is the strongest evidence that plasticity is essential.

Author: CARE Research Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback
except ImportError:
    import lightning as pl
    from lightning.pytorch.callbacks import Callback

try:
    import snntorch as snn
    from snntorch import surrogate
except ImportError:
    raise ImportError("snntorch is required. Install via: pip install snntorch")

from systems.experiment import (
    sabotage_init, normal_init, DeadNeuronCallback, CareLIFConv
)


# =============================================================================
# VGG Configurations (Modified for 28x28 Small Images)
# =============================================================================

# Original VGG has 5 pooling layers (reducing 224->7)
# For 28x28, we can only have 2 pooling layers (28->14->7)
# We keep the same depth but reduce pooling

VGG_CONFIGS = {
    # Small-image VGG variants (2 pooling layers only)
    8: [64, 64, 'M', 128, 128, 'M', 256, 256],  # 8 conv layers
    11: [64, 64, 'M', 128, 128, 256, 256, 'M', 512, 512, 512],  # 11 conv layers  
    13: [64, 64, 'M', 128, 128, 256, 256, 256, 'M', 512, 512, 512, 512],  # 13 conv
    16: [64, 64, 'M', 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],  # 16 conv
}


# =============================================================================
# VGG-Spiking Architecture (NO Skip Connections = Maximum Dead Neurons)
# =============================================================================

class VGGSpiking(nn.Module):
    """
    VGG-style Spiking Neural Network.
    
    NO SKIP CONNECTIONS = Worst case for dead neurons.
    This architecture should show catastrophic dead neuron ratios
    with standard backprop, making it the strongest proof that
    STDP plasticity is essential.
    
    Configurations:
        VGG-11: 8 conv layers
        VGG-13: 10 conv layers
        VGG-16: 13 conv layers
        VGG-19: 16 conv layers
    """
    
    def __init__(
        self,
        depth: int = 11,
        in_channels: int = 1,
        num_classes: int = 10,
        num_steps: int = 25,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        if depth not in VGG_CONFIGS:
            raise ValueError(f"Unsupported VGG depth {depth}. Choose from {list(VGG_CONFIGS.keys())}")
        
        self.depth = depth
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        
        # Build convolutional layers
        self.features, self.conv_layers, last_channels = self._make_layers(
            VGG_CONFIGS[depth], in_channels
        )
        
        # Adaptive pooling for different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier (dynamic based on config)
        self.classifier = nn.Linear(last_channels, num_classes)
        
        # Output LIF
        self.lif_out = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
    
    def _make_layers(
        self,
        config: List,
        in_channels: int,
    ) -> Tuple[nn.ModuleList, List[CareLIFConv], int]:
        """Build VGG layers from config."""
        layers = nn.ModuleList()
        conv_layers = []
        last_channels = in_channels
        
        for v in config:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = CareLIFConv(
                    in_channels, v,
                    kernel_size=3, stride=1, padding=1,
                    beta=self.beta, threshold=self.threshold, slope=self.slope
                )
                layers.append(conv)
                conv_layers.append(conv)
                in_channels = v
                last_channels = v
        
        return layers, conv_layers, last_channels
    
    def reset_spike_records(self) -> None:
        """Reset all spike records."""
        for conv in self.conv_layers:
            conv.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        """Get spike records for dead neuron analysis."""
        records = {}
        for i, conv in enumerate(self.conv_layers):
            record = conv.get_spike_record()
            if record.numel() > 0:
                records[f'conv{i}'] = record
        return records
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        """Apply homeostatic plasticity to all conv layers."""
        for conv in self.conv_layers:
            conv.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass over time steps."""
        batch_size = x.shape[0]
        device = x.device
        
        self.reset_spike_records()
        
        # Initialize membrane states for all conv layers
        mems = {}
        current_size = x.shape[2:]  # (H, W)
        
        for i, layer in enumerate(self.features):
            if isinstance(layer, CareLIFConv):
                h, w = current_size
                mems[i] = layer.init_state(batch_size, h, w, device)
            elif isinstance(layer, nn.MaxPool2d):
                current_size = (current_size[0] // 2, current_size[1] // 2)
        
        mem_out = torch.zeros(batch_size, self.classifier.out_features, device=device)
        spike_count = torch.zeros(batch_size, self.classifier.out_features, device=device)
        
        # Time loop
        for t in range(self.num_steps):
            out = x
            
            for i, layer in enumerate(self.features):
                if isinstance(layer, CareLIFConv):
                    out, mems[i] = layer(out, mems[i])
                else:
                    out = layer(out)
            
            # Classifier
            pooled = self.avgpool(out)
            flat = pooled.view(batch_size, -1)
            current = self.classifier(flat)
            spk_out, mem_out = self.lif_out(current, mem_out)
            
            spike_count += spk_out
        
        return spike_count, mem_out


# =============================================================================
# Spiking Attention Block
# =============================================================================

class SpikingSelfAttention(nn.Module):
    """
    Spike-driven self-attention mechanism.
    
    Inspired by Meta-SpikeFormer (ICLR 2024) and SpikedAttention (NeurIPS 2024).
    Uses spike-based query/key/value computations.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections with LIF
        self.qkv_conv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_lif = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        
        # Output projection
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_lif = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        
        self.dim = dim
        self._spike_record: List[Tensor] = []
        
        self.register_buffer('activity_trace', torch.zeros(dim))
        self.activity_decay = 0.99
    
    def reset_spike_record(self) -> None:
        self._spike_record = []
    
    def get_spike_record(self) -> Tensor:
        if self._spike_record:
            return torch.stack(self._spike_record, dim=0)
        return torch.tensor([])
    
    def apply_homeostatic_update(self, target_rate: float, learning_rate: float) -> None:
        """Apply homeostatic plasticity."""
        with torch.no_grad():
            deviation = target_rate - self.activity_trace
            boost = 1.0 + learning_rate * deviation.view(1, -1, 1, 1)
            self.qkv_conv.weight.mul_(boost.clamp(0.9, 1.1).expand_as(self.qkv_conv.weight[:self.dim]))
    
    def forward(
        self,
        x: Tensor,
        mem_qkv: Tensor,
        mem_proj: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Spike-driven attention.
        
        Args:
            x: Input [B, C, H, W]
            mem_qkv: QKV membrane
            mem_proj: Projection membrane
            
        Returns:
            (output, mem_qkv, mem_proj)
        """
        B, C, H, W = x.shape
        
        # QKV projection
        qkv = self.qkv_conv(x)
        qkv_spk, mem_qkv = self.qkv_lif(qkv, mem_qkv)
        
        # Split into Q, K, V
        qkv_spk = qkv_spk.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv_spk[:, 0], qkv_spk[:, 1], qkv_spk[:, 2]
        
        # Spike-based attention (simplified)
        # Instead of softmax, use spike counts as attention weights
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = torch.clamp(attn, 0, 1)  # Binary-like attention
        
        # Apply attention
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        
        # Output projection
        out = self.proj(out)
        out_spk, mem_proj = self.proj_lif(out, mem_proj)
        
        # Record spikes
        self._spike_record.append(out_spk.detach())
        
        # Update activity trace
        with torch.no_grad():
            channel_activity = out_spk.mean(dim=(0, 2, 3))
            self.activity_trace = (
                self.activity_decay * self.activity_trace +
                (1 - self.activity_decay) * channel_activity
            )
        
        return out_spk, mem_qkv, mem_proj


class SpikingAttentionBlock(nn.Module):
    """
    Full spiking attention block with residual.
    
    Structure: x -> Attention -> LIF + x
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        self.attention = SpikingSelfAttention(
            dim, num_heads, beta, threshold, slope
        )
        
        # MLP
        self.mlp_conv1 = nn.Conv2d(dim, dim * 4, 1, bias=False)
        self.mlp_lif1 = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        self.mlp_conv2 = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.mlp_lif2 = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        
        self.dim = dim
    
    def reset_spike_records(self) -> None:
        self.attention.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return {'attention': self.attention.get_spike_record()}
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.attention.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(
        self,
        x: Tensor,
        mem_qkv: Tensor,
        mem_proj: Tensor,
        mem_mlp1: Tensor,
        mem_mlp2: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward through attention block."""
        
        # Attention with residual
        attn_out, mem_qkv, mem_proj = self.attention(x, mem_qkv, mem_proj)
        x = x + attn_out
        
        # MLP with residual
        mlp = self.mlp_conv1(x)
        mlp, mem_mlp1 = self.mlp_lif1(mlp, mem_mlp1)
        mlp = self.mlp_conv2(mlp)
        mlp, mem_mlp2 = self.mlp_lif2(mlp, mem_mlp2)
        x = x + mlp
        
        return x, mem_qkv, mem_proj, mem_mlp1, mem_mlp2


# =============================================================================
# Spiking Attention Network
# =============================================================================

class SpikingAttentionNet(nn.Module):
    """
    Simple spiking attention network for Fashion-MNIST.
    
    Structure: Stem -> Attention Blocks -> Pool -> Classifier
    """
    
    def __init__(
        self,
        num_blocks: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        num_steps: int = 25,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.embed_dim = embed_dim
        
        # Stem
        self.stem = CareLIFConv(
            in_channels, embed_dim,
            kernel_size=7, stride=2, padding=3,
            beta=beta, threshold=threshold, slope=slope
        )
        
        # Attention blocks
        self.blocks = nn.ModuleList([
            SpikingAttentionBlock(
                embed_dim, num_heads, beta, threshold, slope
            ) for _ in range(num_blocks)
        ])
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.lif_out = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
    
    def reset_spike_records(self) -> None:
        self.stem.reset_spike_record()
        for block in self.blocks:
            block.reset_spike_records()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        records = {'stem': self.stem.get_spike_record()}
        for i, block in enumerate(self.blocks):
            block_records = block.get_spike_records()
            for k, v in block_records.items():
                records[f'block{i}_{k}'] = v
        return records
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.stem.apply_homeostatic_update(target_rate, learning_rate)
        for block in self.blocks:
            block.apply_homeostatic_updates(target_rate, learning_rate)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        batch_size = x.shape[0]
        device = x.device
        
        self.reset_spike_records()
        
        # Calculate sizes
        h, w = (x.shape[2] + 2*3 - 7) // 2 + 1, (x.shape[3] + 2*3 - 7) // 2 + 1
        
        # Initialize states
        stem_mem = self.stem.init_state(batch_size, h, w, device)
        
        # Block states
        block_mems = []
        for _ in self.blocks:
            block_mems.append({
                'qkv': torch.zeros(batch_size, self.embed_dim * 3, h, w, device=device),
                'proj': torch.zeros(batch_size, self.embed_dim, h, w, device=device),
                'mlp1': torch.zeros(batch_size, self.embed_dim * 4, h, w, device=device),
                'mlp2': torch.zeros(batch_size, self.embed_dim, h, w, device=device),
            })
        
        mem_out = torch.zeros(batch_size, self.classifier.out_features, device=device)
        spike_count = torch.zeros(batch_size, self.classifier.out_features, device=device)
        
        # Time loop
        for t in range(self.num_steps):
            out, stem_mem = self.stem(x, stem_mem)
            
            for i, block in enumerate(self.blocks):
                out, block_mems[i]['qkv'], block_mems[i]['proj'], \
                block_mems[i]['mlp1'], block_mems[i]['mlp2'] = block(
                    out,
                    block_mems[i]['qkv'],
                    block_mems[i]['proj'],
                    block_mems[i]['mlp1'],
                    block_mems[i]['mlp2'],
                )
            
            pooled = self.avgpool(out)
            flat = pooled.view(batch_size, -1)
            current = self.classifier(flat)
            spk_out, mem_out = self.lif_out(current, mem_out)
            
            spike_count += spk_out
        
        return spike_count, mem_out


# =============================================================================
# Unified Experiment Module
# =============================================================================

class ModernArchExperiment(pl.LightningModule):
    """
    Lightning module for VGG and Spiking Attention experiments.
    """
    
    def __init__(
        self,
        arch_type: str = "vgg",  # "vgg" or "attention"
        depth: int = 11,  # VGG depth or attention blocks
        embed_dim: int = 64,
        num_heads: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        num_steps: int = 25,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_plasticity: bool = True,
        init_method: str = "sabotage",
        init_std: float = 0.01,
        eta_stdp: float = 0.005,
        target_rate: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.arch_type = arch_type
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_plasticity = use_plasticity
        self.eta_stdp = eta_stdp
        self.target_rate = target_rate
        
        # Build network
        if arch_type == "vgg":
            self.network = VGGSpiking(
                depth=depth,
                in_channels=in_channels,
                num_classes=num_classes,
                num_steps=num_steps,
                beta=beta,
                threshold=threshold,
                slope=slope,
            )
        elif arch_type == "attention":
            self.network = SpikingAttentionNet(
                num_blocks=depth,
                embed_dim=embed_dim,
                num_heads=num_heads,
                in_channels=in_channels,
                num_classes=num_classes,
                num_steps=num_steps,
                beta=beta,
                threshold=threshold,
                slope=slope,
            )
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}")
        
        # Apply initialization
        if init_method == "sabotage":
            self.apply(lambda m: sabotage_init(m, std=init_std))
            print(f"[SABOTAGE INIT] Applied to {arch_type.upper()} network")
        else:
            self.apply(normal_init)
            print(f"[NORMAL INIT] Applied to {arch_type.upper()} network")
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return self.network.get_spike_records()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.network(x)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, targets = batch
        
        spike_counts, _ = self(inputs)
        spike_rates = spike_counts / self.num_steps
        loss = F.cross_entropy(spike_rates, targets)
        
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('arch', hash(self.arch_type) % 100, on_step=False, on_epoch=True)
        
        return loss
    
    def on_after_backward(self) -> None:
        if self.use_plasticity:
            self.network.apply_homeostatic_updates(
                target_rate=self.target_rate,
                learning_rate=self.eta_stdp,
            )
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        inputs, targets = batch
        
        spike_counts, _ = self(inputs)
        spike_rates = spike_counts / self.num_steps
        
        loss = F.cross_entropy(spike_rates, targets)
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', acc, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_accuracy': acc}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 30,
            eta_min=1e-6,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


# =============================================================================
# Factory Function
# =============================================================================

def build_modern_experiment(cfg: Any) -> ModernArchExperiment:
    """Build ModernArchExperiment from Hydra config."""
    return ModernArchExperiment(
        arch_type=cfg.model.architecture.type,
        depth=cfg.model.architecture.depth,
        embed_dim=getattr(cfg.model.architecture, 'embed_dim', 64),
        num_heads=getattr(cfg.model.architecture, 'num_heads', 4),
        in_channels=cfg.model.architecture.in_channels,
        num_classes=cfg.model.architecture.num_classes,
        num_steps=cfg.model.architecture.num_steps,
        beta=cfg.model.neuron.beta,
        threshold=cfg.model.neuron.threshold,
        slope=cfg.model.neuron.slope,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        use_plasticity=cfg.use_plasticity,
        init_method=cfg.initialization.method,
        init_std=cfg.initialization.std,
        eta_stdp=cfg.model.plasticity.eta_stdp,
        target_rate=cfg.model.plasticity.homeostasis.target_rate,
    )
