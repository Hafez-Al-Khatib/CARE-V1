"""
Modern SNN Architecture Experiment Module

Architectures beyond ResNet to prove CARE generalizes:
  1. VGG-SNN:     No skip connections → worst dead neuron case with BN
  2. PlainConvSNN: No skip, no BN → absolute worst case (CARE's strongest showcase)
  3. SpikingAttentionNet: Transformer-style → proves CARE works on attention

All architectures use CareLIFConv with full ablation-configurable CARE parameters
(homeo_target, SNR gating) for consistent cross-architecture comparison.

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

from models.components.neuron import (
    CareLIFConv, CareResNet, sabotage_init, normal_init
)


# =============================================================================
# VGG Configurations (Modified for Small Images)
# =============================================================================

VGG_CONFIGS = {
    # Small-image VGG variants (2 pooling layers only for 32x32)
    8:  [64, 64, 'M', 128, 128, 'M', 256, 256],
    11: [64, 64, 'M', 128, 128, 256, 256, 'M', 512, 512, 512],
    13: [64, 64, 'M', 128, 128, 256, 256, 256, 'M', 512, 512, 512, 512],
    16: [64, 64, 'M', 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 512, 512, 512, 512, 512],
}


# =============================================================================
# VGG-Spiking Architecture (NO Skip Connections)
# =============================================================================

class VGGSpiking(nn.Module):
    """
    VGG-style Spiking Neural Network with ablation-configurable CARE.
    
    NO SKIP CONNECTIONS → dead neurons are catastrophic and irrecoverable
    by gradient descent alone. CARE is the only mechanism that can rescue them.
    """
    
    def __init__(
        self,
        depth: int = 11,
        in_channels: int = 3,
        num_classes: int = 10,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        homeo_target: str = 'gamma',
        snr_enabled: bool = True,
        snr_threshold: float = 2.0,
        snr_steepness: float = 5.0,
    ) -> None:
        super().__init__()
        
        if depth not in VGG_CONFIGS:
            raise ValueError(f"Unsupported VGG depth {depth}. Choose from {list(VGG_CONFIGS.keys())}")
        
        self.depth = depth
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        self.homeo_target = homeo_target
        self.snr_enabled = snr_enabled
        self.snr_threshold = snr_threshold
        self.snr_steepness = snr_steepness
        
        # Build convolutional layers
        self.features, self.conv_layers, last_channels = self._make_layers(
            VGG_CONFIGS[depth], in_channels
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channels, num_classes)
        
        self.lif_out = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
    
    def _make_layers(self, config, in_channels):
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
                    beta=self.beta, threshold=self.threshold, slope=self.slope,
                    homeo_target=self.homeo_target,
                    snr_enabled=self.snr_enabled,
                    snr_threshold=self.snr_threshold,
                    snr_steepness=self.snr_steepness,
                )
                layers.append(conv)
                conv_layers.append(conv)
                in_channels = v
                last_channels = v
        
        return layers, conv_layers, last_channels
    
    def reset_spike_records(self) -> None:
        for conv in self.conv_layers:
            conv.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        records = {}
        for i, conv in enumerate(self.conv_layers):
            record = conv.get_spike_record()
            if record.numel() > 0:
                records[f'conv{i}'] = record
        return records
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        for conv in self.conv_layers:
            conv.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        device = x.device
        self.reset_spike_records()
        
        # Initialize membrane states for all conv layers
        mems = {}
        current_size = list(x.shape[2:])
        
        for i, layer in enumerate(self.features):
            if isinstance(layer, CareLIFConv):
                mems[i] = layer.init_state(batch_size, current_size[0], current_size[1], device)
            elif isinstance(layer, nn.MaxPool2d):
                current_size = [current_size[0] // 2, current_size[1] // 2]
        
        mem_out = torch.zeros(batch_size, self.classifier.out_features, device=device)
        spike_count = torch.zeros(batch_size, self.classifier.out_features, device=device)
        
        for t in range(self.num_steps):
            out = x
            for i, layer in enumerate(self.features):
                if isinstance(layer, CareLIFConv):
                    out, mems[i] = layer(out, mems[i])
                else:
                    out = layer(out)
            
            pooled = self.avgpool(out)
            flat = pooled.view(batch_size, -1)
            current = self.classifier(flat)
            spk_out, mem_out = self.lif_out(current, mem_out)
            spike_count += spk_out
        
        return spike_count, mem_out


# =============================================================================
# Plain ConvNet-SNN (No Skip, No BN — CARE's Strongest Showcase)
# =============================================================================

PLAIN_CONFIGS = {
    # depth: list of (out_channels, stride) tuples
    6:  [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1)],
    8:  [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2), (512, 1)],
    10: [(64, 1), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (256, 1), (512, 2), (512, 1), (512, 1)],
    18: [(64, 1)] + [(64, 1)] * 4 + [(128, 2)] + [(128, 1)] * 3 + [(256, 2)] + [(256, 1)] * 3 + [(512, 2)] + [(512, 1)] * 3,
    34: [(64, 1)] + [(64, 1)] * 6 + [(128, 2)] + [(128, 1)] * 7 + [(256, 2)] + [(256, 1)] * 11 + [(512, 2)] + [(512, 1)] * 5,
}

class PlainConvSNN(nn.Module):
    """
    Plain convolutional SNN: NO skip connections, NO BatchNorm.
    
    This is the absolute worst-case architecture for dead neurons.
    Without CARE, gradient flow is completely severed at dead layers,
    making recovery impossible via backprop alone.
    
    This architecture is the strongest possible evidence that CARE
    provides essential "capacity insurance" for deep SNNs.
    """
    
    def __init__(
        self,
        depth: int = 8,
        in_channels: int = 3,
        num_classes: int = 10,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        homeo_target: str = 'weight',  # Default to weight since no BN
        snr_enabled: bool = True,
        snr_threshold: float = 2.0,
        snr_steepness: float = 5.0,
    ) -> None:
        super().__init__()
        
        if depth not in PLAIN_CONFIGS:
            raise ValueError(f"Unsupported Plain depth {depth}. Choose from {list(PLAIN_CONFIGS.keys())}")
        
        self.depth = depth
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        
        config = PLAIN_CONFIGS[depth]
        self.conv_layers = nn.ModuleList()
        ch = in_channels
        
        for out_ch, stride in config:
            conv = CareLIFConv(
                ch, out_ch,
                kernel_size=3, stride=stride, padding=1,
                beta=beta, threshold=threshold, slope=slope,
                homeo_target=homeo_target,
                snr_enabled=snr_enabled,
                snr_threshold=snr_threshold,
                snr_steepness=snr_steepness,
                use_bn=False,  # KEY: No BatchNorm
            )
            self.conv_layers.append(conv)
            ch = out_ch
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(ch, num_classes)
        self.lif_out = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
    
    def reset_spike_records(self) -> None:
        for conv in self.conv_layers:
            conv.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        records = {}
        for i, conv in enumerate(self.conv_layers):
            record = conv.get_spike_record()
            if record.numel() > 0:
                records[f'plain_conv{i}'] = record
        return records
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        for conv in self.conv_layers:
            conv.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        device = x.device
        self.reset_spike_records()
        
        # Initialize membrane states
        mems = []
        h, w = x.shape[2], x.shape[3]
        config = PLAIN_CONFIGS[self.depth]
        
        for i, (out_ch, stride) in enumerate(config):
            if stride > 1:
                h, w = h // stride, w // stride
            mems.append(self.conv_layers[i].init_state(batch_size, h, w, device))
        
        mem_out = torch.zeros(batch_size, self.classifier.out_features, device=device)
        spike_count = torch.zeros(batch_size, self.classifier.out_features, device=device)
        
        for t in range(self.num_steps):
            out = x
            for i, conv in enumerate(self.conv_layers):
                out, mems[i] = conv(out, mems[i])
            
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
    Inspired by Spikingformer (AAAI 2024) and SpikingResformer (CVPR 2024).
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
        
        self.qkv_conv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_lif = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        
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
        with torch.no_grad():
            deviation = target_rate - self.activity_trace
            boost = 1.0 + learning_rate * deviation.view(1, -1, 1, 1)
            self.qkv_conv.weight.mul_(boost.clamp(0.9, 1.1).expand_as(self.qkv_conv.weight[:self.dim]))
    
    def forward(self, x: Tensor, mem_qkv: Tensor, mem_proj: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        B, C, H, W = x.shape
        
        qkv = self.qkv_conv(x)
        qkv_spk, mem_qkv = self.qkv_lif(qkv, mem_qkv)
        
        qkv_spk = qkv_spk.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv_spk[:, 0], qkv_spk[:, 1], qkv_spk[:, 2]
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = torch.clamp(attn, 0, 1)
        
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, H, W)
        
        out = self.proj(out)
        out_spk, mem_proj = self.proj_lif(out, mem_proj)
        
        self._spike_record.append(out_spk.detach())
        
        with torch.no_grad():
            channel_activity = out_spk.mean(dim=(0, 2, 3))
            self.activity_trace = (
                self.activity_decay * self.activity_trace +
                (1 - self.activity_decay) * channel_activity
            )
        
        return out_spk, mem_qkv, mem_proj


class SpikingAttentionBlock(nn.Module):
    """Full spiking attention block with residual: x -> Attention -> LIF + x"""
    
    def __init__(self, dim: int, num_heads: int = 4, beta: float = 0.9,
                 threshold: float = 1.0, slope: float = 25.0) -> None:
        super().__init__()
        
        self.attention = SpikingSelfAttention(dim, num_heads, beta, threshold, slope)
        
        self.mlp_conv1 = nn.Conv2d(dim, dim * 4, 1, bias=False)
        self.mlp_lif1 = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=False,
        )
        self.mlp_conv2 = nn.Conv2d(dim * 4, dim, 1, bias=False)
        self.mlp_lif2 = snn.Leaky(
            beta=beta, threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=False,
        )
        self.dim = dim
    
    def reset_spike_records(self) -> None:
        self.attention.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return {'attention': self.attention.get_spike_record()}
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.attention.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(self, x, mem_qkv, mem_proj, mem_mlp1, mem_mlp2):
        attn_out, mem_qkv, mem_proj = self.attention(x, mem_qkv, mem_proj)
        x = x + attn_out
        
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
    """Spiking attention network with CARE-configurable stem."""
    
    def __init__(
        self,
        num_blocks: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        homeo_target: str = 'gamma',
        snr_enabled: bool = True,
        snr_threshold: float = 2.0,
        snr_steepness: float = 5.0,
    ) -> None:
        super().__init__()
        
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.embed_dim = embed_dim
        
        # Stem with CARE params
        self.stem = CareLIFConv(
            in_channels, embed_dim,
            kernel_size=3, stride=1, padding=1,  # preserve spatial for small images
            beta=beta, threshold=threshold, slope=slope,
            homeo_target=homeo_target, snr_enabled=snr_enabled,
            snr_threshold=snr_threshold, snr_steepness=snr_steepness,
        )
        
        self.blocks = nn.ModuleList([
            SpikingAttentionBlock(embed_dim, num_heads, beta, threshold, slope)
            for _ in range(num_blocks)
        ])
        
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
            for k, v in block.get_spike_records().items():
                records[f'block{i}_{k}'] = v
        return records
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.stem.apply_homeostatic_update(target_rate, learning_rate)
        for block in self.blocks:
            block.apply_homeostatic_updates(target_rate, learning_rate)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]
        device = x.device
        self.reset_spike_records()
        
        h, w = x.shape[2], x.shape[3]
        stem_mem = self.stem.init_state(batch_size, h, w, device)
        
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
        
        for t in range(self.num_steps):
            out, stem_mem = self.stem(x, stem_mem)
            
            for i, block in enumerate(self.blocks):
                out, block_mems[i]['qkv'], block_mems[i]['proj'], \
                block_mems[i]['mlp1'], block_mems[i]['mlp2'] = block(
                    out,
                    block_mems[i]['qkv'], block_mems[i]['proj'],
                    block_mems[i]['mlp1'], block_mems[i]['mlp2'],
                )
            
            pooled = self.avgpool(out)
            flat = pooled.view(batch_size, -1)
            current = self.classifier(flat)
            spk_out, mem_out = self.lif_out(current, mem_out)
            spike_count += spk_out
        
        return spike_count, mem_out


# =============================================================================
# Unified Experiment Module (All Architectures)
# =============================================================================

class ModernArchExperiment(pl.LightningModule):
    """
    Lightning module for all SNN architecture experiments.
    Supports: vgg, plain, attention, resnet
    """
    
    def __init__(
        self,
        arch_type: str = "vgg",
        depth: int = 11,
        embed_dim: int = 64,
        num_heads: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        num_steps: int = 8,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        use_plasticity: bool = True,
        init_method: str = "normal",
        init_std: float = 0.01,
        eta_stdp: float = 0.005,
        target_rate: float = 0.02,
        homeo_target: str = 'gamma',
        snr_enabled: bool = True,
        snr_threshold: float = 2.0,
        snr_steepness: float = 5.0,
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
        
        # Build network based on architecture type
        if arch_type == "vgg":
            self.network = VGGSpiking(
                depth=depth, in_channels=in_channels, num_classes=num_classes,
                num_steps=num_steps, beta=beta, threshold=threshold, slope=slope,
                homeo_target=homeo_target, snr_enabled=snr_enabled,
                snr_threshold=snr_threshold, snr_steepness=snr_steepness,
            )
        elif arch_type == "plain":
            self.network = PlainConvSNN(
                depth=depth, in_channels=in_channels, num_classes=num_classes,
                num_steps=num_steps, beta=beta, threshold=threshold, slope=slope,
                homeo_target='weight' if homeo_target == 'gamma' else homeo_target,
                snr_enabled=snr_enabled, snr_threshold=snr_threshold,
                snr_steepness=snr_steepness,
            )
        elif arch_type == "attention":
            self.network = SpikingAttentionNet(
                num_blocks=depth, embed_dim=embed_dim, num_heads=num_heads,
                in_channels=in_channels, num_classes=num_classes,
                num_steps=num_steps, beta=beta, threshold=threshold, slope=slope,
                homeo_target=homeo_target, snr_enabled=snr_enabled,
                snr_threshold=snr_threshold, snr_steepness=snr_steepness,
            )
        elif arch_type == "resnet":
            self.network = CareResNet(
                depth=depth, in_channels=in_channels, num_classes=num_classes,
                base_channels=64, num_steps=num_steps,
                beta=beta, threshold=threshold, slope=slope,
                block_type="sew",
                homeo_target=homeo_target, snr_enabled=snr_enabled,
                snr_threshold=snr_threshold, snr_steepness=snr_steepness,
            )
        else:
            raise ValueError(f"Unknown arch_type: {arch_type}. Use: vgg, plain, attention, resnet")
        
        # Apply initialization
        if init_method == "sabotage":
            self.apply(lambda m: sabotage_init(m, std=init_std))
        else:
            self.apply(normal_init)
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return self.network.get_spike_records()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.network(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        spike_counts, mem_out = self(inputs)
        
        # Use membrane potential for loss (consistent with ResNet experiments)
        loss = F.cross_entropy(mem_out, targets)
        
        preds = mem_out.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_after_backward(self) -> None:
        if self.use_plasticity:
            self.network.apply_homeostatic_updates(
                target_rate=self.target_rate,
                learning_rate=self.eta_stdp,
            )
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        spike_counts, mem_out = self(inputs)
        
        loss = F.cross_entropy(mem_out, targets)
        preds = mem_out.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)
        self.log('val/accuracy', acc, on_epoch=True, prog_bar=True)
        
        # PREVENT OOM: Clear spike records after validation batch
        if hasattr(self.network, 'reset_spike_records'):
            self.network.reset_spike_records()
        
        return {'val_loss': loss, 'val_accuracy': acc}
    
    def configure_optimizers(self):
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
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }
