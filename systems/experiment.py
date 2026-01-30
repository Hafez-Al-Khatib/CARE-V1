"""
Dead Neuron Experiment Module - Depth Scaling Edition

Components:
    - DeadNeuronCallback: Enhanced with variance logging across layers
    - CareResidualBlock: ResNet-style skip connections
    - CareResNet: Variable-depth SNN backbone (supports 6, 12, 18, 50 layers)
    - DeadNeuronExperiment: LightningModule with scalable architecture

Author: Hafez Al Khatib
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


# =============================================================================
# Sabotage Weight Initialization
# =============================================================================

def sabotage_init(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    "Sabotage" initialization with very low standard deviation.
    Creates dead neurons that standard backprop struggles to revive.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def normal_init(module: nn.Module) -> None:
    """Standard Kaiming initialization for comparison."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# Enhanced Dead Neuron Callback
# =============================================================================

class DeadNeuronCallback(Callback):
    """
    Enhanced callback with variance logging for depth scaling experiments.
    
    Logs:
        - dead_neuron_ratio: Aggregate dead neurons
        - dead_ratio_variance: Variance across layers (high = Signal Propagation failure)
        - Per-layer dead ratios
    """
    
    def __init__(self, log_per_layer: bool = True):
        super().__init__()
        self.log_per_layer = log_per_layer
    
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: "DeadNeuronExperiment",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Compute and log dead neuron metrics."""
        
        spike_records = pl_module.get_spike_records()
        
        if not spike_records:
            return
        
        total_neurons = 0
        total_dead = 0
        layer_ratios: List[float] = []
        
        for layer_name, spikes in spike_records.items():
            if spikes.numel() == 0:
                continue
                
            # Sum over time and batch to get total spikes per neuron
            if spikes.dim() == 5:
                # Convolutional: [T, B, C, H, W] -> sum over T, B, H, W
                spikes_per_neuron = spikes.sum(dim=(0, 1, 3, 4))  # [C]
            elif spikes.dim() == 3:
                # Linear: [T, B, N] -> sum over T, B
                spikes_per_neuron = spikes.sum(dim=(0, 1))  # [N]
            else:
                continue
            
            num_neurons = spikes_per_neuron.numel()
            num_dead = (spikes_per_neuron == 0).sum().item()
            dead_ratio = num_dead / num_neurons if num_neurons > 0 else 0.0
            
            total_neurons += num_neurons
            total_dead += num_dead
            layer_ratios.append(dead_ratio)
            
            # Log per-layer
            if self.log_per_layer and trainer.logger:
                pl_module.log(
                    f"dead_neurons/{layer_name}",
                    dead_ratio,
                    on_step=True,
                    on_epoch=False,
                )
        
        # Aggregate metrics
        if total_neurons > 0:
            aggregate_ratio = total_dead / total_neurons
            pl_module.log(
                "dead_neuron_ratio",
                aggregate_ratio,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        
        # Variance across layers (KEY METRIC for depth scaling)
        if len(layer_ratios) > 1:
            ratio_tensor = torch.tensor(layer_ratios)
            variance = ratio_tensor.var().item()
            pl_module.log(
                "dead_ratio_variance",
                variance,
                on_step=True,
                on_epoch=True,
            )
            
            # Also log min/max for analysis
            pl_module.log("dead_ratio_min", ratio_tensor.min().item(), on_step=True, on_epoch=True)
            pl_module.log("dead_ratio_max", ratio_tensor.max().item(), on_step=True, on_epoch=True)


# =============================================================================
# Spiking Convolutional Layer (Base Component)
# =============================================================================

class CareLIFConv(nn.Module):
    """
    Convolutional layer + LIF neuron with spike recording and homeostatic plasticity.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        
        self.out_channels = out_channels
        self._spike_record: List[Tensor] = []
        
        self.register_buffer('activity_trace', torch.zeros(out_channels))
        self.activity_decay = 0.99
    
    def reset_spike_record(self) -> None:
        self._spike_record = []
    
    def get_spike_record(self) -> Tensor:
        if self._spike_record:
            return torch.stack(self._spike_record, dim=0)
        return torch.tensor([])
    
    def init_state(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.out_channels, height, width, device=device)
    
    def forward(self, x: Tensor, mem: Tensor) -> Tuple[Tensor, Tensor]:
        current = self.conv(x)
        spikes, mem = self.lif(current, mem)
        
        self._spike_record.append(spikes.detach())
        
        with torch.no_grad():
            channel_activity = spikes.mean(dim=(0, 2, 3))
            self.activity_trace = (
                self.activity_decay * self.activity_trace +
                (1 - self.activity_decay) * channel_activity
            )
        
        return spikes, mem
    
    def apply_homeostatic_update(
        self,
        target_rate: float = 0.1,
        learning_rate: float = 0.001,
    ) -> None:
        with torch.no_grad():
            deviation = target_rate - self.activity_trace
            boost = 1.0 + learning_rate * deviation.view(-1, 1, 1, 1)
            self.conv.weight.mul_(boost.clamp(0.9, 1.1))


# =============================================================================
# Residual Block (ResNet-style)
# =============================================================================

class CareResidualBlock(nn.Module):
    """
    ResNet-style residual block for spiking networks.
    
    Architecture: x -> Conv -> LIF -> Conv -> LIF + x (skip)
    
    The skip connection enables gradient flow through deep networks,
    while plasticity maintains neuron activity at each layer.
    """
    
    expansion = 1  # For basic block
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        # First conv-lif
        self.conv1 = CareLIFConv(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            beta=beta, threshold=threshold, slope=slope
        )
        
        # Second conv-lif
        self.conv2 = CareLIFConv(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            beta=beta, threshold=threshold, slope=slope
        )
        
        # Downsample for skip connection if dimensions change
        self.downsample = downsample
        self.stride = stride
    
    def reset_spike_records(self) -> None:
        self.conv1.reset_spike_record()
        self.conv2.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return {
            'conv1': self.conv1.get_spike_record(),
            'conv2': self.conv2.get_spike_record(),
        }
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.conv1.apply_homeostatic_update(target_rate, learning_rate)
        self.conv2.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(
        self,
        x: Tensor,
        mem1: Tensor,
        mem2: Tensor,
        identity: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward through residual block.
        
        Args:
            x: Input spikes [B, C, H, W]
            mem1: Membrane for conv1
            mem2: Membrane for conv2
            identity: Skip connection value (computed if None)
            
        Returns:
            (output_spikes, mem1, mem2)
        """
        if identity is None:
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
        
        # Block forward
        out, mem1 = self.conv1(x, mem1)
        out, mem2 = self.conv2(out, mem2)
        
        # Residual addition (spike + spike = summed activity)
        out = out + identity
        
        return out, mem1, mem2


# =============================================================================
# Bottleneck Block (for ResNet-50+)
# =============================================================================

class CareBottleneckBlock(nn.Module):
    """
    Bottleneck block for deeper networks (ResNet-50, 101, 152).
    
    Architecture: x -> 1x1 -> 3x3 -> 1x1 + x
    Reduces parameters while maintaining representational power.
    """
    
    expansion = 4
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        # 1x1 reduce
        self.conv1 = CareLIFConv(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0,
            beta=beta, threshold=threshold, slope=slope
        )
        
        # 3x3 conv
        self.conv2 = CareLIFConv(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            beta=beta, threshold=threshold, slope=slope
        )
        
        # 1x1 expand
        self.conv3 = CareLIFConv(
            out_channels, out_channels * self.expansion,
            kernel_size=1, stride=1, padding=0,
            beta=beta, threshold=threshold, slope=slope
        )
        
        self.downsample = downsample
        self.stride = stride
    
    def reset_spike_records(self) -> None:
        self.conv1.reset_spike_record()
        self.conv2.reset_spike_record()
        self.conv3.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return {
            'conv1': self.conv1.get_spike_record(),
            'conv2': self.conv2.get_spike_record(),
            'conv3': self.conv3.get_spike_record(),
        }
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.conv1.apply_homeostatic_update(target_rate, learning_rate)
        self.conv2.apply_homeostatic_update(target_rate, learning_rate)
        self.conv3.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(
        self,
        x: Tensor,
        mem1: Tensor,
        mem2: Tensor,
        mem3: Tensor,
        identity: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if identity is None:
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
        
        out, mem1 = self.conv1(x, mem1)
        out, mem2 = self.conv2(out, mem2)
        out, mem3 = self.conv3(out, mem3)
        
        out = out + identity
        
        return out, mem1, mem2, mem3


# =============================================================================
# CareResNet - Variable Depth SNN Backbone
# =============================================================================

# Configuration for different depths
RESNET_CONFIGS = {
    6: {'block': 'basic', 'layers': [1, 1, 1, 0]},      # 6 layers (very shallow)
    12: {'block': 'basic', 'layers': [2, 2, 2, 0]},     # 12 layers
    18: {'block': 'basic', 'layers': [2, 2, 2, 2]},     # 18 layers (ResNet-18)
    34: {'block': 'basic', 'layers': [3, 4, 6, 3]},     # 34 layers (ResNet-34)
    50: {'block': 'bottleneck', 'layers': [3, 4, 6, 3]},# 50 layers (ResNet-50)
    101: {'block': 'bottleneck', 'layers': [3, 4, 23, 3]}, # ResNet-101
}


class CareResNet(nn.Module):
    """
    Variable-depth ResNet backbone for spiking neural networks.
    
    Supports depths: 6, 12, 18, 34, 50, 101
    
    Key Features:
        - ResNet-style residual connections for deep gradient flow
        - Per-layer spike recording for dead neuron analysis
        - Homeostatic plasticity to revive silent neurons
    """
    
    def __init__(
        self,
        depth: int = 18,
        in_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 64,
        num_steps: int = 25,
        beta: float = 0.9,
        threshold: float = 1.0,
        slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        if depth not in RESNET_CONFIGS:
            raise ValueError(f"Unsupported depth {depth}. Choose from {list(RESNET_CONFIGS.keys())}")
        
        config = RESNET_CONFIGS[depth]
        self.depth = depth
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        
        # Select block type
        if config['block'] == 'bottleneck':
            block_class = CareBottleneckBlock
            self.expansion = 4
        else:
            block_class = CareResidualBlock
            self.expansion = 1
        
        self.block_class = block_class
        self.layers_config = config['layers']
        
        # Stem: Initial convolution
        self.stem = CareLIFConv(
            in_channels, base_channels,
            kernel_size=7, stride=2, padding=3,
            beta=beta, threshold=threshold, slope=slope
        )
        
        self.in_channels = base_channels
        
        # Build residual layers
        self.layer1 = self._make_layer(block_class, base_channels, config['layers'][0], stride=1)
        self.layer2 = self._make_layer(block_class, base_channels * 2, config['layers'][1], stride=2)
        self.layer3 = self._make_layer(block_class, base_channels * 4, config['layers'][2], stride=2)
        
        if config['layers'][3] > 0:
            self.layer4 = self._make_layer(block_class, base_channels * 8, config['layers'][3], stride=2)
            final_channels = base_channels * 8 * self.expansion
        else:
            self.layer4 = None
            final_channels = base_channels * 4 * self.expansion
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)
        
        # Output LIF
        self.lif_out = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
    
    def _make_layer(
        self,
        block_class: type,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.ModuleList:
        """Create a layer of residual blocks."""
        
        if num_blocks == 0:
            return nn.ModuleList()
        
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Conv2d(
                self.in_channels, out_channels * self.expansion,
                kernel_size=1, stride=stride, bias=False
            )
        
        blocks = nn.ModuleList()
        
        # First block may have stride and downsample
        blocks.append(block_class(
            self.in_channels, out_channels,
            stride=stride, downsample=downsample,
            beta=self.beta, threshold=self.threshold, slope=self.slope
        ))
        
        self.in_channels = out_channels * self.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            blocks.append(block_class(
                self.in_channels, out_channels,
                stride=1, downsample=None,
                beta=self.beta, threshold=self.threshold, slope=self.slope
            ))
        
        return blocks
    
    def reset_spike_records(self) -> None:
        """Reset all spike records for new batch."""
        self.stem.reset_spike_record()
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer is not None:
                for block in layer:
                    block.reset_spike_records()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        """Get all spike records for dead neuron analysis."""
        records = {'stem': self.stem.get_spike_record()}
        
        layer_idx = 0
        for layer_name, layer in [('layer1', self.layer1), ('layer2', self.layer2), 
                                   ('layer3', self.layer3), ('layer4', self.layer4)]:
            if layer is None:
                continue
            for block_idx, block in enumerate(layer):
                block_records = block.get_spike_records()
                for conv_name, record in block_records.items():
                    records[f'{layer_name}_b{block_idx}_{conv_name}'] = record
                    layer_idx += 1
        
        return records
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        """Apply homeostatic plasticity to all layers."""
        self.stem.apply_homeostatic_update(target_rate, learning_rate)
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer is not None:
                for block in layer:
                    block.apply_homeostatic_updates(target_rate, learning_rate)
    
    def _init_layer_states(
        self,
        layer: nn.ModuleList,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> List[List[Tensor]]:
        """Initialize membrane states for a layer of blocks."""
        states = []
        h, w = height, width
        
        for block in layer:
            if isinstance(block, CareBottleneckBlock):
                # Bottleneck: 3 conv layers
                mem1 = block.conv1.init_state(batch_size, h, w, device)
                
                # After first 1x1 (no size change)
                mem2 = block.conv2.init_state(batch_size, h, w, device)
                
                # Size change happens in conv2 if stride > 1
                if block.stride > 1:
                    h, w = h // block.stride, w // block.stride
                    mem2 = block.conv2.init_state(batch_size, h, w, device)
                
                mem3 = block.conv3.init_state(batch_size, h, w, device)
                states.append([mem1, mem2, mem3])
            else:
                # Basic block: 2 conv layers
                mem1 = block.conv1.init_state(batch_size, h, w, device)
                
                if block.stride > 1:
                    h, w = h // block.stride, w // block.stride
                
                mem2 = block.conv2.init_state(batch_size, h, w, device)
                states.append([mem1, mem2])
        
        return states
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass over time steps.
        
        Args:
            x: Input [B, C, H, W]
            
        Returns:
            (spike_counts, final_membrane)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Reset spike records
        self.reset_spike_records()
        
        # Initialize all membrane states
        # Stem output size: (H-1)//2, (W-1)//2 for 7x7 stride 2 with padding 3
        # For 28x28 input: 14x14
        h, w = (x.shape[2] + 2*3 - 7) // 2 + 1, (x.shape[3] + 2*3 - 7) // 2 + 1
        
        stem_mem = self.stem.init_state(batch_size, h, w, device)
        
        # Calculate layer output sizes
        l1_h, l1_w = h, w
        l2_h, l2_w = (h + 1) // 2, (w + 1) // 2
        l3_h, l3_w = (l2_h + 1) // 2, (l2_w + 1) // 2
        l4_h, l4_w = (l3_h + 1) // 2, (l3_w + 1) // 2 if self.layer4 else (0, 0)
        
        # Initialize block states
        l1_states = self._init_layer_states(self.layer1, batch_size, l1_h, l1_w, device) if len(self.layer1) > 0 else []
        l2_states = self._init_layer_states(self.layer2, batch_size, l1_h, l1_w, device) if len(self.layer2) > 0 else []  # stride applied in first block
        l3_states = self._init_layer_states(self.layer3, batch_size, l2_h, l2_w, device) if len(self.layer3) > 0 else []
        l4_states = self._init_layer_states(self.layer4, batch_size, l3_h, l3_w, device) if self.layer4 and len(self.layer4) > 0 else []
        
        mem_out = torch.zeros(batch_size, self.fc.out_features, device=device)
        spike_count = torch.zeros(batch_size, self.fc.out_features, device=device)
        
        # Time loop
        for t in range(self.num_steps):
            out, stem_mem = self.stem(x, stem_mem)
            
            # Layer 1
            for i, block in enumerate(self.layer1):
                if isinstance(block, CareBottleneckBlock):
                    out, l1_states[i][0], l1_states[i][1], l1_states[i][2] = block(
                        out, l1_states[i][0], l1_states[i][1], l1_states[i][2]
                    )
                else:
                    out, l1_states[i][0], l1_states[i][1] = block(
                        out, l1_states[i][0], l1_states[i][1]
                    )
            
            # Layer 2
            for i, block in enumerate(self.layer2):
                if isinstance(block, CareBottleneckBlock):
                    out, l2_states[i][0], l2_states[i][1], l2_states[i][2] = block(
                        out, l2_states[i][0], l2_states[i][1], l2_states[i][2]
                    )
                else:
                    out, l2_states[i][0], l2_states[i][1] = block(
                        out, l2_states[i][0], l2_states[i][1]
                    )
            
            # Layer 3
            for i, block in enumerate(self.layer3):
                if isinstance(block, CareBottleneckBlock):
                    out, l3_states[i][0], l3_states[i][1], l3_states[i][2] = block(
                        out, l3_states[i][0], l3_states[i][1], l3_states[i][2]
                    )
                else:
                    out, l3_states[i][0], l3_states[i][1] = block(
                        out, l3_states[i][0], l3_states[i][1]
                    )
            
            # Layer 4 (if exists)
            if self.layer4:
                for i, block in enumerate(self.layer4):
                    if isinstance(block, CareBottleneckBlock):
                        out, l4_states[i][0], l4_states[i][1], l4_states[i][2] = block(
                            out, l4_states[i][0], l4_states[i][1], l4_states[i][2]
                        )
                    else:
                        out, l4_states[i][0], l4_states[i][1] = block(
                            out, l4_states[i][0], l4_states[i][1]
                        )
            
            # Classifier
            pooled = self.avgpool(out)
            flat = pooled.view(batch_size, -1)
            current = self.fc(flat)
            spk_out, mem_out = self.lif_out(current, mem_out)
            
            spike_count += spk_out
        
        return spike_count, mem_out


# =============================================================================
# Dead Neuron Experiment Lightning Module
# =============================================================================

class DeadNeuronExperiment(pl.LightningModule):
    """
    LightningModule for Dead Neuron Experiment with scalable depth.
    
    Supports depths: 6, 12, 18, 34, 50, 101
    """
    
    def __init__(
        self,
        depth: int = 18,
        in_channels: int = 1,
        num_classes: int = 10,
        base_channels: int = 64,
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
        
        self.depth = depth
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_plasticity = use_plasticity
        self.eta_stdp = eta_stdp
        self.target_rate = target_rate
        
        # Build network
        self.network = CareResNet(
            depth=depth,
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            slope=slope,
        )
        
        # Apply initialization
        if init_method == "sabotage":
            self.apply(lambda m: sabotage_init(m, std=init_std))
            print(f"[SABOTAGE INIT] Applied with std={init_std} to depth-{depth} network")
        else:
            self.apply(normal_init)
            print(f"[NORMAL INIT] Applied to depth-{depth} network")
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        """Get spike records from network."""
        return self.network.get_spike_records()
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass."""
        return self.network(x)
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """Training step."""
        inputs, targets = batch
        
        spike_counts, _ = self(inputs)
        spike_rates = spike_counts / self.num_steps
        loss = F.cross_entropy(spike_rates, targets)
        
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('depth', float(self.depth), on_step=False, on_epoch=True)
        
        return loss
    
    def on_after_backward(self) -> None:
        """Apply plasticity updates after backward pass if enabled."""
        if self.use_plasticity:
            self.network.apply_homeostatic_updates(
                target_rate=self.target_rate,
                learning_rate=self.eta_stdp,
            )
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """Validation step."""
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
        """Configure optimizer."""
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
# Factory Function for Hydra
# =============================================================================

def build_experiment(cfg: Any) -> DeadNeuronExperiment:
    """Build DeadNeuronExperiment from Hydra config."""
    return DeadNeuronExperiment(
        depth=cfg.model.architecture.depth,
        in_channels=cfg.model.architecture.in_channels,
        num_classes=cfg.model.architecture.num_classes,
        base_channels=cfg.model.architecture.base_channels,
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
