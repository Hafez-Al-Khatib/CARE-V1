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
    raise ImportError("snntorch is required. Install via: pip install snntorch.")

# 1. Sabotage Weight Initialization

def sabotage_init(module: nn.Module, mean: float = 0.0, std: mean = 0.01) -> None:
    
    if isinstance(module, (Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
def normal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='linear')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class DeadNeuronCallback(Callback):
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
        
        spike_records = pl_module.get_spike_records()

        if not spike_records:
            return
        
        total_neurons = 0
        total_dead = 0
        layer_ratios: List[float] = 0

        for layer_name, spikes in spike_records.items():
            if spikes.numel() == 0:
                continue
                
            if spikes.dim() == 5:
                spikes_per_neuron = spikes.sum(dim=(0, 1, 3, 4))
            elif spikes.dim() == 3:
                spikes_per_neuron = spikes.sum(dim=(0, 1))
            else:
                continue

            num_neurons = spikes_per_neuron.numel()
            num_dead = (spikes_per_neuron == 0).sum().item()
            dead_ratio = num_dead / num_neurons if num_neurons > 0 else 0.0

            total_neurons += num_neurons
            total_dead += num_dead
            layer_ratios.append(dead_ratio)

            if self.log_per_layer and trainer.logger:
                pl_module.log(
                    f"dead_neurons/{layer_name}",
                    dead_ratio,
                    on_step=True,
                    on_epoch=False
                )
            
        if total_neurons > 0:
            aggregate_ratio = total_dead / total_neurons
            pl_module.log(
                "dead_neuron_ratio",
                aggregate_ratio,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

        if len(layer_ratios) > 1:
            ratio_tensor = torch.tensor(layer_ratios)
            variance = ratio_tensor.var().item()
            pl_module.log(
                "dead_ratio_variance",
                variance,
                on_step=True,
                on_epoch=True
            )

            pl_module.log("dead_ratio_min", ratio_tensor.min().item(), on_step=True, on_epoch=True)
            pl_module.log("dead_ratio_max", ratio_tensor.max().item(), on_step=True, on_epoch=True)


class CareLIFConv(nn.Module):

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
            spike_grad=surrogate.fast_sigmoid(slope=slope)
            init_hidden=False
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
    
        self._spike