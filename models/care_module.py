"""
CARE Module - Modular Backbone

CareBlock: The core modular unit containing:
    - Input encoding (rate/latency coding)
    - Recurrent SNN reservoir (LSM-style)
    - Readout layer with sparsity masking
    - Synaptic operation counting

Author: CARE Research Team
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.components.neuron import CARELIFNeuron, PlasticityLayer


class RateEncoder(nn.Module):
    """
    Converts static inputs to spike trains via rate coding.
    
    Higher input values -> higher firing rates.
    Uses Bernoulli sampling for stochastic spikes or deterministic thresholding.
    
    Args:
        num_steps: Number of time steps to generate
        gain: Scaling factor for input normalization
        stochastic: Whether to use stochastic (Bernoulli) or deterministic encoding
    """
    
    def __init__(
        self,
        num_steps: int = 25,
        gain: float = 1.0,
        stochastic: bool = True,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.gain = gain
        self.stochastic = stochastic
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode static input to spike train.
        
        Args:
            x: Input tensor [batch, features]
            
        Returns:
            Spike train [time_steps, batch, features]
        """
        # Normalize to [0, 1] range for spike probability
        x_norm = torch.sigmoid(self.gain * x)
        
        # Expand to time dimension
        x_expanded = x_norm.unsqueeze(0).expand(self.num_steps, -1, -1)
        
        if self.stochastic:
            # Bernoulli sampling based on firing rate
            spikes = torch.bernoulli(x_expanded)
        else:
            # Deterministic: uniform random threshold per timestep
            thresholds = torch.rand_like(x_expanded)
            spikes = (x_expanded > thresholds).float()
        
        return spikes


class LatencyEncoder(nn.Module):
    """
    Converts static inputs to spike trains via latency coding.
    
    Higher input values -> earlier spike times (lower latency).
    Each neuron fires at most once.
    
    Args:
        num_steps: Number of time steps
        tau: Time constant controlling latency spread
        normalize: Whether to normalize input
    """
    
    def __init__(
        self,
        num_steps: int = 25,
        tau: float = 5.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.tau = tau
        self.normalize = normalize
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Encode static input to latency-coded spike train.
        
        Args:
            x: Input tensor [batch, features]
            
        Returns:
            Spike train [time_steps, batch, features]
        """
        if self.normalize:
            # Min-max normalization per sample
            x_min = x.min(dim=-1, keepdim=True).values
            x_max = x.max(dim=-1, keepdim=True).values
            x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        else:
            x_norm = x
        
        # Compute spike times: high values -> low latency
        # latency = tau * (1 - x), clipped to [0, num_steps-1]
        latency = (self.tau * (1 - x_norm)).clamp(0, self.num_steps - 1).long()
        
        # Create spike train: spike at computed latency
        batch_size, features = x.shape
        spikes = torch.zeros(self.num_steps, batch_size, features, device=x.device)
        
        # Vectorized spike placement
        time_idx = torch.arange(self.num_steps, device=x.device).view(-1, 1, 1)
        spikes = (time_idx == latency.unsqueeze(0)).float()
        
        return spikes


class Reservoir(nn.Module):
    """
    Liquid State Machine (LSM) style recurrent reservoir.
    
    Sparse, randomly connected recurrent SNN layer that provides
    rich temporal dynamics for spatiotemporal feature extraction.
    
    Args:
        input_dim: Input feature dimension
        reservoir_size: Number of reservoir neurons
        sparsity: Connection sparsity (fraction of zero weights)
        beta: LIF membrane decay
        threshold: LIF spike threshold
        surrogate_slope: Surrogate gradient slope
    """
    
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 512,
        sparsity: float = 0.9,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.sparsity = sparsity
        
        # Input projection (dense)
        self.input_proj = nn.Linear(input_dim, reservoir_size)
        
        # Recurrent weights (sparse)
        self.recurrent = nn.Linear(reservoir_size, reservoir_size, bias=False)
        
        # Initialize with sparsity
        self._init_sparse_recurrent()
        
        # LIF neurons
        self.lif = CARELIFNeuron(
            beta=beta,
            threshold=threshold,
            spike_grad="fast_sigmoid",
            slope=surrogate_slope,
        )
        
        # Sparsity mask (fixed random pattern)
        self.register_buffer(
            'sparse_mask',
            (torch.rand(reservoir_size, reservoir_size) > sparsity).float()
        )
    
    def _init_sparse_recurrent(self) -> None:
        """Initialize recurrent weights with sparsity."""
        with torch.no_grad():
            mask = torch.rand_like(self.recurrent.weight) > self.sparsity
            self.recurrent.weight.mul_(mask.float())
            # Scale remaining weights
            self.recurrent.weight.div_(
                (1 - self.sparsity) ** 0.5 + 1e-8
            )
    
    def forward(
        self,
        x: Tensor,
        mem: Optional[Tensor] = None,
        prev_spikes: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Single timestep forward through reservoir.
        
        Args:
            x: Input [batch, input_dim]
            mem: Membrane potential [batch, reservoir_size]
            prev_spikes: Previous spikes [batch, reservoir_size]
            
        Returns:
            Tuple of (spikes, membrane, synops_count)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states if needed
        if mem is None:
            mem = torch.zeros(batch_size, self.reservoir_size, device=device)
        if prev_spikes is None:
            prev_spikes = torch.zeros(batch_size, self.reservoir_size, device=device)
        
        # Input current
        input_current = self.input_proj(x)
        
        # Recurrent current (apply sparse mask)
        recurrent_weight = self.recurrent.weight * self.sparse_mask
        recurrent_current = prev_spikes @ recurrent_weight.t()
        
        # Total input to LIF
        total_current = input_current + recurrent_current
        
        # LIF dynamics
        spikes, mem = self.lif(total_current, mem)
        
        # Count synaptic operations (sparse computation)
        # SynOps = number of active pre-synaptic spikes × number of connections
        active_synapses = (prev_spikes.sum() * self.sparse_mask.sum() + 
                          x.abs().gt(0).sum() * self.reservoir_size)
        
        return spikes, mem, active_synapses


class ReadoutLayer(nn.Module):
    """
    Readout layer with sparsity masking.
    
    Aggregates reservoir activity and produces output spikes.
    Applies sparsity mask to zero out low-activity neurons
    before computation to save simulated SynOps.
    
    Args:
        input_dim: Reservoir size
        output_dim: Number of output classes
        beta: LIF membrane decay
        threshold: LIF spike threshold
        sparsity_threshold: Activation threshold for masking
        surrogate_slope: Surrogate gradient slope
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        beta: float = 0.9,
        threshold: float = 1.0,
        sparsity_threshold: float = 0.1,
        surrogate_slope: float = 25.0,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity_threshold = sparsity_threshold
        
        # Readout projection
        self.fc = nn.Linear(input_dim, output_dim)
        
        # Output LIF
        self.lif = CARELIFNeuron(
            beta=beta,
            threshold=threshold,
            spike_grad="fast_sigmoid",
            slope=surrogate_slope,
        )
    
    def forward(
        self,
        x: Tensor,
        mem: Optional[Tensor] = None,
        return_mask: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """
        Forward through readout with sparsity masking.
        
        Args:
            x: Reservoir spikes [batch, reservoir_size]
            mem: Output membrane potential [batch, output_dim]
            return_mask: Whether to return the sparsity mask
            
        Returns:
            Tuple of (spikes, membrane, synops, optional mask)
        """
        batch_size = x.shape[0]
        device = x.device
        
        if mem is None:
            mem = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Sparsity masking: zero out low activations
        # Vectorized: torch.where avoids explicit loops
        mask = (x.abs() > self.sparsity_threshold).float()
        x_masked = x * mask
        
        # Count active inputs (for SynOps)
        active_inputs = mask.sum()
        
        # Linear readout on masked input
        current = self.fc(x_masked)
        
        # LIF dynamics
        spikes, mem = self.lif(current, mem)
        
        # SynOps: active_inputs × output_dim
        synops = active_inputs * self.output_dim
        
        if return_mask:
            return spikes, mem, synops, mask
        return spikes, mem, synops, None


class CareBlock(nn.Module):
    """
    Complete CARE Block: The modular backbone.
    
    Architecture:
        Input -> Encoder -> Reservoir -> Readout -> Output Spikes
    
    Includes:
        - Rate/Latency encoding for static inputs
        - LSM-style recurrent reservoir
        - Sparsity-masked readout
        - STDP-enabled plasticity layers
        - Metrics tracking (sparsity, synops)
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (for plasticity layer)
        output_dim: Number of output classes
        reservoir_size: Reservoir neuron count
        num_steps: Simulation time steps
        encoding: Encoding type ('rate', 'latency', 'direct')
        beta: LIF membrane decay
        threshold: LIF spike threshold
        surrogate_slope: Surrogate gradient slope
        sparsity_threshold: Readout sparsity threshold
        eta_stdp: STDP learning rate
    """
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        reservoir_size: int = 512,
        num_steps: int = 25,
        encoding: str = "rate",
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
        sparsity_threshold: float = 0.1,
        eta_stdp: float = 0.001,
    ) -> None:
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reservoir_size = reservoir_size
        self.num_steps = num_steps
        self.encoding = encoding
        
        # Encoder
        if encoding == "rate":
            self.encoder = RateEncoder(num_steps=num_steps)
        elif encoding == "latency":
            self.encoder = LatencyEncoder(num_steps=num_steps)
        else:
            self.encoder = None  # Direct spike input
        
        # Optional plasticity layer (between input and reservoir)
        self.plasticity = PlasticityLayer(
            input_dim, hidden_dim,
            eta_stdp=eta_stdp,
        )
        
        # Hidden LIF
        self.hidden_lif = CARELIFNeuron(
            beta=beta,
            threshold=threshold,
            slope=surrogate_slope,
        )
        
        # Reservoir
        self.reservoir = Reservoir(
            input_dim=hidden_dim,
            reservoir_size=reservoir_size,
            beta=beta,
            threshold=threshold,
            surrogate_slope=surrogate_slope,
        )
        
        # Readout
        self.readout = ReadoutLayer(
            input_dim=reservoir_size,
            output_dim=output_dim,
            beta=beta,
            threshold=threshold,
            sparsity_threshold=sparsity_threshold,
            surrogate_slope=surrogate_slope,
        )
    
    def forward(
        self,
        x: Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Full forward pass over time.
        
        Vectorized where possible; time loop is unavoidable for recurrent dynamics.
        Internal operations are fully batched.
        
        Args:
            x: Input tensor
                - Static: [batch, features]
                - Temporal: [time, batch, features]
            num_steps: Override default time steps
            
        Returns:
            Tuple of:
                - Output spike counts [batch, output_dim]
                - Metrics dict with 'sparsity', 'synops', 'spike_record'
        """
        steps = num_steps or self.num_steps
        
        # Determine input type and encode if needed
        if x.dim() == 2:
            # Static input: encode to spike train
            if self.encoder is not None:
                self.encoder.num_steps = steps
                spike_train = self.encoder(x)  # [T, B, F]
            else:
                # Repeat static input across time
                spike_train = x.unsqueeze(0).expand(steps, -1, -1)
        elif x.dim() == 3:
            # Temporal input
            if x.shape[0] > x.shape[1]:
                # Already [T, B, F]
                spike_train = x
            else:
                # [B, T, F] -> transpose to [T, B, F]
                spike_train = x.transpose(0, 1).contiguous()
            steps = spike_train.shape[0]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
        
        batch_size = spike_train.shape[1]
        device = spike_train.device
        
        # Initialize states
        hidden_mem = torch.zeros(batch_size, self.hidden_dim, device=device)
        hidden_spikes = torch.zeros(batch_size, self.hidden_dim, device=device)
        res_mem = torch.zeros(batch_size, self.reservoir_size, device=device)
        res_spikes = torch.zeros(batch_size, self.reservoir_size, device=device)
        out_mem = torch.zeros(batch_size, self.output_dim, device=device)
        
        # Initialize plasticity traces
        trace_pre, trace_post = self.plasticity.init_traces(batch_size, device)
        
        # Accumulators
        output_spike_count = torch.zeros(batch_size, self.output_dim, device=device)
        total_synops = torch.tensor(0.0, device=device)
        all_spikes: List[Tensor] = []
        dead_neuron_count = torch.zeros(self.reservoir_size, device=device)
        
        # Time loop (unavoidable for recurrence, but internals are batched)
        for t in range(steps):
            input_t = spike_train[t]  # [B, F]
            
            # Plasticity layer
            hidden_current, trace_pre, trace_post = self.plasticity(
                input_t, hidden_spikes, trace_pre, trace_post
            )
            
            # Hidden LIF
            hidden_spikes, hidden_mem = self.hidden_lif(hidden_current, hidden_mem)
            
            # Reservoir
            res_spikes, res_mem, res_synops = self.reservoir(
                hidden_spikes, res_mem, res_spikes
            )
            
            # Readout with sparsity
            out_spikes, out_mem, read_synops, _ = self.readout(
                res_spikes, out_mem
            )
            
            # Accumulate
            output_spike_count += out_spikes
            total_synops += res_synops + read_synops
            all_spikes.append(res_spikes.detach())
            
            # Track dead neurons (never fired)
            dead_neuron_count += (res_spikes.sum(dim=0) == 0).float()
        
        # Compute sparsity rate: fraction of timesteps neurons didn't fire
        sparsity_rate = dead_neuron_count.mean() / steps
        
        # Stack spike record for analysis
        spike_record = torch.stack(all_spikes, dim=0)  # [T, B, reservoir]
        
        metrics = {
            'sparsity': sparsity_rate,
            'synops': total_synops,
            'spike_record': spike_record,
        }
        
        return output_spike_count, out_mem, metrics
    
    def apply_stdp_updates(self, blend_factor: float = 0.5) -> None:
        """Apply accumulated STDP updates to plasticity layers."""
        self.plasticity.apply_stdp_update(blend_factor)
