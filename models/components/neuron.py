"""
CARE Neuron Components

Core neural building blocks with custom STDP autograd for combining
gradient descent (task objectives) with Hebbian learning (stability).

Author: CARE Research Team
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# =============================================================================
# Custom Autograd Function for STDP
# =============================================================================

class STDPFunction(torch.autograd.Function):
    """
    Custom Autograd Function that computes a standard linear forward pass
    while enabling STDP-based weight updates in the backward pass.
    
    The STDP update is accumulated separately and does NOT replace the 
    standard gradient—it augments it for bio-plausible learning.
    
    Trace-based STDP Rule:
        ΔW = η * (trace_pre @ spike_post.T - spike_pre @ trace_post.T)
    
    Where:
        - trace_pre: Exponentially decaying trace of pre-synaptic activity
        - trace_post: Exponentially decaying trace of post-synaptic activity
        - spike_pre/post: Current spike tensors (binary or soft)
    """
    
    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        trace_pre: Tensor,
        trace_post: Tensor,
        spike_post: Tensor,
        eta_stdp: float,
        a_plus: float,
        a_minus: float,
    ) -> Tensor:
        """
        Forward pass: Standard linear transformation W @ x + b
        
        Args:
            ctx: Autograd context for saving tensors
            input: Pre-synaptic input [batch, in_features]
            weight: Weight matrix [out_features, in_features]
            bias: Optional bias [out_features]
            trace_pre: Pre-synaptic trace [batch, in_features]
            trace_post: Post-synaptic trace [batch, out_features]
            spike_post: Post-synaptic spikes from previous step [batch, out_features]
            eta_stdp: STDP learning rate
            a_plus: LTP magnitude
            a_minus: LTD magnitude
            
        Returns:
            Linear output [batch, out_features]
        """
        # Save for backward
        ctx.save_for_backward(input, weight, bias, trace_pre, trace_post, spike_post)
        ctx.eta_stdp = eta_stdp
        ctx.a_plus = a_plus
        ctx.a_minus = a_minus
        
        # Standard linear forward
        output = input @ weight.t()
        if bias is not None:
            output = output + bias
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass: Compute standard gradients + accumulate STDP update.
        
        The STDP delta is stored in weight.stdp_delta (if it exists) for
        later application. This preserves the computational graph while
        allowing bio-plausible updates.
        """
        input, weight, bias, trace_pre, trace_post, spike_post = ctx.saved_tensors
        eta_stdp = ctx.eta_stdp
        a_plus = ctx.a_plus
        a_minus = ctx.a_minus
        
        # Standard gradients
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        
        if ctx.needs_input_grad[1]:
            # Standard gradient: dL/dW = grad_output.T @ input
            grad_weight = grad_output.t() @ input
            
            # STDP update (vectorized over batch, then averaged)
            # LTP: Pre-synaptic trace × Post-synaptic spike
            # LTD: Pre-synaptic spike × Post-synaptic trace
            batch_size = input.shape[0]
            
            # spike_post: [B, out], trace_pre: [B, in] -> LTP: [out, in]
            ltp = (spike_post.t() @ trace_pre) / batch_size  # [out, in]
            
            # input acts as "spike_pre", trace_post: [B, out] -> LTD: [out, in]
            ltd = (trace_post.t() @ input) / batch_size  # [out, in]
            
            # STDP delta (combined with standard gradient)
            stdp_delta = eta_stdp * (a_plus * ltp - a_minus * ltd)
            
            # Store STDP delta for external access (hybrid learning)
            if hasattr(weight, 'stdp_delta'):
                weight.stdp_delta = stdp_delta.detach()
            
            # Optionally blend STDP into gradient (configurable)
            # grad_weight = grad_weight + stdp_delta  # Uncomment to fuse
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        # Return gradients for all forward inputs (None for non-tensor args)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

# =============================================================================
# Plasticity Layer
# =============================================================================

class PlasticityLayer(nn.Module):
    """
    Linear layer with trace-based STDP plasticity.
    
    In forward pass: behaves like nn.Linear with spike trace updates.
    After backward: exposes apply_stdp_update() for Hebbian weight modification.
    
    This enables hybrid learning:
        1. Gradient descent optimizes task loss
        2. STDP promotes stable, sparse representations
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias
        tau_pre: Pre-synaptic trace time constant
        tau_post: Post-synaptic trace time constant
        eta_stdp: STDP learning rate
        a_plus: LTP magnitude
        a_minus: LTD magnitude
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        eta_stdp: float = 0.001,
        a_plus: float = 0.005,
        a_minus: float = 0.005,
    ) -> None:
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.eta_stdp = eta_stdp
        self.a_plus = a_plus
        self.a_minus = a_minus
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # STDP delta accumulator (non-trainable)
        self.register_buffer('stdp_delta', torch.zeros(out_features, in_features))
        
        # Trace decay factors (computed once)
        self.register_buffer('decay_pre', torch.tensor(1.0 - 1.0 / tau_pre))
        self.register_buffer('decay_post', torch.tensor(1.0 - 1.0 / tau_post))
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self) -> None:
        """Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def init_traces(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Initialize synaptic traces for a new sequence.
        
        Args:
            batch_size: Batch dimension
            device: Target device
            
        Returns:
            Tuple of (trace_pre, trace_post) initialized to zeros
        """
        trace_pre = torch.zeros(batch_size, self.in_features, device=device)
        trace_post = torch.zeros(batch_size, self.out_features, device=device)
        return trace_pre, trace_post
    
    def forward(
        self,
        input: Tensor,
        spike_post: Tensor,
        trace_pre: Optional[Tensor] = None,
        trace_post: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass with trace updates.
        
        Args:
            input: Pre-synaptic input/spikes [batch, in_features]
            spike_post: Post-synaptic spikes from previous timestep [batch, out_features]
            trace_pre: Pre-synaptic trace (optional, created if None)
            trace_post: Post-synaptic trace (optional, created if None)
            
        Returns:
            Tuple of (output, updated_trace_pre, updated_trace_post)
        """
        batch_size = input.shape[0]
        device = input.device
        
        # Initialize traces if not provided
        if trace_pre is None or trace_post is None:
            trace_pre, trace_post = self.init_traces(batch_size, device)
        
        # Update traces (exponential decay + spike contribution)
        # Vectorized: trace = decay * trace + spike
        trace_pre = self.decay_pre * trace_pre + input.detach()
        trace_post = self.decay_post * trace_post + spike_post.detach()
        
        # Attach stdp_delta buffer to weight for backward access
        self.weight.stdp_delta = self.stdp_delta
        
        # Forward through custom autograd function
        output = STDPFunction.apply(
            input,
            self.weight,
            self.bias,
            trace_pre,
            trace_post,
            spike_post,
            self.eta_stdp,
            self.a_plus,
            self.a_minus,
        )
        
        # Retrieve STDP delta if computed
        if hasattr(self.weight, 'stdp_delta'):
            self.stdp_delta = self.weight.stdp_delta
        
        return output, trace_pre, trace_post
    
    def apply_stdp_update(self, blend_factor: float = 0.5) -> None:
        """
        Apply accumulated STDP update to weights.
        
        Call this after optimizer.step() to blend Hebbian updates.
        
        Args:
            blend_factor: How much STDP to apply (0 = none, 1 = full)
        """
        with torch.no_grad():
            self.weight.add_(blend_factor * self.stdp_delta)
            # Reset accumulator
            self.stdp_delta.zero_()
    
    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'bias={self.bias is not None}, tau_pre={self.tau_pre}, tau_post={self.tau_post}'
        )


# =============================================================================
# CARE LIF Neuron Wrapper
# =============================================================================

class CARELIFNeuron(nn.Module):
    """
    Wrapper around snntorch.Leaky with CARE-specific configurations.
    
    Provides a consistent interface for LIF neurons with configurable:
        - Threshold
        - Membrane decay (beta)
        - Surrogate gradient (type and slope)
        - Reset mechanism
    
    Args:
        beta: Membrane potential decay factor (0-1)
        threshold: Spike threshold voltage
        spike_grad: Surrogate gradient type ('fast_sigmoid', 'atan', 'straight_through')
        slope: Surrogate gradient slope/steepness
        reset_mechanism: 'subtract' or 'zero'
    """
    
    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        spike_grad: str = "fast_sigmoid",
        slope: float = 25.0,
        reset_mechanism: str = "subtract",
    ) -> None:
        super().__init__()
        
        self.beta = beta
        self.threshold = threshold
        self.spike_grad_type = spike_grad
        self.slope = slope
        self.reset_mechanism = reset_mechanism
        
        # Lazy import to avoid forcing snntorch dependency at module load
        self._lif: Optional[nn.Module] = None
    
    def _build_lif(self) -> None:
        """Lazily construct the snntorch LIF neuron."""
        try:
            import snntorch as snn
            from snntorch import surrogate
        except ImportError as e:
            raise ImportError(
                "snntorch is required for CARELIFNeuron. "
                "Install via: pip install snntorch"
            ) from e
        
        # Select surrogate gradient
        grad_map = {
            "fast_sigmoid": surrogate.fast_sigmoid(slope=self.slope),
            "atan": surrogate.atan(alpha=self.slope),
            "straight_through": surrogate.straight_through_estimator(),
        }
        spike_grad = grad_map.get(self.spike_grad_type, surrogate.fast_sigmoid(slope=self.slope))
        
        # init_hidden=False allows us to pass membrane state externally
        self._lif = snn.Leaky(
            beta=self.beta,
            threshold=self.threshold,
            spike_grad=spike_grad,
            reset_mechanism=self.reset_mechanism,
            init_hidden=False,  # We manage state externally
        )
    
    def forward(
        self,
        input: Tensor,
        mem: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through LIF neuron.
        
        Args:
            input: Input current [batch, features]
            mem: Membrane potential (required)
            
        Returns:
            Tuple of (spikes, membrane_potential)
        """
        if self._lif is None:
            self._build_lif()
        
        if mem is None:
            # Initialize membrane to zeros if not provided
            mem = torch.zeros_like(input)
        
        return self._lif(input, mem)
    
    def init_mem(self, batch_size: int, features: int, device: torch.device) -> Tensor:
        """
        Initialize membrane potential.
        
        Args:
            batch_size: Batch dimension
            features: Feature dimension
            device: Target device
            
        Returns:
            Zero-initialized membrane potential
        """
        return torch.zeros(batch_size, features, device=device)

# =============================================================================
# Spiking Convolutional Layer Base Component
# =============================================================================

# Additional imports for deep learning components
import snntorch as snn
from snntorch import surrogate
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Any

# Initialization utilities
def sabotage_init(module: nn.Module, std: float = 0.01) -> None:
    """Initialize conv weights with abnormally low std to simulate dead neurons."""
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean=0.0, std=std)

def normal_init(module: nn.Module) -> None:
    """Standard Kaiming initialization."""
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

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
        return_current: bool = False
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        
        self.bn = nn.BatchNorm2d(out_channels) # Gradient Rescue: Pre-Activation BN
        
        # Gradient Rescue: Learnable Threshold
        learnable_thresh = nn.Parameter(torch.tensor(float(threshold), dtype=torch.float32))
        
        self.lif = snn.Leaky(
            beta=beta, threshold=learnable_thresh, spike_grad=surrogate.fast_sigmoid(slope=slope), init_hidden=False
        )
        self.lif.threshold = learnable_thresh
        
        self.out_channels = out_channels
        self._spike_record: List[Tensor] = []
        self.return_current = return_current

        self.register_buffer('activity_trace', torch.zeros(out_channels))
        self.register_buffer('max_membrane_trace', torch.zeros(out_channels)) 
        self.register_buffer('mean_membrane_trace', torch.zeros(out_channels)) # Track mean for SNR
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
        current = self.bn(self.conv(x))
        if self.return_current:
            spikes, mem = self.lif(current, mem)
            self._record_activity(spikes, mem)
            return spikes, mem, current
        
        spikes, mem = self.lif(current, mem)
        self._record_activity(spikes, mem)
        return spikes, mem

    def _record_activity(self, spikes: Tensor, mem: Tensor) -> None:
        self._spike_record.append(spikes.detach())
        with torch.no_grad():
            channel_activity = spikes.mean(dim=(0, 2, 3))
            self.activity_trace = (self.activity_decay * self.activity_trace + (1 - self.activity_decay) * channel_activity)
            
            # Track max and mean membrane potential for Dynamic SNR Gating
            channel_max_mem = mem.amax(dim=(0, 2, 3))
            channel_mean_mem = mem.abs().mean(dim=(0, 2, 3))
            
            self.max_membrane_trace = (self.activity_decay * self.max_membrane_trace + (1 - self.activity_decay) * channel_max_mem)
            self.mean_membrane_trace = (self.activity_decay * self.mean_membrane_trace + (1 - self.activity_decay) * channel_mean_mem)

    def apply_homeostatic_update(
        self,
        target_rate: float = 0.1,
        learning_rate: float = 0.01 
    ) -> None:
        with torch.no_grad():
            deviation = target_rate - self.activity_trace
            
            # Dynamic SNR Gating:
            # We compare Peak Activity (Max) vs Background Noise (Mean Abs).
            # If Peak > 2.0 * Mean, it implies structured "Spikes" (features) exist.
            # If Peak ~ Mean, it implies flat noise.
            
            # Avoid division by zero
            safe_mean = self.mean_membrane_trace + 1e-6
            snr = self.max_membrane_trace / safe_mean
            
            # Dynamic Gate: Soft sigmoid transition around SNR = 2.0
            # If SNR > 2.0, Gate -> 1.0 (Amplify)
            # If SNR < 2.0, Gate -> 0.0 (Suppress/Ignore)
            stimulation_gate = torch.sigmoid((snr - 2.0) * 5.0)
            
            # Also require absolute minimal activity to avoid amplifying pure silence (0/0)
            # e.g. Max must be > 0.01 (tiny, but non-zero)
            absolute_gate = torch.sigmoid((self.max_membrane_trace - 0.01) * 100.0)
            
            final_gate = stimulation_gate * absolute_gate
            
            gate = torch.where(deviation > 0, final_gate, torch.ones_like(final_gate))
            
            update = learning_rate * (deviation * gate).view(-1, 1, 1, 1)
            
            sign_mask = torch.sign(self.conv.weight)
            sign_mask[sign_mask == 0] = 1.0 
            
            self.conv.weight.add_(sign_mask * update)


# =============================================================================
# SEW Residual Block (Modern SOTA, used in my project)
# =============================================================================

class SEWResNetBlock(nn.Module):
    """
    Spike Element-Wise Residual Block.
    Reference: Frang et Al. 2021
    Architecture: x->Conv->LIF->Conv->LIF + x (Integer Spike Addition)
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: Optional[nn.Module] = None, beta: float = 0.9, threshold: float = 1.0, slope: float = 25.0) -> None:
        super().__init__()

        self.conv1 = CareLIFConv(in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            beta=beta, threshold=threshold, slope=slope
        )
        self.conv2 = CareLIFConv(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            beta=beta, threshold=threshold, slope=slope
        )
        self.downsample = downsample
        self.stride = stride
    
    def reset_spike_records(self) -> None:
        self.conv1.reset_spike_record()
        self.conv2.reset_spike_record()
    
    def get_spike_records(self) -> Dict[str, Tensor]:
        return {
            'conv1': self.conv1.get_spike_record(),
            'conv2': self.conv2.get_spike_record()
        }
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.conv1.apply_homeostatic_update(target_rate, learning_rate)
        self.conv2.apply_homeostatic_update(target_rate, learning_rate)
    
    def forward(
        self,
        x: Tensor,
        mem1: Tensor,
        mem2: Tensor,
        identity: Optional = None,
    ) -> Tuple[Tensor, Tensor]:
        
        if identity is None:
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
        
        out, mem1 = self.conv1(x, mem1)
        out, mem2 = self.conv2(out, mem2)

        out = out + identity
        return out, mem1, mem2

class MSResNetBlock(nn.Module):
    """
    Membrane-Shortcut (MS) Residual Block.
    Reference: Hu et al., 2018 / Wu et al., 2019
    
    Architecture: x -> Conv -> LIF -> Conv -> (Add x to Mem) -> LIF
    
    Pros: Output is always binary spikes.
    Cons: Gradient must pass through surrogate in the residual path, causing vanishing gradients.
    
    CARE Enhancement: conv2 uses SNR-gated homeostasis (matching CareLIFConv).
    """
    expansion = 1 
    
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
        
        # First layer is standard CareLIFConv (has full SNR gating)
        self.conv1 = CareLIFConv(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            beta=beta, threshold=threshold, slope=slope
        )
        
        # Second layer components separated for Membrane Shortcut
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        self.lif2 = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
        
        # SNR-gated homeostasis buffers for conv2 (matching CareLIFConv)
        self.register_buffer('activity_trace_2', torch.zeros(out_channels))
        self.register_buffer('max_membrane_trace_2', torch.zeros(out_channels))
        self.register_buffer('mean_membrane_trace_2', torch.zeros(out_channels))
        self.activity_decay = 0.99
        self._spike_record_2: List[Tensor] = []
        
        self.downsample = downsample
        self.stride = stride
    
    def reset_spike_records(self) -> None:
        self.conv1.reset_spike_record()
        self._spike_record_2 = []
    
    def get_spike_records(self) -> Dict:
        rec2 = torch.stack(self._spike_record_2, dim=0) if self._spike_record_2 else torch.zeros(1)
        return {
            'conv1': self.conv1.get_spike_record(),
            'conv2': rec2,
        }
    
    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.conv1.apply_homeostatic_update(target_rate, learning_rate)
        
        # SNR-gated update for conv2 (matches CareLIFConv.apply_homeostatic_update)
        with torch.no_grad():
            deviation = target_rate - self.activity_trace_2
            
            # Dynamic SNR Gating
            safe_mean = self.mean_membrane_trace_2 + 1e-6
            snr = self.max_membrane_trace_2 / safe_mean
            stimulation_gate = torch.sigmoid((snr - 2.0) * 5.0)
            absolute_gate = torch.sigmoid((self.max_membrane_trace_2 - 0.01) * 100.0)
            final_gate = stimulation_gate * absolute_gate
            
            gate = torch.where(deviation > 0, final_gate, torch.ones_like(final_gate))
            update = learning_rate * (deviation * gate).view(-1, 1, 1, 1)
            
            sign_mask = torch.sign(self.conv2.weight)
            sign_mask[sign_mask == 0] = 1.0
            self.conv2.weight.add_(sign_mask * update)
            
    def init_mem2(self, batch_size: int, height: int, width: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.conv2.out_channels, height, width, device=device)
    
    def forward(
        self,
        x: Tensor,
        mem1: Tensor,
        mem2: Tensor,
        identity: Optional = None,
    ) -> Tuple:
        
        if identity is None:
            identity = x
            if self.downsample is not None:
                identity = self.downsample(x)
        
        # 1. Conv1 -> LIF1
        spk1, mem1 = self.conv1(x, mem1)
        
        # 2. Conv2
        cur2 = self.conv2(spk1)
        
        # 3. Membrane Shortcut: Add identity to current
        total_input = cur2 + identity
        
        # 4. LIF2
        spk2, mem2 = self.lif2(total_input, mem2)
        
        # Record output spikes and track membrane for SNR gating
        self._spike_record_2.append(spk2.detach())
        with torch.no_grad():
            self.activity_trace_2 = (
                self.activity_decay * self.activity_trace_2 +
                (1 - self.activity_decay) * spk2.mean(dim=(0, 2, 3))
            )
            # Track max and mean membrane potential for Dynamic SNR Gating
            channel_max_mem = mem2.amax(dim=(0, 2, 3))
            channel_mean_mem = mem2.abs().mean(dim=(0, 2, 3))
            self.max_membrane_trace_2 = (
                self.activity_decay * self.max_membrane_trace_2 +
                (1 - self.activity_decay) * channel_max_mem
            )
            self.mean_membrane_trace_2 = (
                self.activity_decay * self.mean_membrane_trace_2 +
                (1 - self.activity_decay) * channel_mean_mem
            )
            
        return spk2, mem1, mem2


# =============================================================================
# CareResNet - Variable Depth SNN Backbone
# =============================================================================

class CareResNet(nn.Module):
    """
    Variable-depth ResNet backbone with architecture selection.
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
        block_type: str = 'sew',  # 'sew' or 'ms'
    ) -> None:
        super().__init__()
        
        self.num_steps = num_steps
        self.beta = beta
        self.threshold = threshold
        self.slope = slope
        
        # Config
        if depth not in [6, 18, 34, 50]: 
            depth = 18
        
        layers_cfg = {
            6:  [1, 1, 1, 1],
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3]
        }[depth]
        
        # Block Selection
        if block_type.lower() == 'sew':
            self.block_class = SEWResNetBlock
            print(" Using SEW-ResNet Blocks (Spike Addition)")
        elif block_type.lower() == 'ms':
            self.block_class = MSResNetBlock
            print(" Using MS-ResNet Blocks (Membrane Shortcut)")
        else:
            raise ValueError("block_type must be 'sew' or 'ms'")

        self.expansion = 1
        
        # Stem: Use small 3x3 stride-1 for CIFAR-sized inputs, 7x7 stride-2 for ImageNet
        self.cifar_mode = (in_channels <= 3 and base_channels <= 64)
        if base_channels <= 32:
            # CIFAR stem: preserve spatial resolution
            self.stem = CareLIFConv(
                in_channels, base_channels,
                kernel_size=3, stride=1, padding=1,
                beta=beta, threshold=threshold, slope=slope
            )
            self.stem_stride = 1
        else:
            # ImageNet stem: downsample aggressively
            self.stem = CareLIFConv(
                in_channels, base_channels,
                kernel_size=7, stride=2, padding=3,
                beta=beta, threshold=threshold, slope=slope
            )
            self.stem_stride = 2
        
        self.in_channels = base_channels
        
        # Layers
        self.layer1 = self._make_layer(self.block_class, base_channels, layers_cfg[0], stride=1)
        self.layer2 = self._make_layer(self.block_class, base_channels * 2, layers_cfg[1], stride=2)
        self.layer3 = self._make_layer(self.block_class, base_channels * 4, layers_cfg[2], stride=2)
        self.layer4 = self._make_layer(self.block_class, base_channels * 8, layers_cfg[3], stride=2)
        
        final_channels = base_channels * 8 * self.expansion
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, num_classes)
        
        self.lif_out = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=slope),
            init_hidden=False,
        )
    
    def _make_layer(self, block_class, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        
        blocks = []
        blocks.append(block_class(
            self.in_channels, out_channels, stride, downsample,
            self.beta, self.threshold, self.slope
        ))
        self.in_channels = out_channels * self.expansion
        for _ in range(1, num_blocks):
            blocks.append(block_class(
                self.in_channels, out_channels, 1, None,
                self.beta, self.threshold, self.slope
            ))
        return nn.ModuleList(blocks)

    def reset_spike_records(self) -> None:
        self.stem.reset_spike_record()
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.reset_spike_records()

    def get_spike_records(self) -> Dict:
        records = {'stem': self.stem.get_spike_record()}
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            for j, block in enumerate(layer):
                br = block.get_spike_records()
                for k, v in br.items():
                    records[f'layer{i+1}_block{j+1}_{k}'] = v
        return records

    def apply_homeostatic_updates(self, target_rate: float, learning_rate: float) -> None:
        self.stem.apply_homeostatic_update(target_rate, learning_rate)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.apply_homeostatic_updates(target_rate, learning_rate)

    def forward(self, x: Tensor) -> Tuple:
        batch_size = x.shape[0]
        device = x.device
        self.reset_spike_records()
        
        # After stem spatial dimensions
        if self.stem_stride == 2:
            h, w = x.shape[2] // 2, x.shape[3] // 2
        else:
            h, w = x.shape[2], x.shape[3]
        
        # Init states
        stem_mem = self.stem.init_state(batch_size, h, w, device)
        
        # Init layer states
        l_states = []
        curr_h, curr_w = h, w
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            layer_state = []
            for block in layer:
                if block.stride > 1:
                    curr_h, curr_w = curr_h // 2, curr_w // 2
                
                # Mem1 always same size as block input
                # Mem2 depends on block stride. 
                # If MSBlock, init_mem2 is custom. If SEW, it's CareLIFConv.
                if isinstance(block, MSResNetBlock):
                    m1 = block.conv1.init_state(batch_size, curr_h if block.stride==1 else curr_h*2, curr_w if block.stride==1 else curr_w*2, device)
                    m2 = block.init_mem2(batch_size, curr_h, curr_w, device)
                else:
                    m1 = block.conv1.init_state(batch_size, curr_h if block.stride==1 else curr_h*2, curr_w if block.stride==1 else curr_w*2, device)
                    m2 = block.conv2.init_state(batch_size, curr_h, curr_w, device)
                layer_state.append([m1, m2])
            l_states.append(layer_state)
            
        mem_out = torch.zeros(batch_size, self.fc.out_features, device=device)
        spike_count = torch.zeros(batch_size, self.fc.out_features, device=device)
        
        for t in range(self.num_steps):
            out, stem_mem = self.stem(x, stem_mem)
            
            # Layers
            for l_idx, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
                for b_idx, block in enumerate(layer):
                    mem1, mem2 = l_states[l_idx][b_idx]
                    out, mem1, mem2 = block(out, mem1, mem2)
                    l_states[l_idx][b_idx] = [mem1, mem2]
            
            # Head
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
    LightningModule for Dead Neuron Experiment.
    Supports:
        - SEW vs MS ResNet
        - CIFAR-10 vs CIFAR-100
        - Synaptic Scaling (Homeostasis)
    """
    
    def __init__(
        self,
        depth: int = 18,
        dataset_name: str = "cifar10", # 'cifar10' or 'cifar100'
        in_channels: int = 3, # CIFAR is RGB
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
        block_type: str = 'sew',
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        # Determine num_classes
        if dataset_name.lower() == "cifar100":
            self.num_classes = 100
        else:
            self.num_classes = 10
            
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
            num_classes=self.num_classes,
            base_channels=base_channels,
            num_steps=num_steps,
            beta=beta,
            threshold=threshold,
            slope=slope,
            block_type=block_type
        )
        
        # Apply initialization
        if init_method == "sabotage":
            self.apply(lambda m: sabotage_init(m, std=init_std))
            print(f" Applied std={init_std} to {block_type.upper()}-ResNet{depth}")
        else:
            self.apply(normal_init)
            print(f" Applied to {block_type.upper()}-ResNet{depth}")
    
    def get_spike_records(self) -> Dict:
        return self.network.get_spike_records()
    
    def forward(self, x: Tensor) -> Tuple:
        # x shape: -> need to broadcast for Time if input is static
        # But our network loop handles static x inside. 
        # If x is (DVS), we need to adapt CareResNet.
        # Assuming static image for standard CIFAR.
        return self.network(x)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        inputs, targets = batch
        spike_counts, _ = self(inputs)
        spike_rates = spike_counts / self.num_steps
        loss = F.cross_entropy(spike_rates, targets)
        
        preds = spike_counts.argmax(dim=-1)
        acc = (preds == targets).float().mean()
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def on_after_backward(self) -> None:
        """Apply plasticity updates after backward pass if enabled."""
        if self.use_plasticity:
            self.network.apply_homeostatic_updates(
                target_rate=self.target_rate,
                learning_rate=self.eta_stdp,
            )
            
    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict:
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
# Utility: Trace-Based Plasticity Hook (Alternative Implementation)
# =============================================================================

def create_stdp_hook(
    eta: float = 0.001,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
):
    """
    Factory for creating STDP backward hooks.
    
    Alternative to STDPFunction for simpler integration.
    Attach to any nn.Linear layer.
    
    Usage:
        layer = nn.Linear(64, 128)
        hook_state = create_stdp_hook()
        layer.register_full_backward_hook(hook_state['hook'])
    
    Args:
        eta: STDP learning rate
        tau_pre: Pre-synaptic trace decay
        tau_post: Post-synaptic trace decay
        
    Returns:
        Dict with 'hook' function and 'traces' state
    """
    state = {
        'trace_pre': None,
        'trace_post': None,
        'decay_pre': 1.0 - 1.0 / tau_pre,
        'decay_post': 1.0 - 1.0 / tau_post,
    }
    
    def hook(module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """Backward hook that accumulates STDP updates."""
        # This is a simplified hook version - full implementation
        # would track pre/post spikes and apply STDP rule
        pass  # Implement based on specific use case
    
    return {'hook': hook, 'state': state}
