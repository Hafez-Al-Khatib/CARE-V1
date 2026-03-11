"""Quick smoke test to verify consolidated code works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Test 1: Import chain works
print("Test 1: Import chain...")
from systems.experiment import DeadNeuronExperiment, PhdGradeNeuronTracker
from models.components.neuron import CareResNet, CareLIFConv, SEWResNetBlock, MSResNetBlock
print("  PASS: All imports successful")

# Test 2: CareResNet with in_channels=3 (CIFAR-10)
print("\nTest 2: CareResNet with in_channels=3...")
net = CareResNet(depth=18, in_channels=3, num_classes=10, num_steps=4, block_type='sew')
x = torch.randn(2, 3, 32, 32)
out, mem = net(x)
print(f"  PASS: Output shape={out.shape}, expected [2, 10]")

# Test 3: CareResNet with in_channels=1 (Fashion-MNIST) 
print("\nTest 3: CareResNet with in_channels=1...")
net = CareResNet(depth=18, in_channels=1, num_classes=10, num_steps=4, block_type='ms')
x = torch.randn(2, 1, 28, 28)
out, mem = net(x)
print(f"  PASS: Output shape={out.shape}, expected [2, 10]")

# Test 4: DeadNeuronExperiment builds internally 
print("\nTest 4: DeadNeuronExperiment internal build...")
model = DeadNeuronExperiment(
    depth=18, in_channels=3, num_classes=10, num_steps=4,
    block_type='sew', init_method='normal'
)
x = torch.randn(2, 3, 32, 32)
out, mem = model(x)
print(f"  PASS: Output shape={out.shape}")

# Test 5: SNR gating exists on CareLIFConv
print("\nTest 5: SNR gating buffers exist...")
conv = CareLIFConv(3, 64, kernel_size=3, stride=1, padding=1)
assert hasattr(conv, 'max_membrane_trace'), "Missing max_membrane_trace!"
assert hasattr(conv, 'mean_membrane_trace'), "Missing mean_membrane_trace!"
print("  PASS: CareLIFConv has SNR buffers")

# Test 6: MSResNetBlock has SNR gating for conv2
print("\nTest 6: MSResNetBlock SNR gating...")
ms_block = MSResNetBlock(64, 64)
assert hasattr(ms_block, 'max_membrane_trace_2'), "Missing max_membrane_trace_2!"
assert hasattr(ms_block, 'mean_membrane_trace_2'), "Missing mean_membrane_trace_2!"
print("  PASS: MSResNetBlock has SNR buffers for conv2")

print("\n" + "="*60)
print("ALL SMOKE TESTS PASSED")
print("="*60)
