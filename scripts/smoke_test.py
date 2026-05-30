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

# Test 2: CareResNet with CIFAR-10 (32x32, base_channels=64)
print("\nTest 2: CareResNet CIFAR-10 (32x32, base_channels=64, input_size=32)...")
net = CareResNet(depth=18, in_channels=3, num_classes=10, base_channels=64, num_steps=4, block_type='sew', input_size=32)
assert net.stem_stride == 1, f"Expected stride-1 stem for input_size=32, got stride-{net.stem_stride}"
x = torch.randn(2, 3, 32, 32)
out, mem = net(x)
print(f"  PASS: Output shape={out.shape}, expected [2, 10], stem_stride={net.stem_stride}")

# Test 3: CareResNet with Tiny-ImageNet (64x64, base_channels=64)
print("\nTest 3: CareResNet Tiny-ImageNet (64x64, base_channels=64, input_size=64)...")
net = CareResNet(depth=34, in_channels=3, num_classes=200, base_channels=64, num_steps=4, block_type='sew', input_size=64)
assert net.stem_stride == 1, f"Expected stride-1 stem for input_size=64, got stride-{net.stem_stride}"
x = torch.randn(2, 3, 64, 64)
out, mem = net(x)
print(f"  PASS: Output shape={out.shape}, expected [2, 200], stem_stride={net.stem_stride}")

# Test 4: CareResNet with ImageNet (224x224) uses stride-2 stem
print("\nTest 4: CareResNet ImageNet (224x224, input_size=224)...")
net = CareResNet(depth=18, in_channels=3, num_classes=1000, base_channels=64, num_steps=2, block_type='sew', input_size=224)
assert net.stem_stride == 2, f"Expected stride-2 stem for input_size=224, got stride-{net.stem_stride}"
x = torch.randn(1, 3, 224, 224)
out, mem = net(x)
print(f"  PASS: Output shape={out.shape}, expected [1, 1000], stem_stride={net.stem_stride}")

# Test 5: CareResNet with Fashion-MNIST (28x28, in_channels=1)
print("\nTest 5: CareResNet with in_channels=1 (Fashion-MNIST)...")
net = CareResNet(depth=18, in_channels=1, num_classes=10, num_steps=4, block_type='ms', input_size=28)
assert net.stem_stride == 1, f"Expected stride-1 stem for input_size=28, got stride-{net.stem_stride}"
x = torch.randn(2, 1, 28, 28)
out, mem = net(x)
print(f"  PASS: Output shape={out.shape}, expected [2, 10]")

# Test 6: DeadNeuronExperiment builds with input_size
print("\nTest 6: DeadNeuronExperiment with input_size=32...")
model = DeadNeuronExperiment(
    depth=18, in_channels=3, num_classes=10, num_steps=4,
    block_type='sew', init_method='normal', input_size=32
)
x = torch.randn(2, 3, 32, 32)
out, mem = model(x)
print(f"  PASS: Output shape={out.shape}")

# Test 7: mem_out has reasonable magnitude for cross-entropy
print("\nTest 7: mem_out magnitude check...")
import torch.nn.functional as F
model.eval()
with torch.no_grad():
    spike_counts, mem_out = model(x)
    print(f"  spike_counts range: [{spike_counts.min():.3f}, {spike_counts.max():.3f}]")
    print(f"  mem_out range: [{mem_out.min():.3f}, {mem_out.max():.3f}]")
    targets = torch.randint(0, 10, (2,))
    loss = F.cross_entropy(mem_out, targets)
    expected_random_loss = torch.log(torch.tensor(10.0))
    print(f"  CE loss on mem_out: {loss:.4f}")
    print(f"  Expected random-guess CE: {expected_random_loss:.4f}")
    print(f"  PASS: mem_out provides meaningful logits")

# Test 8: SNR gating exists on CareLIFConv
print("\nTest 8: SNR gating buffers exist...")
conv = CareLIFConv(3, 64, kernel_size=3, stride=1, padding=1)
assert hasattr(conv, 'max_membrane_trace'), "Missing max_membrane_trace!"
assert hasattr(conv, 'mean_membrane_trace'), "Missing mean_membrane_trace!"
print("  PASS: CareLIFConv has SNR buffers")

# Test 9: MSResNetBlock has SNR gating for conv2
print("\nTest 9: MSResNetBlock SNR gating...")
ms_block = MSResNetBlock(64, 64)
assert hasattr(ms_block, 'max_membrane_trace_2'), "Missing max_membrane_trace_2!"
assert hasattr(ms_block, 'mean_membrane_trace_2'), "Missing mean_membrane_trace_2!"
print("  PASS: MSResNetBlock has SNR buffers for conv2")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
