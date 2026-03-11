
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_flexible_experiment import get_dataloaders

dataset = 'cifar10'
print(f"Testing dataset: {dataset}")
try:
    _, _, in_channels, num_classes = get_dataloaders(dataset, 64)
    print(f"in_channels: {in_channels}")
    print(f"num_classes: {num_classes}")
except Exception as e:
    print(f"Error: {e}")
