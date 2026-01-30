"""
Generate comprehensive visualizations for the Dead Neuron Experiment.
Reads aggregated results from 'results/all_experiments.json'.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (16, 12),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 2.5,
})

RESULTS_FILE = Path("results/all_experiments.json")
OUTPUT_DIR = Path("results")

def load_data() -> Dict[str, Any]:
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def smooth_curve(values: List[float], window: int = 5) -> np.ndarray:
    """Apply simple moving average"""
    if not values or len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    pad_width = window // 2
    padded = np.pad(values, pad_width, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(values)]

def plot_lazarus_grid(data: Dict[str, Any]):
    """Generate a grid of Lazarus plots for different experiments."""
    
    # Identify pairs
    experiments = ['vgg8', 'depth6', 'depth12', 'depth18']
    titles = {
        'vgg8': 'Spiking VGG-8',
        'depth6': 'Spiking ResNet-6',
        'depth12': 'Spiking ResNet-12',
        'depth18': 'Spiking ResNet-18',
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, exp_base in enumerate(experiments):
        ax = axes[i]
        control_key = f"{exp_base}_control"
        hybrid_key = f"{exp_base}_hybrid"
        
        has_control = control_key in data
        has_hybrid = hybrid_key in data
        
        if not has_control and not has_hybrid:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            continue
            
        if has_control:
            # Check if we have valid dead neuron data
            d = data[control_key]
            if 'dead_neuron_ratio_epoch' in d and d['dead_neuron_ratio_epoch']:
                dead = np.array([x for x in d['dead_neuron_ratio_epoch'] if x is not None]) * 100
                epochs = d['epoch'][:len(dead)]
                # Fix: Ensure shapes match if data was sparse
                # epochs = np.arange(1, len(dead) + 1) # Force sequential epoch numbering if mismatched
                if len(dead) > 0:
                    ax.plot(epochs, dead, 'o--', color='#e74c3c', label='Control (Backprop)')
                    ax.fill_between(epochs, 0, dead, color='#e74c3c', alpha=0.1)
                    
                    # Store final for comparison
                    final_control = dead[-1]
                    ax.annotate(f'{final_control:.1f}%', xy=(epochs[-1], dead[-1]), 
                                xytext=(epochs[-1]-2, dead[-1]+5), color='#e74c3c', fontweight='bold')

        if has_hybrid:
            d = data[hybrid_key]
            if 'dead_neuron_ratio_epoch' in d and d['dead_neuron_ratio_epoch']:
                dead = np.array([x for x in d['dead_neuron_ratio_epoch'] if x is not None]) * 100
                epochs = d['epoch'][:len(dead)]
                if len(dead) > 0:
                    ax.plot(epochs, dead, 's-', color='#3498db', label='CARE Hybrid')
                    ax.fill_between(epochs, 0, dead, color='#3498db', alpha=0.1)
                    
                    final_hybrid = dead[-1]
                    ax.annotate(f'{final_hybrid:.1f}%', xy=(epochs[-1], dead[-1]), 
                                xytext=(epochs[-1]-2, dead[-1]+5), color='#3498db', fontweight='bold')
        
        ax.set_title(titles.get(exp_base, exp_base))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dead Neurons (%)')
        ax.set_ylim(0, 100) # Assuming percentage 0-100
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend()
        
    plt.tight_layout()
    plt.suptitle("The Lazarus Effect: Dead Neuron Recovery Across Architectures", y=1.02, fontsize=20, fontweight='bold')
    
    out_path = OUTPUT_DIR / "lazarus_grid.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")

def plot_depth_scaling(data: Dict[str, Any]):
    """Bar chart of final performance vs Depth."""
    depths = [6, 12, 18]
    
    final_acc = {'control': [], 'hybrid': []}
    final_dead = {'control': [], 'hybrid': []}
    
    valid_depths = []
    
    for d in depths:
        c_key = f"depth{d}_control"
        h_key = f"depth{d}_hybrid"
        
        if c_key in data and h_key in data:
            valid_depths.append(d)
            
            # Accuracy
            c_acc = data[c_key].get('final_val_acc', 0) or 0
            h_acc = data[h_key].get('final_val_acc', 0) or 0
            final_acc['control'].append(c_acc * 100)
            final_acc['hybrid'].append(h_acc * 100)
            
            # Dead Ratio
            c_dead = data[c_key].get('final_dead_ratio', 0) or 0
            h_dead = data[h_key].get('final_dead_ratio', 0) or 0
            final_dead['control'].append(c_dead * 100)
            final_dead['hybrid'].append(h_dead * 100)
            
    if not valid_depths:
        print("No paired depth data found for scaling plot.")
        return

    x = np.arange(len(valid_depths))
    width = 0.35
    
    # Plot 1: Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    rects1 = ax1.bar(x - width/2, final_acc['control'], width, label='Control', color='#e74c3c', alpha=0.8)
    rects2 = ax1.bar(x + width/2, final_acc['hybrid'], width, label='CARE', color='#3498db', alpha=0.8)
    
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Scaling with Depth')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'ResNet-{d}' for d in valid_depths])
    ax1.legend()
    ax1.set_ylim(0, 100) # Accuracy is usually high, maybe 60-100? Let's check ranges. 
    # Actually scaling to 0-100 is safe.
    
    # Add labels
    ax1.bar_label(rects1, padding=3, fmt='%.1f')
    ax1.bar_label(rects2, padding=3, fmt='%.1f')

    # Plot 2: Dead Neurons
    rects3 = ax2.bar(x - width/2, final_dead['control'], width, label='Control', color='#e74c3c', alpha=0.8)
    rects4 = ax2.bar(x + width/2, final_dead['hybrid'], width, label='CARE', color='#3498db', alpha=0.8)
    
    ax2.set_ylabel('Dead Neurons (%)')
    ax2.set_title('Dead Neuron Ratio Scaling with Depth')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'ResNet-{d}' for d in valid_depths])
    ax2.legend()
    ax2.set_ylim(0, 100)
    
    ax2.bar_label(rects3, padding=3, fmt='%.1f')
    ax2.bar_label(rects4, padding=3, fmt='%.1f')
    
    plt.tight_layout()
    out_path = OUTPUT_DIR / "depth_scaling.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved {out_path}")

def plot_recovery_speed(data: Dict[str, Any]):
    """Comparison of epoch to reach <10% dead neurons."""
    # This is a bit advanced, let's stick to the grid and bars first.
    pass

def main():
    if not RESULTS_FILE.exists():
        print(f"Results file not found: {RESULTS_FILE}")
        return
        
    data = load_data()
    print(f"Loaded data for: {list(data.keys())}")
    
    plot_lazarus_grid(data)
    plot_depth_scaling(data)
    print("Done generating plots.")

if __name__ == "__main__":
    main()
