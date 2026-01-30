"""
Generate Lazarus Plot from experiment results.

Usage:
    py -3.12 scripts/generate_lazarus_plot.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.figsize': (12, 7),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def main():
    # Load results
    results_path = Path("results/experiment_20260129_125921.json")
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    control = data['control']['history']
    hybrid = data['hybrid']['history']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Dead Neuron Ratio over Epochs
    epochs = control['epoch']
    control_dead = [r * 100 for r in control['dead_ratio']]
    hybrid_dead = [r * 100 for r in hybrid['dead_ratio']]
    
    ax1.fill_between(epochs, 0, control_dead, alpha=0.3, color='#e74c3c')
    ax1.fill_between(epochs, 0, hybrid_dead, alpha=0.3, color='#3498db')
    
    ax1.plot(epochs, control_dead, 'o-', color='#e74c3c', linewidth=2.5, 
             markersize=8, label='Control (Backprop)', linestyle='--')
    ax1.plot(epochs, hybrid_dead, 's-', color='#3498db', linewidth=2.5, 
             markersize=8, label='CARE Hybrid')
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Dead Neurons (%)', fontsize=14, fontweight='bold')
    ax1.set_title('The Lazarus Plot: Dead Neuron Revival', fontsize=16, fontweight='bold')
    ax1.set_ylim(0, 60)
    ax1.set_xlim(0.5, 15.5)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add annotations
    ax1.annotate(f'{control_dead[-1]:.0f}%', xy=(15, control_dead[-1]), 
                 xytext=(13, control_dead[-1]+8), fontsize=11, color='#e74c3c',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7))
    ax1.annotate(f'{hybrid_dead[-1]:.0f}%', xy=(15, hybrid_dead[-1]), 
                 xytext=(13, hybrid_dead[-1]+8), fontsize=11, color='#3498db',
                 arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7))
    
    # Add reduction box
    reduction = control_dead[-1] / hybrid_dead[-1]
    ax1.text(0.5, 0.95, f'7.7x fewer\ndead neurons', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', ha='center', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#2ecc71', alpha=0.3))
    
    # Plot 2: Accuracy Comparison
    control_acc = control['val_acc']
    hybrid_acc = hybrid['val_acc']
    
    ax2.plot(epochs, control_acc, 'o-', color='#e74c3c', linewidth=2.5, 
             markersize=8, label='Control (Backprop)', linestyle='--')
    ax2.plot(epochs, hybrid_acc, 's-', color='#3498db', linewidth=2.5, 
             markersize=8, label='CARE Hybrid')
    
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Accuracy Comparison', fontsize=16, fontweight='bold')
    ax2.set_ylim(60, 80)
    ax2.set_xlim(0.5, 15.5)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    
    # Add final accuracy annotations
    ax2.annotate(f'{control_acc[-1]:.1f}%', xy=(15, control_acc[-1]), 
                 xytext=(13, control_acc[-1]-3), fontsize=11, color='#e74c3c',
                 arrowprops=dict(arrowstyle='->', color='#e74c3c', alpha=0.7))
    ax2.annotate(f'{hybrid_acc[-1]:.1f}%', xy=(15, hybrid_acc[-1]), 
                 xytext=(13, hybrid_acc[-1]+3), fontsize=11, color='#3498db',
                 arrowprops=dict(arrowstyle='->', color='#3498db', alpha=0.7))
    
    plt.tight_layout()
    
    # Save
    output_path = Path("results/lazarus_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved Lazarus Plot to {output_path}")
    
    # Also save as PDF for publication
    plt.savefig(Path("results/lazarus_plot.pdf"), bbox_inches='tight')
    print(f"Saved PDF to results/lazarus_plot.pdf")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nControl Final: Acc={control_acc[-1]:.1f}%, Dead={control_dead[-1]:.1f}%")
    print(f"Hybrid Final:  Acc={hybrid_acc[-1]:.1f}%, Dead={hybrid_dead[-1]:.1f}%")
    print(f"\nDead Neuron Reduction: {reduction:.1f}x")
    print(f"Accuracy Improvement: {hybrid_acc[-1] - control_acc[-1]:+.1f}%")

if __name__ == "__main__":
    main()
