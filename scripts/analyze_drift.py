"""
Analyze the 'Drift Velocity' Hypothesis.

Hypothesis: The rate at which dead neurons are recovered (Drift Velocity) 
correlates with the improvement in accuracy.

Metrics:
- Drift Velocity: Slope of (Dead Neuron Ratio vs Epoch) during the recovery phase (epochs 0-5).
- Accuracy Gain: (Hybrid Acc - Control Acc) or Slope of Hybrid Acc.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from typing import Dict, List, Tuple

RESULTS_FILE = Path("results/all_experiments.json")
OUTPUT_DIR = Path("results")

def load_data() -> Dict:
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def calculate_drift_velocity(dead_ratios: List[float], epochs: List[float], window: int = 5) -> float:
    """
    Calculate the rate of decrease in dead neurons over the first 'window' epochs.
    Returns a positive value indicating speed of recovery (reduction/epoch).
    """
    if len(dead_ratios) < 2:
        return 0.0
    
    # Take first 'window' points or all if fewer
    n = min(len(dead_ratios), window)
    y = np.array(dead_ratios[:n])
    x = np.array(epochs[:n])
    
    # Linear regression: dead = slope * epoch + intercept
    # We expect slope to be negative (reduction).
    # Drift Velocity = -slope
    slope, _, _, _, _ = linregress(x, y)
    return -slope

def calculate_accuracy_metrics(accs: List[float]) -> Dict[str, float]:
    if not accs:
        return {'final': 0, 'max': 0}
    return {
        'final': accs[-1],
        'max': max(accs)
    }

def main():
    if not RESULTS_FILE.exists():
        print("Results file not found. Run compile_all_results.py first.")
        return

    data = load_data()
    experiments = set([k.replace('_control', '').replace('_hybrid', '') for k in data.keys()])
    
    stats = []
    
    print(f"{'Experiment':<20} | {'Mode':<8} | {'Drift Vel':<10} | {'Final Acc':<10} | {'Dead %':<10}")
    print("-" * 75)
    
    for exp in experiments:
        # Check for both parts
        c_key = f"{exp}_control"
        h_key = f"{exp}_hybrid"
        
        # Analyze Hybrid (Recovery)
        if h_key in data:
            h_data = data[h_key]
            dead = h_data.get('dead_neuron_ratio_epoch', [])
            eps = h_data.get('epoch', [])
            acc = h_data.get('val/accuracy', [])
            
            # Filter None
            valid_idx = [i for i, v in enumerate(dead) if v is not None]
            dead_clean = [dead[i] for i in valid_idx]
            eps_clean = [eps[i] for i in valid_idx]
            
            # Accuracy might have different length or Nones
            acc_clean = [a for a in acc if a is not None]
            
            velocity = calculate_drift_velocity(dead_clean, eps_clean)
            acc_stats = calculate_accuracy_metrics(acc_clean)
            final_dead = dead_clean[-1] if dead_clean else 0
            
            stats.append({
                'experiment': exp,
                'mode': 'hybrid',
                'velocity': velocity,
                'final_acc': acc_stats['final'],
                'final_dead': final_dead
            })
            
            print(f"{exp:<20} | Hybrid   | {velocity:.4f}     | {acc_stats['final']:.4f}     | {final_dead:.2%}")

        # Analyze Control (Stagnation)
        if c_key in data:
            c_data = data[c_key]
            dead = c_data.get('dead_neuron_ratio_epoch', [])
            eps = c_data.get('epoch', [])
            acc = c_data.get('val/accuracy', [])
            
            valid_idx = [i for i, v in enumerate(dead) if v is not None]
            dead_clean = [dead[i] for i in valid_idx]
            eps_clean = [eps[i] for i in valid_idx]
            acc_clean = [a for a in acc if a is not None]
            
            velocity = calculate_drift_velocity(dead_clean, eps_clean)
            acc_stats = calculate_accuracy_metrics(acc_clean)
            final_dead = dead_clean[-1] if dead_clean else 0
            
            stats.append({
                'experiment': exp,
                'mode': 'control',
                'velocity': velocity,
                'final_acc': acc_stats['final'],
                'final_dead': final_dead
            })
            
            print(f"{exp:<20} | Control  | {velocity:.4f}     | {acc_stats['final']:.4f}     | {final_dead:.2%}")

    # Plot Scatter: Velocity vs Accuracy
    df = pd.DataFrame(stats)
    hybrids = df[df['mode'] == 'hybrid']
    
    plt.figure(figsize=(10, 6))
    
    # Plot Hybrid points
    plt.scatter(hybrids['velocity'], hybrids['final_acc'], s=100, c='#3498db', label='Hybrid (CARE)')
    
    # Add labels
    for _, row in hybrids.iterrows():
        plt.annotate(row['experiment'], (row['velocity'], row['final_acc']), 
                     xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    plt.xlabel('Drift Velocity (Dead Neuron Reduction / Epoch)', fontsize=12)
    plt.ylabel('Final Accuracy', fontsize=12)
    plt.title('Drift Velocity Hypothesis: Recovery Speed vs Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = OUTPUT_DIR / "drift_hypothesis.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"\nSaved drift analysis plot to {out_path}")

if __name__ == "__main__":
    main()
