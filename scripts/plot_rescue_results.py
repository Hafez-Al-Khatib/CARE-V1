import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_rescue_results():
    base_dir = Path('results/sabotage_bio') # Phase 9c: Bio/Smart Rescue Results
    plots_dir = base_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load Data
    try:
        care_df = pd.read_csv(base_dir / 'care/neuron_metrics.csv')
        control_df = pd.read_csv(base_dir / 'control/neuron_metrics.csv')
    except FileNotFoundError:
        print("Waiting for results...")
        return

    # 1. Gradient Health (Dead Neurons)
    plt.figure(figsize=(12, 6))
    
    # CARE
    care_dead_grads = [c for c in care_df.columns if 'grad_dead_mean' in c and 'layer4' in c]
    if care_dead_grads:
        col = care_dead_grads[0] # Pick deep layer
        sns.lineplot(data=care_df, x='epoch', y=col, label=f'CARE {col}', marker='o')
        
    # Control
    control_dead_grads = [c for c in control_df.columns if 'grad_dead_mean' in c and 'layer4' in c]
    if control_dead_grads:
        col = control_dead_grads[0]
        sns.lineplot(data=control_df, x='epoch', y=col, label=f'Control {col}', marker='x', linestyle='--')
        
    plt.title('Gradient Rescue: Gradient Norms of Dead Neurons (Layer 4)')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(plots_dir / 'rescue_gradient_health.png')
    plt.close()
    
    # 2. Accuracy Comparison
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=care_df, x='epoch', y='val_accuracy', label='CARE (Rescue)', marker='o')
    sns.lineplot(data=control_df, x='epoch', y='val_accuracy', label='Control (Rescue)', marker='x')
    plt.title('Gradient Rescue: Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(plots_dir / 'rescue_accuracy.png')
    plt.close()
    
    print(f"Rescue plots saved to {plots_dir}")

if __name__ == "__main__":
    plot_rescue_results()
