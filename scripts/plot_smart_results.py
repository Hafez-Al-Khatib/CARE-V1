import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_smart_results():
    results_dir = Path('results/sabotage_smart')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    try:
        control_df = pd.read_csv(results_dir / 'control/neuron_metrics.csv')
        control_df['Model'] = 'Control (SEW)'
    except Exception:
        control_df = pd.DataFrame()

    try:
        care_df = pd.read_csv(results_dir / 'care/neuron_metrics.csv')
        care_df['Model'] = 'CARE (MS)'
    except Exception:
        care_df = pd.DataFrame()

    if control_df.empty and care_df.empty:
        print("No data found.")
        return

    df = pd.concat([control_df, care_df])
    
    # ---------------------------------------------------------
    # Helper: Aggregate Layer Metrics to Global
    # ---------------------------------------------------------
    def aggregate_metric(dataframe, metric_suffix):
        cols = [c for c in dataframe.columns if c.endswith(metric_suffix)]
        if not cols: return None
        return dataframe[cols].mean(axis=1)

    # 1. Gradient Health (Dead vs Alive)
    # We only have this for CARE usually? Or both?
    # Both should have it if PhdTracker was used.
    
    plt.figure(figsize=(12, 6))
    
    for model_name, model_df in df.groupby('Model'):
        alive_grad = aggregate_metric(model_df, '_grad_alive_mean')
        dead_grad = aggregate_metric(model_df, '_grad_dead_mean')
        
        if alive_grad is not None:
            plt.plot(model_df['epoch'], alive_grad, label=f'{model_name} (Alive)', linestyle='-')
        if dead_grad is not None:
            plt.plot(model_df['epoch'], dead_grad, label=f'{model_name} (Dead)', linestyle='--')
            
    plt.title('Gradient Norms: Living vs Dead Neurons')
    plt.ylabel('Mean Gradient Norm')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'gradient_health.png', dpi=300)
    plt.close()

    # 2. Revival Impact (CARE Only)
    if not care_df.empty and 'revival_loss_impact' in care_df.columns:
        plt.figure(figsize=(10, 6))
        # Filter for non-null impact
        impact_df = care_df.dropna(subset=['revival_loss_impact'])
        
        if not impact_df.empty:
            colors = ['red' if x > 0 else 'green' for x in impact_df['revival_loss_impact']]
            plt.bar(impact_df['epoch'], impact_df['revival_loss_impact'], color=colors)
            plt.axhline(0, color='black', linewidth=0.8)
            plt.title('Revival Impact on Validation Loss (Negative = Good)')
            plt.ylabel('Change in Val Loss')
            plt.xlabel('Epoch')
            plt.tight_layout()
            plt.savefig(plots_dir / 'revival_impact.png', dpi=300)
        plt.close()

    # 3. Weight Kurtosis (Sparsity/Structure)
    plt.figure(figsize=(10, 6))
    for model_name, model_df in df.groupby('Model'):
        kurtosis = aggregate_metric(model_df, '_w_kurtosis')
        if kurtosis is not None:
            plt.plot(model_df['epoch'], kurtosis, label=model_name)
            
    plt.title('Weight Kurtosis (Structure Evolution)')
    plt.ylabel('Mean Kurtosis')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'weight_kurtosis.png', dpi=300)
    plt.close()
    
    print(f"Smart plots saved to {plots_dir}")

if __name__ == "__main__":
    plot_smart_results()
