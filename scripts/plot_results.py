import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_results():
    results_dir = Path('results/sabotage_comparison')
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Load data
    try:
        control_df = pd.read_csv(results_dir / 'control/neuron_metrics.csv')
        control_df['Model'] = 'Control (SEW)'
    except Exception as e:
        print(f"Could not load Control data: {e}")
        return

    try:
        care_df = pd.read_csv(results_dir / 'care/neuron_metrics.csv')
        care_df['Model'] = 'CARE (MS)'
    except Exception as e:
        print(f"Could not load CARE data: {e}")
        return

    # Combine
    df = pd.concat([control_df, care_df])
    
    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)
    
    # 1. Dead Neuron Ratio
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='global_dead_ratio', hue='Model', marker='o', linewidth=2.5)
    plt.title('Dead Neuron Ratio: Sabotage Initialization (std=0.01)')
    plt.ylabel('Dead Neuron Ratio')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(plots_dir / 'dead_neuron_ratio.png', dpi=300)
    plt.close()
    
    # 2. Validation Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='epoch', y='val_accuracy', hue='Model', marker='o', linewidth=2.5)
    plt.title('Validation Accuracy: Sabotage Initialization (std=0.01)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(plots_dir / 'accuracy.png', dpi=300)
    plt.close()
    
    # 3. Layer-wise Dead Ratio (Final Epoch)
    final_epoch = df['epoch'].max()
    final_df = df[df['epoch'] == final_epoch]
    
    # Extract layer columns
    layer_cols = [c for c in df.columns if 'dead_ratio' in c and 'global' not in c and 'stem' not in c]
    
    layer_data = []
    for _, row in final_df.iterrows():
        model = row['Model']
        for col in layer_cols:
            layer_name = col.replace('_dead_ratio', '').replace('layer', 'L').replace('_b', '.B').replace('_conv', '.C')
            layer_data.append({
                'Model': model,
                'Layer': layer_name,
                'Dead Ratio': row[col]
            })
            
    layer_df = pd.DataFrame(layer_data)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=layer_df, x='Layer', y='Dead Ratio', hue='Model')
    plt.title(f'Layer-wise Dead Neurons (Epoch {final_epoch})')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plots_dir / 'layer_dead_ratio.png', dpi=300)
    plt.close()

    print(f"Plots saved to {plots_dir}")

if __name__ == "__main__":
    plot_results()
