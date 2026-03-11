import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base = Path('results/final_v2')

care_path = base / 'fmnist_sab_care_v2' / 'neuron_metrics.csv'
ctrl_path = base / 'fmnist_sab_control_v2' / 'neuron_metrics.csv'

if care_path.exists() and ctrl_path.exists():
    df_care = pd.read_csv(care_path)
    df_ctrl = pd.read_csv(ctrl_path)
    
    epochs = df_care['epoch'] + 1
    
    # 1. Revival Events Plot
    plt.figure(figsize=(10, 5))
    plt.bar(epochs - 0.2, df_ctrl['revival_event_count'].fillna(0), width=0.4, label='Control (Plasticity OFF)', color='#e74c3c', alpha=0.7)
    plt.bar(epochs + 0.2, df_care['revival_event_count'].fillna(0), width=0.4, label='CARE (Plasticity ON)', color='#2ecc71', alpha=0.9)
    
    plt.title('Targeted Interventions: SNR-Gated Revival Events (Sabotage Init)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Number of Neurons Revived', fontsize=12)
    plt.xticks(epochs[::2])
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/final_v2_revival_events.png', dpi=300)
    print("Saved final_v2_revival_events.png")
    
    # 2. Kurtosis Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, df_ctrl['stem_w_kurtosis'], label='Control (Plasticity OFF)', color='#e74c3c', marker='o', linestyle='--', linewidth=2)
    plt.plot(epochs, df_care['stem_w_kurtosis'], label='CARE (Plasticity ON)', color='#2ecc71', marker='s', linewidth=2)
    
    plt.title('Structural Adaptation: Stem Weight Kurtosis (Sabotage Init)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Kurtosis (Heavy-Tails)', fontsize=12)
    plt.xticks(epochs[::2])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/final_v2_kurtosis.png', dpi=300)
    print("Saved final_v2_kurtosis.png")
else:
    print("Data files not found.")
