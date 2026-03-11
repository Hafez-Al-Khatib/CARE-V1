import pandas as pd
from pathlib import Path
import sys

def main():
    results = {}
    base = Path('results/v3')

    experiments = [
        'cifar10_norm_control_v3',
        'cifar10_norm_care_v3',
        'cifar10_sab_control_v3',
        'cifar10_sab_care_v3',
    ]

    print('='*80)
    print(f"{'Experiment':<30} {'Best Acc %':<12} {'Final Dead %':<12} {'Epochs':<8}")
    print('='*80)

    for name in experiments:
        metrics_file = base / name / 'neuron_metrics.csv'
        if metrics_file.exists() and metrics_file.stat().st_size > 0:
            df = pd.read_csv(metrics_file)
            if len(df) > 0:
                best_acc = df['val_accuracy'].max() if 'val_accuracy' in df.columns else None
                final_dead = df['global_dead_ratio'].iloc[-1] if 'global_dead_ratio' in df.columns else None
                acc = f"{best_acc*100:.2f}" if best_acc is not None else 'N/A'
                dead = f"{final_dead*100:.2f}" if final_dead is not None else 'N/A'
                print(f"{name:<30} {acc:<12} {dead:<12} {len(df):<8}")
        else:
            print(f"{name:<30} MISSING")

    print('='*80)

if __name__ == '__main__':
    main()
