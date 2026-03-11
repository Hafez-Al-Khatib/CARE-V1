"""Parse and display results from all completed V2 experiments."""
import pandas as pd
from pathlib import Path

results = {}
base = Path(r'c:\Users\hafez\Desktop\AUB Research\CARE\results\final_v2')

experiments = [
    'fmnist_norm_control_v2',
    'fmnist_norm_care_v2',
    'fmnist_sab_control_v2',
    'fmnist_sab_care_v2',
    'cifar10_norm_control_v2',
    'cifar10_norm_care_v2',
]

for name in experiments:
    exp_dir = base / name
    metrics_file = exp_dir / 'neuron_metrics.csv'
    if metrics_file.exists() and metrics_file.stat().st_size > 0:
        df = pd.read_csv(metrics_file)
        if len(df) > 0:
            best_acc = df['val_accuracy'].max() if 'val_accuracy' in df.columns else None
            final_dead = df['global_dead_ratio'].iloc[-1] if 'global_dead_ratio' in df.columns else None
            accs = df['val_accuracy'].tolist() if 'val_accuracy' in df.columns else []
            results[name] = {
                'best_acc': best_acc,
                'final_dead': final_dead,
                'epochs': len(df),
                'accs': accs
            }

print('='*70)
print(f"{'Experiment':<30} {'Best Acc %':<12} {'Final Dead %':<12} {'Epochs':<8}")
print('='*70)
for name in experiments:
    if name not in results:
        print(f"{name:<30} {'RUNNING/MISSING':<12}")
        continue
    r = results[name]
    acc = f"{r['best_acc']*100:.2f}" if r['best_acc'] is not None else 'N/A'
    dead = f"{r['final_dead']*100:.2f}" if r['final_dead'] is not None else 'N/A'
    print(f"{name:<30} {acc:<12} {dead:<12} {r['epochs']:<8}")
print('='*70)

# Per-epoch breakdown
print("\nPER-EPOCH VAL ACCURACY:")
for name in experiments:
    if name not in results: continue
    accs = results[name]['accs']
    acc_str = "  ".join(f"{a*100:.1f}%" for a in accs)
    print(f"  {name}: {acc_str}")

# Summary comparisons
print("\nKEY COMPARISONS (SEW Blocks | Plasticity OFF vs ON):")
pairs = [
    ('fmnist_norm_control_v2', 'fmnist_norm_care_v2', 'Fashion-MNIST | Normal Init'),
    ('fmnist_sab_control_v2',  'fmnist_sab_care_v2',  'Fashion-MNIST | Sabotage Init'),
    ('cifar10_norm_control_v2','cifar10_norm_care_v2', 'CIFAR-10 | Normal Init'),
]
for ctrl, care, label in pairs:
    if ctrl in results and care in results:
        ctrl_acc = results[ctrl]['best_acc']
        care_acc = results[care]['best_acc']
        if ctrl_acc and care_acc:
            delta = (care_acc - ctrl_acc) * 100
            sign = '+' if delta >= 0 else ''
            ctrl_dead = results[ctrl]['final_dead']
            care_dead = results[care]['final_dead']
            print(f"  {label}")
            print(f"    Control (Plasticity OFF):  {ctrl_acc*100:.2f}%  [{ctrl_dead*100:.1f}% dead]")
            print(f"    CARE    (Plasticity ON):   {care_acc*100:.2f}%  [{care_dead*100:.1f}% dead]")
            print(f"    Delta: {sign}{delta:.2f}%")
            print()

