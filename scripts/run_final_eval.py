import sys
import os
import subprocess
from pathlib import Path
import time
import pandas as pd

# Configurations
EXPERIMENTS = [
    # 1. Fashion-MNIST Normal Init (Baseline)
    {'dataset': 'fashion_mnist', 'depth': 18, 'init': 'normal', 'use_plasticity': False, 'name': 'fmnist_norm_control_v2'},
    {'dataset': 'fashion_mnist', 'depth': 18, 'init': 'normal', 'use_plasticity': True,  'name': 'fmnist_norm_care_v2'},
    
    # 2. Fashion-MNIST Sabotage Init (Robustness)
    {'dataset': 'fashion_mnist', 'depth': 18, 'init': 'sabotage', 'use_plasticity': False, 'name': 'fmnist_sab_control_v2'},
    {'dataset': 'fashion_mnist', 'depth': 18, 'init': 'sabotage', 'use_plasticity': True,  'name': 'fmnist_sab_care_v2'},
    
    # 3. CIFAR-10 Normal Init (Generalization)
    {'dataset': 'cifar10', 'depth': 18, 'init': 'normal', 'use_plasticity': False, 'name': 'cifar10_norm_control_v2'},
    {'dataset': 'cifar10', 'depth': 18, 'init': 'normal', 'use_plasticity': True,  'name': 'cifar10_norm_care_v2'},
]

def run_cmd(cmd, log_file):
    print(f"Running: {cmd} > {log_file}")
    with open(log_file, 'w') as f:
        subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)

def main():
    print("Starting Final Evaluations V2...")
    Path("results/final_v2").mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for exp in EXPERIMENTS:
        print(f"\n>>> Starting Experiment: {exp['name']}")
        start_time = time.time()
        
        plasticity_flag = "" if exp['use_plasticity'] else "--no_plasticity"
        
        cmd = f"py -3.12 -u scripts/run_flexible_experiment.py --dataset {exp['dataset']} --depth {exp['depth']} --init {exp['init']} --block sew --name {exp['name']} --batch_size 64 --time_steps 4 --epochs 30 --eta_stdp 0.001 {plasticity_flag}"
        
        log_path = Path("results/final_v2") / f"{exp['name']}.log"
        run_cmd(cmd, log_path)
        
        duration = time.time() - start_time
        print(f"<<< Finished {exp['name']} in {duration:.1f}s")
        
        # Parse log for best accuracy
        acc = 0.0
        try:
            with open(log_path, 'r') as f:
                content = f.read()
                pass
        except:
            pass
            
if __name__ == "__main__":
    main()
