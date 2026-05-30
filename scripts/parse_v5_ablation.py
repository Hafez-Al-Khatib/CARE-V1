import os
import glob
import ast
import re

base_dir = "/media/samsung8T/Hafez/CARE-V1/results/v5_neurips_ablation"
run_dirs = glob.glob(os.path.join(base_dir, "*"))

results = []

for d in run_dirs:
    if not os.path.isdir(d):
        continue
    exp_name = os.path.basename(d)
    log_file = os.path.join(d, "run.log")
    
    if not os.path.exists(log_file):
        results.append((exp_name, "N/A", "N/A"))
        continue
        
    acc = "N/A"
    dead = "N/A"
    
    with open(log_file, "r") as f:
        for line in f:
            if "Training complete. Summary:" in line:
                try:
                    summary_str = line.split("Summary:")[1].strip()
                    summary_dict = ast.literal_eval(summary_str)
                    acc = f"{summary_dict.get('best_accuracy', 0)*100:.2f}%"
                    dead = f"{summary_dict.get('final_dead_ratio', 0)*100:.1f}%"
                except Exception as e:
                    acc = "Error"
                    dead = "Error"
    
    results.append((exp_name, acc, dead))

# Sort logically
results.sort(key=lambda x: x[0])

print("| Experiment Name | Best Accuracy | Final Dead Neurons |")
print("|----------------|---------------|--------------------|")
for exp_name, acc, dead in results:
    print(f"| `{exp_name}` | {acc} | {dead} |")
