"""
Compile results from all experiments in the logs directory.
Walks through logs/, parses metrics.csv, and aggregates data into a single JSON file.
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

# Define paths
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("results")
OUTPUT_FILE = RESULTS_DIR / "all_experiments.json"

def parse_csv(file_path: Path) -> Dict[str, List[float]]:
    """Parse a Lightning metrics.csv file and extract relevant time series."""
    metrics = defaultdict(list)
    headers = []
    
    try:
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                return {} # Empty file
            
            # Map headers to indices
            header_map = {h: i for i, h in enumerate(headers)}
            
            # Key columns we care about
            keys_to_extract = [
                'epoch', 'step', 
                'val/accuracy', 'val/loss',
                'dead_neuron_ratio_epoch', 'dead_neuron_ratio_step',
                'train/accuracy_epoch', 'train/loss_epoch'
            ]
            
            # Filter for keys that actually exist
            valid_keys = [k for k in keys_to_extract if k in header_map]
            
            for row in reader:
                # Skip empty rows or rows with mismatched length
                if not row or len(row) != len(headers):
                    continue
                    
                for key in valid_keys:
                    val_str = row[header_map[key]]
                    if val_str and val_str.strip():
                        try:
                            val = float(val_str)
                            metrics[key].append(val)
                        except ValueError:
                            pass
                            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    # Consolidate epoch-level metrics (since they might be sparse)
    # We want to return cleaner lists where possible
    clean_metrics = {}
    
    # Process epoch-based metrics: take the last non-NaN value per epoch
    if 'epoch' in metrics:
        epochs = sorted(list(set(metrics['epoch'])))
        clean_metrics['epochs'] = epochs
        
        # Helper to group by epoch
        epoch_map = defaultdict(list)
        for i, ep in enumerate(metrics['epoch']):
            for key in valid_keys:
                if key != 'epoch' and i < len(metrics[key]):
                    epoch_map[ep].append((key, metrics[key][i])) # This logic is flawed due to separate lists
        
        # Better approach: Iterate rows again? No, let's just extract what we have.
        # The CSV lists in 'metrics' dictionary are NOT aligned if we just append non-empty values.
        # We need to re-read aligned rows.
        pass

    return reconstruct_aligned_metrics(file_path, valid_keys)

def reconstruct_aligned_metrics(file_path: Path, keys: List[str]) -> Dict[str, List[float]]:
    """Re-read CSV to get aligned epoch-end metrics."""
    data = defaultdict(list)
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    # Group rows by epoch to find the final validation metrics for each epoch
    epoch_rows = defaultdict(list)
    for row in rows:
        if 'epoch' in row and row['epoch']:
            epoch_rows[float(row['epoch'])].append(row)
            
    sorted_epochs = sorted(epoch_rows.keys())
    
    final_metrics = defaultdict(list)
    final_metrics['epoch'] = sorted_epochs
    
    for ep in sorted_epochs:
        # Get the last row for this epoch that has val/accuracy (usually the validation step)
        # Or merging strategy: take the last non-empty value for each column in this epoch
        
        ep_rows = epoch_rows[ep]
        merged_row = {}
        
        # Go through rows in order and update known values
        for r in ep_rows:
            for k in keys:
                if k in r and r[k] and r[k].strip():
                    merged_row[k] = float(r[k])
        
        # Key metrics to persist
        for k in keys:
            if k != 'epoch':
                final_metrics[k].append(merged_row.get(k, None)) # Use None for missing
                
    return dict(final_metrics)

def main():
    print(f"Scanning logs in {LOGS_DIR.absolute()}...")
    
    experiments = {}
    
    # Iterate over all directories
    for root, dirs, files in os.walk(LOGS_DIR):
        if "metrics.csv" in files:
            csv_path = Path(root) / "metrics.csv"
            
            # Determine experiment name from parent folder name
            # Structure: logs/exp_name/version_x/metrics.csv
            parent_dir = csv_path.parent
            grandparent_dir = parent_dir.parent
            
            exp_name = grandparent_dir.name
            version = parent_dir.name
            
            print(f"Found {exp_name}/{version}")
            
            # Parse
            data = parse_csv(csv_path)
            
            # Identify if this is a "best" run or just add to list
            # For now, store all, but maybe prioritize later version
            
            if exp_name not in experiments:
                experiments[exp_name] = {}
            
            experiments[exp_name][version] = data

    # Summarize best run for each experiment
    # Criteria: Longest run (most epochs)
    consolidated = {}
    
    for exp_name, versions in experiments.items():
        best_version = None
        max_epochs = -1
        
        for v, data in versions.items():
            if 'epoch' in data and len(data['epoch']) > max_epochs:
                max_epochs = len(data['epoch'])
                best_version = v
        
        if best_version:
            print(f"Selected {best_version} for {exp_name} ({max_epochs} epochs)")
            consolidated[exp_name] = versions[best_version]
            
            # Compute some summary stats
            if 'val/accuracy' in consolidated[exp_name]:
                valid_accs = [x for x in consolidated[exp_name]['val/accuracy'] if x is not None]
                if valid_accs:
                    consolidated[exp_name]['best_val_acc'] = max(valid_accs)
                    consolidated[exp_name]['final_val_acc'] = valid_accs[-1]
            
            if 'dead_neuron_ratio_epoch' in consolidated[exp_name]:
                dead_ratios = [x for x in consolidated[exp_name]['dead_neuron_ratio_epoch'] if x is not None]
                if dead_ratios:
                    consolidated[exp_name]['final_dead_ratio'] = dead_ratios[-1]

    # Save to JSON
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir(parents=True)
        
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(consolidated, f, indent=2)
        
    print(f"\nSaved compiled results to {OUTPUT_FILE}")
    print(f"Total experiments: {len(consolidated)}")
    print("Experiments found:", list(consolidated.keys()))

if __name__ == "__main__":
    main()
