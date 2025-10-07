#!/usr/bin/env python
"""
Visualize NGAFID CoT predictions alongside the underlying sensor data.

This script creates comprehensive plots showing:
1. The original sensor data from the NGAFID dataset
2. The model's generated predictions/rationale
3. The ground truth gold standard answers
4. Side-by-side comparison for analysis

The key insight is that the test_predictions.jsonl file contains predictions
in the same order as the test dataset, so we can iterate through both simultaneously.
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import textwrap

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from time_series_datasets.ngafid_cot.ngafid_cot_loader import load_ngafid_cot_splits

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Sensor labels and colors for consistent plotting
SENSOR_LABELS = [
    "engine_temperature",  # CHT combined proxy
    "engine_performance",  # EGT combined proxy  
    "oil_parameters",      # OilT/OilP
    "fuel_system",         # Fuel qty/flow
    "electrical",          # Volt/Amp
    "engine_condition",    # RPM
]

SENSOR_COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
SENSOR_DISPLAY_NAMES = [
    "Engine Temperature (CHT)",
    "Engine Performance (EGT)", 
    "Oil Parameters",
    "Fuel System",
    "Electrical",
    "Engine Condition (RPM)"
]

def load_predictions(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load predictions from the JSONL file."""
    predictions = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions

def parse_rationale(rationale_text: str) -> Dict[str, str]:
    """Parse the structured rationale text into sections."""
    sections = {
        'state_summary': '',
        'recommended_action': '',
        'expected_outcome': ''
    }
    
    # Try to extract sections from the text
    text = rationale_text.lower()
    
    # Look for section headers
    if 'state summary:' in text:
        start = rationale_text.lower().find('state summary:')
        end = rationale_text.lower().find('recommended action:')
        if end == -1:
            end = rationale_text.lower().find('expected outcome:')
        if end == -1:
            end = len(rationale_text)
        sections['state_summary'] = rationale_text[start+14:end].strip()
    
    if 'recommended action:' in text:
        start = rationale_text.lower().find('recommended action:')
        end = rationale_text.lower().find('expected outcome:')
        if end == -1:
            end = len(rationale_text)
        sections['recommended_action'] = rationale_text[start+18:end].strip()
    
    if 'expected outcome:' in text:
        start = rationale_text.lower().find('expected outcome:')
        sections['expected_outcome'] = rationale_text[start+16:].strip()
    
    return sections

def create_sensor_plots(sensor_data: Dict[str, List[float]], sample_idx: int, 
                       plots_dir: str) -> str:
    """Create sensor data visualization plots."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'NGAFID Sensor Data - Sample {sample_idx:03d}', 
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (sensor_name, display_name, color) in enumerate(zip(SENSOR_LABELS, SENSOR_DISPLAY_NAMES, SENSOR_COLORS)):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        if sensor_name in sensor_data:
            data = np.array(sensor_data[sensor_name])
            
            # Handle NaN values
            if np.isnan(data).any():
                # Create a masked array for plotting
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    time_axis = np.arange(len(data)) / 60  # Convert to minutes
                    ax.plot(time_axis, data, color=color, linewidth=2, alpha=0.8)
                    # Mark NaN regions
                    nan_mask = np.isnan(data)
                    if nan_mask.any():
                        ax.axvspan(time_axis[nan_mask][0] if nan_mask.any() else 0, 
                                 time_axis[nan_mask][-1] if nan_mask.any() else 0, 
                                 alpha=0.3, color='red', label='Missing Data')
                else:
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
            else:
                time_axis = np.arange(len(data)) / 60  # Convert to minutes
                ax.plot(time_axis, data, color=color, linewidth=2, alpha=0.8)
            
            # Calculate statistics
            if len(data) > 0 and not np.isnan(data).all():
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    min_val = np.min(valid_data)
                    max_val = np.max(valid_data)
                    
                    # Add statistics text box
                    stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nRange: {min_val:.2f} - {max_val:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8), fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No {display_name} data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
        
        ax.set_title(display_name, fontweight='bold')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Sensor Reading')
        ax.grid(True, alpha=0.3)
        if i == 0:  # Only show legend on first plot
            ax.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"ngafid_cot_sample_{sample_idx:03d}_sensors.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def create_rationale_plot(prediction: Dict[str, Any], sample_idx: int, 
                         plots_dir: str) -> str:
    """Create a plot showing the model's rationale and ground truth."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'NGAFID CoT Analysis - Sample {sample_idx:03d}', 
                fontsize=16, fontweight='bold')
    
    # Parse generated rationale
    generated_rationale = parse_rationale(prediction.get('generated', ''))
    gold_rationale = parse_rationale(prediction.get('gold', ''))
    
    # Display generated prediction
    ax1.set_title('Model Prediction (Generated)', fontweight='bold', fontsize=14)
    ax1.axis('off')
    
    y_pos = 0.95
    for section, content in generated_rationale.items():
        if content:
            section_title = section.replace('_', ' ').title()
            ax1.text(0.02, y_pos, f"{section_title}:", fontweight='bold', 
                    fontsize=12, transform=ax1.transAxes)
            y_pos -= 0.08
            
            # Wrap text for better display
            wrapped_text = textwrap.fill(content, width=80)
            ax1.text(0.05, y_pos, wrapped_text, fontsize=10, 
                    transform=ax1.transAxes, verticalalignment='top')
            y_pos -= 0.15
    
    # Display ground truth
    ax2.set_title('Ground Truth (Gold Standard)', fontweight='bold', fontsize=14)
    ax2.axis('off')
    
    y_pos = 0.95
    for section, content in gold_rationale.items():
        if content:
            section_title = section.replace('_', ' ').title()
            ax2.text(0.02, y_pos, f"{section_title}:", fontweight='bold', 
                    fontsize=12, transform=ax2.transAxes)
            y_pos -= 0.08
            
            # Wrap text for better display
            wrapped_text = textwrap.fill(content, width=80)
            ax2.text(0.05, y_pos, wrapped_text, fontsize=10, 
                    transform=ax2.transAxes, verticalalignment='top')
            y_pos -= 0.15
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"ngafid_cot_sample_{sample_idx:03d}_rationale.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def create_comprehensive_plot(sensor_data: Dict[str, List[float]], 
                            prediction: Dict[str, Any], sample_idx: int, 
                            plots_dir: str) -> str:
    """Create a comprehensive plot with both sensor data and rationale."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 2], hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle(f'NGAFID CoT Comprehensive Analysis - Sample {sample_idx:03d}', 
                fontsize=18, fontweight='bold')
    
    # Sensor plots (top 3 rows)
    sensor_axes = []
    for i in range(6):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        sensor_axes.append(ax)
    
    # Plot sensor data
    for i, (sensor_name, display_name, color) in enumerate(zip(SENSOR_LABELS, SENSOR_DISPLAY_NAMES, SENSOR_COLORS)):
        if i >= len(sensor_axes):
            break
            
        ax = sensor_axes[i]
        
        if sensor_name in sensor_data:
            data = np.array(sensor_data[sensor_name])
            
            # Handle NaN values
            if np.isnan(data).any():
                time_axis = np.arange(len(data)) / 60
                ax.plot(time_axis, data, color=color, linewidth=2, alpha=0.8)
                # Mark NaN regions
                nan_mask = np.isnan(data)
                if nan_mask.any():
                    ax.axvspan(time_axis[nan_mask][0] if nan_mask.any() else 0, 
                             time_axis[nan_mask][-1] if nan_mask.any() else 0, 
                             alpha=0.3, color='red', label='Missing Data')
            else:
                time_axis = np.arange(len(data)) / 60
                ax.plot(time_axis, data, color=color, linewidth=2, alpha=0.8)
            
            # Add statistics
            if len(data) > 0 and not np.isnan(data).all():
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    mean_val = np.mean(valid_data)
                    std_val = np.std(valid_data)
                    stats_text = f'μ={mean_val:.2f}\nσ={std_val:.2f}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', 
                           facecolor='white', alpha=0.8), fontsize=8)
        else:
            ax.text(0.5, 0.5, f'No {display_name} data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=10)
        
        ax.set_title(display_name, fontweight='bold', fontsize=10)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Reading')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    # Rationale plots (bottom row)
    ax_generated = fig.add_subplot(gs[3, 0])
    ax_gold = fig.add_subplot(gs[3, 1])
    ax_comparison = fig.add_subplot(gs[3, 2])
    
    # Parse rationales
    generated_rationale = parse_rationale(prediction.get('generated', ''))
    gold_rationale = parse_rationale(prediction.get('gold', ''))
    
    # Generated prediction
    ax_generated.set_title('Model Prediction', fontweight='bold', fontsize=12)
    ax_generated.axis('off')
    
    y_pos = 0.95
    for section, content in generated_rationale.items():
        if content:
            section_title = section.replace('_', ' ').title()
            ax_generated.text(0.02, y_pos, f"{section_title}:", fontweight='bold', 
                            fontsize=10, transform=ax_generated.transAxes)
            y_pos -= 0.1
            
            wrapped_text = textwrap.fill(content, width=60)
            ax_generated.text(0.05, y_pos, wrapped_text, fontsize=8, 
                            transform=ax_generated.transAxes, verticalalignment='top')
            y_pos -= 0.2
    
    # Ground truth
    ax_gold.set_title('Ground Truth', fontweight='bold', fontsize=12)
    ax_gold.axis('off')
    
    y_pos = 0.95
    for section, content in gold_rationale.items():
        if content:
            section_title = section.replace('_', ' ').title()
            ax_gold.text(0.02, y_pos, f"{section_title}:", fontweight='bold', 
                        fontsize=10, transform=ax_gold.transAxes)
            y_pos -= 0.1
            
            wrapped_text = textwrap.fill(content, width=60)
            ax_gold.text(0.05, y_pos, wrapped_text, fontsize=8, 
                        transform=ax_gold.transAxes, verticalalignment='top')
            y_pos -= 0.2
    
    # Comparison/Summary
    ax_comparison.set_title('Analysis Summary', fontweight='bold', fontsize=12)
    ax_comparison.axis('off')
    
    # Add some basic comparison metrics
    gen_text = prediction.get('generated', '')
    gold_text = prediction.get('gold', '')
    
    # Simple similarity metrics
    gen_words = set(gen_text.lower().split())
    gold_words = set(gold_text.lower().split())
    
    if len(gold_words) > 0:
        overlap = len(gen_words.intersection(gold_words)) / len(gold_words)
        ax_comparison.text(0.05, 0.9, f'Word Overlap: {overlap:.2%}', 
                          fontsize=10, transform=ax_comparison.transAxes)
    
    # Add key terms found in both
    common_terms = gen_words.intersection(gold_words)
    key_terms = [term for term in common_terms if len(term) > 4][:10]
    if key_terms:
        terms_text = 'Key Terms: ' + ', '.join(key_terms)
        wrapped_terms = textwrap.fill(terms_text, width=50)
        ax_comparison.text(0.05, 0.7, wrapped_terms, fontsize=8, 
                          transform=ax_comparison.transAxes)
    
    # Save the plot
    plot_filename = f"ngafid_cot_sample_{sample_idx:03d}_comprehensive.png"
    plot_path = os.path.join(plots_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def load_test_dataset_unshuffled():
    """Load test dataset in the same order as used during evaluation."""
    from time_series_datasets.ngafid_cot.ngafid_cot_loader import _ensure_dataset_available, _find_default_files
    from datasets import Dataset
    
    # Ensure dataset exists
    _ensure_dataset_available()
    
    # Get the raw records
    train_records, val_records, test_records = _find_default_files()
    
    # The issue is that _find_default_files() shuffles the test records
    # We need to get the original order. Let's try a different approach:
    # Load the raw data and create our own test split without shuffling
    
    import os
    from time_series_datasets.constants import RAW_DATA
    from time_series_datasets.ngafid_cot.ngafid_cot_loader import _read_jsonl, _coerce_sample
    
    NGAFID_COT_DIR = os.path.join(RAW_DATA, "ngafid")
    
    # Load raw records in original order
    flat_candidates = [
        os.path.join(NGAFID_COT_DIR, "ngafid_cot.jsonl"),
        os.path.join(NGAFID_COT_DIR, "dataset.jsonl"),
        os.path.join(NGAFID_COT_DIR, "ngafid_cot.json"),
        os.path.join(NGAFID_COT_DIR, "dataset.json"),
    ]
    
    records = []
    for path in flat_candidates:
        if os.path.exists(path) and path.endswith(".jsonl"):
            records = [_coerce_sample(r) for r in _read_jsonl(path)]
            break
        if os.path.exists(path) and path.endswith(".json"):
            with open(path, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                records = [_coerce_sample(obj)]
            elif isinstance(obj, list):
                records = [_coerce_sample(r) for r in obj]
            break
    
    if not records:
        # Fallback: gather many per-sample JSON files
        multi_json = []
        for root, _dirs, files in os.walk(NGAFID_COT_DIR):
            for f in files:
                if f.endswith(".json"):
                    p = os.path.join(root, f)
                    try:
                        obj = json.load(open(p, "r"))
                        if isinstance(obj, dict):
                            multi_json.append(_coerce_sample(obj))
                    except Exception:
                        continue
        records = multi_json
    
    # Create splits WITHOUT shuffling to preserve original order
    from collections import defaultdict
    label_groups = defaultdict(list)
    for record in records:
        label = record.get("target_class_raw", "unknown")
        label_groups[label].append(record)
    
    train, val, test = [], [], []
    
    for label, group_records in label_groups.items():
        # Calculate split sizes for this label
        n_total = len(group_records)
        n_train = int(0.9 * n_total)
        n_val = int(0.1 * n_total)
        
        # Split WITHOUT shuffling to preserve order
        train.extend(group_records[:n_train])
        val.extend(group_records[n_train:n_train + n_val])
        test.extend(group_records[n_train:n_train + n_val])  # Same as val for test
    
    return Dataset.from_list(test)

def main():
    """Main function to create NGAFID CoT prediction visualizations."""
    print("=== NGAFID CoT Prediction Visualization ===\n")
    
    # Paths
    predictions_path = "/Users/planger/Development/OpenTSLM/results/Llama_3_2_1B/OpenTSLMSP/stage6_ngafid_cot/results/test_predictions.jsonl"
    plots_dir = "ngafid_cot_prediction_plots"
    
    # Create plots directory
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        # Load predictions
        print("Loading predictions...")
        predictions = load_predictions(predictions_path)
        print(f"Loaded {len(predictions)} predictions")
        
        # Load test dataset (now unshuffled)
        print("Loading NGAFID CoT test dataset...")
        train_dataset, val_dataset, test_dataset = load_ngafid_cot_splits()
        print(f"Loaded {len(test_dataset)} test samples")
        
        # Verify we have matching counts
        if len(predictions) != len(test_dataset):
            print(f"Warning: Mismatch between predictions ({len(predictions)}) and test dataset ({len(test_dataset)})")
            min_count = min(len(predictions), len(test_dataset))
            predictions = predictions[:min_count]
            print(f"Using first {min_count} samples for visualization")
        
        # Create visualizations for first N samples
        num_samples = min(10, len(predictions))  # Limit to first 10 for initial testing
        print(f"Creating visualizations for {num_samples} samples...")
        
        created_plots = []
        
        for i in range(num_samples):
            print(f"Processing sample {i+1}/{num_samples}...")
            
            # Get prediction and corresponding dataset sample
            prediction = predictions[i]
            dataset_sample = test_dataset[i]
            
            # Extract sensor data from raw dataset
            sensor_data = dataset_sample.get('sensor_data', {})
            
            # Create different types of plots
            sensor_plot = create_sensor_plots(sensor_data, i, plots_dir)
            rationale_plot = create_rationale_plot(prediction, i, plots_dir)
            comprehensive_plot = create_comprehensive_plot(sensor_data, prediction, i, plots_dir)
            
            created_plots.extend([sensor_plot, rationale_plot, comprehensive_plot])
            
            # Print some basic info about this sample
            print(f"  Sample {i}: {len(sensor_data)} sensor types")
            print(f"  Generated rationale length: {len(prediction.get('generated', ''))}")
            print(f"  Gold rationale length: {len(prediction.get('gold', ''))}")
        
        print(f"\n=== Visualization Complete ===")
        print(f"Created {len(created_plots)} plots in {plots_dir}/")
        print(f"Plot types created:")
        print(f"  - Sensor data plots: {num_samples}")
        print(f"  - Rationale comparison plots: {num_samples}")
        print(f"  - Comprehensive analysis plots: {num_samples}")
        
        # Print sample of created files
        print(f"\nSample of created files:")
        for plot_file in created_plots[:6]:  # Show first 6
            print(f"  - {plot_file}")
        if len(created_plots) > 6:
            print(f"  ... and {len(created_plots) - 6} more")
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
