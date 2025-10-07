#!/usr/bin/env python
"""
Visualize NGAFID CoT test predictions alongside sensor data.

This script creates comprehensive plots showing:
1. Sensor data from the NGAFID test dataset
2. Model predictions (generated rationale)
3. Ground truth rationale
4. Side-by-side comparison for analysis

The key insight is that the test_predictions.jsonl file corresponds 1:1 with the 
NGAFID test dataset samples in the same order (no shuffling during evaluation).
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import pandas as pd
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from time_series_datasets.ngafid_cot.NGAFIDCoTQADataset import NGAFIDCoTQADataset
from time_series_datasets.ngafid_cot.ngafid_cot_loader import load_ngafid_cot_splits

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# Sensor labels in the order they appear in the dataset
SENSOR_LABELS_ORDERED = [
    "engine_temperature",  # CHT (Cylinder Head Temperature)
    "engine_performance",  # EGT (Exhaust Gas Temperature)
    "oil_parameters",      # OilT/OilP (Oil Temperature/Pressure)
    "fuel_system",         # Fuel qty/flow
    "electrical",          # Volt/Amp (Voltage/Amperage)
    "engine_condition",    # RPM (Revolutions Per Minute)
]

# Sensor display names with full descriptions
SENSOR_DISPLAY_NAMES = {
    "engine_temperature": "Cylinder Head Temperature (CHT)",
    "engine_performance": "Exhaust Gas Temperature (EGT)",
    "oil_parameters": "Oil Temperature/Pressure",
    "fuel_system": "Fuel System",
    "electrical": "Electrical (Voltage/Amperage)",
    "engine_condition": "Engine RPM"
}

def load_test_predictions(predictions_file: str) -> List[Dict[str, Any]]:
    """Load test predictions from JSONL file."""
    predictions = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions

def load_test_dataset():
    """Load the NGAFID test dataset - return both raw and processed versions."""
    from time_series_datasets.ngafid_cot.ngafid_cot_loader import load_ngafid_cot_splits
    train, val, test = load_ngafid_cot_splits()
    return test  # Return the raw test dataset

def extract_sensor_data_from_time_series_text(time_series_text: List[str]) -> Dict[str, List[float]]:
    """
    Extract sensor data from the time_series_text format used in predictions.
    
    The time_series_text contains strings like:
    "The following is the engine temperature sensor stream, it has mean 28.7031 and std 0.0000:"
    followed by the actual sensor data.
    """
    sensor_data = {}
    
    for i, text in enumerate(time_series_text):
        if i < len(SENSOR_LABELS_ORDERED):
            sensor_name = SENSOR_LABELS_ORDERED[i]
            
            # Check if this sensor has NaN values
            if "contains NaN (not available) values" in text:
                # For NaN sensors, we'll create a placeholder
                sensor_data[sensor_name] = [0.0] * 300  # Placeholder length
            else:
                # Extract mean and std from the text
                try:
                    # This is a simplified extraction - in practice, the actual sensor data
                    # would need to be reconstructed from the model's internal representation
                    # For now, we'll create synthetic data based on the statistics
                    if "mean" in text and "std" in text:
                        # Extract mean and std values
                        parts = text.split("mean")[1].split("std")
                        mean_str = parts[0].strip().rstrip(" and")
                        std_str = parts[1].strip().rstrip(":")
                        
                        mean_val = float(mean_str)
                        std_val = float(std_str)
                        
                        # Generate synthetic sensor data based on statistics
                        # In a real implementation, you'd need access to the actual sensor data
                        np.random.seed(42 + i)  # Reproducible synthetic data
                        synthetic_data = np.random.normal(mean_val, std_val, 300)
                        sensor_data[sensor_name] = synthetic_data.tolist()
                    else:
                        # Fallback: create placeholder data
                        sensor_data[sensor_name] = [0.0] * 300
                except (ValueError, IndexError):
                    # Fallback: create placeholder data
                    sensor_data[sensor_name] = [0.0] * 300
    
    return sensor_data

def wrap_text(text: str, max_width: int = 80) -> str:
    """Wrap text to fit within a specified width."""
    import textwrap
    return textwrap.fill(text, width=max_width)

def parse_rationale_sections(rationale_text: str) -> Dict[str, str]:
    """Parse the rationale text into State Summary, Recommended Action, and Expected Outcome sections."""
    sections = {
        "State Summary": "",
        "Recommended Action": "",
        "Expected Outcome": ""
    }
    
    # Split by section headers
    text = rationale_text.strip()
    
    # Look for section headers
    state_summary_start = text.find("State Summary:")
    recommended_action_start = text.find("Recommended Action:")
    expected_outcome_start = text.find("Expected Outcome:")
    
    if state_summary_start != -1:
        if recommended_action_start != -1:
            sections["State Summary"] = text[state_summary_start + len("State Summary:"):recommended_action_start].strip()
        else:
            sections["State Summary"] = text[state_summary_start + len("State Summary:"):].strip()
    
    if recommended_action_start != -1:
        if expected_outcome_start != -1:
            sections["Recommended Action"] = text[recommended_action_start + len("Recommended Action:"):expected_outcome_start].strip()
        else:
            sections["Recommended Action"] = text[recommended_action_start + len("Recommended Action:"):].strip()
    
    if expected_outcome_start != -1:
        sections["Expected Outcome"] = text[expected_outcome_start + len("Expected Outcome:"):].strip()
    
    return sections

def create_comprehensive_plot(sample_idx: int, dataset_sample: Dict[str, Any], 
                            prediction: Dict[str, Any], output_dir: str) -> str:
    """Create a comprehensive plot for a single sample."""
    
    # Extract sensor data from the dataset sample
    sensor_data = dataset_sample.get("sensor_data", {})
    
    # Get model prediction only
    generated_rationale = prediction.get("generated", "")
    
    # Parse the rationale into sections
    rationale_sections = parse_rationale_sections(generated_rationale)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout: 3 rows for sensors, 1 row for rationale sections
    gs = fig.add_gridspec(4, 3, height_ratios=[2, 2, 2, 2], hspace=0.3, wspace=0.3)
    
    # Plot sensor data in the first 3 rows
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, sensor_name in enumerate(SENSOR_LABELS_ORDERED):
        if i >= 6:  # Only plot first 6 sensors
            break
            
        row = i // 2
        col = i % 2
        
        ax = fig.add_subplot(gs[row, col])
        
        if sensor_name in sensor_data:
            sensor_values = sensor_data[sensor_name]
            
            # Convert to numpy array and handle any issues
            if isinstance(sensor_values, list):
                sensor_array = np.array(sensor_values, dtype=np.float64)
            else:
                sensor_array = np.array(sensor_values, dtype=np.float64)
            
            # Handle NaN values
            if np.isnan(sensor_array).any():
                # Create a masked array for NaN values
                valid_mask = ~np.isnan(sensor_array)
                if np.any(valid_mask):
                    # Plot valid data points
                    time_axis = np.arange(len(sensor_array))
                    ax.plot(time_axis[valid_mask], sensor_array[valid_mask], 
                           color=colors[i], linewidth=2, alpha=0.8)
                    # Mark NaN regions
                    nan_regions = ~valid_mask
                    if np.any(nan_regions):
                        ax.axvspan(np.min(time_axis[nan_regions]), 
                                  np.max(time_axis[nan_regions]), 
                                  alpha=0.3, color='red', label='NaN Region')
                else:
                    # All NaN - create placeholder
                    ax.text(0.5, 0.5, f'{sensor_name.replace("_", " ").title()}\n(No Data)', 
                           transform=ax.transAxes, ha='center', va='center', fontsize=12)
            else:
                # Normal data
                time_axis = np.arange(len(sensor_array))
                ax.plot(time_axis, sensor_array, color=colors[i], linewidth=2, alpha=0.8)
            
            # Calculate statistics
            if not np.isnan(sensor_array).all():
                mean_val = np.nanmean(sensor_array)
                std_val = np.nanstd(sensor_array)
                ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5, 
                          label=f'Mean: {mean_val:.2f}')
            
        else:
            # No data for this sensor
            ax.text(0.5, 0.5, f'{sensor_name.replace("_", " ").title()}\n(Not Available)', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
        
        # Formatting
        ax.set_title(f'{sensor_name.replace("_", " ").title()}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Sensor Reading')
        ax.grid(True, alpha=0.3)
        
        # Only add legend if there are artists to show
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    
    # Add three boxes for rationale sections
    section_colors = ['#e8f4fd', '#fff2e8', '#f0f8e8']  # Light blue, light orange, light green
    section_titles = ['State Summary', 'Recommended Action', 'Expected Outcome']
    
    for i, (section_name, section_text) in enumerate(rationale_sections.items()):
        ax_section = fig.add_subplot(gs[3, i])
        
        # Add colored background
        ax_section.set_facecolor(section_colors[i])
        
        # Add section title
        ax_section.text(0.05, 0.95, section_name.upper(), transform=ax_section.transAxes, 
                       fontweight='bold', fontsize=14, verticalalignment='top', color='#2c3e50')
        
        # Add section text with proper wrapping
        if section_text:
            # Wrap text to fit in the box
            wrapped_text = wrap_text(section_text, max_width=80)
            ax_section.text(0.05, 0.85, wrapped_text, transform=ax_section.transAxes, 
                           fontsize=11, verticalalignment='top', wrap=True, color='#34495e')
        else:
            ax_section.text(0.05, 0.85, 'No content available', transform=ax_section.transAxes, 
                           fontsize=11, verticalalignment='top', style='italic', color='#7f8c8d')
        
        ax_section.set_xlim(0, 1)
        ax_section.set_ylim(0, 1)
        ax_section.axis('off')
    
    # Main title
    fig.suptitle(f'NGAFID CoT Test Sample {sample_idx}: Sensor Data and Model Prediction', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    plot_filename = f"ngafid_cot_sample_{sample_idx:03d}_comprehensive.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def create_sensor_only_plot(sample_idx: int, dataset_sample: Dict[str, Any], 
                          output_dir: str) -> str:
    """Create a plot showing only the sensor data."""
    
    # Extract sensor data from the dataset sample
    sensor_data = dataset_sample.get("sensor_data", {})
    
    # Debug: print sensor data structure
    print(f"Sample {sample_idx} sensor data keys: {list(sensor_data.keys())}")
    for key, values in sensor_data.items():
        if isinstance(values, list):
            print(f"  {key}: {len(values)} values, first few: {values[:5]}")
        else:
            print(f"  {key}: {type(values)} - {values}")
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'NGAFID CoT Test Sample {sample_idx}: Sensor Data', 
                fontsize=16, fontweight='bold')
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    
    for i, sensor_name in enumerate(SENSOR_LABELS_ORDERED):
        if i >= 6:
            break
            
        ax = axes[i // 2, i % 2]
        
        if sensor_name in sensor_data:
            sensor_values = sensor_data[sensor_name]
            
            if isinstance(sensor_values, list):
                sensor_array = np.array(sensor_values, dtype=np.float64)
            else:
                sensor_array = np.array(sensor_values, dtype=np.float64)
            
            # Handle NaN values
            if np.isnan(sensor_array).any():
                valid_mask = ~np.isnan(sensor_array)
                if np.any(valid_mask):
                    time_axis = np.arange(len(sensor_array))
                    ax.plot(time_axis[valid_mask], sensor_array[valid_mask], 
                           color=colors[i], linewidth=2, alpha=0.8)
                    # Mark NaN regions
                    nan_regions = ~valid_mask
                    if np.any(nan_regions):
                        ax.axvspan(np.min(time_axis[nan_regions]), 
                                  np.max(time_axis[nan_regions]), 
                                  alpha=0.3, color='red')
                else:
                    ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
            else:
                time_axis = np.arange(len(sensor_array))
                ax.plot(time_axis, sensor_array, color=colors[i], linewidth=2, alpha=0.8)
                
                # Add statistics
                mean_val = np.mean(sensor_array)
                std_val = np.std(sensor_array)
                ax.axhline(y=mean_val, color=colors[i], linestyle='--', alpha=0.5)
                ax.text(0.02, 0.98, f'μ={mean_val:.2f}, σ={std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Not Available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
        
        ax.set_title(f'{sensor_name.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Sensor Reading')
        ax.grid(True, alpha=0.3)
        
        # Only add legend if there are artists to show
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"ngafid_cot_sample_{sample_idx:03d}_sensors.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def create_rationale_comparison_plot(sample_idx: int, prediction: Dict[str, Any], 
                                   output_dir: str) -> str:
    """Create a plot showing the model prediction rationale in three sections."""
    
    generated_rationale = prediction.get("generated", "")
    
    # Parse the rationale into sections
    rationale_sections = parse_rationale_sections(generated_rationale)
    
    # Create figure with three columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 12))
    fig.suptitle(f'NGAFID CoT Test Sample {sample_idx}: Model Prediction Analysis', 
                fontsize=16, fontweight='bold')
    
    # Section colors
    section_colors = ['#e8f4fd', '#fff2e8', '#f0f8e8']  # Light blue, light orange, light green
    
    for i, (section_name, section_text) in enumerate(rationale_sections.items()):
        ax = axes[i]
        
        # Add colored background
        ax.set_facecolor(section_colors[i])
        
        # Add section title
        ax.text(0.05, 0.95, section_name.upper(), transform=ax.transAxes, 
               fontweight='bold', fontsize=16, verticalalignment='top', color='#2c3e50')
        
        # Add section text with proper wrapping
        if section_text:
            # Wrap text to fit in the box
            wrapped_text = wrap_text(section_text, max_width=70)
            ax.text(0.05, 0.85, wrapped_text, transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', wrap=True, color='#34495e')
        else:
            ax.text(0.05, 0.85, 'No content available', transform=ax.transAxes, 
                   fontsize=12, verticalalignment='top', style='italic', color='#7f8c8d')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"ngafid_cot_sample_{sample_idx:03d}_rationale.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_filename

def main():
    """Main function to create NGAFID CoT prediction visualizations."""
    print("=== NGAFID CoT Test Predictions Visualization ===\n")
    
    # Configuration
    predictions_file = "/Users/planger/Development/OpenTSLM/results/Llama_3_2_1B/OpenTSLMSP/stage6_ngafid_cot/results/test_predictions.jsonl"
    output_dir = "ngafid_cot_test_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load test predictions
        print("Loading test predictions...")
        predictions = load_test_predictions(predictions_file)
        print(f"Loaded {len(predictions)} predictions")
        
        # Load test dataset
        print("Loading NGAFID test dataset...")
        test_dataset = load_test_dataset()
        print(f"Loaded test dataset with {len(test_dataset)} samples")
        
        # Verify that we have matching numbers
        if len(predictions) != len(test_dataset):
            print(f"⚠️  Warning: Mismatch between predictions ({len(predictions)}) and dataset ({len(test_dataset)})")
            min_samples = min(len(predictions), len(test_dataset))
            print(f"Will process first {min_samples} samples")
        else:
            min_samples = len(predictions)
        
        # Create visualizations for first few samples
        num_samples_to_plot = min(10, min_samples)  # Plot first 10 samples
        print(f"\nCreating visualizations for first {num_samples_to_plot} samples...")
        
        created_plots = []
        
        for i in range(num_samples_to_plot):
            print(f"Processing sample {i+1}/{num_samples_to_plot}...")
            
            # Get dataset sample and prediction
            dataset_sample = test_dataset[i]
            prediction = predictions[i]
            
            # Create comprehensive plot
            comprehensive_plot = create_comprehensive_plot(i, dataset_sample, prediction, output_dir)
            created_plots.append(comprehensive_plot)
            
            # Create sensor-only plot
            sensor_plot = create_sensor_only_plot(i, dataset_sample, output_dir)
            created_plots.append(sensor_plot)
            
            # Create rationale comparison plot
            rationale_plot = create_rationale_comparison_plot(i, prediction, output_dir)
            created_plots.append(rationale_plot)
        
        print(f"\n=== Visualization Complete ===")
        print(f"Created {len(created_plots)} plots in {output_dir}/")
        print(f"Plots created:")
        for plot in created_plots:
            print(f"  - {plot}")
        
        # Print summary statistics
        print(f"\n=== Dataset Summary ===")
        print(f"Total test samples: {len(test_dataset)}")
        print(f"Total predictions: {len(predictions)}")
        print(f"Samples visualized: {num_samples_to_plot}")
        
        # Show sample of predictions
        print(f"\n=== Sample Predictions ===")
        for i in range(min(3, len(predictions))):
            pred = predictions[i]
            generated = pred.get("generated", "")[:200] + "..." if len(pred.get("generated", "")) > 200 else pred.get("generated", "")
            print(f"Sample {i}: {generated}")
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()