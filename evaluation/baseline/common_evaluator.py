# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import json
import os
import io
import re
import sys
import base64
from typing import Type, Callable, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.pipelines import pipeline
import matplotlib.pyplot as plt
from time import sleep

from opentslm.logger import get_logger

from .openai_pipeline import OpenAIPipeline


class CommonEvaluator:
    """
    A common evaluation framework for testing LLMs on time series datasets.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            device: Device to use for inference ('cuda', 'mps', 'cpu', or None for auto)
        """
        self.device = device or self._get_best_device()

        if self.device == "mps":
            print(
                "⚠️ Warning!! MPS is available but not recommended for evaluation. Many LLMs do not produce reasonable output!"
            )
            print(
                "⚠️ Warning!! MPS is available but not recommended for evaluation. Many LLMs do not produce reasonable output!"
            )
            print("⚠️ Better use CPU or CUDA for evaluation.")
            sleep(10)

    def _get_best_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def load_model(self, model_name: str, **pipeline_kwargs) -> pipeline:
        """
        Load a model using transformers pipeline or OpenAI API.
        """
        self.current_model_name = (
            model_name  # Track the current model name for formatter selection
        )
        if model_name.startswith("openai-"):
            # Use OpenAI API
            openai_model = model_name.replace("openai-", "")
            return OpenAIPipeline(model_name=openai_model, **pipeline_kwargs)
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        # Default pipeline arguments
        default_kwargs = {
            "task": "text-generation",
            "device": self.device,
            "temperature": 0.1,
        }
        default_kwargs.update(pipeline_kwargs)
        pipe = pipeline(model=model_name, **default_kwargs)
        print(f"Model loaded successfully: {model_name}")
        return pipe

    def load_dataset(
        self,
        dataset_class: Type[Dataset],
        split: str = "test",
        format_sample_str: bool = True,
        max_samples: Optional[int] = None,
        **dataset_kwargs,
    ) -> Dataset:
        """
        Load a dataset with proper formatting.
        """
        print(f"Loading dataset: {dataset_class.__name__}")

        # Import the gruver formatters
        from .gruver_llmtime_tokenizer import gpt_formatter, llama_formatter

        # Choose formatter based on model type
        model_name = getattr(self, "current_model_name", None)
        if model_name is None and "model_name" in dataset_kwargs:
            model_name = dataset_kwargs["model_name"]
        if model_name is not None:
            if model_name.startswith("openai-") or "gpt" in model_name.lower():
                formatter = gpt_formatter
                print(f"Using GPT formatter for model: {model_name}")
            elif "llama" in model_name.lower():
                formatter = llama_formatter
                print(f"Using Llama formatter for model: {model_name}")
            else:
                print(f"Defaulting to Llama formatter for model: {model_name}")

                formatter = llama_formatter
        else:
            formatter = llama_formatter

        # Default dataset arguments
        default_kwargs = {
            "split": split,
            "EOS_TOKEN": "",
            "format_sample_str": format_sample_str,
            "time_series_format_function": formatter,
        }

        # Add max_samples if provided
        if max_samples is not None:
            default_kwargs["max_samples"] = max_samples

        # Update with provided kwargs
        default_kwargs.update(dataset_kwargs)

        dataset = dataset_class(**default_kwargs)
        print(f"Loaded {len(dataset)} {split} samples")

        return dataset

    def evaluate_model_on_dataset(
        self,
        model_name: str,
        dataset_class: Type[Dataset],
        evaluation_function: Callable[[str, str], Dict[str, Any]],
        max_samples: Optional[int] = None,
        use_plot: bool = False,
        **pipeline_kwargs,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a dataset using a custom evaluation function.

        Args:
            model_name: Name of the model to evaluate
            dataset_class: Dataset class to use
            evaluation_function: Function that takes (ground_truth, prediction) and returns metrics
            max_samples: Maximum number of samples to evaluate (None for all)
            **pipeline_kwargs: Additional arguments for model pipeline

        Returns:
            Dictionary containing evaluation results
        """
        print(
            f"Starting evaluation with model {model_name} on dataset {dataset_class.__name__}"
        )
        print("=" * 60)

        # Load model
        pipe = self.load_model(model_name, **pipeline_kwargs)

        # Load dataset
        dataset = self.load_dataset(dataset_class, max_samples=max_samples)

        # Check for existing results to resume from
        existing_count = self._get_existing_results_count(
            model_name, dataset_class.__name__
        )
        start_idx = existing_count

        # Limit samples if specified
        if max_samples is not None:
            dataset_size = min(len(dataset), max_samples)
            print(f"Processing samples {start_idx} to {dataset_size}...")
        else:
            dataset_size = len(dataset)
            print(f"Processing samples {start_idx} to {dataset_size}...")

        if start_idx >= dataset_size:
            print(f"✅ All {dataset_size} samples already processed!")
            return self._consolidate_jsonl_results(model_name, dataset_class.__name__)

        # Initialize tracking
        total_samples = dataset_size
        successful_inferences = existing_count  # Start with existing count
        all_metrics = []
        results = []
        first_error_printed = False  # Track if we've printed the first error

        print("\nRunning inference...")
        print("=" * 80)

        # Get max_new_tokens for generation (default 1000)
        max_new_tokens = pipeline_kwargs.pop("max_new_tokens", 1000)

        # Load existing metrics if resuming
        if start_idx > 0:
            print(f"📂 Loading existing results from {start_idx} completed samples...")
            jsonl_file = self._get_jsonl_file_path(model_name, dataset_class.__name__)
            if os.path.exists(jsonl_file):
                with open(jsonl_file, "r") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line.strip())
                            all_metrics.append(result["metrics"])
                            results.append(result)

        # Process each sample starting from where we left off
        for idx in tqdm(range(start_idx, dataset_size), desc="Processing samples"):
            try:
                sample = dataset[idx]
                plot_data = None

                # Clean up prompt for TSQADataset (if needed)
                if use_plot and hasattr(sample, "get") and sample.get("prompt"):
                    plot_data = self.get_plot_from_prompt(sample["prompt"])
                    pattern = r"The following is the accelerometer data on the [xyz]-axis\n([\-0-9, ]+)"
                    sample["prompt"] = re.sub(pattern, "", sample["prompt"])

                # Clean up prompt for TSQADataset (if needed)
                if hasattr(sample, "get") and sample.get("prompt"):
                    pattern = r"This is the time series, it has mean (-?\d+\.\d{4}) and std (-?\d+\.\d{4})\."
                    replacement = "This is the time series:"
                    sample["prompt"] = re.sub(pattern, replacement, sample["prompt"])

                # Create input text
                input_text = sample["prompt"]
                target_answer = sample["answer"]

                # Generate prediction
                outputs = pipe(
                    input_text,
                    max_new_tokens=max_new_tokens,
                    return_full_text=False,
                    plot_data=plot_data,
                )

                # Extract generated text
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0]["generated_text"].strip()
                    successful_inferences += 1

                    # Evaluate using custom function (optionally with sample)
                    try:
                        import inspect

                        sig = inspect.signature(evaluation_function)
                        if len(sig.parameters) >= 3:
                            metrics = evaluation_function(
                                target_answer, generated_text, sample
                            )
                        else:
                            metrics = evaluation_function(target_answer, generated_text)
                    except Exception:
                        # Fallback to 2-arg call
                        metrics = evaluation_function(target_answer, generated_text)
                    all_metrics.append(metrics)

                    # Store detailed results
                    result = {
                        "sample_idx": idx,
                        "input_text": input_text,
                        "target_answer": target_answer,
                        "generated_answer": generated_text,
                        "metrics": metrics,
                    }
                    # Include template_id if present in sample for downstream analysis
                    if isinstance(sample, dict) and "template_id" in sample:
                        result["template_id"] = sample["template_id"]
                    results.append(result)

                    # Save individual result immediately to prevent data loss
                    self._save_individual_result(
                        result, model_name, dataset_class.__name__
                    )

                    # Print progress for first few samples
                    if idx < 10:
                        print(f"\nSAMPLE {idx + 1}:")
                        print(f"PROMPT: {input_text}...")
                        print(f"TARGET: {target_answer}")
                        print(f"PREDICTION: {generated_text}")
                        print(f"METRICS: {metrics}")
                        print("=" * 80)

                    # Print first error for debugging
                    if not first_error_printed and metrics.get("accuracy", 1) == 0:
                        print(f"\n❌ FIRST ERROR (Sample {idx + 1}):")
                        print(f"TARGET: {target_answer}")
                        print(f"PREDICTION: {generated_text}")
                        print("=" * 80)
                        first_error_printed = True
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        # Calculate aggregate metrics
        if successful_inferences > 0:
            # Aggregate metrics across all samples
            aggregate_metrics = self._aggregate_metrics(all_metrics)

            # Calculate success rate
            success_rate = successful_inferences / total_samples

            # Prepare final results
            final_results = {
                "model_name": model_name,
                "dataset_name": dataset_class.__name__,
                "total_samples": total_samples,
                "successful_inferences": successful_inferences,
                "success_rate": success_rate,
                "metrics": aggregate_metrics,
                "detailed_results": results,
            }

            # Print summary
            self._print_summary(final_results)

            # Consolidate JSONL results into final JSON file
            consolidated_file = self._consolidate_jsonl_results(
                model_name, dataset_class.__name__
            )
            if consolidated_file:
                # Update the consolidated file with correct total_samples
                with open(consolidated_file, "r") as f:
                    consolidated_data = json.load(f)
                consolidated_data["total_samples"] = total_samples
                consolidated_data["success_rate"] = success_rate
                with open(consolidated_file, "w") as f:
                    json.dump(consolidated_data, f, indent=2)

            return final_results
        else:
            print("❌ No successful inferences completed!")
            return {
                "model_name": model_name,
                "dataset_name": dataset_class.__name__,
                "total_samples": total_samples,
                "successful_inferences": 0,
                "success_rate": 0.0,
                "metrics": {},
                "detailed_results": [],
            }

    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across all samples.

        Args:
            metrics_list: List of metric dictionaries

        Returns:
            Aggregated metrics
        """
        if not metrics_list:
            return {}

        # Get all unique metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        aggregated = {}
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in metrics_list]
            if all(isinstance(v, (int, float)) for v in values):
                # Calculate overall accuracy/percentage
                accuracy = np.mean(values) * 100
                aggregated[key] = accuracy
            else:
                # For non-numeric metrics, just count occurrences
                aggregated[key] = {
                    "values": values,
                    "count": len(values),
                }

        return aggregated

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Model: {results['model_name']}")
        print(f"Dataset: {results['dataset_name']}")
        print(f"Total samples processed: {results['total_samples']}")
        print(f"Successful inferences: {results['successful_inferences']}")
        print(f"Success rate: {results['success_rate']:.2%}")

        if results["metrics"]:
            print("\nAggregated Metrics:")
            for metric_name, metric_values in results["metrics"].items():
                if isinstance(metric_values, (int, float)):
                    print(f"  {metric_name}: {metric_values:.1f}%")
                else:
                    print(f"  {metric_name}: {metric_values}")

    def _save_results(self, results: Dict[str, Any]):
        """Save detailed results to file."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_dir = os.path.join(
            current_dir, "..", "results", "baseline", "detailed"
        )
        os.makedirs(detailed_dir, exist_ok=True)
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", results["model_name"].lower())
        normalized_dataset_name = re.sub(
            r"[^a-z0-9]", "-", results["dataset_name"].lower()
        )
        results_file = os.path.join(
            detailed_dir,
            f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.json",
        )

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nDetailed results saved to: {results_file}")

    def _save_individual_result(
        self, result: Dict[str, Any], model_name: str, dataset_name: str
    ):
        """Save individual result incrementally to prevent data loss."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_dir = os.path.join(
            current_dir, "..", "results", "baseline", "detailed"
        )
        os.makedirs(detailed_dir, exist_ok=True)
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", model_name.lower())
        normalized_dataset_name = re.sub(r"[^a-z0-9]", "-", dataset_name.lower())
        results_file = os.path.join(
            detailed_dir,
            f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.jsonl",
        )

        # Append individual result as JSONL
        with open(results_file, "a") as f:
            json.dump(result, f)
            f.write("\n")

    def _consolidate_jsonl_results(self, model_name: str, dataset_name: str) -> str:
        """Consolidate JSONL results into final JSON file."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_dir = os.path.join(
            current_dir, "..", "results", "baseline", "detailed"
        )
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", model_name.lower())
        normalized_dataset_name = re.sub(r"[^a-z0-9]", "-", dataset_name.lower())

        jsonl_file = os.path.join(
            detailed_dir,
            f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.jsonl",
        )
        json_file = os.path.join(
            detailed_dir,
            f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.json",
        )

        # Read all JSONL results
        individual_results = []
        if os.path.exists(jsonl_file):
            with open(jsonl_file, "r") as f:
                for line in f:
                    if line.strip():
                        individual_results.append(json.loads(line.strip()))

        # Create consolidated results structure
        if individual_results:
            # Calculate aggregate metrics
            all_metrics = [result["metrics"] for result in individual_results]
            aggregate_metrics = self._aggregate_metrics(all_metrics)

            # Calculate success rate
            successful_inferences = len(individual_results)
            total_samples = len(individual_results)  # This will be updated by caller

            consolidated_results = {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "total_samples": total_samples,
                "successful_inferences": successful_inferences,
                "success_rate": (
                    successful_inferences / total_samples if total_samples > 0 else 0.0
                ),
                "metrics": aggregate_metrics,
                "detailed_results": individual_results,
            }

            # Save consolidated results
            with open(json_file, "w") as f:
                json.dump(consolidated_results, f, indent=2)

            print(f"\nConsolidated results saved to: {json_file}")
            return json_file

        return None

    def _get_existing_results_count(self, model_name: str, dataset_name: str) -> int:
        """Get count of existing results from JSONL file for resuming interrupted evaluations."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_dir = os.path.join(
            current_dir, "..", "results", "baseline", "detailed"
        )
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", model_name.lower())
        normalized_dataset_name = re.sub(r"[^a-z0-9]", "-", dataset_name.lower())
        jsonl_file = os.path.join(
            detailed_dir,
            f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.jsonl",
        )

        if os.path.exists(jsonl_file):
                    if line.strip():
                        count += 1
            return count
        return 0

    def _get_jsonl_file_path(self, model_name: str, dataset_name: str) -> str:
        """Get the JSONL file path for a model-dataset combination."""
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        detailed_dir = os.path.join(
            current_dir, "..", "results", "baseline", "detailed"
        )
        normalized_model_id = re.sub(r"[^a-z0-9]", "-", model_name.lower())
        normalized_dataset_name = re.sub(r"[^a-z0-9]", "-", dataset_name.lower())
        return os.path.join(
            detailed_dir,
            f"evaluation_results_{normalized_model_id}_{normalized_dataset_name}.jsonl",
        )

    def evaluate_multiple_models(
        self,
        model_names: List[str],
        dataset_classes: List[Type[Dataset]],
        evaluation_functions: Dict[str, Callable[[str, str], Dict[str, Any]]],
        max_samples: Optional[int] = None,
        **pipeline_kwargs,
    ) -> pd.DataFrame:
        """
        Evaluate multiple models on multiple datasets.

        Args:
            model_names: List of model names to evaluate
            dataset_classes: List of dataset classes to evaluate on
            evaluation_functions: Dictionary mapping dataset class names to evaluation functions
            max_samples: Maximum number of samples per evaluation
            **pipeline_kwargs: Additional arguments for model pipeline

        Returns:
            DataFrame with results for all model-dataset combinations
        """
        all_results = []

        # Generate filename once at the beginning
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(current_dir, "..", "results", "baseline")
        os.makedirs(results_dir, exist_ok=True)
        df_filename = os.path.join(results_dir, "evaluation_results.csv")
        print(f"Results will be saved to: {df_filename}")

        # Load existing results if file exists
        existing_df = None
        if os.path.exists(df_filename):
            try:
                existing_df = pd.read_csv(df_filename)
                print(f"Found existing results file with {len(existing_df)} entries")
            except Exception as e:
                print(f"Warning: Could not read existing results file: {e}")

        for model_name in model_names:
            for dataset_class in dataset_classes:
                dataset_name = dataset_class.__name__

                if dataset_name not in evaluation_functions:
                    print(f"Warning: No evaluation function found for {dataset_name}")
                    continue

                # Check if this model-dataset combination already exists in results
                if existing_df is not None:
                    existing_result = existing_df[
                        (existing_df["model"] == model_name)
                        & (existing_df["dataset"] == dataset_name)
                    ]
                    if not existing_result.empty:
                        print(
                            f"⏭️  Skipping {model_name} on {dataset_name} (already evaluated)"
                        )
                        continue

                evaluation_function = evaluation_functions[dataset_name]

                print(f"\n{'=' * 80}")
                print(f"Evaluating {model_name} on {dataset_name}")
                print(f"{'=' * 80}")

                try:
                    results = self.evaluate_model_on_dataset(
                        model_name=model_name,
                        dataset_class=dataset_class,
                        evaluation_function=evaluation_function,
                        max_samples=max_samples,
                        use_plot=False,
                        **pipeline_kwargs,
                    )

                    # Extract key metrics for DataFrame
                    row = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "total_samples": results["total_samples"],
                        "successful_inferences": results["successful_inferences"],
                        "success_rate": results["success_rate"],
                    }

                    # Add specific metrics
                    if results["metrics"]:
                        for metric_name, metric_values in results["metrics"].items():
                            if isinstance(metric_values, (int, float)):
                                row[metric_name] = metric_values
                            else:
                                row[metric_name] = str(metric_values)

                    all_results.append(row)

                    # Combine with existing results and save
                    current_df = pd.DataFrame(all_results)
                    if existing_df is not None:
                        # Append new results
                        final_df = pd.concat(
                            [existing_df, current_df], ignore_index=True
                        )
                    else:
                        final_df = current_df

                    final_df.to_csv(df_filename, index=False)
                    print(f"✅ Results updated: {df_filename}")

                except Exception as e:
                    print(f"Error evaluating {model_name} on {dataset_name}: {e}")
                    all_results.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "status": "Failed",
                        }
                    )

                    # Save DataFrame even after errors
                    current_df = pd.DataFrame(all_results)
                    if existing_df is not None:
                        final_df = pd.concat(
                            [existing_df, current_df], ignore_index=True
                        )
                    else:
                        final_df = current_df
                    final_df.to_csv(df_filename, index=False)
                    print(f"⚠️  Results updated (with error): {df_filename}")

        print(f"\nFinal results saved to: {df_filename}")
        return final_df

    def get_plot_from_prompt(self, prompt: str):
        """
        Parse time series data from the prompt and return a base64 image.
        """
        # Parse the time series data from the prompt
        time_series_data = []

        # Extract data for each axis using regex
        axes = ["x-axis", "y-axis", "z-axis"]
        for axis in axes:
            pattern = f"accelerometer data on the {axis}\\n([\\-0-9, ]+)"
            match = re.search(pattern, prompt.lower())
            if match:
                # Extract the data and convert to a list of integers
                data_str = match.group(1).strip()
                data_str = data_str.replace(" ", "")
                data = [int(val.strip()) for val in data_str.split(",") if val.strip()]
                time_series_data.append(data)

        # Create the plot
        num_series = len(time_series_data)
        fig, axes = plt.subplots(
            num_series, 1, figsize=(10, 4 * num_series), sharex=True
        )
        # If there's only one series, axes won't be an array
        if num_series == 1:
            axes = [axes]

        # Plot each time series in its own subplot
        axis_names = {0: "X-axis", 1: "Y-axis", 2: "Z-axis"}
        for i, series in enumerate(time_series_data):
            axes[i].plot(series, marker="o", linestyle="-", markersize=0)
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"Accelerometer - {axis_names.get(i)}")

        plt.tight_layout()

        # Convert plot to base64 image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", bbox_inches="tight", dpi=100)
        plt.close()
        img_buffer.seek(0)
        image_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        return image_data
