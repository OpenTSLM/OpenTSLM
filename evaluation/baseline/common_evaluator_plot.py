# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import io
import base64
from typing import Type, Callable, Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers.pipelines import pipeline
import matplotlib.pyplot as plt
from time import sleep
from PIL import Image
import pandas as pd


from openai_pipeline import OpenAIPipeline
from common_evaluator import CommonEvaluator


class CommonEvaluatorPlot(CommonEvaluator):
    """
    A common evaluation framework for testing LLMs on time series datasets with plot generation.
    """

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
            "task": "image-text-to-text",
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
        formatter = None

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

    def evaluate_multiple_models(
        self,
        model_names: List[str],
        dataset_classes: List[Type[Dataset]],
        evaluation_functions: Dict[str, Callable[[str, str], Dict[str, Any]]],
        plot_functions: Optional[Dict[str, Callable[[Any], str]]] = None,
        max_samples: Optional[int] = None,
        **pipeline_kwargs,
    ) -> pd.DataFrame:
        """
        Evaluate multiple models on multiple datasets with plotting support.

        Args:
            model_names: List of model names to evaluate
            dataset_classes: List of dataset classes to evaluate on
            evaluation_functions: Dictionary mapping dataset class names to evaluation functions
            plot_functions: Optional dict mapping dataset class names to plot functions
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
                plot_fn = None
                if plot_functions is not None and dataset_name in plot_functions:
                    plot_fn = plot_functions[dataset_name]
                else:
                    print(f"Warning: No plot function found for {dataset_name}")

                print(f"\n{'=' * 80}")
                print(f"Evaluating {model_name} on {dataset_name}")
                print(f"{'=' * 80}")

                try:
                    results = self.evaluate_model_on_dataset(
                        model_name=model_name,
                        dataset_class=dataset_class,
                        evaluation_function=evaluation_function,
                        plot_function=plot_fn,
                        max_samples=max_samples,
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

    def evaluate_model_on_dataset(
        self,
        model_name: str,
        dataset_class: Type[Dataset],
        evaluation_function: Callable[[str, str], Dict[str, Any]],
        plot_function: Optional[Callable[[Any], str]] = None,
        max_samples: Optional[int] = None,
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
        dataset = self.load_dataset(
            dataset_class, format_sample_str=False, max_samples=max_samples
        )

        # Limit samples if specified
        if max_samples is not None:
            dataset_size = min(len(dataset), max_samples)
            print(f"Processing first {dataset_size} samples...")
        else:
            dataset_size = len(dataset)
            print(f"Processing all {dataset_size} samples...")
        # Initialize tracking
        total_samples = 0
        successful_inferences = 0
        all_metrics = []
        results = []
        first_error_printed = False  # Track if we've printed the first error

        print("\nRunning inference...")
        print("=" * 80)

        # Get max_new_tokens for generation (default 1000)
        max_new_tokens = pipeline_kwargs.pop("max_new_tokens", 1000)

        for idx in tqdm(range(dataset_size), desc="Processing samples"):
            try:
                sample = dataset[idx]
                plot_data = None

                if isinstance(sample, dict) and "time_series" in sample:
                    plot_data = plot_function(sample["time_series"])
                else:
                    raise ValueError(
                        f"Sample {sample} does not contain 'time_series' key"
                    )

                target_answer = sample["answer"]
                input_text = sample["pre_prompt"] + sample["post_prompt"]

                # Generate prediction
                if isinstance(pipe, OpenAIPipeline):
                    outputs = pipe(
                        input_text,
                        max_new_tokens=max_new_tokens,
                        return_full_text=False,
                        plot_data=plot_data,
                    )
                else:  # For Hugging Face pipelines, convert plot_data (base64) to PIL and pass via images
                    try:
                        img_bytes = base64.b64decode(plot_data)
                        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                        # Check if this is a pretrained model (pt) vs instruction-tuned (it)
                        if "pt" in model_name.lower() and "gemma" in model_name.lower():
                            # For pretrained Gemma models, use model and processor directly
                            # to avoid pipeline issues with chat templates

                            try:
                                outputs = pipe(
                                    text=f"{sample['pre_prompt']} <start_of_image> {sample['post_prompt']}",
                                    images=img,
                                    max_new_tokens=max_new_tokens,
                                    return_full_text=False,
                                )
                            except Exception as e:
                                raise RuntimeError(
                                    f"Failed to call pretrained pipeline: {e}"
                                )
                        else:
                            # For instruction-tuned models, use chat template format
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "image", "image": img},
                                        {"type": "text", "text": input_text},
                                    ],
                                }
                            ]
                            outputs = pipe(
                                text=messages,
                                max_new_tokens=max_new_tokens,
                                return_full_text=False,
                            )
                    except Exception as e:
                        raise RuntimeError(f"Failed to decode plot image: {e}")

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
                    results.append(result)
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
                else:
                    raise ValueError(f"Unexpectedly found empty outputs")

                total_samples += 1

                # Save results every 100 samples
                if total_samples % 50 == 0:
                    # Calculate current aggregate metrics
                    current_aggregate_metrics = (
                        self._aggregate_metrics(all_metrics) if all_metrics else {}
                    )
                    current_success_rate = (
                        successful_inferences / total_samples
                        if total_samples > 0
                        else 0.0
                    )

                    # Prepare intermediate results
                    intermediate_results = {
                        "model_name": model_name,
                        "dataset_name": dataset_class.__name__,
                        "total_samples": total_samples,
                        "successful_inferences": successful_inferences,
                        "success_rate": current_success_rate,
                        "metrics": current_aggregate_metrics,
                        "detailed_results": results,
                    }

                    # Save intermediate results
                    self._save_results(intermediate_results)
                    print(
                        f"💾 Intermediate results saved after {total_samples} samples"
                    )

            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                total_samples += 1
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

            # Save results
            self._save_results(final_results)

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
