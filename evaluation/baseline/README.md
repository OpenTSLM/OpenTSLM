<!--
SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.

SPDX-License-Identifier: MIT
-->

# Time Series LLM Evaluation Framework

This directory contains a modular evaluation framework for testing LLMs on time series datasets. The system is designed to be easily extensible to new datasets and models.

## Overview

The evaluation framework consists of:

1. **`common_evaluator.py`** - Core evaluation logic that can be reused across datasets
2. **`evaluate_tsqa.py`** - TSQA-specific evaluation
3. **`evaluate_pamap.py`** - PAMAP-specific evaluation  
4. **`evaluate_all.py`** - Combined evaluation across all datasets
5. **`test_baseline.py`** - Original baseline test (legacy)

## Quick Start

### Running Individual Dataset Evaluations

```bash
# Evaluate on TSQA dataset
python evaluate_tsqa.py

# Evaluate on PAMAP datasets
python evaluate_pamap.py

# Evaluate on all datasets
python evaluate_all.py
```

### Adding New Models

To add new models, simply update the `model_names` list in any evaluation script:

```python
model_names = [
    "meta-llama/Llama-3.2-1B",
    "google/gemma-3n-e2b",
    "google/gemma-3n-e2b-it",
    "microsoft/DialoGPT-medium",
    "gpt2",
]
```

### Adding New Datasets

To add a new dataset:

1. Create a new evaluation file (e.g., `evaluate_newdataset.py`)
2. Define an evaluation function that takes `(ground_truth, prediction)` and returns metrics
3. Import your dataset class and evaluation function in `evaluate_all.py`

Example evaluation function:

```python
def evaluate_newdataset(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """Evaluate predictions for new dataset."""
    gt_clean = ground_truth.lower().strip()
    pred_clean = prediction.lower().strip()
    
    exact_match = gt_clean == pred_clean
    similarity = SequenceMatcher(None, gt_clean, pred_clean).ratio()
    
    return {
        "exact_match": int(exact_match),
        "similarity": similarity,
        "ground_truth": gt_clean,
        "prediction": pred_clean,
    }
```

## Output Files

The evaluation system generates several output files:

1. **Individual Results**: `evaluation_results_{model}_{dataset}.json` - Detailed results for each model-dataset combination
2. **Summary CSV**: `evaluation_results_{timestamp}.csv` - Pandas DataFrame with all results
3. **Console Output**: Real-time progress and summary statistics

## Metrics

The framework calculates several metrics for each prediction:

- **exact_match**: Binary indicator for exact string match
- **partial_match**: Binary indicator for partial string match
- **contains_answer**: Binary indicator if prediction contains ground truth
- **similarity**: Character-level similarity score (0-1)
- **has_reasoning**: For CoT datasets, indicates if prediction contains reasoning

## Configuration

### Sample Limits

For faster testing, you can limit the number of samples:

```python
max_samples=50  # Process only first 50 samples
max_samples=None  # Process all samples
```

### Model Parameters

You can customize model parameters:

```python
results_df = evaluator.evaluate_multiple_models(
    model_names=model_names,
    dataset_classes=dataset_classes,
    evaluation_functions=evaluation_functions,
    max_samples=50,
    temperature=0.1,  # Model temperature
    max_new_tokens=100,  # Maximum tokens to generate
)
```

## Dataset-Specific Considerations

### TSQADataset
- Extracts answers after "Answer:" in predictions
- Uses built-in time series formatting from dataset class
- Calculates similarity metrics

### PAMAP2AccQADataset
- Activity classification from accelerometer data
- Standard exact/partial match metrics

### PAMAP2CoTQADataset
- Chain-of-thought reasoning evaluation
- Additional "has_reasoning" metric
- More complex answer extraction

## Extending the Framework

### Adding Custom Metrics

To add custom metrics, modify the evaluation function:

```python
def evaluate_custom(ground_truth: str, prediction: str) -> Dict[str, Any]:
    # ... existing metrics ...
    
    # Add custom metric
    custom_metric = calculate_custom_metric(ground_truth, prediction)
    
    return {
        # ... existing metrics ...
        "custom_metric": custom_metric,
    }
```

### Adding New Dataset Classes

1. Ensure your dataset class inherits from `QADataset`
2. Implement the required abstract methods
3. Create an evaluation function
4. Add to the evaluation scripts

### Batch Processing

For large-scale evaluations, you can modify the framework to process models in batches or use distributed processing.

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Check if the model name is correct and accessible
2. **Memory Issues**: Reduce `max_samples` or use smaller models
3. **Import Errors**: Ensure all dataset classes are properly imported

### Debug Mode

To enable detailed debugging, modify the evaluation scripts to print more information:

```python
# In common_evaluator.py, set debug=True
if idx < 10:  # Print more samples for debugging
    print(f"Sample {idx}: {input_text[:200]}...")
```

## Performance Tips

1. **Use GPU**: The framework automatically detects and uses CUDA/MPS if available
2. **Batch Processing**: For multiple models, consider running them in parallel
3. **Sample Limiting**: Use `max_samples` for quick testing before full evaluation
4. **Model Caching**: Models are loaded once per evaluation run 

## Using OpenAI Models (ChatGPT, GPT-4, etc.)

You can now evaluate OpenAI models (e.g., ChatGPT, GPT-4) using the same evaluation scripts. To do so:

1. **Set your OpenAI API key** (required):
   
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

2. **Run the evaluation script with an OpenAI model name prefixed by `openai-`**:
   
   ```bash
   python evaluate_tsqa.py openai-gpt-4
   python evaluate_pamap.py openai-gpt-3.5-turbo
   ```
   This will use the OpenAI API instead of a local HuggingFace model.

3. **Notes:**
   - The model name after `openai-` should match the OpenAI API model name (e.g., `gpt-4`, `gpt-3.5-turbo`).
   - You can adjust `max_new_tokens` and other parameters as usual. 