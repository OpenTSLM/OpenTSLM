<!--
SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.

SPDX-License-Identifier: MIT
-->

# Clinician ECG Evaluation

This directory contains tools for the clinician-based analysis of the ECG-QA-based model evaluation. It contains a random subset of 84 samples based on the total templates with suitable predictions (42 in total, 2 random samples from each).

## Dataset Structure

```
clinicianecg/
├── README.md
├── REVIEWER_INSTRUCTIONS.md
├── data/
│   ├── template_01/
│   │   ├── sample1/
│   │   │   ├── ecg_plot.png
│   │   │   ├── evaluation_info.txt
│   │   │   ├── lead_I.csv
│   │   │   ├── lead_II.csv
│   │   │   ├── lead_III.csv
│   │   │   ├── lead_aVF.csv
│   │   │   ├── lead_aVL.csv
│   │   │   ├── lead_aVR.csv
│   │   │   ├── lead_V1.csv
│   │   │   ├── lead_V2.csv
│   │   │   ├── lead_V3.csv
│   │   │   ├── lead_V4.csv
│   │   │   ├── lead_V5.csv
│   │   │   └── lead_V6.csv
│   │   └── sample2/
│   │       └── ... (same structure)
│   ├── template_02/
│   │   └── ... (same structure)
│   ├── ... (templates 03-42)
│   └── template_42/
├── pipeline/
│   ├── config.json
│   ├── 1_dataset_analyzer.py
│   ├── 2_excel_generator.py
│   ├── 2.5_demo_responses.py
│   ├── 3_response_analyzer.py
│   ├── 4_model_performance_analysis.ipynb
│   └── requirements.txt
└── reviewer_workbooks/
    ├── review_assignments_summary.csv
    ├── reviewer_assignments.pkl
    └── ECG_Assessment_[Reviewer_Name].xlsx (configurable number)
```

## Configuration

The pipeline uses a simple `config.json` file in the `pipeline/` directory to configure the evaluation setup:

```json
{
  "reviewer_count": 6,
  "reviews_per_sample": 2
}
```

- **`reviewer_count`**: Number of reviewers to generate (default: 6, generates Reviewer_A through Reviewer_F)
- **`reviews_per_sample`**: Number of reviews each sample receives (default: 2 for dual-review reliability)

You can modify these values to scale the evaluation system for different numbers of reviewers or review requirements.

**Notes:**
- All predictions are CORRECT (model answer matches expected answer)
- ECG data is downsampled to 100Hz for consistency
- Each sample includes clinical context, question, answer options, and model reasoning
- CSV files contain time series data for each of the 12 ECG leads
- Templates 01-42 each contain 2 samples for a total of 84 ECG cases
- All reviewer assignments and generated workbooks are stored in the `reviewer_workbooks/` directory

## Analysis Pipeline

The `pipeline/` directory contains a comprehensive workflow for analyzing ECG model performance through expert clinician review. The pipeline consists of several Python scripts and a Jupyter notebook that work together to facilitate data analysis, review generation, and performance evaluation.

### Pipeline Components

#### 1. Dataset Analyzer (`1_dataset_analyzer.py`)
The initial component that analyzes the ECG dataset and generates reviewer assignments:
- **Purpose**: Parses ECG evaluation files and creates a configurable review system for 84 ECG samples
- **Key Features**:
  - Extracts metadata from `evaluation_info.txt` files (template ID, ECG ID, questions, answers)
  - Assigns samples to reviewers with balanced workload distribution (number configurable via `config.json`)
  - Ensures each sample receives the configured number of reviews for reliability assessment
  - Generates reviewer assignment files in the `reviewer_workbooks/` folder (`reviewer_assignments.pkl`, `review_assignments_summary.csv`)
- **Output**: Structured assignment data for downstream processing in `reviewer_workbooks/` directory

#### 2. Excel Generator (`2_excel_generator.py`)
Creates professionally formatted Excel workbooks for clinician review:
- **Purpose**: Generates individual Excel files for each reviewer containing their assigned ECG samples
- **Key Features**:
  - Embeds large, high-quality ECG plots directly in Excel sheets
  - Creates structured assessment forms with dropdown menus for standardized responses
  - Includes three assessment categories: ECG Pattern Recognition, Clinical Reasoning, Context Integration
  - Applies professional styling and formatting for optimal reviewer experience
  - Adapts to configurable number of reviewers
- **Output**: Individual Excel workbooks in `reviewer_workbooks/` directory (filename format: `ECG_Assessment_[Reviewer_Name].xlsx`)

#### 2.5. [Optional] Demo Response Generator (`2.5_demo_responses.py`)
Generates mock responses for testing and demonstration purposes:
- **Purpose**: Creates sample responses in Excel workbooks to test the analysis pipeline
- **Key Features**:
  - Fills assessment questions with realistic mock responses
  - Uses the same 3-option response scale as actual reviews
  - Enables testing of the response analyzer without requiring completed reviews
  - Useful for pipeline validation and demonstration
- **Output**: Excel workbooks populated with demo responses

#### 3. Response Analyzer (`3_response_analyzer.py`)
Comprehensive analysis of completed reviewer responses:
- **Purpose**: Parses completed Excel workbooks and generates statistical analysis of reviewer responses
- **Key Features**:
  - Extracts responses from all reviewer Excel files
  - Calculates inter-rater agreement statistics (Cohen's kappa, percentage agreement)
  - Generates response distribution visualizations
  - Creates completion status reports and summary statistics
  - Exports analysis results to CSV and JSON formats
- **Output**: Statistical summaries, visualizations, and detailed response data

### Workflow Usage

#### Initial Setup

Before running the pipeline, set up the Python environment and install dependencies:

```bash
# Navigate to the project directory
cd /path/to/clinicianecg

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install required dependencies
pip install -r pipeline/requirements.txt

# Install Jupyter kernel for the virtual environment
python -m ipykernel install --user --name=clinicianecg --display-name="ECG Analysis"
```

#### Pipeline Execution

1. **Configuration**: (Optional) Modify the number of reviewers and reviews per sample
   ```bash
   cd pipeline
   # Edit config.json to adjust reviewer_count and reviews_per_sample as needed
   # Default: 6 reviewers, 2 reviews per sample
   ```

2. **Setup**: Run the dataset analyzer to generate reviewer assignments
   ```bash
   python3 1_dataset_analyzer.py
   ```
   - Creates `../reviewer_workbooks/` directory 
   - Generates `reviewer_assignments.pkl` and `review_assignments_summary.csv`
   - Reads ECG data from `../data/` directory
   - Uses configuration from `config.json` for reviewer assignment

3. **Review Generation**: Create Excel workbooks for clinicians
   ```bash
   python3 2_excel_generator.py
   ```
   - Generates individual Excel files for each reviewer in `../reviewer_workbooks/`
   - Files: `Reviewer_A_ECG_Review.xlsx`, `Reviewer_B_ECG_Review.xlsx`, etc.

3. **Testing** (Optional): Generate mock responses for pipeline testing
   ```bash
   python3 2.5_demo_responses.py
   ```
   - Fills Excel workbooks with sample responses for testing

4. **Analysis**: Extract and analyze completed reviewer responses
   ```bash
   python3 3_response_analyzer.py
   ```
   - Parses completed Excel workbooks from `../reviewer_workbooks/`
   - Generates statistical summaries and visualizations in `../analysis_results/`

5. **Deep Analysis**: Open the Jupyter notebook for comprehensive analysis
   ```bash
   # Start Jupyter Lab or Jupyter Notebook (from pipeline directory)
   jupyter lab 4_model_performance_analysis.ipynb
   # OR
   jupyter notebook 4_model_performance_analysis.ipynb
   ```
   - Select the "ECG Analysis" kernel when prompted
   - Run cells for statistical analysis and visualization
   - Notebook reads from `../analysis_results/` directory

### Requirements
See `pipeline/requirements.txt` for Python dependencies.
