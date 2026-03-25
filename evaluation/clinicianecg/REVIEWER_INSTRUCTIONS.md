<!--
SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.

SPDX-License-Identifier: MIT
-->

# ECG Model Output Review Instructions

## Overview

You are participating in a scientific evaluation of an AI model's performance on ECG interpretation tasks. This model uses multimodal input (ECG tracings + clinical context) to provide textual diagnostic outputs. Your expert assessment is crucial for validating the model's clinical reasoning capabilities.

## Study Design

- **Total samples**: 84 ECG cases across 42 different clinical scenarios
- **Review system**: Each sample is reviewed by multiple independent reviewers (configurable)
- **Reviewer assignment**: Automated balanced assignment system ensures fair distribution

## Assessment Criteria

For each sample, you will evaluate the model's output across three dimensions:

### 1. ECG Pattern Recognition Accuracy
**Question**: "Did the model correctly identify the relevant ECG features needed to answer the question?"

**Response Options**:
- **Yes**: Model correctly identified all key ECG features relevant to the diagnostic question
- **Some, but not all**: Model identified some relevant features but missed key ones or included irrelevant features
- **None identified**: Model failed to identify any of the key ECG features

### 2. Clinical Reasoning Quality
**Question**: "Did the model appropriately connect the identified ECG features to the final answer?"

**Response Options**:
- **Yes**: Model demonstrated clear, logical connection between ECG findings and diagnostic conclusion
- **Some incorrect logic**: Model showed some logical reasoning but with notable gaps, inconsistencies, or errors
- **Completely incorrect logic**: Model failed to establish logical connection or reasoning was fundamentally flawed

### 3. Clinical Context Integration
**Question**: "Did the model appropriately incorporate patient clinical background (age, recording conditions, artifacts) in its interpretation?"

**Response Options**:
- **Yes**: Model appropriately considered patient demographics, recording conditions, and technical factors
- **Used some key background**: Model acknowledged some but not all relevant clinical context factors
- **No, didn't use any of the relevant background**: Model ignored or inappropriately used clinical context information

## Review Process

### Step 1: Initial Setup
1. Open your assigned Excel file: `[Your_Name]_ECG_Review_v2.xlsx`
2. **IMPORTANT**: Enter your initials in the designated field on the Summary sheet (cell B4)
3. Your initials will automatically appear on all 28 individual review sheets
4. Review the updated assessment criteria and sample list

### Step 2: Enhanced Review Interface
Each sample worksheet now features an improved layout:
- **Left side**: Clinical information, diagnostic question, and model output
- **Right side**: Large, high-resolution ECG tracing positioned next to assessment questions
- **Enhanced formatting**: Better text wrapping and visual organization
- **Automated features**: Your initials auto-populate, dropdowns for consistent responses

### Step 3: Review Each Sample
For each sample worksheet:

1. **Study the clinical context** (left side) - Note patient age, recording conditions, artifacts
2. **Examine the large ECG tracing** (middle) - Enhanced visibility for detailed analysis
3. **Read the diagnostic question** (left side) - Understand what the model was asked to determine
4. **Analyze the model's output** (left side) - Full model reasoning and conclusion provided
5. **Complete the assessment** (right side) - Three dropdown questions positioned next to ECG

### Step 4: Complete Assessment
1. **Use dropdown menus** for the 3 assessment questions (consistent response options)
2. **Add comments** in the provided text fields to explain your reasoning
3. **Review date** will be automatically recorded when you make selections
4. **Overall comments** field for additional observations

### Step 5: Quality Assurance
- Ensure all 28 samples are reviewed
- Verify your initials appear correctly on all sheets
- Save the Excel file frequently during review

## Scoring Guidelines

### Best Practices for Consistent Evaluation

**Maintain Clinical Standards**:
- Apply the same diagnostic criteria you would use in clinical practice
- Consider the model output as you would evaluate a resident's interpretation
- Focus on clinical accuracy and reasoning quality

**Objective Assessment**:
- Base evaluations on medical evidence and established guidelines
- Avoid being influenced by whether you personally agree with the final answer
- Focus on the quality of reasoning process, not just the conclusion

**Comment Quality** (when selecting "Partially"):
- Be specific about what was correct vs. incorrect
- Explain why the partial score was assigned
- Suggest what would improve the assessment to "Yes"

**Examples of Good Comments**:
- "Model correctly identified QRS widening but failed to note specific LBBB morphology in V1"
- "Acknowledged patient age but didn't consider how pacemaker spikes might affect interpretation"
- "Logical reasoning about ST changes but overlooked clinical context of recent MI"

## Quality Control Measures

### Before Submission
- [ ] All 28 samples reviewed
- [ ] All Yes/Partially/No questions answered
- [ ] Comments provided for all "Partially" responses
- [ ] Reviewer ID and review dates completed
- [ ] File saved with correct filename
