#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
ECG Dataset Analyzer and Review Assignment Generator
Creates a configurable review system for ECG samples with configurable reviewers
"""

import os
import pandas as pd
import random
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

class ECGDatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.samples_data = []
        self.config = self._load_config()
        self.reviewers = self._generate_reviewer_names()
    
    def _load_config(self) -> dict:
        """Load configuration from config.json"""
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _generate_reviewer_names(self) -> List[str]:
        """Generate reviewer names based on config"""
        count = self.config["reviewer_count"]
        return [f"Reviewer_{chr(65 + i)}" for i in range(count)]
        
    def parse_evaluation_file(self, eval_file_path: Path) -> Dict[str, Any]:
        """Parse evaluation_info.txt file and extract all relevant information"""
        try:
            with open(eval_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract template and ECG ID
            template_match = re.search(r'Template ID:\s*(\d+)', content)
            ecg_match = re.search(r'ECG ID:\s*(\d+)', content)
            
            # Extract question
            question_match = re.search(r'Question:\s*(.+?)(?=\n\nAnswer Options:|$)', content, re.DOTALL)
            
            # Extract answer options
            options_match = re.search(r'Answer Options:\s*(.+?)(?=\n\nClinical Context:|$)', content, re.DOTALL)
            
            # Extract clinical context
            clinical_match = re.search(r'Clinical Context:\s*(.+?)(?=\n\nModel Output|$)', content, re.DOTALL)
            
            # Extract model output
            model_match = re.search(r'Model Output \([^)]+\):\s*(.+?)(?=\n\nExpected Answer:|$)', content, re.DOTALL)
            
            # Extract expected answer
            expected_match = re.search(r'Expected Answer:\s*(.+?)(?=\n|$)', content)
            
            return {
                'template_id': int(template_match.group(1)) if template_match else None,
                'ecg_id': int(ecg_match.group(1)) if ecg_match else None,
                'question': question_match.group(1).strip() if question_match else None,
                'answer_options': options_match.group(1).strip() if options_match else None,
                'clinical_context': clinical_match.group(1).strip() if clinical_match else None,
                'model_output': model_match.group(1).strip() if model_match else None,
                'expected_answer': expected_match.group(1).strip() if expected_match else None,
                'file_path': eval_file_path
            }
        except Exception as e:
            print(f"Error parsing {eval_file_path}: {e}")
            return None
    
    def collect_all_samples(self):
        """Collect all samples from the dataset"""
        print("Analyzing dataset structure...")
        
        template_dirs = sorted([d for d in self.dataset_path.iterdir() 
                               if d.is_dir() and d.name.startswith('template_')])
        
        for template_dir in template_dirs:
            template_num = int(template_dir.name.split('_')[1])
            
            sample_dirs = sorted([d for d in template_dir.iterdir() 
                                 if d.is_dir() and d.name.startswith('sample')])
            
            for sample_dir in sample_dirs:
                sample_num = int(sample_dir.name.replace('sample', ''))
                eval_file = sample_dir / 'evaluation_info.txt'
                ecg_plot = sample_dir / 'ecg_plot.png'
                
                if eval_file.exists() and ecg_plot.exists():
                    eval_data = self.parse_evaluation_file(eval_file)
                    if eval_data:
                        eval_data.update({
                            'template_number': template_num,
                            'sample_number': sample_num,
                            'sample_id': f"T{template_num:02d}_S{sample_num}",
                            'ecg_plot_path': ecg_plot,
                            'sample_dir': sample_dir
                        })
                        self.samples_data.append(eval_data)
        
        print(f"Found {len(self.samples_data)} samples across {len(template_dirs)} templates")
        return self.samples_data
    
    def create_dual_review_assignments(self, seed: int = 42) -> Dict[str, List[Dict]]:
        """Create assignments ensuring each sample is reviewed exactly twice"""
        random.seed(seed)  # For reproducibility
        
        # Create a list of all samples
        all_samples = list(range(len(self.samples_data)))
        
        # Initialize reviewer assignments
        reviewer_assignments = {reviewer: [] for reviewer in self.reviewers}
        sample_assignments = {i: [] for i in all_samples}
        
        # Shuffle samples for random distribution
        random.shuffle(all_samples)
        
        # Assign each sample to the configured number of reviewers
        reviewer_cycle = 0
        for sample_idx in all_samples:
            assigned_reviewers = []
            reviews_per_sample = self.config["reviews_per_sample"]
            
            # Assign reviews_per_sample different reviewers to this sample
            for i in range(reviews_per_sample):
                reviewer = self.reviewers[(reviewer_cycle + i) % len(self.reviewers)]
                
                # Ensure we don't assign the same reviewer twice to the same sample
                while reviewer in assigned_reviewers:
                    reviewer_cycle += 1
                    reviewer = self.reviewers[(reviewer_cycle + i) % len(self.reviewers)]
                
                assigned_reviewers.append(reviewer)
                reviewer_assignments[reviewer].append(sample_idx)
            
            # Track which reviewers are assigned to each sample
            sample_assignments[sample_idx] = assigned_reviewers
            
            reviewer_cycle += 1
        
        # Convert sample indices to actual sample data
        final_assignments = {}
        for reviewer, sample_indices in reviewer_assignments.items():
            final_assignments[reviewer] = [self.samples_data[idx] for idx in sample_indices]
        
        return final_assignments, sample_assignments
    
    def generate_assignment_summary(self, assignments: Dict[str, List[Dict]], 
                                   sample_assignments: Dict[int, List[str]]) -> pd.DataFrame:
        """Generate a summary DataFrame of assignments"""
        summary_data = []
        
        for sample_idx, reviewers in sample_assignments.items():
            sample = self.samples_data[sample_idx]
            
            # Create base record
            record = {
                'Sample_ID': sample['sample_id'],
                'Template_ID': sample['template_id'],
                'ECG_ID': sample['ecg_id'],
                'Question_Type': sample['question'][:50] + '...' if sample['question'] else 'N/A'
            }
            
            # Add reviewer assignments dynamically
            for i, reviewer in enumerate(reviewers):
                record[f'Reviewer_{i+1}'] = reviewer
            
            # Fill in any missing reviewer columns with empty strings
            reviews_per_sample = self.config["reviews_per_sample"]
            for i in range(len(reviewers), reviews_per_sample):
                record[f'Reviewer_{i+1}'] = ''
            
            summary_data.append(record)
        
        return pd.DataFrame(summary_data)
    
    def print_assignment_statistics(self, assignments: Dict[str, List[Dict]]):
        """Print statistics about the assignments"""
        print("\n" + "="*60)
        print("REVIEW ASSIGNMENT STATISTICS")
        print("="*60)
        
        total_samples = len(self.samples_data)
        total_reviews = sum(len(samples) for samples in assignments.values())
        reviews_per_sample = self.config["reviews_per_sample"]
        
        print(f"Total samples in dataset: {total_samples}")
        print(f"Total reviews to be conducted: {total_reviews}")
        print(f"Reviews per sample: {total_reviews / total_samples:.1f}")
        print(f"Number of reviewers: {len(self.reviewers)}")
        
        print("\nReviews per reviewer:")
        for reviewer, samples in assignments.items():
            print(f"  {reviewer}: {len(samples)} samples")
        
        # Check template distribution
        template_distribution = {}
        for reviewer, samples in assignments.items():
            template_counts = {}
            for sample in samples:
                template_id = sample['template_id']
                template_counts[template_id] = template_counts.get(template_id, 0) + 1
            template_distribution[reviewer] = template_counts
        
        print(f"\nTemplate distribution across reviewers:")
        all_templates = set()
        for template_counts in template_distribution.values():
            all_templates.update(template_counts.keys())
        
        for template_id in sorted(all_templates):
            counts = [template_distribution[reviewer].get(template_id, 0) 
                     for reviewer in self.reviewers]
            print(f"  Template {template_id:2d}: {counts} (total: {sum(counts)})")
        
        print("="*60)

def main():
    print("=== ECG Dataset Analyzer ===")
    
    # Initialize analyzer with data path relative to the clinicianecg root directory  
    analyzer = ECGDatasetAnalyzer("../data")
    
    print(f"Configuration:")
    print(f"  - Reviewer count: {analyzer.config['reviewer_count']}")
    print(f"  - Reviews per sample: {analyzer.config['reviews_per_sample']}")
    print(f"  - Generated reviewers: {', '.join(analyzer.reviewers)}")
    print()
    
    # Create reviewer_workbooks directory if it doesn't exist
    reviewer_workbooks_dir = Path("../reviewer_workbooks")
    reviewer_workbooks_dir.mkdir(exist_ok=True)
    
    # Collect all samples
    samples = analyzer.collect_all_samples()
    
    if not samples:
        print("No samples found! Check dataset structure.")
        return
    
    # Create review assignments
    assignments, sample_assignments = analyzer.create_dual_review_assignments()
    
    # Print statistics
    analyzer.print_assignment_statistics(assignments)
    
    # Generate summary
    summary_df = analyzer.generate_assignment_summary(assignments, sample_assignments)
    
    # Save assignment summary to CSV in reviewer_workbooks folder
    summary_path = reviewer_workbooks_dir / 'review_assignments_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nAssignment summary saved to: {summary_path}")
    
    # Save detailed assignments for later use in reviewer_workbooks folder
    import pickle
    assignments_path = reviewer_workbooks_dir / 'reviewer_assignments.pkl'
    with open(assignments_path, 'wb') as f:
        pickle.dump({
            'assignments': assignments,
            'sample_assignments': sample_assignments,
            'samples_data': samples
        }, f)
    
    print(f"Detailed assignments saved to: {assignments_path}")
    
    return analyzer, assignments, sample_assignments

if __name__ == "__main__":
    main()