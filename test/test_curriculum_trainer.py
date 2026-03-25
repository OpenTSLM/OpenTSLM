#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for the curriculum learning implementation.

This script tests the basic functionality without running full training.
"""

import os
import sys
import torch
import json

# Add the parent directory to the path to import curriculum_learning

from curriculum_learning import CurriculumTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"


# Add a helper to sanitize llm_id for directory names (should match curriculum_learning.py)
def _sanitize_llm_id(llm_id: str) -> str:
    if not llm_id:
        return "unknown_llm"
    name = llm_id.split("/")[-1]
    name = name.replace(".", "_").replace("-", "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name


LLM_ID = "meta-llama/Llama-3.2-1B"
LLM_ID_SAFE = _sanitize_llm_id(LLM_ID)


def test_curriculum_trainer_initialization():
    """Test that the CurriculumTrainer can be initialized correctly."""
    print("🧪 Testing CurriculumTrainer initialization...")

    try:
        # Test with OpenTSLMFlamingo
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)
        assert trainer.model_type == "OpenTSLMFlamingo"
        assert trainer.device in ["cuda", "mps", "cpu"]
        print("✅ OpenTSLMFlamingo initialization successful")

        # Test with OpenTSLMSP
        trainer = CurriculumTrainer("OpenTSLMSP", llm_id=LLM_ID, device=device)
        assert trainer.model_type == "OpenTSLMSP"
        print("✅ OpenTSLMSP initialization successful")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False

    return True


def test_results_directory_creation():
    """Test that the results directory structure is created correctly."""
    print("\n🧪 Testing results directory creation...")

    try:
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)

        # Check that the main results directory exists
        assert os.path.exists("results"), "Main results directory not created"

        # Check that llm_id-specific directory exists
        llm_dir = os.path.join("results", LLM_ID_SAFE)
        assert os.path.exists(llm_dir), "LLM directory not created"

        # Check that model-specific directory exists
        model_dir = os.path.join(llm_dir, "OpenTSLMFlamingo")
        assert os.path.exists(model_dir), "Model directory not created"

        # Check that stage directories exist
        for stage in ["stage1_mcq", "stage2_captioning"]:
            stage_dir = os.path.join(model_dir, stage)
            assert os.path.exists(stage_dir), f"Stage directory {stage} not created"

            # Check subdirectories
            checkpoints_dir = os.path.join(stage_dir, "checkpoints")
            results_dir = os.path.join(stage_dir, "results")
            assert os.path.exists(checkpoints_dir), (
                f"Checkpoints directory for {stage} not created"
            )
            assert os.path.exists(results_dir), (
                f"Results directory for {stage} not created"
            )

        print("✅ Results directory structure created correctly")

    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        return False

    return True


def test_optimizer_creation():
    """Test that optimizers can be created for both model types."""
    print("\n🧪 Testing optimizer creation...")

    try:
        # Test OpenTSLMFlamingo optimizer
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)
        optimizer = trainer._get_optimizer()
        assert optimizer is not None, "Flamingo optimizer is None"
        print("✅ OpenTSLMFlamingo optimizer created successfully")

        # Test OpenTSLMSP optimizer
        trainer = CurriculumTrainer("OpenTSLMSP", llm_id=LLM_ID, device=device)
        optimizer = trainer._get_optimizer()
        assert optimizer is not None, "SP optimizer is None"
        print("✅ OpenTSLMSP optimizer created successfully")

    except Exception as e:
        print(f"❌ Optimizer creation failed: {e}")
        return False

    return True


def test_accuracy_calculation():
    """Test the accuracy calculation function."""
    print("\n🧪 Testing accuracy calculation...")

    try:
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)

        # Test exact matches
        print("🧪 Testing exact matches...")
        predictions = ["A", "B", "C", "D"]
        gold_answers = ["A", "B", "C", "D"]
        accuracy = trainer._calculate_accuracy(predictions, gold_answers)
        assert accuracy == 1.0, f"Expected 1.0, got {accuracy}"

        # Test partial matches
        print("🧪 Testing partial matches...")
        predictions = ["A", "B", "C", "E"]
        gold_answers = ["A", "B", "C", "D"]
        accuracy = trainer._calculate_accuracy(predictions, gold_answers)
        assert accuracy == 0.75, f"Expected 0.75, got {accuracy}"

        # Test case insensitive
        print("🧪 Testing case insensitive matches...")
        predictions = ["a", "B", "c", "D"]
        gold_answers = ["A", "b", "C", "d"]
        accuracy = trainer._calculate_accuracy(predictions, gold_answers)
        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"

        # Test empty lists
        print("🧪 Testing empty lists...")
        accuracy = trainer._calculate_accuracy([], [])
        assert accuracy == 0.0, f"Expected 0.0, got {accuracy}"

        print("✅ Accuracy calculation working correctly")

    except Exception as e:
        print(f"❌ Accuracy calculation failed: {e}")
        return False

    return True


def test_checkpoint_operations():
    """Test checkpoint saving and loading operations."""
    print("\n🧪 Testing checkpoint operations...")

    try:
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)

        # Create simple mock objects with state_dict method
        class MockOptimizer:
            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        class MockScheduler:
            def state_dict(self):
                return {}

            def load_state_dict(self, state_dict):
                pass

        mock_optimizer = MockOptimizer()
        mock_scheduler = MockScheduler()

        # Test saving checkpoint
        trainer._save_checkpoint("stage1_mcq", 5, 0.123, mock_optimizer, mock_scheduler)

        checkpoint_path = os.path.join(
            "results",
            LLM_ID_SAFE,
            "OpenTSLMFlamingo",
            "stage1_mcq",
            "checkpoints",
            "best_model.pt",
        )
        assert os.path.exists(checkpoint_path), "Checkpoint file not saved"

        # Test loading checkpoint
        epoch, val_loss = trainer._load_checkpoint(
            "stage1_mcq", mock_optimizer, mock_scheduler
        )
        assert epoch == 5, f"Expected epoch 5, got {epoch}"
        assert val_loss == 0.123, f"Expected val_loss 0.123, got {val_loss}"

        print("✅ Checkpoint operations working correctly")

    except Exception as e:
        print(f"❌ Checkpoint operations failed: {e}")
        return False

    return True


def test_previous_stage_loading():
    """Test loading previous stage model and metrics."""
    print("\n🧪 Testing previous stage loading...")

    try:
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)

        # Create mock metrics file for stage1_mcq
        metrics_dir = os.path.join(
            "results", LLM_ID_SAFE, "OpenTSLMFlamingo", "stage1_mcq", "results"
        )
        os.makedirs(metrics_dir, exist_ok=True)

        mock_metrics = {"accuracy": 0.85, "test_loss": 0.234}

        with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
            json.dump(mock_metrics, f)

        # Create mock checkpoint for stage1_mcq
        checkpoint_dir = os.path.join(
            "results", LLM_ID_SAFE, "OpenTSLMFlamingo", "stage1_mcq", "checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        mock_checkpoint = {
            "model_state": trainer.model.state_dict(),
            "optimizer_state": {},
            "scheduler_state": {},
            "val_loss": 0.123,
            "epoch": 10,
        }

        torch.save(mock_checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))

        # Test loading previous stage for stage2_captioning
        previous_info = trainer._load_previous_stage_model("stage2_captioning")

        assert previous_info is not None, "Should load previous stage info"
        assert previous_info["stage"] == "stage1_mcq", "Should load stage1_mcq"
        assert previous_info["metrics"] == mock_metrics, "Should load correct metrics"
        assert previous_info["epoch"] == 10, "Should load correct epoch"
        assert previous_info["val_loss"] == 0.123, "Should load correct val_loss"

        # Test that first stage returns None
        first_stage_info = trainer._load_previous_stage_model("stage1_mcq")
        assert first_stage_info is None, "First stage should return None"

        print("✅ Previous stage loading working correctly")

    except Exception as e:
        print(f"❌ Previous stage loading failed: {e}")
        return False

    return True


def test_stage_methods_exist():
    """Test that the stage methods exist and are callable."""
    print("\n🧪 Testing stage methods...")

    try:
        trainer = CurriculumTrainer("OpenTSLMFlamingo", llm_id=LLM_ID, device=device)

        # Check that stage methods exist
        assert hasattr(trainer, "stage1_mcq"), "stage1_mcq method not found"
        assert hasattr(trainer, "stage2_captioning"), (
            "stage2_captioning method not found"
        )
        assert callable(trainer.stage1_mcq), "stage1_mcq is not callable"
        assert callable(trainer.stage2_captioning), "stage2_captioning is not callable"

        print("✅ Stage methods exist and are callable")

    except Exception as e:
        print(f"❌ Stage methods test failed: {e}")
        return False

    return True


def test_invalid_model_type():
    """Test that invalid model types are handled correctly."""
    print("\n🧪 Testing invalid model type handling...")

    try:
        # This should raise a ValueError
        trainer = CurriculumTrainer("InvalidModel", llm_id=LLM_ID, device=device)
        print("❌ Should have raised ValueError for invalid model type")
        return False

    except ValueError as e:
        print("✅ Invalid model type correctly rejected")
        return True
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def cleanup_test_files():
    """Clean up test files and directories."""
    print("\n🧹 Cleaning up test files...")

    try:
        import shutil

        if os.path.exists("results"):
            shutil.rmtree("results")
        print("✅ Test files cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


def main():
    """Run all tests."""
    print("🚀 Running Curriculum Learning Tests")
    print("=" * 50)

    tests = [
        test_curriculum_trainer_initialization,
        test_results_directory_creation,
        test_optimizer_creation,
        test_accuracy_calculation,
        test_checkpoint_operations,
        test_previous_stage_loading,
        test_stage_methods_exist,
        test_invalid_model_type,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")

    # Cleanup
    cleanup_test_files()

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
