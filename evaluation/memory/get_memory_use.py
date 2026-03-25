# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import os as _os
import psutil

import pynvml  # type: ignore


# Models
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP

# Datasets
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.har_cot.HARCoTQADataset import HARCoTQADataset
from opentslm.time_series_datasets.sleep.SleepEDFCoTQADataset import SleepEDFCoTQADataset
from opentslm.time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
from opentslm.time_series_datasets.simulation.SimulationQADataset import SimulationQADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)

_NVML_AVAILABLE = True

def get_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    return "cpu"  # Default to CPU for easier testing


def measure_peak_cuda_bytes() -> int:
    if not torch.cuda.is_available():
        return -1
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_allocated())


def measure_peak_cuda_reserved_bytes() -> int:
    if not torch.cuda.is_available():
        return -1
    torch.cuda.synchronize()
    return int(torch.cuda.max_memory_reserved())


def measure_peak_cpu_bytes() -> int:
    """Measure peak CPU memory usage in bytes"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return int(memory_info.rss)  # Resident Set Size (physical memory)
    except Exception:
        return -1


def nvml_current_process_bytes() -> int:
    if not _NVML_AVAILABLE or not torch.cuda.is_available():
        return -1
    try:
        pynvml.nvmlInit()
        pid = _os.getpid()
        total_bytes = 0
        found = False
        device_count = pynvml.nvmlDeviceGetCount()
        for device_index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            # Try compute procs first
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            except Exception:
                procs = []
            # Fallback to graphics procs
            try:
                procs_gfx = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3(handle)
            except Exception:
                procs_gfx = []
            for p in list(procs) + list(procs_gfx):
                if (
                    int(p.pid) == pid
                    and p.usedGpuMemory is not None
                    and p.usedGpuMemory >= 0
                ):
                    total_bytes += int(p.usedGpuMemory)
                    found = True
        return total_bytes if found else -1
    except Exception:
        return -1


def get_first_batch(dataset, batch_size: int = 1) -> List[Dict[str, any]]:
    # QADataset returns dict samples compatible with model.compute_loss
    batch: List[Dict[str, any]] = []
    for i in range(min(batch_size, len(dataset))):
        batch.append(dataset[i])
    # Ensure time series tensors are padded and converted
    batch = extend_time_series_to_match_patch_size_and_aggregate(batch)
    return batch


def build_optimizer(model, model_type: str, base_lr: float = 2e-4):
    if model_type == "OpenTSLMSP":
        enc_params = [
            p for p in getattr(model, "encoder").parameters() if p.requires_grad
        ]
        proj_params = [
            p for p in getattr(model, "projector").parameters() if p.requires_grad
        ]
        param_groups = []
        if len(enc_params) > 0:
            param_groups.append({"params": enc_params, "weight_decay": 0.1})
        if len(proj_params) > 0:
            param_groups.append({"params": proj_params, "weight_decay": 0.1})
        return (
            torch.optim.AdamW(param_groups, lr=base_lr)
            if len(param_groups) > 0
            else None
        )
    # Flamingo-like
    named_params = list(model.named_parameters())
    trainable = list(
        filter(
            lambda np: np[1].requires_grad
            and not getattr(np[1], "exclude_from_optimizer", False),
            named_params,
        )
    )
    params_with_wd, params_without_wd = [], []
    for name, p in trainable:
        if "gated_cross_attn" in name:
            params_with_wd.append(p)
        else:
            params_without_wd.append(p)
    if len(params_with_wd) + len(params_without_wd) == 0:
        return None
    return torch.optim.AdamW(
        [
            {"params": params_with_wd, "weight_decay": 0.1},
            {"params": params_without_wd, "weight_decay": 0.0},
        ],
        lr=2e-4,
    )


def train_for_steps(
    model, model_type: str, dataset, steps: int
) -> Tuple[float, int, int, int]:
    model.train()
    optimizer = build_optimizer(model, model_type)

    # Initialize memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Track CPU memory baseline
    cpu_memory_baseline = measure_peak_cpu_bytes()

    last_loss = 0.0
    # DataLoader with shuffle and collate that pads series
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b),
        drop_last=False,
    )

    pbar = tqdm(total=steps, desc="Training", leave=False)
    max_peak_bytes = -1
    max_reserved_bytes = -1
    max_nvml_bytes = -1
    max_cpu_bytes = cpu_memory_baseline
    step = 0
    # Initialize postfix
    pbar.set_postfix(
        {
            "alloc_gb": 0.0,
            "res_gb": 0.0,
            "nvml_gb": 0.0,
            "cpu_gb": 0.0,
        }
    )
    for batch in loader:
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        if optimizer and loss.requires_grad:
            print(f"Backpropagating loss of {loss.item()} for step {step}")
            loss.backward()
            optimizer.step()
        last_loss = float(loss.detach().item())

        # Track peak memory across steps
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            current_peak = int(torch.cuda.max_memory_allocated())
            current_reserved = int(torch.cuda.max_memory_reserved())
            if current_peak > max_peak_bytes:
                max_peak_bytes = current_peak
            if current_reserved > max_reserved_bytes:
                max_reserved_bytes = current_reserved
            nvml_bytes = nvml_current_process_bytes()
            if nvml_bytes > max_nvml_bytes:
                max_nvml_bytes = nvml_bytes

        # Track CPU memory
        current_cpu_bytes = measure_peak_cpu_bytes()
        if current_cpu_bytes > max_cpu_bytes:
            max_cpu_bytes = current_cpu_bytes

        # Update progress bar postfix in GB
        def _to_gb(val: int) -> float:
            return (
                float(val) / (1024.0**3)
                if isinstance(val, (int, float)) and val >= 0
                else 0.0
            )

        pbar.set_postfix(
            {
                "alloc_gb": f"{_to_gb(max_peak_bytes):.2f}",
                "res_gb": f"{_to_gb(max_reserved_bytes):.2f}",
                "nvml_gb": f"{_to_gb(max_nvml_bytes):.2f}",
                "cpu_gb": f"{_to_gb(max_cpu_bytes):.2f}",
            }
        )
        step += 1
        pbar.update(1)
        if step >= steps:
            break
    pbar.close()

    if torch.cuda.is_available():
        peak_bytes = max_peak_bytes
        peak_reserved_bytes = max_reserved_bytes
        nvml_peak_bytes = max_nvml_bytes
    else:
        peak_bytes = max_cpu_bytes  # Use CPU memory as fallback
        peak_reserved_bytes = max_cpu_bytes
        nvml_peak_bytes = max_cpu_bytes
    return last_loss, peak_bytes, peak_reserved_bytes, nvml_peak_bytes


def ensure_csv(path: str, header: List[str]):
    exists = os.path.exists(path)
    if not exists:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_row(path: str, row: List[any]):
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def run_for_dataset(
    model_name: str, model, dataset_name: str, dataset_obj
) -> Dict[str, any]:
    result: Dict[str, any] = {
        "model": model_name,
        "dataset": dataset_name,
        "loss": None,
        "peak_cuda_bytes": None,
        "status": "ok",
        "error": "",
    }
    try:
        # Train for half an epoch, capped at 10000 steps
        steps = max(1, min(len(dataset_obj), 100))
        loss, peak, peak_reserved, nvml_peak = train_for_steps(
            model, model_name, dataset_obj, steps
        )
        result["loss"] = loss
        result["peak_cuda_bytes"] = peak
        result["peak_cuda_reserved_bytes"] = peak_reserved
        result["nvml_peak_bytes"] = nvml_peak
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure memory use for a single training iteration for a chosen model and dataset."
    )
    parser.add_argument(
        "-llm_id", required=True, help="HuggingFace model id for the language model"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["OpenTSLMFlamingo", "OpenTSLMSP"],
        help="Model to instantiate",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "TSQADataset",
            "HARCoTQADataset",
            "SleepEDFCoTQADataset",
            "ECGQACoTQADataset",
            "SimulationQADataset",
        ],
        help="Dataset to use",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run on (e.g., cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Length of time series for SimulationQADataset (default: 100)",
    )
    parser.add_argument(
        "--num_series",
        type=int,
        default=1,
        help="Number of time series for SimulationQADataset (default: 1)",
    )
    parser.add_argument(
        "--results_csv",
        default=os.path.join(REPO_DIR, "memory_use.csv"),
        help="Path to CSV file to append results",
    )
    args = parser.parse_args()

    device = get_device(args.device)

    # CSV header and file
    header = [
        "timestamp",
        "llm_id",
        "device",
        "model",
        "dataset",
        "loss",
        "peak_cuda_bytes",
        "peak_cuda_gb",
        "peak_cuda_reserved_bytes",
        "peak_cuda_reserved_gb",
        "nvml_peak_bytes",
        "nvml_peak_gb",
        "status",
        "error",
    ]
    ensure_csv(args.results_csv, header)

    # Instantiate selected model
    if args.model == "OpenTSLMFlamingo":
        model = OpenTSLMFlamingo(
            device=device,
            llm_id=args.llm_id,
            cross_attn_every_n_layers=1,
            gradient_checkpointing=True,
        )
        eos = model.get_eos_token()
    elif args.model == "OpenTSLMSP":
        model = OpenTSLMSP(llm_id=args.llm_id, device=device)
        eos = model.get_eos_token()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Make absolutely sure parameters are on the requested device
    model.to(device)

    # Instantiate selected dataset
    if args.dataset == "TSQADataset":
        dataset = TSQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "TSQA"
    elif args.dataset == "HARCoTQADataset":
        dataset = HARCoTQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "HAR-CoT"
    elif args.dataset == "SleepEDFCoTQADataset":
        dataset = SleepEDFCoTQADataset(split="train", EOS_TOKEN=eos)
        dataset_name = "SleepEDF-CoT"
    elif args.dataset == "ECGQACoTQADataset":
        dataset = ECGQACoTQADataset(
            split="train", EOS_TOKEN=eos, max_samples=1, preload_processed_data=False
        )
        dataset_name = "ECG-QA-CoT"
    elif args.dataset == "SimulationQADataset":
        dataset = SimulationQADataset(
            split="train", EOS_TOKEN=eos, length=args.length, num_series=args.num_series
        )
        dataset_name = f"Simulation-L{args.length}-N{args.num_series}"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Run one iteration and append results
    res = run_for_dataset(args.model, model, dataset_name, dataset)
    peak_bytes = res["peak_cuda_bytes"]
    peak_gb = (
        (float(peak_bytes) / (1024.0**3))
        if isinstance(peak_bytes, (int, float)) and peak_bytes >= 0
        else -1
    )
    peak_reserved_bytes = res.get("peak_cuda_reserved_bytes", -1)
    peak_reserved_gb = (
        (float(peak_reserved_bytes) / (1024.0**3))
        if isinstance(peak_reserved_bytes, (int, float)) and peak_reserved_bytes >= 0
        else -1
    )
    nvml_peak_bytes = res.get("nvml_peak_bytes", -1)
    nvml_peak_gb = (
        (float(nvml_peak_bytes) / (1024.0**3))
        if isinstance(nvml_peak_bytes, (int, float)) and nvml_peak_bytes >= 0
        else -1
    )
    append_row(
        args.results_csv,
        [
            datetime.utcnow().isoformat(),
            args.llm_id,
            device,
            res["model"],
            res["dataset"],
            res["loss"],
            res["peak_cuda_bytes"],
            f"{peak_gb:.4f}"
            if isinstance(peak_gb, float) and peak_gb >= 0
            else peak_gb,
            res.get("peak_cuda_reserved_bytes", -1),
            f"{peak_reserved_gb:.4f}"
            if isinstance(peak_reserved_gb, float) and peak_reserved_gb >= 0
            else peak_reserved_gb,
            res.get("nvml_peak_bytes", -1),
            f"{nvml_peak_gb:.4f}"
            if isinstance(nvml_peak_gb, float) and nvml_peak_gb >= 0
            else nvml_peak_gb,
            res["status"],
            res["error"],
        ],
    )

    print(f"Done. Results appended to: {args.results_csv}")


if __name__ == "__main__":
    main()
