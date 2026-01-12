#!/usr/bin/env python3
"""
Common SFT (LoRA) fine-tuning helper for HF Image-Text-to-Text models.
Exposes run_sft(train_examples, **kwargs) where each example is a dict with a
"messages" field (chat template) and optional images embedded in the messages.

Dependencies:
  transformers, datasets, peft, trl, accelerate (and bitsandbytes if you want QLoRA)
"""
from __future__ import annotations
import os
from typing import List
import io

from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoModelForCausalLM
import torch
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer.sft_config import SFTConfig
from PIL import Image

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

def run_sft(
    train_examples: List[dict],
    *,
    output_dir: str,
    llm_id: str = "google/gemma-3-4b-pt",
    epochs: int = 1,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_len: int = 4096,
    logging_steps: int = 10,
    save_steps: int = 10000,  # Save less frequently to save disk space
    bf16: bool = True,
) -> None:
    """Run LoRA SFT on chat-style examples (with images) and save adapters.

    Args:
        train_examples: List of dicts, each containing a "messages" list compatible
            with the processor's chat template. Image elements should be PIL Images
            placed as dicts with {"type": "image", "image": PIL.Image}.
        output_dir: Where to save adapters and processor
        llm_id: HF model id (e.g., google/gemma-3-4b-pt)
        epochs, learning_rate, per_device_train_batch_size, gradient_accumulation_steps,
        max_seq_len: Usual training hyperparameters
        logging_steps, save_steps, bf16: Trainer settings
    """
    if not train_examples:
        raise ValueError("train_examples is empty; provide at least one training example")

    os.makedirs(output_dir, exist_ok=True)

    # Build a tiny HF dataset with chat messages
    ds = Dataset.from_list(train_examples)

    # Tokenizer / model
    # tokenizer = AutoTokenizer.from_pretrained(llm_id)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it") # for some reason it is preferred 


    model = AutoModelForImageTextToText.from_pretrained( 
        llm_id,
        attn_implementation="eager",  # Use eager attention (flash_attention_2 requires flash-attn package)
        torch_dtype=torch.bfloat16,  # Explicit bfloat16 for memory savings
        device_map="auto",
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
    )

    # LoRA config - reduce rank to save memory
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        lora_alpha=8,  # Reduced from 16
        lora_dropout=0.05,
        r=8,  # Reduced rank from 16 to 8 for memory savings
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    # defaults from https://ai.google.dev/gemma/docs/core/huggingface_vision_finetune_qlora
    # args = SFTConfig(
    #     output_dir="gemma-product-description",     # directory to save and repository id
    #     num_train_epochs=1,                         # number of training epochs
    #     per_device_train_batch_size=1,              # batch size per device during training
    #     gradient_accumulation_steps=4,              # number of steps before performing a backward/update pass
    #     gradient_checkpointing=True,                # use gradient checkpointing to save memory
    #     optim="adamw_torch_fused",                  # use fused adamw optimizer
    #     logging_steps=5,                            # log every 5 steps
    #     save_strategy="epoch",                      # save checkpoint every epoch
    #     learning_rate=2e-4,                         # learning rate, based on QLoRA paper
    #     bf16=True,                                  # use bfloat16 precision
    #     max_grad_norm=0.3,                          # max gradient norm based on QLoRA paper
    #     warmup_ratio=0.03,                          # warmup ratio based on QLoRA paper
    #     lr_scheduler_type="constant",               # use constant learning rate scheduler
    #     push_to_hub=True,                           # push model to hub
    #     report_to="tensorboard",                    # report metrics to tensorboard
    #     gradient_checkpointing_kwargs={
    #         "use_reentrant": False
    #     },  # use reentrant checkpointing
    #     dataset_text_field="",                      # need a dummy field for collator
    #     dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
    # )

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy="epoch",  # Only save at end of epoch to save disk space
        bf16=bf16,
        report_to=[],
        # SFT-specific fields: we provide our own collator, so skip text field processing
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=max_seq_len,
        packing=False,
        remove_unused_columns=False,
        # Memory optimization settings
        gradient_checkpointing=True,  # Trade compute for memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",  # Use 8-bit optimizer to save memory
        max_grad_norm=0.3,  # Gradient clipping
    )

    def process_vision_info(messages: List[dict]):
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            for element in content:
                if isinstance(element, dict) and (
                    "image" in element or element.get("type") == "image"
                ):
                    image = element.get("image", element)
                    # Skip None images
                    if image is None:
                        continue
                    
                    # Handle dict with 'bytes' key (HuggingFace datasets format)
                    if isinstance(image, dict):
                        if "bytes" in image and image["bytes"] is not None:
                            image = Image.open(io.BytesIO(image["bytes"]))
                        elif "path" in image and image["path"] is not None:
                            image = Image.open(image["path"])
                        else:
                            continue  # Skip if no valid image data
                    
                    # Expect PIL.Image, convert to RGB
                    if hasattr(image, "convert"):
                        image = image.convert("RGB")
                        image_inputs.append(image)
        return image_inputs

    def collate_fn(examples: List[dict]):
        # Build chat-formatted text and gather images per example
        texts = []
        images = []
        for ex in examples:
            msgs = ex["messages"]
            text = processor.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(process_vision_info(msgs))

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Create labels and mask out padding and image tokens
        labels = batch["input_ids"].clone()

        # Mask padding tokens
        pad_token_id = processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        # Mask image begin tokens and any known special image token ids
        special_map = processor.tokenizer.special_tokens_map
        boi_id = None
        if isinstance(special_map, dict) and "boi_token" in special_map:
            boi_id = processor.tokenizer.convert_tokens_to_ids(special_map["boi_token"])
        if boi_id is not None:
            labels[labels == boi_id] = -100
        # Some processors use a large reserved id for image placeholder
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch

    trainer = SFTTrainer(
        model=model,
        peft_config=lora_cfg,
        processing_class=processor,
        train_dataset=ds,
        args=training_args,
        data_collator=collate_fn,
    )

    # Resume from checkpoint if it exists and is valid
    resume_from_checkpoint = None
    import glob
    import os as os_module
    checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
    if checkpoints:
        # Get the latest checkpoint by sorting
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
        # Check if trainer_state.json exists (valid checkpoint)
        if os_module.path.exists(os_module.path.join(latest_checkpoint, "trainer_state.json")):
            resume_from_checkpoint = latest_checkpoint
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        else:
            print(f"Checkpoint {latest_checkpoint} is incomplete, starting from scratch")
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"Saved LoRA adapters and processor to: {output_dir}")

    # Free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

