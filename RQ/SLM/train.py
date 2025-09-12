#!/usr/bin/env python3
"""
Unified Training Script for Fine-tuning Language Models

Usage:
    python train.py --config configs/phi.yml
    python train.py --config configs/qwen.yml
    python train.py  # Default: config/phi.yml
"""

import argparse
import json
import sys
import logging
import os
import gc
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from huggingface_hub import login
from dotenv import load_dotenv
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig

# Import custom utilities
import sys
# Add parent's parent directory to path (for utils/ access from RQ/SLM/)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.prompt import get_system_prompt

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Load environment variables
load_dotenv()
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", None)
HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", None)
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", None)


# Dataset processing functions
def create_message_column(row: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """Wrap raw commit into system/user/assistant chat format."""
    user_content = (
        f"- given commit message:\n {row['commit_message']}\n Diff: {row['diff']}"
    )
    assistant_content = json.dumps(
        {"types": json.loads(row["types"])}, ensure_ascii=False
    )

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def apply_chat_template(row: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Convert messages into chatML-formatted text."""
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    }


def prepare_dataset(
    dataset, 
    tokenizer, 
    system_prompt: str, 
    num_workers: int = 4
):
    """Pipeline: raw → messages → chatML text."""
    cols = dataset.column_names
    
    # Use lambda to pass system_prompt to create_message_column
    return dataset.map(
        lambda row: create_message_column(row, system_prompt),
        num_proc=num_workers,
        desc="Building messages",
    ).map(
        lambda row: apply_chat_template(row, tokenizer),
        remove_columns=cols,
        num_proc=num_workers,
        desc="Applying chatML",
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def prepare_datasets_from_config(config: Dict[str, Any]):
    """Load and process training datasets based on configuration"""
    dataset_name = config['model']['dataset_name']
    
    train_dataset = load_dataset(
        dataset_name, split="train", cache_dir=HF_DATASETS_CACHE
    )
    test_dataset = load_dataset(
        dataset_name, split="test", cache_dir=HF_DATASETS_CACHE
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['id'],
        trust_remote_code=True,
        use_fast=True,
        cache_dir=HF_CACHE_DIR,
    )
    tokenizer.padding_side = "left"
    
    # Get system prompt
    system_prompt = get_system_prompt()
    
    # Process datasets using shared utility
    processed_train_dataset = prepare_dataset(
        train_dataset, 
        tokenizer, 
        system_prompt,
        config['technical']['num_workers']
    )
    processed_test_dataset = prepare_dataset(
        test_dataset, 
        tokenizer, 
        system_prompt,
        config['technical']['num_workers']
    )
    
    return processed_train_dataset, processed_test_dataset, tokenizer


def train_model(processed_train_dataset, tokenizer, config: Dict[str, Any]):
    """Train the model with given configuration"""
    
    # Determine compute dtype
    if torch.cuda.is_bf16_supported():
        compute_dtype, attn_implementation = torch.bfloat16, "flash_attention_2"
    else:
        compute_dtype, attn_implementation = torch.float16, "sdpa"
    
    learning_rate = float(config['training']['learning_rate'])
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['id'],
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map=config['technical']['device_map'],
        attn_implementation=attn_implementation,
        cache_dir=HF_CACHE_DIR,
    )
    
    # Prepare output directories
    model_output_dir = f"./outputs/{config['model']['name']}-LoRA"
    hf_model_repo = f"Berom0227/{config['model']['new_model']}"
    hf_adapter_repo = f"{hf_model_repo}-adapter"
    
    # Training arguments
    args = SFTConfig(
        output_dir=model_output_dir,
        eval_strategy="no",
        optim="adamw_torch",
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        gradient_checkpointing=True,
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        log_level="debug",
        save_strategy=config['training']['save_strategy'],
        logging_steps=config['training']['logging_steps'],
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_steps=config['training']['eval_steps'],
        num_train_epochs=config['training']['num_train_epochs'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type="linear",
        report_to="wandb",
        seed=config['training']['seed'],
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=hf_adapter_repo,
        max_length=config['training']['max_seq_length'],
        packing=config['training']['packing'],
        # deepspeed="RQ/SLM/configs/deepspeed.json",
    )
    
    # Update W&B config
    wandb.config.update(
        {
            "model_id": config['model']['id'],
            "learning_rate": learning_rate,
            "num_train_epochs": config['training']['num_train_epochs'],
            "logging_steps": config['training']['logging_steps'],
            "eval_steps": config['training']['eval_steps'],
            "per_device_train_batch_size": config['training']['per_device_train_batch_size'],
            "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
            "lora_r": config['lora']['rank'],
            "lora_alpha": config['lora']['alpha'],
            "lora_dropout": config['lora']['dropout'],
            "max_length": config['training']['max_seq_length'],
        },
        allow_val_change=True,
    )
    
    # LoRA configuration
    peft_config = LoraConfig(
        r=config['lora']['rank'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        task_type=TaskType.CAUSAL_LM,
        target_modules=config['lora']['target_modules'],
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=args,
    )
    
    # Check for checkpoint
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
    
    # Train
    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Log metadata to W&B (artifact upload delegated to Hugging Face Hub via push_to_hub=True)
    wandb.log({
        "model_metadata": {
            "base_model": config['model']['id'],
            "hf_adapter_repo": hf_adapter_repo,
            "peft": {
                "r": config['lora']['rank'], 
                "alpha": config['lora']['alpha'], 
                "dropout": config['lora']['dropout']
            },
            "training": {
                "learning_rate": config['training']['learning_rate'],
                "epochs": config['training']['num_train_epochs'],
                "per_device_train_batch_size": config['training']['per_device_train_batch_size'],
                "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
                "max_length": config['training']['max_seq_length'],
            },
        }
    })
    
    # Create model card
    logger.info("📝 Creating model card and uploading to Hub...")
    trainer.create_model_card(
        model_name=hf_model_repo,
        tags=config['tags'],
        dataset_name=[config['model']['dataset_name']],
    )
    
    # Close W&B run
    wandb.finish()
    
    return model, trainer, args, compute_dtype, hf_model_repo


def merge_and_upload_model(model, trainer, args, compute_dtype, tokenizer, hf_model_repo):
    """Merge LoRA adapter with base model and upload to HF Hub"""
    logger.info("🔄 Merging LoRA adapter with base model...")
    
    # Free memory
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
    # Load and merge
    new_model = AutoPeftModelForCausalLM.from_pretrained(
        args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map="auto",
        cache_dir=HF_CACHE_DIR,
    )
    
    merged_model = new_model.merge_and_unload()
    
    # Upload to Hub
    merged_model.push_to_hub(hf_model_repo)
    tokenizer.push_to_hub(hf_model_repo)
    
    logger.info(f"🚀 Model successfully uploaded to: https://huggingface.co/{hf_model_repo}")
    
    return merged_model


def main():
    """Main training pipeline execution"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train language models with configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="RQ/SLM/configs/phi.yml",
        help="Path to configuration file (default: RQ/SLM/configs/phi.yml)"
    )
    # Add wandb sweep parameters
    parser.add_argument("--learning_rate", type=float, help="Learning rate for sweep")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank for sweep")
    parser.add_argument("--lora_alpha", type=int, help="LoRA alpha for sweep")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    model_name = config['model']['new_model']
    logger.info(f"🚀 Starting {model_name} fine-tuning pipeline...")
    
    # 1. Setup environment and authentication
    login(token=HF_HUB_TOKEN)
    wandb.login()
    wandb.init(
        project=config['wandb']['project'], 
        name=config['wandb']['experiment_name']
    )
    
    # 2. Set seed for reproducibility
    set_seed(config['training']['seed'])
    
    # 3. Prepare datasets
    processed_train_dataset, processed_test_dataset, tokenizer = prepare_datasets_from_config(config)
    
    # 4. Override config with command line arguments (wandb sweep parameters)
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.lora_rank is not None:
        config['lora']['rank'] = args.lora_rank
    if args.lora_alpha is not None:
        config['lora']['alpha'] = args.lora_alpha
    
    # Log the hyperparameters being used
    logger.info(f"Using hyperparameters: lr={config['training']['learning_rate']}, "
                f"rank={config['lora']['rank']}, alpha={config['lora']['alpha']}")
    
    # 4. Train the model
    model, trainer, training_args, compute_dtype, hf_model_repo = train_model(
        processed_train_dataset, tokenizer, config
    )
    
    # 5. Merge and upload model
    merged_model = merge_and_upload_model(
        model, trainer, training_args, compute_dtype, tokenizer, hf_model_repo
    )
    
    # Clean up
    del merged_model
    gc.collect()
    torch.cuda.empty_cache()
    
    logger.info("🎉 Training completed successfully!")
    logger.info(f"✅ Model uploaded to: https://huggingface.co/{hf_model_repo}")
    logger.info("")
    logger.info("📦 To convert to GGUF format, run:")
    logger.info(f"   python RQ/convert_to_gguf.py --model {config['model']['new_model'].split('-')[-1].lower()}")
    

if __name__ == "__main__":
    main()