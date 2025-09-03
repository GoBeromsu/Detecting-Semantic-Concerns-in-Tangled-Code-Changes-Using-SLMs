"""
Fine-tuning Qwen3-14B for Untangling Multi-Concern Commits

Dataset: Untangling Multi-Concern Commits with Small Language Models
Task: Predict reasoning and concern types from commit messages and diffs
Input: commit_message, diff → Output: types

Usage: python train.py
"""

import json
import sys
import logging
import os
import subprocess
import gc
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from utils.prompt import get_system_prompt
import wandb

from datasets import load_dataset
from peft import LoraConfig, TaskType


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint

from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

MODEL_ID: str = "Qwen/Qwen3-14B"
MODEL_NAME: str = "Qwen/Qwen3-14B"
DATASET_NAME: str = (
    "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
)

NEW_MODEL: str = "Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
HF_MODEL_REPO: str = "Berom0227/" + NEW_MODEL


# Training output configuration - HF Hub delegation
MODEL_OUTPUT_DIR = f"./outputs/{MODEL_NAME}-LoRA"  # Local training output only
HF_ADAPTER_REPO = f"Berom0227/{NEW_MODEL}-adapter"

# HF Cache delegation to environment variables
# Cache directories managed via environment variables and symlinks
HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", None)
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", None)

# Experiment tracking configuration
WANDB_PROJECT: str = "Untangling-Multi-Concern-Commits-with-Small-Language-Models"
EXPERIMENT_NAME: str = f"qwen3-14b-{NEW_MODEL.lower()}-lora"

DEVICE_MAP: str = "auto"

# LoRA hyperparameters optimized for Qwen3-14B (hidden_dim=5120)
LORA_RANK: int = 128
LORA_ALPHA: int = 256
LORA_DROPOUT: float = 0.05

# 'target_modules' is a list of the modules in the model that will be replaced with LoRA layers.
TARGET_MODULES: list[str] = [
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]

# Training configuration
MAX_SEQ_LENGTH: int = 16_384
NUM_WORKERS: int = 4

set_seed(1234)

######################
# Connect to Hugging Face Hub
######################
from huggingface_hub import login, create_repo, upload_folder
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face Hub using token from environment
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", None)
login(token=HF_HUB_TOKEN)

######################
# Setup Experiment Tracking
######################
# Initialize Weights & Biases following reference notebook pattern
wandb.login()
wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME)

## Dataset Loading
train_dataset = load_dataset(
    DATASET_NAME,
    split="train",
    cache_dir=HF_DATASETS_CACHE,
)

test_dataset = load_dataset(
    DATASET_NAME,
    split="test",
    cache_dir=HF_DATASETS_CACHE,
)

######################
# Step 1: Dataset Preprocessing Setup
######################
# Load tokenizer for data formatting
tokenizer_id = MODEL_ID
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=HF_CACHE_DIR)
tokenizer.padding_side = "right"

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


######################
# Step 2: Dataset Processing for Fine-tuning
######################
def create_message_column(row) -> Dict[str, Any]:
    """Create messages column for multi-concern commit classification."""
    # Create structured prompt for commit analysis
    # user_content = f"# Commit Message\n{row['commit_message']}\n\n# Diff\n```diff\n{row['diff']}\n```\n"
    user_content = (
        f"- given commit message:\n {row['commit_message']}\n Diff: {row['diff']}"
    )
    parsed_types = json.loads(row["types"])
    assistant_content = json.dumps({"types": parsed_types}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}


def format_dataset_chatml(row) -> Dict[str, Any]:
    """Format dataset with chat template for multi-concern commit classification."""
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    }


column_names = list(train_dataset.features)

# Apply chat template formatting to training data
train_dataset_with_messages = train_dataset.map(
    create_message_column,
    num_proc=NUM_WORKERS,
    desc="Creating messages column for multi-concern commit train data",
)

processed_train_dataset = train_dataset_with_messages.map(
    format_dataset_chatml,
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit train data",
)

# Apply chat template formatting to test data (optional)
test_dataset_with_messages = test_dataset.map(
    create_message_column,
    num_proc=NUM_WORKERS,
    desc="Creating messages column for multi-concern commit test data",
)

processed_test_dataset = test_dataset_with_messages.map(
    format_dataset_chatml,
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit test data",
)

######################
# Step 3: Model and Tokenizer Setup for Training
######################
# Configure precision and attention implementation
if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    compute_dtype = torch.float16
    attn_implementation = "sdpa"

# Load tokenizer for training (with specific settings for causal LM)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    add_eos_token=True,
    use_fast=True,
    cache_dir=HF_CACHE_DIR,
)
tokenizer.padding_side = "left"  # Left padding for causal LM

# Load pre-trained Qwen3-14B model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
    attn_implementation=attn_implementation,
    cache_dir=HF_CACHE_DIR,
)


######################
# Step 4: LoRA Configuration and Training Setup
######################


# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.CAUSAL_LM,
    target_modules=TARGET_MODULES,
)

# Training configuration
args = SFTConfig(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="no",
    optim="adamw_torch",
    per_device_train_batch_size=1,  # Reduce memory usage
    gradient_accumulation_steps=16,  # Compensate for small batch size
    gradient_checkpointing=True,  # Further reduce memory usage
    per_device_eval_batch_size=2,
    log_level="debug",
    save_strategy="no",  # HPC guarantees full training; checkpoint loading requires PyTorch 2.6.0+ (CUDA 12.1+ only) - see https://pytorch.org/get-started/previous-versions/
    logging_steps=100,
    learning_rate=5e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    eval_steps=100,
    num_train_epochs=5,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="wandb",
    seed=42,
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=HF_MODEL_REPO + "-adapter",
    max_length=MAX_SEQ_LENGTH,
    packing=True,
)

# Update W&B config with hyperparameters
wandb.config.update(
    {
        "model_id": MODEL_ID,
        "learning_rate": 1e-4,
        "num_train_epochs": 3,
        "logging_steps": 100,
        "eval_steps": 100,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "lora_r": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "max_length": MAX_SEQ_LENGTH,
    },
    allow_val_change=True,
)

######################
# Step 5: Fine-tuning Execution
######################
# Initialize SFT trainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=args,
)

# Check for existing checkpoints and resume if available
last_checkpoint = None
if os.path.isdir(args.output_dir):
    last_checkpoint = get_last_checkpoint(args.output_dir)

# Start training
if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# Save the trained LoRA adapter
trainer.save_model()

# Log the trained adapter checkpoint directory as a W&B Artifact for traceability
adapter_artifact = wandb.Artifact(
    name=f"{NEW_MODEL.lower()}-adapter",
    type="model",
    metadata={
        "base_model": MODEL_ID,
        "peft": {"r": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        "training": {
            "learning_rate": 1e-4,
            "epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_length": MAX_SEQ_LENGTH,
        },
    },
)
adapter_artifact.add_dir(args.output_dir)
wandb.log_artifact(adapter_artifact)

# Create model card before freeing trainer
print("📝 Creating model card and uploading to Hub...")
trainer.create_model_card(
    model_name=HF_MODEL_REPO,
    tags=["qwen3-14b", "fine-tuned", "commit-analysis", "software-engineering"],
    dataset_name=[
        "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
    ],
)

wandb.finish()

######################
# Step 6: Model Merging and Upload to Hub
######################
# Free up GPU memory before merging
del model
del trainer
import gc

gc.collect()
torch.cuda.empty_cache()
gc.collect()

from peft import AutoPeftModelForCausalLM

# Load the trained LoRA adapter and merge with base model
new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
    cache_dir=HF_CACHE_DIR,
)

# Merge LoRA weights into the base model
merged_model = new_model.merge_and_unload()

# Upload merged model to Hugging Face Hub
merged_model.push_to_hub(HF_MODEL_REPO)
tokenizer.push_to_hub(HF_MODEL_REPO)

print(f"🚀 Model successfully uploaded to: https://huggingface.co/{HF_MODEL_REPO}")

######################
# Step 7: GGUF Conversion (Optional)
######################
logger.info("Starting GGUF conversion process...")

# Import GGUF-related modules
import subprocess
import shutil
import time

# GGUF conversion configuration
LLAMA_CPP_DIR = os.getenv(
    "LLAMA_CPP_DIR", f"/mnt/parscratch/users/{os.getenv('USER', 'acq24bk')}/llama.cpp"
)
WORK_DIR = os.getenv("TMPDIR", "./tmp_gguf_conversion")
MERGED_MODEL_DIR = f"{WORK_DIR}/merged_model"
GGUF_OUTPUT_DIR = f"{WORK_DIR}/gguf_output"
QUANT_TYPES = ["q4_K_M", "q8_0"]
MODEL_NAME = NEW_MODEL  # Use consistent naming with conver_to_gguf.py
HF_REPO_NAME = f"Berom0227/{MODEL_NAME}-gguf"


# GGUF utility functions
def clear_memory() -> None:
    """Memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_dependencies() -> bool:
    """Check llama.cpp dependencies"""
    logger.info("Checking dependencies...")

    if not Path(LLAMA_CPP_DIR).exists():
        logger.error(f"llama.cpp not found at {LLAMA_CPP_DIR}")
        return False

    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.error(f"convert_hf_to_gguf.py not found")
        return False

    logger.info("Dependencies check completed")
    return True


def prepare_model_for_gguf() -> bool:
    """Save merged model for GGUF conversion"""
    try:
        os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
        os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

        merged_model.save_pretrained(
            MERGED_MODEL_DIR, safe_serialization=True, max_shard_size="2GB"
        )
        tokenizer.save_pretrained(MERGED_MODEL_DIR)

        logger.info("✅ Model prepared for GGUF conversion")
        return True
    except Exception as e:
        logger.error(f"Failed to prepare model: {e}")
        return False


def convert_to_gguf_fp16() -> Optional[str]:
    """Convert to GGUF FP16 format"""
    logger.info("Converting to GGUF FP16...")

    output_file = f"{GGUF_OUTPUT_DIR}/{MODEL_NAME}-f16.gguf"
    convert_script = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"

    cmd = [
        "python",
        convert_script,
        MERGED_MODEL_DIR,
        "--outfile",
        output_file,
        "--outtype",
        "f16",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✅ FP16 conversion completed")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        return None


def quantize_model(fp16_file: str, quant_type: str) -> Optional[str]:
    """Quantize GGUF model"""
    logger.info(f"Quantizing to {quant_type}...")

    output_file = f"{GGUF_OUTPUT_DIR}/{MODEL_NAME}-{quant_type}.gguf"
    quantize_binary = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"

    if not Path(quantize_binary).exists():
        logger.error(f"Quantize binary not found: {quantize_binary}")
        return None

    cmd = [quantize_binary, fp16_file, output_file, quant_type]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {quant_type} quantization completed")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed: {e}")
        return None


def upload_to_huggingface() -> bool:
    """Upload GGUF models to Hub"""
    logger.info(f"Uploading to {HF_REPO_NAME}...")

    try:
        create_repo(
            HF_REPO_NAME,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=HF_HUB_TOKEN,
        )
        upload_folder(
            folder_path=GGUF_OUTPUT_DIR,
            repo_id=HF_REPO_NAME,
            repo_type="model",
            commit_message="Upload GGUF quantized models",
            token=HF_HUB_TOKEN,
        )
        logger.info("✅ GGUF models uploaded")
        return True
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


def run_gguf_conversion_workflow() -> None:
    """Execute complete GGUF conversion workflow"""
    logger.info("Starting GGUF conversion workflow...")

    if not check_dependencies():
        logger.error("Dependencies check failed, skipping GGUF conversion")
        logger.info(
            "💡 You can run GGUF conversion separately later using: python RQ/Qwen/conver_to_gguf.py"
        )
        return

    if not prepare_model_for_gguf():
        logger.error("Model preparation failed")
        return

    # Clean up memory before conversion
    del merged_model, new_model
    clear_memory()

    # Convert to FP16
    fp16_file = convert_to_gguf_fp16()
    if not fp16_file:
        logger.error("FP16 conversion failed")
        return

    # Quantize models
    success_count = 1
    for quant_type in QUANT_TYPES:
        quantized_file = quantize_model(fp16_file, quant_type)
        if quantized_file:
            success_count += 1

    logger.info(f"✅ {success_count} model(s) created successfully")

    # Upload to Hub
    if upload_to_huggingface():
        logger.info("🎉 GGUF conversion completed!")
        logger.info(f"GGUF models available at: https://huggingface.co/{HF_REPO_NAME}")

    # Cleanup
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    logger.info("🧹 Temporary workspace cleaned up")


# Execute GGUF conversion workflow
try:
    run_gguf_conversion_workflow()
except Exception as e:
    logger.error(f"GGUF conversion failed: {e}")
    logger.info(
        "💡 You can run GGUF conversion separately later using: python RQ/Qwen/conver_to_gguf.py"
    )

print("🎉 Training and GGUF conversion workflow completed!")
