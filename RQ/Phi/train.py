"""
Fine-tuning Phi-4 for Untangling Multi-Concern Commits

Dataset: Untangling Multi-Concern Commits with Small Language Models
Task: Predict reasoning and concern types from commit messages and diffs
Input: commit_message, diff → Output: types

Usage: python train.py
"""

# Reference : https://github.com/microsoft/PhiCookBook/blob/main/code/03.Finetuning/Phi-3-finetune-lora-python.ipynb
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

# 'load_dataset' is a function from the 'datasets' library by Hugging Face which allows you to load a dataset.
from datasets import load_dataset

# 'LoraConfig' and 'prepare_model_for_kbit_training' are from the 'peft' library.
# 'LoraConfig' is used to configure the LoRA (Learning from Random Architecture) model.
# 'prepare_model_for_kbit_training' is a function that prepares a model for k-bit training.
# 'TaskType' contains differenct types of tasks supported by PEFT
# 'PeftModel' base model class for specifying the base Transformer model and configuration to apply a PEFT method to.
from peft import LoraConfig, TaskType

# Several classes and functions are imported from the 'transformers' library by Hugging Face.
# 'AutoModelForCausalLM' is a class that provides a generic transformer model for causal language modeling.
# 'AutoTokenizer' is a class that provides a generic tokenizer class.
# 'BitsAndBytesConfig' is a class for configuring the Bits and Bytes optimizer.
# 'TrainingArguments' is a class that defines the arguments used for training a model.
# 'set_seed' is a function that sets the seed for generating random numbers.
# 'pipeline' is a function that creates a pipeline that can process data and make predictions.
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

# Import utilities for checkpoint management following HF best practices
from transformers.trainer_utils import get_last_checkpoint

# 'SFTTrainer' is a class from the 'trl' library that provides a trainer for soft fine-tuning.
from trl import SFTTrainer, SFTConfig

logger = logging.getLogger(__name__)

# Model and dataset configuration
MODEL_ID: str = "microsoft/phi-4"
MODEL_NAME: str = "microsoft/phi-4"
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
EXPERIMENT_NAME: str = f"phi4-{NEW_MODEL.lower()}-lora"

DEVICE_MAP: str = "auto"

# LoRA hyperparameters optimized for Phi-4 (hidden_dim=5120)
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

# Load tokenizer to prepare the dataset (First tokenizer - for data formatting only)
tokenizer_id = MODEL_ID

# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'tokenizer_id' is passed as an argument to specify which tokenizer to load.
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, cache_dir=HF_CACHE_DIR)

# 'tokenizer.padding_side' is a property that specifies which side to pad when the input sequence is shorter than the maximum sequence length.
# Setting it to 'right' means that padding tokens will be added to the right (end) of the sequence.
# This is done to prevent warnings that can occur when the padding side is not explicitly set.
tokenizer.padding_side = "right"

###############
# Setup logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


##################
# Data Processing
##################
def create_message_column(row) -> Dict[str, Any]:
    """Create messages column for multi-concern commit classification."""
    # Create structured prompt for commit analysis
    # user_content = f"# Commit Message\n{row['commit_message']}\n\n# Diff\n```diff\n{row['diff']}\n```\n"
    user_content = f"- given commit message:\n {row['commit_message']}\n Diff: {row['diff']}"
    parsed_types = json.loads(row["types"])
    assistant_content = json.dumps({"types": parsed_types}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}




# 'format_dataset_chatml' is a function that takes a row from the dataset and returns a dictionary
# with a 'text' key and a string of formatted chat messages as its value.
# Uses the first tokenizer with tokenize=False to create formatted strings (not token IDs)
def format_dataset_chatml(row) -> Dict[str, Any]:
    """Format dataset with chat template for multi-concern commit classification."""
    return {
        "text": tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    }


column_names = list(train_dataset.features)

# Step 1: Create messages column
train_dataset_with_messages = train_dataset.map(
    create_message_column,
    num_proc=NUM_WORKERS,
    desc="Creating messages column for multi-concern commit train data",
)

# Step 2: Format with chat template
processed_train_dataset = train_dataset_with_messages.map(
    format_dataset_chatml,
    num_proc=NUM_WORKERS,
    remove_columns=column_names,
    desc="Applying chat template to multi-concern commit train data",
)

# Process test dataset
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

###########
# Training
###########

if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    compute_dtype = torch.float16
    attn_implementation = "sdpa"


# Second tokenizer - for actual model training (different from data formatting tokenizer above)
# 'AutoTokenizer.from_pretrained' is a method that loads a tokenizer from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which tokenizer to load.
# 'trust_remote_code' is set to True to trust the remote code in the tokenizer files.
# 'add_eos_token' is set to True to add an end-of-sentence token to the tokenizer.
# 'use_fast' is set to True to use the fast version of the tokenizer.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, trust_remote_code=True, add_eos_token=True, use_fast=True, cache_dir=HF_CACHE_DIR
)

# The padding token is set to the unknown token.
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
# The padding side is set to 'left', meaning that padding tokens will be added to the left (start) of the sequence.
# Left padding is preferred for causal LM training (different from 'right' padding used in data formatting)
tokenizer.padding_side = "left"

# 'AutoModelForCausalLM.from_pretrained' is a method that loads a pre-trained model for causal language modeling from the Hugging Face Model Hub.
# 'model_id' is passed as an argument to specify which model to load.
# 'torch_dtype' is set to the compute data type determined earlier.
# 'trust_remote_code' is set to True to trust the remote code in the model files.
# 'device_map' is passed as an argument to specify the device mapping for distributed training.
# 'attn_implementation' is set to the attention implementation determined earlier.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
    attn_implementation=attn_implementation,
    cache_dir=HF_CACHE_DIR,
)


# This code block is used to define the training arguments for the model.

# 'TrainingArguments' is a class that holds the arguments for training a model.
# 'output_dir' is the directory where the model and its checkpoints will be saved.
# 'evaluation_strategy' is set to "steps", meaning that evaluation will be performed after a certain number of training steps.
# 'do_eval' is set to True, meaning that evaluation will be performed.
# 'optim' is set to "adamw_torch", meaning that the AdamW optimizer from PyTorch will be used.
# 'per_device_train_batch_size' and 'per_device_eval_batch_size' are set to 8, meaning that the batch size for training and evaluation will be 8 per device.
# 'gradient_accumulation_steps' is set to 4, meaning that gradients will be accumulated over 4 steps before performing a backward/update pass.
# 'log_level' is set to "debug", meaning that all log messages will be printed.
# 'save_strategy' is set to "epoch", meaning that the model will be saved after each epoch.
# 'logging_steps' is set to 100, meaning that log messages will be printed every 100 steps.
# 'learning_rate' is set to 1e-4, which is the learning rate for the optimizer.
# 'fp16' is set to the opposite of whether bfloat16 is supported on the current CUDA device.
# 'bf16' is set to whether bfloat16 is supported on the current CUDA device.
# 'eval_steps' is set to 100, meaning that evaluation will be performed every 100 steps.
# 'num_train_epochs' is set to 3, meaning that the model will be trained for 3 epochs.
# 'warmup_ratio' is set to 0.1, meaning that 10% of the total training steps will be used for the warmup phase.
# 'lr_scheduler_type' is set to "linear", meaning that a linear learning rate scheduler will be used.
# 'report_to' is set to "wandb", meaning that training and evaluation metrics will be reported to Weights & Biases.
# 'seed' is set to 42, which is the seed for the random number generator.

# LoraConfig object is created with the following parameters:
# 'r' (rank of the low-rank approximation) is set to 16,
# 'lora_alpha' (scaling factor) is set to 16,
# 'lora_dropout' dropout probability for Lora layers is set to 0.05,
# 'task_type' (set to TaskType.CAUSAL_LM indicating the task type),
# 'target_modules' (the modules to which LoRA is applied) choosing linear layers except the output layer..


args = SFTConfig(
    output_dir=MODEL_OUTPUT_DIR,
    eval_strategy="no",
    # do_eval=True,
    optim="adamw_torch",
    per_device_train_batch_size=1,  # Reduce memory usage
    gradient_accumulation_steps=16,  # Compensate for small batch size
    gradient_checkpointing=True,  # Further reduce memory usage
    per_device_eval_batch_size=2,
    log_level="debug",
    save_strategy="epoch",
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

# Make the most relevant hyperparameters visible in the W&B run config
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

peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    task_type=TaskType.CAUSAL_LM,
    target_modules=TARGET_MODULES,
)

# 'model' is the model that will be trained.
# 'train_dataset' and 'eval_dataset' are the datasets that will be used for training and evaluation, respectively.
# 'peft_config' is the configuration for peft, which is used for instruction tuning.
# 'processing_class' is the tokenizer that will be used to tokenize the input text.
# This uses the second tokenizer (training tokenizer) to convert text strings to token IDs
# 'args' are the training arguments that were defined earlier.

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_train_dataset,
    # eval_dataset=processed_test_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=args,
)

# 'trainer.train()' is a method that starts the training of the model.
# It uses the training dataset, evaluation dataset, and training arguments that were provided when the trainer was initialized.

# trainer.train() 전에 체크포인트 확인
last_checkpoint = None
if os.path.isdir(args.output_dir):
    last_checkpoint = get_last_checkpoint(args.output_dir)

if last_checkpoint is not None:
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    trainer.train()

# 'trainer.save_model()' is a method that saves the trained model locally.
# The model will be saved in the directory specified by 'output_dir' in the training arguments.
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
    tags=["phi-4", "fine-tuned", "commit-analysis", "software-engineering"],
    dataset_name=["Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"],
)

# Close the W&B run cleanly
wandb.finish()
###############
# Merge Model and Adapter
###############
# Free up GPU memory before merging
del model
del trainer

import gc

gc.collect()
gc.collect()

torch.cuda.empty_cache()
gc.collect()

# Import AutoPeftModelForCausalLM for merging
from peft import AutoPeftModelForCausalLM

# Load the trained adapter model
new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
    device_map=DEVICE_MAP,
    cache_dir=HF_CACHE_DIR,
)

# Merge the model and adapter
merged_model = new_model.merge_and_unload()

merged_model.push_to_hub(HF_MODEL_REPO)
tokenizer.push_to_hub(HF_MODEL_REPO)

print(f"🚀 Model successfully uploaded to: https://huggingface.co/{HF_MODEL_REPO}")

###############
# GGUF Conversion Workflow - Integrated with Training
###############
logger.info("Starting GGUF conversion process...")
logger.info("🌐 Using already loaded model from training (memory efficient)")

# Import GGUF-related modules
import subprocess
import shutil
import time

# GGUF conversion configuration  
LLAMA_CPP_DIR = os.getenv("LLAMA_CPP_DIR", f"/mnt/parscratch/users/{os.getenv('USER', 'acq24bk')}/llama.cpp")
WORK_DIR = os.getenv("TMPDIR", "./tmp_gguf_conversion")
MERGED_MODEL_DIR = f"{WORK_DIR}/merged_model"
GGUF_OUTPUT_DIR = f"{WORK_DIR}/gguf_output"
QUANT_TYPES = ["q4_K_M","q8_0"]
MODEL_NAME = NEW_MODEL  # Use consistent naming with conver_to_gguf.py
HF_REPO_NAME = f"Berom0227/{MODEL_NAME}-gguf"

def clear_memory() -> None:
    """CPU memory cleanup for CPU-only workflow"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_dependencies() -> bool:
    """Check if required tools are available"""
    logger.info("Checking dependencies...")
    
    if not Path(LLAMA_CPP_DIR).exists():
        logger.error(f"llama.cpp not found at {LLAMA_CPP_DIR}")
        logger.info("Please set LLAMA_CPP_DIR environment variable or build llama.cpp")
        return False

    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.error(f"convert_hf_to_gguf.py not found at {convert_script}")
        return False

    quantize_binary = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
    if not quantize_binary.exists():
        logger.warning(f"quantize binary not found at {quantize_binary}")
        logger.info("Build llama.cpp first: cmake -B build && cmake --build build --config Release")

    logger.info("Dependencies check completed")
    return True

def prepare_model_for_gguf() -> bool:
    """Prepare already loaded model for GGUF conversion"""
    try:
        os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
        os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
        
        # Save the already merged model (no need to re-download from HF Hub)
        merged_model.save_pretrained(MERGED_MODEL_DIR, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        
        logger.info("✅ Model prepared for GGUF conversion")
        return True
    except Exception as e:
        logger.error(f"Failed to prepare model: {e}")
        return False

def convert_to_gguf_fp16() -> Optional[str]:
    """Convert merged model to GGUF FP16 format"""
    logger.info("Converting to GGUF FP16...")
    
    output_file = f"{GGUF_OUTPUT_DIR}/{MODEL_NAME}-f16.gguf"
    convert_script = f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py"

    cmd = [
        "python", convert_script, MERGED_MODEL_DIR,
        "--outfile", output_file, "--outtype", "f16"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {quant_type} quantization completed")
        logger.debug(f"Quantize output: {result.stdout}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Quantization failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None

def upload_to_huggingface() -> bool:
    """Upload GGUF folder to Hugging Face Hub"""
    logger.info(f"Uploading to {HF_REPO_NAME}...")

    try:
        create_repo(HF_REPO_NAME, repo_type="model", private=False, exist_ok=True, token=HF_HUB_TOKEN)
        logger.info(f"✅ Repository {HF_REPO_NAME} ready")

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
    """Execute GGUF conversion workflow - Top-down clean approach"""
    logger.info("Starting GGUF conversion workflow...")
    
    # Check prerequisites first
    if not check_dependencies():
        logger.error("Dependencies check failed, skipping GGUF conversion")
        logger.info("💡 You can run GGUF conversion separately later using: python RQ/Phi/conver_to_gguf.py")
        return
    
    # Prepare model for conversion
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
    success_count = 1  # Count FP16 as success
    for quant_type in QUANT_TYPES:
        quantized_file = quantize_model(fp16_file, quant_type)
        if quantized_file:
            success_count += 1
        else:
            logger.warning(f"Skipping {quant_type} quantization")
    
    logger.info(f"✅ {success_count} model(s) created successfully")
    
    # Upload to HF Hub
    if not upload_to_huggingface():
        logger.error("GGUF upload failed")
        return
    
    # Success - cleanup and finish
    logger.info("🎉 GGUF conversion and upload completed successfully!")
    logger.info(f"GGUF models available at: https://huggingface.co/{HF_REPO_NAME}")
    
    # Cleanup temporary workspace
    shutil.rmtree(WORK_DIR, ignore_errors=True)
    logger.info("🧹 Temporary workspace cleaned up")

# Execute GGUF conversion workflow
try:
    run_gguf_conversion_workflow()
except Exception as e:
    logger.error(f"GGUF conversion failed: {e}")
    logger.info("💡 You can run GGUF conversion separately later using: python RQ/Phi/conver_to_gguf.py")

print("🎉 Training and GGUF conversion workflow completed!")

