"""
Fine-tuning Phi-4 for Untangling Multi-Concern Commits

Dataset: Untangling Multi-Concern Commits with Small Language Models
Task: Predict reasoning and concern types from commit messages and diffs
Input: commit_message, diff → Output: reason, types

Usage: python train.py
"""

# Reference : https://github.com/microsoft/PhiCookBook/blob/main/code/03.Finetuning/Phi-3-finetune-lora-python.ipynb
import json
import sys
import logging
import os
from typing import Dict, Any
import torch
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


def get_system_prompt():
    return """
You are a software engineer classifying each minimal logical code unit — a semantically cohesive change — extracted from a tangled commit using the Conventional Commits specification (CCS).
# CCS Labels
- Purpose labels : the motivation behind making a code change
    - feat: Introduces new features to the codebase.
    - fix: Fixes bugs or faults in the codebase.
    - refactor: Restructures existing code without changing external behavior (e.g., improves readability, simplifies complexity, removes unused code).
- Object labels : the essence of the code changes that have been made
    - docs: Modifies documentation or text (e.g., fixes typos, updates comments or docs).
    - test: Modifies test files (e.g., adds or updates tests).
    - cicd: Updates CI (Continuous Integration) configuration files or scripts (e.g., `.travis.yml`, `.github/workflows`).
    - build: Affects the build system (e.g., updates dependencies, changes build configs or scripts).

# Instructions
1. For each code unit, review the change and determine the most appropriate CCS label.
2. If multiple CCS labels are possible, resolve the overlap by applying the following rule:
    - Purpose + Purpose: Choose the label that best reflects *why* the change was made — `fix` if resolving a bug, `feat` if adding new capability, `refactor` if improving structure without changing behavior.
    - Object + Object: Choose the label that reflects the *functional role* of the artifact being modified — e.g., even if changing build logic, editing a CI script should be labeled as `cicd`.
    - Purpose + Object: Use purpose labels **only** when the change affects application behavior or structure. Otherwise, assign the object label based on what was changed — not why or where.
3. Repeat step 1–2 for each code unit.
4. After all code units are labeled, return a unique set of assigned CCS labels for the entire commit
"""


logger = logging.getLogger(__name__)

# Model and dataset configuration
MODEL_ID: str = "microsoft/phi-4"
MODEL_NAME: str = "microsoft/phi-4"
DATASET_NAME: str = (
    "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
)

NEW_MODEL: str = "Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
HF_MODEL_REPO: str = "Berom0227/" + NEW_MODEL

# HPC storage configuration - Use fastdata area for large files
# Reference: Sheffield HPC Storage Guidelines
# https://docs.hpc.shef.ac.uk/en/latest/hpc/filestore.html
# Fastdata areas (/mnt/parscratch/users/$USER/) are designed for large files and avoid "Disk quota exceeded" errors in home directories
USER = os.getenv("USER", "acq24bk")  # Fallback to your username if USER env var not set
FASTDATA_BASE = os.getenv("FASTDATA_BASE", f"/mnt/parscratch/users/{USER}")
MODEL_OUTPUT_DIR = f"{FASTDATA_BASE}/models/{MODEL_NAME}-LoRA"
MERGED_MODEL_DIR = f"{FASTDATA_BASE}/models/merged_model"
HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", f"{FASTDATA_BASE}/.cache/huggingface/transformers")
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", f"{FASTDATA_BASE}/.cache/huggingface/datasets")

# Create necessary directories
os.makedirs(f"{FASTDATA_BASE}/models", exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)

# Experiment tracking configuration
WANDB_PROJECT: str = "Untangling-Multi-Concern-Commits-with-Small-Language-Models"
EXPERIMENT_NAME: str = f"phi4-{NEW_MODEL.lower()}-lora"

DEVICE_MAP: str = "auto"

# 'lora_r' is the dimension of the LoRA attention.
LORA_RANK: int = 16

# 'lora_alpha' is the alpha parameter for LoRA scaling.
LORA_ALPHA: int = 16

# 'lora_dropout' is the dropout probability for LoRA layers.
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
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Login to Hugging Face Hub using token from environment
login(token=os.getenv("HF_HUB_TOKEN"))

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
    user_content = f"# Commit Message\n{row['commit_message']}\n\n# Diff\n```diff\n{row['diff']}\n```\n"
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
    eval_strategy="steps",
    do_eval=True,
    optim="adamw_torch",
    per_device_train_batch_size=1,  # Reduce memory usage
    gradient_accumulation_steps=16,  # Compensate for small batch size
    gradient_checkpointing=True,  # Further reduce memory usage
    per_device_eval_batch_size=2,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    eval_steps=100,
    num_train_epochs=3,
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
    eval_dataset=processed_test_dataset,
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

# Create model card before freeing trainer
print("📝 Creating model card and uploading to Hub...")
trainer.create_model_card(
    tags=["phi-4", "fine-tuned", "commit-analysis", "software-engineering"],
    dataset_tags=["Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"],
    language="en",
    license="mit",
    base_model="microsoft/phi-4",
)

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

# Save the merged model locally
merged_model.save_pretrained(
    MERGED_MODEL_DIR, trust_remote_code=True, safe_serialization=True
)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

# Push the merged model to Hugging Face Hub
merged_model.push_to_hub(HF_MODEL_REPO)
tokenizer.push_to_hub(HF_MODEL_REPO)

print(f"🚀 Model successfully uploaded to: https://huggingface.co/{HF_MODEL_REPO}")
