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

from datasets import load_dataset
from peft import LoraConfig, TaskType

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from huggingface_hub import login, create_repo, upload_folder
from dotenv import load_dotenv
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTTrainer, SFTConfig


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", None)
load_dotenv()
login(token=HF_HUB_TOKEN)

# Fixed Configuration (Infrastructure & Model Architecture)
MODEL_ID: str = "microsoft/phi-4"
MODEL_NAME: str = "microsoft/phi-4"
DATASET_NAME: str = (
    "Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
)

NEW_MODEL: str = "Semantic-Concern-SLM-Phi"
HF_MODEL_REPO: str = "Berom0227/" + NEW_MODEL
MODEL_OUTPUT_DIR = f"./outputs/{MODEL_NAME}-LoRA"  # Local training output only
HF_ADAPTER_REPO = f"Berom0227/{NEW_MODEL}-adapter"

HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", None)
HF_DATASETS_CACHE = os.getenv("HF_DATASETS_CACHE", None)

WANDB_PROJECT: str = "Untangling-Multi-Concern-Commits-with-Small-Language-Models"
EXPERIMENT_NAME: str = f"phi4-{NEW_MODEL.lower()}-lora"

DEVICE_MAP: str = "auto"
MAX_SEQ_LENGTH: int = 16_384
SEED: int = 42

TARGET_MODULES: list[str] = [
    "k_proj",
    "q_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]

# Experimental Variables (Tune these for different experiments)
LORA_RANK: int = 16  # Experiment with: 16, 32, 64, 128
LORA_ALPHA: int = 16  # Experiment with: 16, 32, 64, 128
LORA_DROPOUT: float = 0.05  # Experiment with: 0.05, 0.1, 0.2

LEARNING_RATE: float = 5e-5  # Experiment with: 5e-5, 1e-4, 2e-4
NUM_TRAIN_EPOCHS: int = 5  # Experiment with: 3, 5, 10
PER_DEVICE_TRAIN_BATCH_SIZE: int = 1  # Adjust based on GPU memory
GRADIENT_ACCUMULATION_STEPS: int = 16  # Adjust to maintain effective batch size
WARMUP_RATIO: float = 0.1  # Experiment with: 0.1, 0.2

# Dataset processing configuration
NUM_WORKERS: int = 4          # Parallel processing for dataset mapping

# Fixed Training Configuration
PER_DEVICE_EVAL_BATCH_SIZE: int = 2
LOGGING_STEPS: int = 100
EVAL_STEPS: int = 100

set_seed(SEED)

def prepare_datasets():
    """Load and process training datasets"""
    train_dataset = load_dataset(
        DATASET_NAME, split="train", cache_dir=HF_DATASETS_CACHE
    )
    test_dataset = load_dataset(DATASET_NAME, split="test", cache_dir=HF_DATASETS_CACHE)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        use_fast=True,
        cache_dir=HF_CACHE_DIR,
    )
    tokenizer.padding_side = "left"

    processed_train_dataset = prepare_dataset(train_dataset, tokenizer)
    processed_test_dataset = prepare_dataset(test_dataset, tokenizer)

    return processed_train_dataset, processed_test_dataset, tokenizer

def create_message_column(row: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap raw commit into system/user/assistant chat format."""
    user_content = (
        f"- given commit message:\n {row['commit_message']}\n Diff: {row['diff']}"
    )
    assistant_content = json.dumps(
        {"types": json.loads(row["types"])}, ensure_ascii=False
    )

    return {
        "messages": [
            {"role": "system", "content": get_system_prompt()},
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


def prepare_dataset(dataset, tokenizer) -> Any:
    """Pipeline: raw → messages → chatML text."""
    cols = dataset.column_names
    return dataset.map(
        create_message_column, 
        num_proc=NUM_WORKERS,  # Parallel processing for better performance
        desc=f"Building messages"
    ).map(
        lambda row: apply_chat_template(row, tokenizer),
        remove_columns=cols,
        num_proc=NUM_WORKERS,  # Parallel processing for better performance
        desc=f"Applying chatML",
    )

###########
# Training
###########

def train_model(processed_train_dataset, tokenizer):
    if torch.cuda.is_bf16_supported():
        compute_dtype, attn_implementation = torch.bfloat16, "flash_attention_2"
    else:
        compute_dtype, attn_implementation = torch.float16, "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map=DEVICE_MAP,
        attn_implementation=attn_implementation,
        cache_dir=HF_CACHE_DIR,
    )

    args = SFTConfig(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="no",
        optim="adamw_torch",
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=True,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        log_level="debug",
        save_strategy="epoch",
        logging_steps=LOGGING_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        eval_steps=EVAL_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        report_to="wandb",
        seed=SEED,
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
            "learning_rate": LEARNING_RATE,
            "num_train_epochs": NUM_TRAIN_EPOCHS,
            "logging_steps": LOGGING_STEPS,
            "eval_steps": EVAL_STEPS,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=args,
    )

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)

    if last_checkpoint is not None:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    trainer.save_model()

    adapter_artifact = wandb.Artifact(
        name=f"{NEW_MODEL.lower()}-adapter",
        type="model",
        metadata={
            "base_model": MODEL_ID,
            "peft": {"r": LORA_RANK, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
            "training": {
                "learning_rate": LEARNING_RATE,
                "epochs": NUM_TRAIN_EPOCHS,
                "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
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
        dataset_name=[DATASET_NAME],
    )

    # Close the W&B run cleanly
    wandb.finish()

    return model, trainer, args, compute_dtype


def merge_and_upload_model(model, trainer, args, compute_dtype, tokenizer):
    """Merge LoRA adapter with base model and upload to HF Hub"""
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    from peft import AutoPeftModelForCausalLM

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

    return merged_model

###############
# GGUF Conversion Workflow - Integrated with Training
###############
logger.info("Starting GGUF conversion process...")
logger.info("🌐 Using already loaded model from training (memory efficient)")

import subprocess
import shutil

LLAMA_CPP_DIR = os.getenv(
    "LLAMA_CPP_DIR", f"/mnt/parscratch/users/{os.getenv('USER', 'acq24bk')}/llama.cpp"
)
WORK_DIR = os.getenv("TMPDIR", "./tmp_gguf_conversion")
MERGED_MODEL_DIR = f"{WORK_DIR}/merged_model"
GGUF_OUTPUT_DIR = f"{WORK_DIR}/gguf_output"
QUANT_TYPES = ["q4_K_M", "q8_0"]
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
        logger.info(
            "Build llama.cpp first: cmake -B build && cmake --build build --config Release"
        )

    logger.info("Dependencies check completed")
    return True


def prepare_model_for_gguf(merged_model, tokenizer) -> bool:
    """Prepare already loaded model for GGUF conversion"""
    try:
        os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
        os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

        # Save the already merged model (no need to re-download from HF Hub)
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
    """Convert merged model to GGUF FP16 format"""
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
        create_repo(
            HF_REPO_NAME,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=HF_HUB_TOKEN,
        )
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


def run_gguf_conversion(merged_model, tokenizer) -> None:
    """Execute GGUF conversion workflow - Top-down clean approach"""
    logger.info("Starting GGUF conversion workflow...")

    # Check prerequisites first
    if not check_dependencies():
        logger.error("Dependencies check failed, skipping GGUF conversion")
        logger.info(
            "You can run GGUF conversion separately later using: python RQ/Phi/conver_to_gguf.py"
        )
        return

    # Prepare model for conversion
    if not prepare_model_for_gguf(merged_model, tokenizer):
        logger.error("Model preparation failed")
        return

    # Clean up memory before conversion
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


def main():
    """Main training pipeline execution"""
    logger.info("🚀 Starting Phi-4 fine-tuning pipeline...")
    
    # 1. Setup environment and authentication
    wandb.login()
    wandb.init(project=WANDB_PROJECT, name=EXPERIMENT_NAME)

    # 2. Prepare datasets
    processed_train_dataset, processed_test_dataset, tokenizer = prepare_datasets()
    
    # 3. Train the model
    model, trainer, args, compute_dtype = train_model(processed_train_dataset, tokenizer)
    
    # 4. Merge and upload model
    merged_model = merge_and_upload_model(model, trainer, args, compute_dtype, tokenizer)
    
    # 5. Optional GGUF conversion
    try:
        run_gguf_conversion(merged_model, tokenizer)
    except Exception as e:
        logger.error(f"GGUF conversion failed: {e}")

if __name__ == "__main__":
    main()