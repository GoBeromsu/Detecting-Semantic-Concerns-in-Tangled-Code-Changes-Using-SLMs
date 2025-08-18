#!/usr/bin/env python3
"""
GGUF Conversion Script for Phi-4 Fine-tuned Models
Convert merged LoRA models to GGUF format and upload to Hugging Face
"""

import os
import subprocess
import logging
import sys
import gc
from pathlib import Path
from typing import Optional
from huggingface_hub import create_repo, upload_folder
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - Match train.py settings
USER = os.getenv("USER", "acq24bk")
FASTDATA_BASE = f"/mnt/parscratch/users/{USER}"
BASE_MODEL_ID = "microsoft/phi-4"
MERGED_MODEL_DIR = f"{FASTDATA_BASE}/models/merged_model"
GGUF_OUTPUT_DIR = f"{FASTDATA_BASE}/models/gguf"
LLAMA_CPP_DIR = f"{FASTDATA_BASE}/llama.cpp"
HF_CACHE_DIR = f"{FASTDATA_BASE}/.cache/huggingface/transformers"

# Model naming (align with train.py NEW_MODEL)
MODEL_NAME = "Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs"
HF_REPO_NAME = f"Berom0227/{MODEL_NAME}-gguf"
HF_ADAPTER_REPO = f"Berom0227/{MODEL_NAME}-adapter"  # LoRA adapter repository from train.py

# Memory optimization - CPU offloading enabled by default
USE_CPU_OFFLOAD = True

# Quantization options
QUANT_TYPES = ["q4_K_M","q8_0"]

def log_memory_usage(stage: str) -> None:
    """Simple memory logging"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        logger.info(f"[{stage}] GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")


def clear_memory() -> None:
    """Memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def check_dependencies() -> bool:
    """Check if required tools are available"""
    logger.info("Checking dependencies...")

    # Check if llama.cpp exists
    if not Path(LLAMA_CPP_DIR).exists():
        logger.error(f"llama.cpp not found at {LLAMA_CPP_DIR}")
        logger.info(
            "Please clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp"
        )
        return False

    # Check if convert_hf_to_gguf.py exists
    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.error(f"convert_hf_to_gguf.py not found at {convert_script}")
        return False

    # Check if quantize binary exists (cmake build structure)
    quantize_binary = Path(LLAMA_CPP_DIR) / "build" / "bin" / "llama-quantize"
    if not quantize_binary.exists():
        logger.warning(f"quantize binary not found at {quantize_binary}")
        logger.info(
            "Build llama.cpp first: cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release"
        )

    logger.info("Dependencies check completed")
    return True



def create_output_dir() -> None:
    """Create GGUF output directory"""
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)
    logger.info(f"Output directory: {GGUF_OUTPUT_DIR}")


def merge_lora_adapter() -> bool:
    """Load LoRA adapter from HF Hub and merge with base model with memory optimization"""
    logger.info(f"Loading LoRA adapter from {HF_ADAPTER_REPO} and merging with base model...")
    
    # Log initial memory state
    log_memory_usage("Initial")

    # Check if merged model already exists
    merged_model_path = Path(MERGED_MODEL_DIR)
    if (merged_model_path / "config.json").exists():
        logger.info("✅ Merged model already exists, skipping merge")
        return True

    try:
        # Clear memory before starting
        clear_memory()
        
        # Determine compute dtype
        if torch.cuda.is_bf16_supported():
            compute_dtype = torch.bfloat16
        else:
            compute_dtype = torch.float16
        
        # Smart CPU offloading: move memory-heavy, compute-light components
        device_map = {
            # Always CPU: memory-heavy but simple operations
            "model.embed_tokens": "cpu",  # Lookup table, pure memory
            "model.norm": "cpu",          # Small layer norm
            "lm_head": "cpu",            # Output projection, memory-heavy
        }
        
        # For transformer layers: keep computation on GPU, offload memory-heavy parts
        # GPU gets fewer layers but keeps compute-intensive operations
        for i in range(0, 12):  # Core layers with heavy computation stay on GPU
            device_map[f"model.layers.{i}"] = 0
        for i in range(12, 40):  # Upper layers to CPU (less critical for computation)
            device_map[f"model.layers.{i}"] = "cpu"
        
        logger.info(f"Loading LoRA adapter from {HF_ADAPTER_REPO} with CPU offloading...")
        model = AutoPeftModelForCausalLM.from_pretrained(
            HF_ADAPTER_REPO,
            low_cpu_mem_usage=True,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map=device_map,
            cache_dir=HF_CACHE_DIR,
        )
        
        log_memory_usage("After model loading")

        # Merge the adapter with base model
        logger.info("Starting LoRA merge process...")
        
        # Move model to CPU for merge
        logger.info("Moving model to CPU for merge...")
        model = model.cpu()
        clear_memory()
        
        merged_model = model.merge_and_unload()
        log_memory_usage("After merge")
        
        # Clean up original model immediately
        del model
        clear_memory()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR
        )
        
        # Save merged model and tokenizer
        logger.info("Saving merged model...")
        merged_model.save_pretrained(
            MERGED_MODEL_DIR, 
            trust_remote_code=True, 
            safe_serialization=True,
            max_shard_size="5GB"  # Limit shard size for memory efficiency
        )
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        
        # Clean up GPU memory aggressively
        del merged_model
        clear_memory()
        log_memory_usage("After cleanup")
        
        logger.info("✅ LoRA adapter merged successfully")
        return True

    except Exception as e:
        logger.error(f"LoRA merge failed: {e}")
        return False


def convert_to_gguf_fp16() -> Optional[str]:
    """Convert merged model to GGUF FP16 format"""
    logger.info("Converting to GGUF FP16...")

    output_file = f"{GGUF_OUTPUT_DIR}/{MODEL_NAME}-f16.gguf"

    # Check if fp16 file already exists
    if Path(output_file).exists():
        logger.info(f"✅ FP16 file already exists, skipping conversion: {output_file}")
        return output_file

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
        logger.debug(f"Convert output: {result.stdout}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Conversion failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return None


def quantize_model(fp16_file: str, quant_type: str) -> Optional[str]:
    """Quantize GGUF model"""
    logger.info(f"Quantizing to {quant_type}...")

    output_file = f"{GGUF_OUTPUT_DIR}/{MODEL_NAME}-{quant_type}.gguf"

    # Check if quantized file already exists
    if Path(output_file).exists():
        logger.info(
            f"✅ {quant_type} file already exists, skipping quantization: {output_file}"
        )
        return output_file

    # Use build/bin/llama-quantize from cmake build
    quantize_binary = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"

    if not Path(quantize_binary).exists():
        logger.error(f"Quantize binary not found: {quantize_binary}")
        logger.info(
            "Make sure llama.cpp is built with: cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release"
        )
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
        # Create repository if it doesn't exist
        create_repo(HF_REPO_NAME, repo_type="model", private=False, exist_ok=True)
        logger.info(f"✅ Repository {HF_REPO_NAME} ready")

        # Upload entire GGUF directory
        upload_folder(
            folder_path=GGUF_OUTPUT_DIR,
            repo_id=HF_REPO_NAME,
            repo_type="model",
            commit_message="Upload GGUF quantized models",
        )

        logger.info("✅ GGUF models uploaded")
        return True

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return False


def main():
    """Main conversion workflow"""
    logger.info("Starting GGUF conversion process with CPU offloading...")

    # Check prerequisites
    if not check_dependencies():
        logger.error("Dependencies check failed")
        sys.exit(1)

    # Create output directory
    create_output_dir()

    # Merge LoRA adapter with base model
    if not merge_lora_adapter():
        logger.error("LoRA adapter merge failed")
        sys.exit(1)

    # Convert to FP16
    fp16_file = convert_to_gguf_fp16()
    if not fp16_file:
        logger.error("FP16 conversion failed")
        sys.exit(1)

    # Quantize models
    # success_count = 1  # Count FP16 as success

    # for quant_type in QUANT_TYPES:
    #     quantized_file = quantize_model(fp16_file, quant_type)
    #     if quantized_file:
    #         success_count += 1
    #     else:
    #         logger.warning(f"Skipping {quant_type} quantization")

    # logger.info(f"✅ {success_count} model(s) created successfully")

    # Upload to Hugging Face Hub
    if upload_to_huggingface():
        logger.info("🎉 GGUF conversion and upload completed successfully!")
        logger.info(f"Model available at: https://huggingface.co/{HF_REPO_NAME}")
    else:
        logger.error("Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
