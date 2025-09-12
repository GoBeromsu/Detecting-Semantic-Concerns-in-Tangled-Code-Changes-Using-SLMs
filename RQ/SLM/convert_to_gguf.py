#!/usr/bin/env python3
"""
Unified GGUF Conversion Script for Fine-tuned Models
Convert merged LoRA models to GGUF format and upload to Hugging Face

Usage:
    python convert_to_gguf.py --model phi
    python convert_to_gguf.py --model qwen
"""

import argparse
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
import dotenv

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configurations - THE ONLY DIFFERENCE BETWEEN PHI AND QWEN
MODEL_CONFIGS = {
    "phi": {
        "base_model_id": "microsoft/phi-4",
        "model_name": "Semantic-Concern-SLM-Phi",
    },
    "qwen": {
        "base_model_id": "Qwen/Qwen3-14B",
        "model_name": "Semantic-Concern-SLM-Qwen",
    }
}

# Get model selection from command line
parser = argparse.ArgumentParser(
    description="Convert fine-tuned models to GGUF format"
)
parser.add_argument(
    "--model",
    type=str,
    default="phi",  # Default to phi model
    choices=["phi", "qwen"],
    help="Model to convert (default: phi)"
)
args = parser.parse_args()

# Set model-specific configuration
config = MODEL_CONFIGS[args.model]
BASE_MODEL_ID = config["base_model_id"]
MODEL_NAME = config["model_name"]

# Configuration - HF Hub delegation
HF_REPO_NAME = f"Berom0227/{MODEL_NAME}-gguf"
HF_ADAPTER_REPO = f"Berom0227/{MODEL_NAME}-adapter"
HF_HUB_TOKEN = os.getenv("HF_HUB_TOKEN", None)

# Temporary workspace - environment dependent
WORK_DIR = os.getenv("TMPDIR", "./tmp_gguf_conversion")
MERGED_MODEL_DIR = f"{WORK_DIR}/merged_model"
GGUF_OUTPUT_DIR = f"{WORK_DIR}/gguf_output"

# Dependencies - environment dependent  
LLAMA_CPP_DIR = os.getenv("LLAMA_CPP_DIR", os.path.expanduser("~/llama.cpp"))

# HF Cache delegation to environment variables
HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", None)

# Quantization options
QUANT_TYPES = ["q4_K_M","q8_0"]

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
    """Load LoRA adapter from HF Hub and merge with base model"""
    logger.info(f"Loading LoRA adapter from {HF_ADAPTER_REPO} and merging...")
    
    logger.info(f"Memory checkpoint")

    try:
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # bfloat16: 2x memory reduction vs float32, native H100 support  
        compute_dtype = torch.bfloat16
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            HF_ADAPTER_REPO,
            low_cpu_mem_usage=True,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map="auto",
            cache_dir=HF_CACHE_DIR,
        )
        
        logger.info(f"Memory checkpoint")

        logger.info("Starting LoRA merge...")
        merged_model = model.merge_and_unload()
        logger.info(f"Memory checkpoint")
        
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE_DIR
        )
        
        logger.info("Saving merged model...")
        merged_model.save_pretrained(
            MERGED_MODEL_DIR, 
            trust_remote_code=True, 
            safe_serialization=True,
            max_shard_size="2GB"
        )
        tokenizer.save_pretrained(MERGED_MODEL_DIR)
        
        del merged_model, tokenizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info(f"Memory checkpoint")
        
        logger.info("✅ LoRA adapter merged successfully")
        return True

    except Exception as e:
        logger.error(f"LoRA merge failed: {e}")
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
    quantize_binary = f"{LLAMA_CPP_DIR}/build/bin/llama-quantize"

    if not Path(quantize_binary).exists():
        logger.error(f"Quantize binary not found: {quantize_binary}")
        logger.info(
            "Build llama.cpp: cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release"
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
        create_repo(HF_REPO_NAME, repo_type="model", private=False, exist_ok=True,token=HF_HUB_TOKEN)
        logger.info(f"✅ Repository {HF_REPO_NAME} ready")

        # Upload entire GGUF directory
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


def main():
    """Main conversion workflow"""
    logger.info(f"Starting GGUF conversion for {args.model.upper()} model...")
    logger.info(f"Base model: {BASE_MODEL_ID}")
    logger.info(f"Model name: {MODEL_NAME}")

    # Check prerequisites
    if not check_dependencies():
        logger.error("Dependencies check failed")
        sys.exit(1)

    # Create output directory
    create_output_dir()

    # Download and merge LoRA adapter from HF Hub
    if not merge_lora_adapter():
        logger.error("LoRA adapter merge failed")
        sys.exit(1)

    # Convert to FP16
    fp16_file = convert_to_gguf_fp16()
    if not fp16_file:
        logger.error("FP16 conversion failed")
        sys.exit(1)

    # Quantize models
    success_count = 1  # Count FP16 as success

    for quant_type in QUANT_TYPES:
        quantized_file = quantize_model(fp16_file, quant_type)
        if quantized_file:
            success_count += 1
        else:
            logger.warning(f"Skipping {quant_type} quantization")

    logger.info(f"✅ {success_count} model(s) created successfully")

    # Upload to Hugging Face Hub
    if upload_to_huggingface():
        logger.info("🎉 GGUF conversion and upload completed successfully!")
        logger.info(f"Model available at: https://huggingface.co/{HF_REPO_NAME}")
    else:
        logger.error("Upload failed")
        sys.exit(1)


if __name__ == "__main__":
    main()