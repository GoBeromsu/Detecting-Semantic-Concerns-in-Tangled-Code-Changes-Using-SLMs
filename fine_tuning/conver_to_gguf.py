#!/usr/bin/env python3
"""
GGUF Conversion Script for Phi-4 Fine-tuned Models
Convert merged LoRA models to GGUF format and upload to Hugging Face
"""

import os
import subprocess
import logging
import sys
from pathlib import Path
from typing import Optional
from huggingface_hub import create_repo, upload_folder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration - Match train.py settings
USER = os.getenv("USER", "acq24bk")
FASTDATA_BASE = f"/mnt/parscratch/users/{USER}"
MERGED_MODEL_DIR = f"{FASTDATA_BASE}/models/merged_model"
GGUF_OUTPUT_DIR = f"{FASTDATA_BASE}/models/gguf"
LLAMA_CPP_DIR = f"{FASTDATA_BASE}/llama.cpp"

# Model naming
MODEL_NAME = "phi4-commit"
HF_REPO_NAME = "Berom0227/phi4-commit-gguf"

# Quantization options
QUANT_TYPES = ["q4_K_M", "q5_K_S", "q8_0"]


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

    # Check if convert.py exists
    convert_script = Path(LLAMA_CPP_DIR) / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        logger.error(f"convert.py not found at {convert_script}")
        return False

    # Check if quantize binary exists
    quantize_binary = Path(LLAMA_CPP_DIR) / "quantize"
    if not quantize_binary.exists():
        logger.warning(f"quantize binary not found at {quantize_binary}")
        logger.info("Build llama.cpp first: cd ~/llama.cpp && make")

    logger.info("✅ Dependencies check completed")
    return True


def check_merged_model() -> bool:
    """Check if merged model exists and has required files"""
    logger.info(f"Checking merged model at {MERGED_MODEL_DIR}")

    required_files = [
        "config.json",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    model_path = Path(MERGED_MODEL_DIR)
    if not model_path.exists():
        logger.error(f"Merged model directory not found: {MERGED_MODEL_DIR}")
        logger.info("Run train.py first to create merged model")
        return False

    missing_files = []
    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    logger.info("✅ Merged model files found")
    return True


def create_output_dir() -> None:
    """Create GGUF output directory"""
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {GGUF_OUTPUT_DIR}")


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
    quantize_binary = f"{LLAMA_CPP_DIR}/quantize"

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
    logger.info("Starting GGUF conversion process...")

    # Check prerequisites
    if not check_dependencies():
        logger.error("Dependencies check failed")
        sys.exit(1)

    if not check_merged_model():
        logger.error("Merged model check failed")
        sys.exit(1)

    # Create output directory
    create_output_dir()

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
