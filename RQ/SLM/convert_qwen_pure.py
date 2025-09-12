#!/usr/bin/env python3

import os
import subprocess
import logging
import sys
from huggingface_hub import create_repo, upload_folder, snapshot_download

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-14B"
WORK_DIR = os.getenv("TMPDIR", "./tmp_gguf_conversion")
MODEL_DIR = os.path.join(WORK_DIR, "qwen_model")
OUTPUT_DIR = os.path.join(WORK_DIR, "gguf_output")
FASTDATA_BASE = "/mnt/parscratch/users/" + os.getenv("USER", os.getenv("USERNAME", ""))
LLAMA_CPP_DIR = os.getenv("LLAMA_CPP_DIR", os.path.join(FASTDATA_BASE, "llama.cpp"))
HF_REPO = "Berom0227/Qwen3-14B-Pure-gguf"
HF_TOKEN = os.getenv("HF_HUB_TOKEN")
HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", None)

def main():
    logger.info("LLAMA_CPP_DIR: " + LLAMA_CPP_DIR)
    
    # Check dependencies
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        logger.error("convert_hf_to_gguf.py not found at: " + convert_script)
        sys.exit(1)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download model first
    logger.info("Downloading model from HuggingFace...")
    local_model_path = snapshot_download(
        repo_id=BASE_MODEL,
        local_dir=MODEL_DIR,
        cache_dir=HF_CACHE_DIR,
        token=HF_TOKEN
    )
    logger.info("Model downloaded to: " + local_model_path)
    
    output_file = os.path.join(OUTPUT_DIR, "Qwen3-14B-Pure-bf16.gguf")
    
    cmd = [
        "python", convert_script, local_model_path,
        "--outfile", output_file,
        "--outtype", "bf16"
    ]
    
    logger.info("Converting to GGUF BF16...")
    subprocess.run(cmd, check=True)
    logger.info("Created: " + output_file)
    
    logger.info("Uploading to HuggingFace...")
    create_repo(HF_REPO, repo_type="model", exist_ok=True, token=HF_TOKEN)
    upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO,
        token=HF_TOKEN,
        commit_message="Add Qwen3-14B pure BF16 GGUF"
    )
    logger.info("Done: https://huggingface.co/" + HF_REPO)

if __name__ == "__main__":
    main()