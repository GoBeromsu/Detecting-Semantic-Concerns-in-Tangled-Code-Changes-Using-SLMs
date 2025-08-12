# Inference on UoS Stanage for RQ/Phi/infer_huggingface.py

## 1) Current State
- fine_tuning/setup_env.sh: Proven working HPC env bootstrap (CUDA 12.1.1, PyTorch installed, Flash-Attn, project requirements via requirements.txt).
- requirements.txt: Transformers stack, datasets, accelerate, huggingface_hub, peft, trl, wandb, python-dotenv.
- Missing for inference: llama-cpp-python (required by `utils/llms/hugging_face.py`).
- RQ/Phi/infer_huggingface.py: Uses `llms.load_model(...)` and `llms.hugging_face_api_call(...)` with an extra `chat_format` argument not supported by their current signatures.
- Data: Inference loads public HF dataset (`Berom0227/Detecting-Semantic-Concerns-in-Tangled-Code-Changes-Using-SLMs`). No private token needed.

## 2) Final State
- Same HPC env from `setup_env.sh` plus llama-cpp-python GPU wheel (CUDA 12.1): `llama-cpp-python-cu121` (or `llama-cpp-python` if cu121 wheel unavailable).
- Minimal edits to `RQ/Phi/infer_huggingface.py` to align with `utils/llms` function signatures (remove `chat_format` args). No behavioral changes.
- New SLURM script `fine_tuning/run_inference_hf.sh` to activate env and run the inference script on A100 (single GPU), saving results under `results/Phi-4/huggingface/`.

## 3) Files To Change
- `fine_tuning/setup_env.sh`
  - Add: `pip install llama-cpp-python-cu121` after PyTorch install. If wheel unavailable on Stanage, fallback to `pip install llama-cpp-python` CPU-only.
- `RQ/Phi/infer_huggingface.py`
  - Remove unsupported `chat_format=...` argument in `llms.load_model(...)` and `llms.hugging_face_api_call(...)` calls.
- Create `fine_tuning/run_inference_hf.sh`
  - Mirror module loads and `source activate phi4_env`.
  - Optional: `source .env` if present.
  - Run: `python RQ/Phi/infer_huggingface.py`.

## 4) Checklist
- [ ] Environment
  - [ ] Run `sbatch fine_tuning/setup_env.sh` to (re)create `phi4_env`.
  - [ ] Ensure llama-cpp installed:
    - [ ] Prefer: `pip install llama-cpp-python-cu121` (GPU). If unavailable: `pip install llama-cpp-python` (CPU).
  - [ ] Verify import works: `python -c "import llama_cpp; print('ok')"` inside env.
- [ ] Code alignment
  - [ ] Edit `RQ/Phi/infer_huggingface.py` to remove `chat_format` from `llms.load_model(...)` and `llms.hugging_face_api_call(...)` calls.
  - [ ] Keep `SEED`, `TEMPERATURE`, `INCLUDE_MESSAGE`, `CONTEXT_WINDOWS` as-is.
- [ ] Job script
  - [ ] Create `fine_tuning/run_inference_hf.sh` with Stanage modules, `source activate phi4_env`, and `python RQ/Phi/infer_huggingface.py`.
  - [ ] Set reasonable SLURM limits (e.g., 2–4 hours, 1× A100, 16 CPU, 64–128GB RAM).
- [ ] Runtime
  - [ ] Submit: `sbatch fine_tuning/run_inference_hf.sh`.
  - [ ] Confirm HF model download succeeds (`microsoft/phi-4-gguf`, e.g., `phi-4-Q4_K.gguf` or `phi-4-bf16.gguf`).
  - [ ] Check logs in `fine_tuning/logs/*.out` for progress and errors.
- [ ] Results
  - [ ] Verify CSVs written under `results/Phi-4/huggingface/`.
  - [ ] Spot-check metrics columns: `precision`, `recall`, `f1`, `exact_match`.

## Ideas (not in scope unless requested)
- Parameterize model filename via env (`HF_GGUF_FILENAME`) to avoid device detection logic.
- Add retries around HF downloads and inference to handle transient failures.
- Optional: add `HF_HUB_ENABLE_HF_TRANSFER=1` for faster downloads via hf_transfer if available.