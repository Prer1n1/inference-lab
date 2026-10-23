# Session A runbook: Labs 01-04 on an inexpensive GPU

Goal: run the four first-principles labs in one short session. About 2 hours of pod time. Read `docs/pod-workflow.md` once first.

## Before you rent anything (laptop, free)
1. `git push` so the pod can clone the latest code.
2. Read the *Mental model* sections of Labs 01-04 and fill in each **Predictions** table. Do this *before* seeing any numbers.
3. Have your Hugging Face token ready (Llama is gated; it was approved on your account).
4. Check your RunPod balance (about $30).

## Choosing the GPU
- You want **at least 40 GB of memory** so Llama-3.1-8B (about 16 GB) leaves room for the KV cache in Lab 03: a 48 GB class card (for example A40, L40S, RTX A6000). A 24 GB card works for Labs 01, 02, 04 but Lab 03 will run out of memory at large batch × context; that is recorded, not fatal.
- Use the same **PyTorch template** as before. Container disk 50 GB. **Do not attach a network volume**: in the first Session A run the `/workspace` area silently stored every model file as 0 bytes (see "What went wrong" below). Download the model to the container disk; it takes about 4 minutes and costs a few cents.
- Write down the price per hour and the start time.

## On the pod
Open Connect → Jupyter → File → New → Terminal.

**Do not create a venv this time.** The PyTorch template's own Python already has torch; a venv would hide it.

### Setup (about 3 minutes)
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
*Why:* confirms torch sees the GPU before anything else. It must print `True` and the GPU name.
```bash
git clone https://github.com/Prer1n1/inference-lab.git /root/inference-lab     # clone to /root, not /workspace
cd /root/inference-lab
pip install -r requirements-labs.txt
export HF_HOME=/root/hf             # NOT /workspace
export HF_TOKEN=hf_xxxxxxxx          # paste your token in the terminal only; never in chat or a file
nvidia-smi                           # record GPU, driver, CUDA for your README
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir /root/models/llama-3.1-8b-instruct --exclude "original/*"
du -sh /root/models/llama-3.1-8b-instruct     # must say about 15G; if it says 0, stop
```
*Why each line:* `/root` is on the pod's own disk and has no permission quirks; `requirements-labs.txt` adds transformers and accelerate; `HF_HOME` keeps the Hugging Face cache on that disk; `--local-dir` writes plain files into one folder (no cache links); `--exclude "original/*"` skips a duplicate 16 GB copy of the weights; `du -sh` proves the files are real. Every `--model` flag below uses that folder.
Every new terminal forgets `HF_HOME` and `HF_TOKEN`, so set them again each time.

### Lab 01 (about 2 minutes)
```bash
cd /root/inference-lab/labs/01_gpu_architecture
python gpu_info.py --params-b 8 --out results/gpu_info.json
```
*Look for:* `measured_bandwidth_gb_s`, `measured_matmul_tflops`, `ridge_point_flop_per_byte`, and `decode_step_floor_ms` for BF16.
*Compare with your predictions.* **Paste the JSON output into the chat** (it is short).

### Lab 02 (about 5-8 minutes; the first model load may take a few minutes)
```bash
cd /root/inference-lab/labs/02_kv_cache
python kv_cache.py --model /root/models/llama-3.1-8b-instruct --out results/kv_cache.json
```
*Look for:* `formula_kib_per_token` (should be 128), each `ratio_measured_over_formula` (near 1.0), and `speedup_from_cache`.
*Paste the JSON.*

### Lab 03 (about 10-20 minutes)
```bash
cd /root/inference-lab/labs/03_batching
python batching.py --model /root/models/llama-3.1-8b-instruct --out results/batching.jsonl
```
It prints one JSON line per (context, batch) cell. A cell with `"oom": true` hit the memory limit: that is data.
*Look for:* `decode_ms_per_step` staying nearly flat while `decode_tok_per_s` climbs with batch; how the numbers change with context; where the first OOM appears.
*Paste all the lines* (or run `cat results/batching.jsonl`).

### Lab 04 (about 5-10 minutes)
```bash
cd /root/inference-lab/labs/04_prefill_decode
python prefill_decode.py --model /root/models/llama-3.1-8b-instruct --gpu-info ../01_gpu_architecture/results/gpu_info.json --out results/prefill_decode.json
```
*Look for:* `prefill_pct_of_peak_tflops`, `decode_ms_per_step` against `decode_floor_ms`, and `decode_pct_of_bandwidth`.
*Paste the JSON.*

## If something fails
| Symptom | Likely cause | What to do |
|---|---|---|
| `torch.cuda.is_available()` is False | wrong template or driver issue | stop, check the pod's GPU is attached; do not continue |
| 401/403 downloading Llama | token not set in this terminal, or wrong account | `export HF_TOKEN=...` again; confirm access on the model page |
| `ModuleNotFoundError: transformers` | requirements not installed | `pip install -r /root/inference-lab/requirements-labs.txt` |
| `OutOfMemoryError` at model load | GPU too small | use a 48 GB card |
| Warnings about `torch_dtype` deprecated | newer transformers | harmless |
| Any Python traceback in my scripts | the scripts were never run on a GPU | paste the full traceback; fixing it is part of the session |
| Script prints but numbers look absurd | measurement bug | do not trust it; paste output and we check against a hand calculation |

## What went wrong the first time (2026-10-23, L40S)
| Symptom | Cause | Fix |
|---|---|---|
| `config.json is not a valid JSON file`, and a "complete" download that finished instantly the second time | the `/workspace` area stored every model file as **0 bytes** (`ls -laL` showed size 0 for config and all weight files); the downloader then trusted those empty files as already present | do not use `/workspace` for the model; `HF_HOME=/root/hf` and `hf download --local-dir /root/models/...`; verify with `du -sh` |
| A stopped pod could not restart: "GPUs no longer available" | another user took the GPU while the pod was stopped | do not Stop to resume later; **Terminate** and redeploy (setup is about 10 minutes) |
| After "Automatically migrate", `/root/inference-lab` and the pip packages were gone | migration kept only `/workspace` | do not migrate; redeploy and re-run the setup block |
| `python: can't open file ... kv_cache.py` | the terminal was still inside the cache folder | `cd` to the lab folder explicitly before each run |
| `HT_HOME` typo | mistyped variable | check with `echo $HF_HOME` |
| Hugging Face token pasted into the chat | copy-paste | revoke it on huggingface.co/settings/tokens, create a new one, never paste it outside the pod's terminal |
| Lab 04 could not find `gpu_info.json` | the file was lost when the repo was re-cloned | run Lab 01 before Lab 04 on every new pod |

## Ending the session
1. Make sure you pasted all outputs into the chat (the pod's files vanish on terminate).
2. In the RunPod console: **Stop**, then **Terminate**. Closing the tab does not stop billing.
3. Check the balance a few minutes later and tell me the spend; I update the budget log.

## After the session
Fill in Results, Interpretation, Production takeaway and Interview answers for Labs 01-04, score your predictions, and note anything that surprised you or nearly fooled you.
