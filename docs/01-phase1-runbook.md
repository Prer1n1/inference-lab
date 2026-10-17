# Phase 1 runbook: baseline vLLM + concurrency sweep

Goal: produce the throughput-vs-latency curve for Llama-3.1-8B-Instruct on one GPU, and explain its shape.
Budget for this session: about 2-3 GPU hours.

## 0. Before you rent anything (laptop)
- Put the repo on a private GitHub repo (so the pod can `git clone` it). Never commit your HF token.
- Write down your predictions (section 6). Predicting first is what makes the real numbers teach you something.

## 1. Rent the pod (RunPod)
- GPU: **1x H100 80GB** (FP8 in Phase 3 needs Hopper; A100 would force a re-baseline later). Check the hourly price shown and note it.
- Template: **RunPod PyTorch** (recent CUDA). We install vLLM ourselves so you see every moving part.
- Disk: container 50 GB, volume 100 GB (model is about 16 GB; leave room for later phases).
- Note the start time. Cost = hours x price.

## 2. Install and record the environment
```bash
nvidia-smi                       # record GPU, driver, CUDA
python -m venv /workspace/venv && source /workspace/venv/bin/activate
pip install vllm                 # large download; takes a few minutes
vllm --version                   # record this: results are meaningless without versions
export HF_TOKEN=hf_xxxxxxxx      # your READ token; never put it in a file in the repo
```

## 3. Serve the model (inside tmux so it survives a dropped connection)
```bash
tmux new -s server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000
```
Wait for "Application startup complete". In the startup log find the line **GPU KV cache size: N tokens**.
That number is your KV-cache budget (concept 3). Write it down.

Detach with `Ctrl-b d`. Open a second pane/window: `tmux new -s work`.

## 4. Smoke test
```bash
curl localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct","prompt":"The capital of France is","max_tokens":16}'
```

## 5. Run the benchmark (client runs ON the pod, so network does not pollute the numbers)
```bash
git clone <your-private-repo> && cd inference-lab
pip install aiohttp numpy
M=meta-llama/Llama-3.1-8B-Instruct

# warm-up: discard (CUDA graphs, cold caches)
python bench/bench.py --model $M --num-requests 32 --concurrency 8 --out results/warmup.jsonl

# single-user latency (fewer requests: this is slow by design)
python bench/bench.py --model $M --num-requests 32 --concurrency 1 --out results/baseline.jsonl

# the sweep
python bench/bench.py --model $M --num-requests 256 --concurrency 2 4 8 16 32 64 128 \
  --input-words 256 --output-tokens 128 --out results/baseline.jsonl
```
While it runs, in a third pane: `watch -n1 nvidia-smi` (watch memory and utilization change).
Re-run the sweep once more into `results/baseline_run2.jsonl`. If the two runs differ by more than a few
percent, something is noisy and you should find out why before trusting either.

## 6. Predict first, then compare
Fill in before running:
- c=1 ITL: floor is 16 GB / 3.35 TB/s = about 4.8 ms. Real engines land above the floor. Your guess: ___ ms
- Logged KV cache size: (0.90 x 80 GB - 16 GB weights - overhead) / 128 KB per token = roughly 350-450k tokens. Your guess: ___
- Throughput at c=128 vs c=1: your guess for the ratio: ___x
- Where does the curve flatten, and which resource is the limit? ___

## 7. Save results, then stop the bill
```bash
git add results && git commit -m "phase1 baseline" && git push   # results only, no secrets
```
(or download results/ from the RunPod web UI)
Then in the RunPod console: **Terminate** the pod when you are done with it. A stopped pod can still bill for
its disk, so check the billing page afterward and confirm nothing is still running.

## 8. What to write down (this becomes your report)
Environment table (GPU, driver, vLLM version, model, flags), the sweep table, one plot of
throughput vs p99 ITL, and 3-5 sentences on what limited each region of the curve.
