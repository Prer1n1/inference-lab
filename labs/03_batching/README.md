# Lab 03: batching and context length (plain PyTorch)

**Status:** Scripts ready (not yet run)  
**Baseten posting items:** "LLM optimization techniques (continuous batching)", "tokens per GPU-hour"  
**Needs:** CUDA GPU, about 30 minutes. · **Depends on:** Labs 01-02

## 1. The question
Why does serving more sequences at once raise total throughput so much, what stops it, and what does a longer context do?

## 2. Concepts

### 2.1 The batching idea
One decode step reads all the weights. If B sequences share the step, the weight bytes are read once but B tokens come out. While the step is memory-bound (arithmetic intensity ≈ B, far below the ridge, Lab 01), step time stays nearly flat and **throughput grows almost linearly with B**.

### 2.2 What ends the free ride
1. **Compute**: when B approaches the ridge point (about 300 on an H100), the matmuls become compute-bound and step time starts to grow with B.
2. **KV bytes**: every sequence's cache must also be read each step. Bytes per step ≈ `weights + B × context × KV_bytes_per_token`. Worked example, 8B BF16, batch 32, context 4,096: weights 16 GB + KV `32 × 4096 × 128 KiB` = 17.2 GB → about 33 GB per step, twice the batch-1 figure, so the floor rises from about 4.8 ms to about 10 ms.
3. **KV capacity**: `B × context × KV_bytes_per_token` must fit in the memory left after weights. On an 80 GB card with a 16 GB model that is roughly 50-56 GB; on a 48 GB card roughly 25 GB. Beyond that, out of memory.

### 2.3 Prefill and decode behave differently
Prefill of B prompts is one big compute-heavy pass (time grows about linearly with total prompt tokens); decode is the memory-bound loop. The script times them separately by measuring a 1-token generation (prefill + one step) and subtracting it from an N-token generation.

### 2.4 Static vs continuous batching
- **Static batching** (what this lab does): a fixed set of requests starts together and the batch runs until the *longest* one finishes. Short requests finish early but their slots idle (and padded prompts waste compute). New requests wait for the whole batch.
- **Continuous (iteration-level) batching** (Orca; vLLM, SGLang, TensorRT-LLM): the scheduler re-forms the batch *every step*: finished sequences leave and waiting ones join immediately. The GPU stays full and waiting time drops. Lab 05 measures an engine doing this; comparing it with this lab's numbers shows what the engine adds.

### 2.5 Throughput vs latency
Bigger batches raise tokens/s but make each step (and each user's per-token time) slower once past the flat region, and queueing raises time-to-first-token. Serving is a *curve*, not one number. Operating at the knee is usually the economic optimum; the worst latency target decides how far up the curve you may go.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Decode tok/s growth from batch 1 to 8 at context 128 | | | |
| Decode ms/step at batch 1 vs batch 8 | | | |
| Effect of raising context from 128 to 4,096 at batch 32 (step time, peak memory) | | | |
| First batch size that runs out of memory at context 4,096 on your GPU (use the KV formula) | | | |

## 4. Experiment, step by step
```bash
cd /root/inference-lab/labs/03_batching
python batching.py --model meta-llama/Llama-3.1-8B-Instruct --out results/batching.jsonl
```
**What the script does and why:**
1. Loads the model once and sweeps batch {1, 2, 4, 8, 16, 32} × context {128, 1,024, 4,096} using random token IDs (speed does not depend on the words). *A clean grid isolates each effect.*
2. For each cell: a warm-up generation, then a 1-token generation (prefill time) and a 64-token generation; decode time = difference. *Separates the two phases.*
3. Records prefill tokens/s, decode ms per step, decode tokens/s and peak memory; out-of-memory cells are recorded and skipped. *The OOM boundary is a result, not a failure.*

Expected time: 10-20 minutes depending on the GPU.

## 5. Results
_To fill in. Plot decode tokens/s against batch size, one line per context length._

## 6. Interpretation
_To fill in: where does scaling bend, and is that compute, KV bytes or something else? What evidence distinguishes them (peak memory, step time vs the formula in 2.2)?_

## 7. Production takeaway
_To fill in._ Hint: why static batching wastes capacity, what continuous batching changes, and what that implies for the throughput numbers in Lab 05.

## 8. Pitfalls to watch
- Hugging Face `generate` has Python overhead per step that a serving engine removes; absolute numbers here understate what an engine reaches. Compare *shapes*, not absolute values.
- Random token IDs are fine for speed, meaningless for quality.
- `min_new_tokens` is set so every sequence generates exactly N tokens; otherwise early stops distort throughput.

## 9. Interview questions
1. Why does decode throughput scale almost linearly with batch size at first?
2. What limits batch size on a given GPU, and how would you compute it?
3. What is the difference between static and continuous batching, and why does it matter for tail latency?
4. Why does a longer context slow every decode step even though the weights are unchanged?
