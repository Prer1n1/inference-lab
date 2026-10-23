# Lab 03: batching and context length (plain PyTorch)

**Status:** Done on an L40S (2026-10-23)  
**Baseten posting items:** "LLM optimization techniques (continuous batching)", "tokens per GPU-hour"  
**Needs:** CUDA GPU, about 30 minutes. · **Depends on:** Labs 01-02

## 1. The question
Why does serving more sequences at once raise total throughput so much, what stops it, and what does a longer context do?

## 2. Mental model

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

## 3. Predictions

Predictions were **not written down before this run** (a verbal prediction question was asked but not answered), so the "Real" column is a plain result, not a scored guess. Write them before the next GPU lab.

| Quantity | Prediction | Real (L40S) | Verdict |
|---|---|---|---|
| Decode tok/s growth from batch 1 to 8 at context 128 | not recorded | 40.1 → 303.9 = **7.6×** | n/a |
| Decode ms/step at batch 1 vs batch 8 | not recorded | 24.94 → 26.32 ms (**+5.5%**) | n/a |
| Effect of raising context from 128 to 4,096 at batch 16 (step time, peak memory) | not recorded | 27.36 → 64.31 ms (**2.35×**); peak 15.44 → 30.25 GiB | n/a |
| First batch size that runs out of memory at context 4,096 | the KV formula alone says batch 32 needs 15 + 16.3 = 31 GiB, which would fit in 44.4 GiB | **batch 32 ran out of memory** | The formula alone was too optimistic; see 6 |

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
**Environment:** 1× L40S (44.39 GiB), PyTorch 2.8.0+cu128, transformers 5.20.2, Llama-3.1-8B-Instruct BF16, Hugging Face `generate` (eager), static batches, random token IDs, 64 new tokens. Raw lines: `results/batching.jsonl`. Single run; not repeated.

Decode ms per step (decode tokens/s in brackets):

| Batch | Context 128 | Context 1,024 | Context 4,096 |
|---|---|---|---|
| 1 | 24.94 (40.1) | 24.83 (40.3) | 26.40 (37.9) |
| 2 | 26.08 (76.7) | 26.59 (75.2) | 29.25 (68.4) |
| 4 | 25.86 (154.7) | 27.77 (144.0) | 32.23 (124.1) |
| 8 | 26.32 (303.9) | 29.40 (272.1) | 43.07 (185.7) |
| 16 | 27.36 (584.9) | 33.70 (474.7) | 64.31 (248.8) |
| 32 | **29.58 (1,081.9)** | 45.08 (709.8) | **out of memory** |

Peak memory (GiB) at batch 32: 15.92 (context 128), 22.61 (context 1,024). At context 4,096: 15.92 (batch 1), 18.79 (4), 22.61 (8), 30.25 (16). Prefill ran at about 10,000-12,700 tokens/s at every cell with at least 1,024 tokens.
(plot not yet made)

## 6. Interpretation
- **Batching is nearly free at short context.** Batch 1 → 32 at context 128: step time +19% (24.94 → 29.58 ms), throughput **27×** (1,081.9 ÷ 40.1). Batch 1 itself is 98% of the Lab 01 floor (24.4 ms). The math at batch 32 is 32 × 16 GFLOP ≈ 0.51 TFLOP, about 2.6 ms at the measured 196 TFLOP/s, hidden behind the weight read, as expected far below the ridge point (299).
- **Longer context bends the curve.** At batch 32 the step grows 29.6 → 45.1 ms from context 128 to 1,024, and throughput falls from 1,082 to 710 tokens/s. The KV read alone (32 × about 1,088 tokens × 128 KiB ≈ 4.3 GiB ≈ 7 ms at 655 GB/s) explains only part of the +15.5 ms. The rest is probably eager Hugging Face attention and cache handling; not verified.
- **Throughput and per-user speed trade off.** At context 4,096, batch 16 gives 6.6× the throughput of batch 1 (248.8 vs 37.9 tokens/s) but a step time 2.4× longer (64.3 vs 26.4 ms): each user waits longer per token.
- **The out-of-memory cell shows a gap in the capacity formula.** Weights (about 15 GiB) plus KV for batch 32 × 4,160 tokens (16.25 GiB) is about 31 GiB, under the 44.4 GiB card. But measured peak memory exceeds weights + KV by about **0.44 GiB per sequence** at context 4,096 (3.55 GiB at batch 8, 7.1 GiB at batch 16), close to the KV size per sequence (0.51 GiB). Extrapolating to batch 32 gives about 15 + 16.3 + 14.2 ≈ 45.4 GiB, above 44.4 GiB. This matches an implementation that temporarily holds a second copy of the cache as it grows, but I did not confirm that; it is an extrapolation from three points.
- **Prefill rates (about 10-12.7k tokens/s) are roughly constant** across batch sizes once the prompt is at least 1,024 tokens: prefill is compute-bound (Lab 04).

## 7. Engineering takeaway
Batching converts one weight read into many tokens, so throughput rises almost linearly while step time stays nearly flat. It stops being free when per-user KV reads grow with context, when KV memory (plus implementation overhead) runs out, and, on faster hardware or larger batches, when the step becomes compute-bound. Static batching, as measured here, holds every sequence until the longest finishes and reserves memory for all of them; a continuous-batching engine with paged memory (Lab 05) re-forms the batch every step and wastes less, which is why its numbers at the same load are higher than a loop of `generate` calls. Operate near the knee of the throughput-versus-latency curve and choose the batch by the latency target.

## 8. Pitfalls to watch
- Hugging Face `generate` has Python overhead per step that a serving engine removes; absolute numbers here understate what an engine reaches. Compare *shapes*, not absolute values.
- Random token IDs are fine for speed, meaningless for quality.
- `min_new_tokens` is set so every sequence generates exactly N tokens; otherwise early stops distort throughput.

## 9. Interview questions
1. Why does decode throughput scale almost linearly with batch size at first?
2. What limits batch size on a given GPU, and how would you compute it?
3. What is the difference between static and continuous batching, and why does it matter for tail latency?
4. Why does a longer context slow every decode step even though the weights are unchanged?

### Worked answers
1. **Linear scaling at first.** Each step reads all the weights (16 GB for an 8B BF16 model) once, whatever the batch. Extra sequences add FLOPs but not weight reads, and the GPU has math capacity to spare far below the ridge point. Measured: batch 1 → 32 raised step time only 19% and throughput 27×.
2. **What limits batch size.** KV memory: `B × context × KV_bytes_per_token` plus weights plus working memory must fit; here 32 × 4,160 tokens × 128 KiB = 16.25 GiB of KV was not enough headroom on a 44 GiB card once the measured overhead (about 0.44 GiB per sequence) was included. Also the latency target and, at large batch, the ridge point. Compute capacity as `(memory − weights − working memory) ÷ KV per token`, then verify by measuring the peak.
3. **Static vs continuous batching.** Static: a fixed set starts together and runs until the longest finishes; short sequences idle and new arrivals wait for the whole batch, which inflates tail latency. Continuous: the scheduler adds and removes sequences every step, so slots refill immediately and waiting is short.
4. **Longer context slows every step.** Each step reads the weights plus every sequence's KV cache, and the cache grows with context: at batch 32 the step rose from 29.6 ms (128 tokens) to 45.1 ms (1,024 tokens) on this setup.

## Further reading
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models" (OSDI 2022).
- Pope et al., "Efficiently Scaling Transformer Inference" (2022).
- Kipply, "Transformer Inference Arithmetic" (blog).
