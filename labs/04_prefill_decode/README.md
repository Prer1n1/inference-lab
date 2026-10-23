# Lab 04: prefill vs decode against the roofline

**Status:** Done on an L40S (2026-10-23)  
**Baseten posting items:** "Profile and optimize inference end to end", "prefill/decode disaggregation" (the reason it exists), "chunked prefill"  
**Needs:** CUDA GPU, about 30 minutes. · **Depends on:** Labs 01-03

## 1. The question
Prefill and decode run the same model but are limited by different resources. How far is each from its hardware limit, and what does that mean for a mixed workload?

## 2. Mental model

### 2.1 Two phases of one request
1. **Prefill**: the model processes all `L` prompt tokens in one forward pass and fills the KV cache. It determines **TTFT** (time to first token), together with any queueing.
2. **Decode**: tokens are generated one at a time, each step reading weights and the whole cache. Its pace is **ITL/TPOT** (inter-token latency / time per output token).

### 2.2 Prefill is compute-bound
Per token, a forward pass costs about `2 × parameters` FLOPs (one multiply and one add per weight). For an 8B model that is about 16 GFLOP per token, so a prompt of `L` tokens costs `16 GFLOP × L` (attention adds a term that grows with `L²`, small until contexts get long).
- Upper bound on prefill speed = `peak FLOP/s ÷ 16 GFLOP` = 989 TFLOP/s ÷ 16 GFLOP ≈ **62,000 tokens/s** on an H100. Real engines reach perhaps 40-60% of peak.
- A 4,096-token prompt at 50% of peak: `4096 ÷ 31,000 ≈ 130 ms`. TTFT grows roughly linearly with prompt length.

### 2.3 Decode is memory-bound
At batch 1, each step moves `weights + KV` bytes for one token of work (Lab 01 §2.4). The floor is `(weight_bytes + KV_bytes) ÷ bandwidth`. The ratio `achieved ÷ floor` (or achieved GB/s ÷ peak GB/s) is called **MBU** (memory bandwidth utilization); for prefill, `achieved FLOP/s ÷ peak` is **MFU** (model FLOPs utilization). These two numbers are the standard way to say how well an engine uses the hardware.

### 2.4 Why real numbers sit below the limits
- **Launch overhead**: a decode step runs several hundred small kernels. If the CPU cannot issue them as fast as the GPU finishes them, the GPU idles between kernels. Eager PyTorch (as in this lab) is especially exposed; **CUDA graphs** record the whole sequence once and replay it with a single launch (Lab 06).
- Attention and sampling kernels are not at peak; small matrices underfill the GPU.
- Python and framework overhead.
A small model on a fast GPU is *more* launch-bound than a big one, because less time is spent on useful work per kernel.

### 2.5 Interference in a shared GPU
If one engine runs both phases, a long prefill occupies the GPU and delays every other user's next decode step: a tail-latency spike for them. Two families of fix:
- **Chunked prefill**: split the prompt into chunks and run each chunk *alongside* other users' decode tokens in the same step (Lab 08).
- **Prefill/decode disaggregation**: run prefill and decode on *separate* GPUs or pools, each tuned for its own bottleneck, and ship the KV cache between them (Lab 16).

## 3. Predictions

Predictions were **not written down before this run**, so "Real" is a plain result here.

| Quantity | Prediction | Real (L40S) | Verdict |
|---|---|---|---|
| Prefill throughput as a % of your GPU's measured matmul ceiling, 4k prompt | not recorded | **94.1%** (184.2 of 195.7 TFLOP/s) | n/a |
| Prefill time growth from 1k to 8k tokens | not recorded | 88.8 → 818.0 ms = **9.2×** for 8× the tokens | n/a |
| Decode step time vs the bandwidth floor, 8B model in eager PyTorch | not recorded | 23.6-26.5 ms against a floor of 24.5-26.2 ms: **94-104%** | n/a |
| Decode step growth from 128 to 8,192 context (use KV bytes ÷ bandwidth) | not recorded | floor 24.53 → 26.15 ms (+1.6 ms); measured 25.98 → 26.49 ms | n/a |

## 4. Experiment, step by step
```bash
cd /root/inference-lab/labs/04_prefill_decode
python prefill_decode.py --gpu-info ../01_gpu_architecture/results/gpu_info.json --out results/prefill_decode.json
```
**What the script does and why:**
1. Loads the model and counts parameters and weight bytes. *Inputs to the FLOP and byte formulas.*
2. For each prompt length `L`: times a prefill (several repetitions) → ms, tokens/s, achieved TFLOP/s, and the percentage of the Lab 01 matmul ceiling (MFU-like).
3. Prefills once, then runs 64 decode steps with the cache at batch 1 → ms/step, achieved GB/s, percentage of the Lab 01 bandwidth (MBU-like), and the floor in ms.
Run Lab 01 first so the roofline columns are filled in.

## 5. Results
**Environment:** 1× L40S, PyTorch 2.8.0+cu128, transformers 5.20.2, Llama-3.1-8B-Instruct BF16, eager PyTorch, batch 1, 64 decode steps. Roofline columns use the Lab 01 numbers (655.3 GB/s, 195.7 TFLOP/s). Command used `--model /root/models/llama-3.1-8b-instruct`. Raw lines: `results/prefill_decode.jsonl`. Single run; not repeated.

| Prompt tokens | Prefill ms | Prefill tokens/s | Prefill TFLOP/s | % of 195.7 | Decode ms/step | Decode GB/s | Floor ms | Floor ÷ measured |
|---|---|---|---|---|---|---|---|---|
| 128 | 30.7 | 4,164 | 66.9 | 34.2% | 25.98 | 618.8 | 24.53 | 94.4% |
| 512 | 49.0 | 10,449 | 167.8 | 85.7% | 23.59 | 683.7 | 24.61 | 104.3% |
| 1,024 | 88.8 | 11,533 | 185.2 | 94.6% | 23.80 | 680.5 | 24.71 | 103.8% |
| 2,048 | 168.3 | 12,170 | 195.5 | 99.9% | 24.16 | 675.9 | 24.92 | 103.1% |
| 4,096 | 357.1 | 11,469 | 184.2 | 94.1% | 24.89 | 666.9 | 25.33 | 101.8% |
| 8,192 | 818.0 | 10,014 | 160.8 | 82.2% | 26.49 | 646.9 | 26.15 | 98.7% |

**How the 2,048-token row is computed.** Prefill work = 2,048 × 16.06 GFLOP = 32.9 TFLOP; 32.9 ÷ 0.1683 s = 195.5 TFLOP/s. Decode bytes = 16.06 GB of weights + 2,048 × 128 KiB (0.27 GB) of KV = 16.33 GB; 16.33 GB ÷ 0.02416 s = 676 GB/s; the floor at 655.3 GB/s is 16.33 ÷ 655.3 = 24.92 ms.

## 6. Interpretation
- **Prefill is compute-bound.** From 1,024 tokens up it runs at 82-100% of the measured matmul ceiling (99.9% at 2,048 tokens). Its arithmetic intensity (about 2,048 FLOP/byte at 2,048 tokens) is far above the ridge point (299).
- **Short prompts under-use the GPU**: 128 tokens reached only 34%, probably because there is too little work per kernel (not tested). **8,192 tokens fell to 82%**, plausibly attention, whose cost grows with the square of the length and which runs below matmul efficiency (not tested). TTFT grew 9.2× for 8× the tokens, slightly super-linear.
- **Decode is memory-bound.** Step times of 23.6-26.5 ms are 94-104% of the floor computed from weights plus KV bytes.
- **Over 100% means the "floor" is an estimate, not a hard limit.** It was built from the Lab 01 copy test (read plus write); a read-only decode may achieve more than that, so measured decode can beat it. This was not verified. The spec-sheet bandwidth (864 GB/s) would give about 18.9 ms for the same bytes, so 23.6 ms is about 80% of that.
- **The 128-token decode row (25.98 ms) is slower than the 512-token row (23.59 ms)**, which a bigger context should not cause; most likely warm-up or run-to-run noise in the first measurement (not confirmed). The run was not repeated, so treat differences of about 2 ms as noise.
- **Compared with Lab 05** (not yet done for this GPU): a long prompt occupies the GPU for hundreds of milliseconds (8,192 tokens = 818 ms here), during which other users' decode steps wait; that is the interference chunked prefill removes (Lab 08).

## 7. Engineering takeaway
The two phases hit different limits, so they need different fixes: **TTFT (prefill) is a compute problem** (more FLOP/s, shorter prompts, prefix caching, chunking, more GPUs), **inter-token latency (decode) is a bytes problem** (quantization, batching, higher-bandwidth hardware). Batching cannot help prefill, because it already runs at about 100% of the math ceiling. Because a long prefill delays everyone else's decode, systems either chunk it into the decode steps (Lab 08) or move it to a separate pool (Lab 16).

## 8. Pitfalls to watch
- Prefill timing must exclude one-time costs (the script warms up).
- Eager decode includes framework overhead; do not read its percentage of bandwidth as the engine's potential. Lab 06 separates the two.
- The FLOP formula ignores attention; at 8k tokens the error is small but nonzero.

## 9. Interview questions
1. Why is prefill compute-bound and decode memory-bound? Give the arithmetic-intensity argument.
2. Why does a long prompt hurt other users' latency on a shared GPU, and name two fixes.
3. If a profile shows decode at 40% of the bandwidth floor, what are your first three suspects?
4. Define MFU and MBU. Which one is the right yardstick for decode?

### Worked answers
1. **Arithmetic intensity.** A BF16 weight is 2 bytes and each token does 2 FLOPs with it, so FLOP/byte equals the number of tokens that share one weight read. Prefill processes thousands of tokens at once (about 2,048 FLOP/byte at 2,048 tokens) against a ridge point of about 299, so it is compute-bound: measured 195.5 TFLOP/s, 99.9% of the matmul ceiling. Decode at batch 1 shares each read with one token (about 1 FLOP/byte), so it is memory-bound: measured 24.2 ms for 16.3 GB, about 676 GB/s.
2. **Long prompts and other users.** One engine runs one step at a time; a long prefill occupies it for hundreds of milliseconds (818 ms at 8,192 tokens here), so every other user's next token waits. Fixes: chunked prefill (run the prompt in pieces alongside decode tokens) and prefill/decode disaggregation (separate pools, KV cache shipped between them).
3. **Decode at 40% of the floor.** (a) Kernel-launch and CPU overhead (many small kernels, eager mode; fix with CUDA graphs); (b) inefficient attention or small-matrix kernels that underfill the GPU; (c) extra bytes beyond what you counted (KV cache, padding, un-fused ops that re-read activations), or contention from other work on the GPU. Profile before guessing (Lab 06).
4. **MFU and MBU.** MFU = achieved FLOP/s ÷ peak FLOP/s; MBU = achieved bytes/s ÷ peak bandwidth. MBU is the right yardstick for decode because decode is limited by memory traffic; MFU is the right one for prefill.

## Further reading
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (2023).
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving" (OSDI 2024).
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI 2024).
