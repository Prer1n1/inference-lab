# Lab 04: prefill vs decode against the roofline

**Status:** Scripts ready (not yet run)  
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

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Prefill throughput as a % of your GPU's measured matmul ceiling, 4k prompt | | | |
| Prefill time growth from 1k to 8k tokens | | | |
| Decode step time vs the bandwidth floor, 8B model in eager PyTorch | | | |
| Decode step growth from 128 to 8,192 context (use KV bytes ÷ bandwidth) | | | |

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
_To fill in._

## 6. Interpretation
_To fill in. Compare with Lab 05: given a prompt-heavy workload, how much GPU time is prefill, and does that explain the p99 ITL spikes seen there?_

## 7. Engineering takeaway
_To fill in._ Hint: this is the analysis behind chunked prefill (Lab 08) and disaggregation (Lab 16).

## 8. Pitfalls to watch
- Prefill timing must exclude one-time costs (the script warms up).
- Eager decode includes framework overhead; do not read its percentage of bandwidth as the engine's potential. Lab 06 separates the two.
- The FLOP formula ignores attention; at 8k tokens the error is small but nonzero.

## 9. Interview questions
1. Why is prefill compute-bound and decode memory-bound? Give the arithmetic-intensity argument.
2. Why does a long prompt hurt other users' latency on a shared GPU, and name two fixes.
3. If a profile shows decode at 40% of the bandwidth floor, what are your first three suspects?
4. Define MFU and MBU. Which one is the right yardstick for decode?

## Further reading
- Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting" (2023).
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving" (OSDI 2024).
- Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve" (OSDI 2024).
