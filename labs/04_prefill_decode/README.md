# Lab 04: prefill vs decode against the roofline

**Status:** Scripts ready (not yet run)  
**Baseten posting items:** "Profile and optimize inference end to end", "prefill/decode disaggregation" (the reason it exists), "chunked prefill"  
**Needs:** CUDA GPU, ~30 minutes. · **Depends on:** Labs 01-03

## 1. The question
Prefill and decode run on the same model but are limited by different things. How far is each from its hardware limit, and what does that mean for a mixed workload?

## 2. Mental model
- **Prefill**: thousands of tokens at once → big matmuls → compute-bound. Ceiling = matmul TFLOP/s from Lab 01. FLOPs ≈ 2 × params × tokens.
- **Decode** (batch 1): one token per step → every weight read for one token of work → memory-bound. Floor = (weights + KV bytes) ÷ bandwidth.
- Real engines do not reach either limit. The gap has causes you should be able to name: kernel launch overhead (many small kernels), attention kernels below peak, sampling, Python.
- When prefill and decode share a GPU, a long prefill step stalls every user's decode. Chunked prefill and prefill/decode disaggregation exist to fix that.

## 3. Predictions (fill in before running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Prefill throughput as % of peak matmul TFLOP/s, 4k prompt | 40-70% | | |
| Prefill time scaling, 1k → 8k tokens | slightly worse than linear | | |
| Decode step vs bandwidth floor (8B model) | well above floor in eager PyTorch (launch-bound) | | |
| Decode step growth with context 128 → 8,192 | small (KV is small next to weights) | | |

## 4. Experiment
```bash
cd labs/04_prefill_decode
python prefill_decode.py --gpu-info ../01_gpu_architecture/results/gpu_info.json --out results/prefill_decode.json
```
For each prompt length it prints prefill ms / tok/s / achieved TFLOP/s (and % of peak), decode ms per step, achieved GB/s,
and % of measured bandwidth. Run Lab 01 first so the roofline comparison is filled in.

## 5. Results
_To fill in._

## 6. Interpretation
_To fill in. Compare with Lab 05: given a prompt-heavy workload, how much of the GPU's time is prefill, and does that explain the p99 latency spikes there?_

## 7. Production takeaway
_To fill in. Hint: this is the analysis behind chunked prefill (Lab 08) and disaggregation (Lab 11)._

## 8. Pitfalls found
_To fill in._

## 9. Interview questions
1. Why is prefill compute-bound and decode memory-bound? Give the arithmetic-intensity argument.
2. Why does a long prompt hurt other users' latency on a shared GPU, and name two fixes.
3. If a profile shows decode at 40% of the bandwidth floor, what are your first three suspects?
