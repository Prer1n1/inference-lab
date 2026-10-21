# Lab 03: batching and context length (plain PyTorch)

**Status:** Scripts ready (not yet run)  
**Baseten posting items:** "LLM optimization techniques (continuous batching)", "tokens per GPU-hour"  
**Needs:** CUDA GPU, ~30 minutes. · **Depends on:** Labs 01-02

## 1. The question
Why does serving more sequences at once raise total throughput so much, and what stops it?

## 2. Mental model
- Decode reads all the weights every step no matter how many sequences share the step. Step time is roughly flat while memory-bound, so tokens/s scales almost with batch size.
- Two things end the free ride: the math units fill up (compute-bound), or the KV cache fills memory (each sequence needs its own, and longer contexts mean more bytes to read per step).
- Longer context = more KV bytes read per step = slower steps and fewer sequences that fit.
- This lab uses static batches in plain PyTorch. A serving engine (Lab 05) adds continuous batching on top, which keeps batches full as requests finish and arrive.

## 3. Predictions (fill in before running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Decode tok/s growth, batch 1 → 8 at context 128 | close to 8× | | |
| Decode ms/step, batch 1 vs batch 8 | nearly equal (weights dominate) | | |
| Effect of context 128 → 4,096 at batch 32 | slower steps, higher peak memory | | |
| First batch size that OOMs at context 4,096 (your GPU) | from the KV formula | | |

## 4. Experiment
```bash
cd labs/03_batching
python batching.py --model meta-llama/Llama-3.1-8B-Instruct --out results/batching.jsonl
```
Sweeps batch {1,2,4,8,16,32} × context {128, 1,024, 4,096}. Prefill and decode are timed separately
(decode = N-token time − 1-token time). Out-of-memory points are recorded as `"oom": true`.

## 5. Results
_To fill in. Plot decode tok/s vs batch for each context._

## 6. Interpretation
_To fill in: where does scaling bend? Is that compute or memory? How do you know?_

## 7. Production takeaway
_To fill in. Hint: why static batching wastes capacity and what continuous batching changes (compare with Lab 05)._

## 8. Pitfalls found
_To fill in. Watch for: random token IDs are fine for speed but meaningless for quality; HF `generate` has Python overhead that a serving engine removes._

## 9. Interview questions
1. Why does decode throughput scale almost linearly with batch size at first?
2. What limits batch size on a given GPU, and how would you compute it?
3. What is the difference between static and continuous batching and why does it matter for tail latency?
