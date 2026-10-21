# Lab 07: Quantization: speed, capacity, quality

**Status:** Planned  
**Baseten posting items:** Quantization (FP8/FP4, AWQ, GPTQ); tokens per GPU-hour; latency/throughput/cost tradeoffs  
**Needs:** 1x H100 (FP8 needs Hopper or Ada), ~2 h (~$8) · **Depends on:** Labs 01, 05

## 1. The question
What do FP8 weights, INT4 weight-only quantization (AWQ/GPTQ) and an FP8 KV cache buy in speed and capacity, and what do they cost in quality?

## 2. Plan
1. Configs: BF16 baseline; FP8 weights; AWQ INT4; GPTQ INT4; FP8 KV cache; best combination. Confirm vLLM flags and checkpoint availability for the installed version.
2. Speed: decode step at 1 user (predict from bytes / bandwidth from Lab 01), throughput at 64 users, TTFT.
3. Capacity: KV-cache tokens reported by the server, and what that means for users per GPU.
4. Quality: perplexity and a small task eval on a fixed subset for every config, same prompts, same settings.
5. Cost: $ per million tokens per config at a fixed latency target (goodput), not at peak throughput.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- One table: config x (ITL, tokens/s at target, KV capacity, quality delta, $/M tokens)
- Prediction vs measurement for the decode speedup from fewer bytes
- A recommendation per workload type

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. Why does halving weight bytes not halve end-to-end latency?
2. When is weight-only INT4 a bad idea compared with FP8?
3. How do you decide a quality loss is acceptable for a customer?
