# Lab 10: Structured output and multi-LoRA serving

**Status:** Planned  
**Baseten posting items:** LoRA; guided generation for structured outputs  
**Needs:** inexpensive GPU, ~1 h (~$3) · **Depends on:** Lab 05

## 1. The question
What do JSON-schema guided decoding and serving many LoRA adapters cost in latency and throughput?

## 2. Plan
1. Guided decoding: compare unconstrained vs JSON-schema constrained on TTFT, ITL, throughput and the share of valid outputs (confirm the vLLM guided-decoding option and backend).
2. LoRA: serve the base model with 1, 4 and 16 adapters (`--enable-lora`, `--max-loras`), send mixed-adapter traffic, compare with base-only.
3. Report overhead as a percentage and find the adapter count where it starts to hurt.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Overhead table for guided decoding and for N adapters
- Recommendation on caps and when to merge adapters instead

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. How does constrained decoding mask tokens and where does its cost come from?
2. Why can many LoRA adapters share one base model's batch?
3. What breaks first as the adapter count grows?
