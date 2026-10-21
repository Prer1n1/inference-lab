# Lab 11: Cache-aware routing and prefill/decode disaggregation

**Status:** Planned  
**Baseten posting items:** Cache-aware routing; custom routing algorithms; prefill/decode disaggregation; load balancing  
**Needs:** 2-3 inexpensive GPUs (or 2 H100 for the disaggregation test), ~$6-10 · **Depends on:** Labs 05, 08

## 1. The question
Does routing by prompt-prefix affinity beat round-robin, and what does separating prefill from decode buy?

## 2. Plan
1. Write a small router (Go or Python) in front of 2-3 replicas with three policies: round-robin, least-loaded, prefix-hash affinity with a load bound.
2. Workload: multi-turn conversations and shared system prompts. Measure cache hit rate, TTFT, p99 latency and throughput per policy.
3. Disaggregation prototype: one prefill worker plus one decode worker using vLLM's KV-transfer support (confirm availability in the installed version), compared with two colocated replicas on the same hardware.
4. Report when each wins and why.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Per-policy table: hit rate, TTFT p50/p99, tokens/s
- Disaggregated vs colocated comparison at equal GPU count
- The router code with tests

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. Why can affinity routing hurt load balance, and how do you bound that?
2. What has to move between a prefill and a decode worker, and how big is it?
3. When is disaggregation not worth it?
