# Lab 08: Scheduling and caching: chunked prefill, token budget, prefix reuse

**Status:** Planned  
**Baseten posting items:** Chunked prefill; KV-cache reuse; custom scheduling; tail latency  
**Needs:** 1x H100, ~2 h (~$8) · **Depends on:** Labs 04, 05, 06

## 1. The question
Do chunked prefill, the per-step token budget and prefix caching tame the tail-latency spikes and lower cost, and by how much?

## 2. Plan
1. Build a mixed workload: many short requests plus some very long prompts (separate runs with different `--input-words`, or a prompt file).
2. Sweep chunked prefill on/off and `--max-num-batched-tokens`; record p99 ITL and TTFT, throughput, goodput at a fixed rate (open loop).
3. Prefix caching on purpose: use `--shared-prefix-words`; plot cache hit rate vs throughput vs TTFT; reproduce the Lab 05 contamination deliberately and explain it.
4. Check the effect of `--max-num-seqs` on the concurrency ceiling.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- p99 ITL with and without chunked prefill, with the trace evidence from Lab 06
- Cache-hit-rate vs $/M tokens curve
- A recommended scheduler config per workload shape

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. Why does a long prompt hurt other users, and how does chunked prefill fix it? What does it cost?
2. Prefix caching gave a 30% win in a benchmark. How do you tell whether production traffic would see it?
3. What does `max-num-batched-tokens` trade off?
