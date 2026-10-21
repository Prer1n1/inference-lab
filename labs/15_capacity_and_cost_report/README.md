# Lab 15: capacity and cost report

**Status:** Planned (no GPU needed)  
**Baseten posting items:** "Turn performance wins into measurable outcomes: tokens per GPU-hour, utilization, latency and cost"; "give customers and internal teams clear latency/throughput/cost tradeoffs"  
**Needs:** your measured results from Labs 05-11 · **Depends on:** Labs 05-11

## 1. The question
Given a workload and a latency SLA, which configuration is cheapest and how many GPUs does it need?

## 2. Concepts

### 2.1 Unit economics
```
$ per million output tokens = ($/GPU-hour) ÷ (tokens/s × 3600 ÷ 1e6)
tokens per GPU-hour          = tokens/s × 3600
```
Both depend on the **operating point**, not the peak: use the throughput the GPU sustains *while meeting the latency target* (goodput).

### 2.2 Input vs output tokens
Prefill (input) tokens are cheap per token (compute-bound, parallel); decode (output) tokens are expensive per token (memory-bound, sequential). Providers often price output tokens several times input tokens for this reason. A cost model should include both: `GPU-seconds = prompt_tokens ÷ prefill_rate + output_tokens ÷ decode_rate_at_target_load`. A prompt-heavy workload costs more than an output-only figure suggests.

### 2.3 From a rate and an SLA to a number of GPUs
- **Little's law**: concurrency needed = arrival rate × time in system. Example: 50 requests/s at 2 s each → 100 requests in flight.
- **Per-GPU capacity** at the latency target comes from your measurements (for example 64 concurrent users meet p99 ITL ≤ 20 ms on one H100 in Lab 05).
- **GPUs = ceil(peak concurrency ÷ per-GPU capacity ÷ target utilization)**, where target utilization (for example 0.7) leaves **headroom** for bursts, failures and the queueing hockey stick (Lab 08).
- Add redundancy (N+1) so one failure does not breach the SLA.

### 2.4 Utilization and headroom
GPUs are expensive, so idle capacity is costly, but near-100% utilization makes tail latency explode. The headroom factor is a business decision; the report should show cost at several headroom levels.

### 2.5 Sensitivity
Recommendations depend on assumptions. Show what changes if prompts are twice as long, if output is half as long, if the prefix-cache hit rate is 0% vs 50%, if the GPU price changes. A recommendation that flips under a small change should say so.

## 3. Predictions (fill in BEFORE building)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| GPUs for 50 req/s, 300-token prompts, 128-token outputs, p99 ITL ≤ 20 ms | | | |
| Cost change from FP8 | | | |
| Cost change from 50% prefix-cache hits | | | |

## 4. Steps and why
1. **Gather the data**: every result file with config label, load, goodput, GPU telemetry. *One dataset, one source of truth.*
2. **Calculator** (script or notebook): inputs are prompt/output distributions, request rate, SLA, GPU type and price; outputs are the best config, GPUs needed, $/M tokens and utilization.
3. **Regenerate every table in the repo from `results/`** with one command. *Reproducibility.*
4. **Sensitivity analysis** over the assumptions in 2.5.
5. **Write a one-page customer-style report**: recommendation, cost, risks, what was measured vs assumed.

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: using peak instead of goodput; extrapolating beyond measured loads; mixing numbers from runs with different prompts; ignoring prefill cost; hiding the assumptions.

## 9. Done when
- A script that regenerates every table from `results/`.
- A short customer-style report with a recommendation and caveats.

## 10. Interview questions
1. How do you turn a throughput benchmark into a GPU count for a customer?
2. Why is goodput a better headline number than peak tokens/s?
3. What would make your recommendation wrong?
4. Why are output tokens usually priced above input tokens?
