# Lab 15: Capacity and cost report

**Status:** Planned  
**Baseten posting items:** Turn performance into tokens per GPU-hour, utilization, latency/throughput/cost tradeoffs for customers  
**Needs:** no GPU · **Depends on:** Labs 05-11

## 1. The question
Given a workload and a latency SLA, which configuration is cheapest and how many GPUs does it need?

## 2. Plan
1. Collect measured results from Labs 05-11 into one dataset (the harness records config labels, GPU telemetry and goodput).
2. Write a calculator: inputs are prompt/output length distribution, request rate, SLA; outputs are the best config, GPU count, $/M tokens and headroom.
3. Compare GPU types using measured numbers only; add a sensitivity analysis (what if prompts are twice as long?).

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Script that regenerates every table from `results/`
- A short customer-style report with a recommendation and caveats

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. How do you turn a throughput benchmark into a GPU count for a customer?
2. Why is goodput a better headline number than peak tokens/s?
3. What would make your recommendation wrong?
