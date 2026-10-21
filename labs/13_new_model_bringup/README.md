# Lab 13: New model bring-up, time-boxed

**Status:** Planned  
**Baseten posting items:** Bring up and tune new model architectures on new hardware quickly, often the same week they are released  
**Needs:** 1x H100, ~3 h (~$12) · **Depends on:** Labs 05-08

## 1. The question
How fast can I take a newly released architecture from 'does not run' to 'tuned and benchmarked', and what checklist makes the next one faster?

## 2. Plan
1. Pick a recent open model with a new feature (MoE, new attention variant, new tokenizer) that the engine supports only in a recent or nightly build.
2. Day 1: get it serving and verify outputs against the reference implementation (token-level or logit comparison).
3. Day 2: benchmark with the harness (baseline table). Day 3: tune (parallelism, quantization, scheduler flags).
4. Keep a log of every failure and how it was found. Repeat the benchmark on a second GPU type as the 'new hardware' analogue.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Timeline log with time spent per stage
- Reusable bring-up checklist
- Benchmark table on two GPU types

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. What do you check first when a new model produces garbage in the engine?
2. How does an MoE model change the memory-vs-compute picture?
3. What would you automate to cut bring-up from days to hours?
