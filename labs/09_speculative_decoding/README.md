# Lab 09: Speculative decoding

**Status:** Planned  
**Baseten posting items:** Speculative decoding (in production); live draft-model training (example initiative)  
**Needs:** 1x H100 or an inexpensive GPU, ~2 h (~$8) · **Depends on:** Labs 05, 06

## 1. The question
When does speculative decoding speed up decode, and when does it hurt?

## 2. Plan
1. Baseline, then prompt-lookup/n-gram speculation (no extra model), then a small draft model (for example a 1B model for the 8B target) at k = 2, 4, 8; EAGLE-style heads if supported. Confirm flags for the installed version.
2. Use TPOT and server-reported token counts, not chunk gaps: speculative steps emit several tokens per chunk (the harness now supports this).
3. Measure acceptance rate and speedup at 1, 8, 32, 64 users. Expect the benefit to shrink as batching already uses the spare memory bandwidth.
4. Two workload types: high-acceptance (code, JSON, repetitive) and low-acceptance (open chat).

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Speedup-vs-concurrency curves with acceptance rates
- A rule of thumb: when to enable it, at which k
- Note on what training a draft model on live traffic would improve

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. Why can verification of k draft tokens cost about the same as one normal step?
2. Why does speculation help less at high batch sizes?
3. What determines the acceptance rate and how would you raise it?
