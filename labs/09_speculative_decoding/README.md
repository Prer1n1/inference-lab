# Lab 09: speculative decoding

**Status:** Planned (flags confirmed against the installed vLLM version when we run it)  
**Baseten posting items:** "speculative decoding" (implement and productionize; also in production); example initiative "Live draft model training for speculative decoding"  
**Needs:** 1× H100 or an inexpensive GPU for the smaller experiments, about 2 hours (~$8) · **Depends on:** Labs 05, 06

## 1. The question
When does speculative decoding speed up decode, and when does it hurt?

## 2. Concepts

### 2.1 The idea
Decode is memory-bound: verifying several tokens in one forward pass costs almost the same as producing one (the weights are read once either way). So let something *cheap* guess the next `k` tokens, then have the big (**target**) model check all `k` guesses in a single pass.

### 2.2 The algorithm
1. A **draft** proposes `k` tokens.
2. The target runs one forward pass over those `k` tokens (plus the context) and computes its own probabilities at each position.
3. Tokens are accepted left to right using a rejection-sampling rule that makes the accepted output **distributed exactly as if the target had generated it alone**. At the first rejection, the target's own token is used and the rest of the draft is discarded.
4. Each step therefore yields between 1 and `k + 1` tokens.
Output quality is unchanged; only speed changes.

### 2.3 How much speedup
If each draft token is accepted independently with probability `α`, the expected tokens per target step is
`E = (1 − α^(k+1)) ÷ (1 − α)`. Let `c` be the cost of one draft token relative to one target step. A step costs about `1 + c·k` target-steps, so
`speedup ≈ E ÷ (1 + c·k)`.
Worked examples at `k = 4`, `c = 0.05`:
- `α = 0.8`: `E = (1 − 0.8⁵)/0.2 = 3.36` tokens per step; cost 1.2 → speedup about **2.8×**.
- `α = 0.5`: `E = (1 − 0.5⁵)/0.5 = 1.94`; cost 1.2 → about **1.6×**.
- `α = 0.2`: `E ≈ 1.25`; cost 1.2 → about **1.04×**; with a costlier draft it is a net loss.

### 2.4 Types of drafters
| Drafter | Idea | Trade-off |
|---|---|---|
| Separate small model (for example a 1B model for an 8B target) | independent model guesses | needs a compatible tokenizer; extra memory and compute |
| Prompt lookup / n-gram | copy likely continuations from the prompt | free, very good on repetitive or extractive text, useless elsewhere |
| Medusa-style heads | extra prediction heads on the target | needs training; one pass |
| EAGLE-style | a small head that predicts using the target's hidden features | higher acceptance; needs training |
| Multi-token-prediction heads (some newer models) | model trained to predict several tokens | built in when the model has them |

### 2.5 Why speedups shrink at high concurrency
At batch 1 the GPU has spare compute while it waits for weights, so extra verified tokens are nearly free. With many sequences per step, decode is already closer to compute-bound (Lab 03), so verifying `k + 1` tokens per sequence costs real compute and the gain falls, possibly below 1. Speculation is mostly a **low-latency, low-batch** tool.

### 2.6 What decides the acceptance rate
How well the draft matches the target *on your traffic*: code, JSON and templated text accept highly; open-ended chat less. This is why a draft trained or adapted on live traffic can beat a generic one (the example initiative in the posting: continuously training a draft model on production data).

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Speedup at 1 user, n-gram speculation on a repetitive/extractive task | | | |
| Speedup at 1 user, draft model, chat prompts | | | |
| Speedup at 64 users | | | |
| Acceptance rate: code vs chat | | | |

## 4. Steps and why
1. **Baseline** at 1, 8, 32, 64 users. *Everything is a ratio against it.*
2. **Switch the harness to TPOT and server-reported token counts.** *With speculation, one streamed chunk can hold several tokens, so chunk gaps no longer equal token gaps (the upgraded harness records both and flags mismatches).*
3. **n-gram / prompt lookup** speculation. *No extra model: the cheapest thing to try.*
4. **Draft model** at `k = 2, 4, 8`. *Find the best `k`: more guesses help only while acceptance stays high.*
5. **Two workload types**: high acceptance (code/JSON/repetitive) and low (open chat). Read acceptance rate from the server's metrics.
6. **Fit the formula** from 2.3 to your measured acceptance rates and see how close the measured speedup is.

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: benchmarking speculation with random-word prompts (acceptance is meaningless); judging by chunk-gap ITL; ignoring the draft's memory and compute cost; evaluating only at batch 1 when production runs at higher load.

## 9. Done when
- Speedup-vs-concurrency curves with acceptance rates.
- A rule of thumb: when to enable it and at which `k`.
- A note on what live-traffic draft training would improve.

## 10. Interview questions
1. Why can verifying `k` draft tokens cost about the same as generating one?
2. Why does speculation help less at high batch sizes?
3. What determines the acceptance rate, and how would you raise it?
4. Why is the output distribution identical to normal decoding?
