# Lab NN: <topic>

**Status:** Planned | Scripts ready | Measured | Done  
**Baseten posting items covered:** <which lines of the job posting this lab answers>  
**Needs:** <GPU class, rough hourly price, expected session length> · **Depends on:** <earlier labs>

## 1. The question
One sentence a customer or teammate could ask. Example: "Why does decode cost the same for 1 or 32 users?"

## 2. Mental model
The 5-10 lines of theory needed to predict the answer. Include the back-of-envelope arithmetic.

## 3. Predictions (write BEFORE running anything)
| Quantity | Prediction (with the arithmetic) | Real | Verdict |
|---|---|---|---|

## 4. Experiment
Exact commands, what each flag does, how long it takes, what it costs, and what could go wrong.
State what is held constant and what is varied. Fresh prompts, warm-up, and repeats are mandatory.

## 5. Results
Raw data lives in `results/`. Tables here, with p50 **and** p99 where latency matters.
Say how many repeats agree and how closely. Report noisy numbers as ranges.

## 6. Interpretation
What the numbers say, what they do NOT say, and which explanations are still hypotheses.
Check every result against a number you can compute by hand.

## 7. Production takeaway
"If a customer ran this workload, I would ..." Tie it to tokens per GPU-hour, latency, or cost.

## 8. Pitfalls found
Anything that nearly gave a wrong answer (cache contamination, client caps, cold starts...) and the guard added.

## 9. Interview questions
Three questions on this lab, answered in your own words, without notes.
