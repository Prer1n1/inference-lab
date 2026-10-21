# Lab NN: <topic>

**Status:** Planned | Scripts ready | Part A done | Measured | Done  
**Baseten posting items:** <the lines of the job posting this lab answers>  
**Needs:** <GPU class, rough hourly price, session length> · **Depends on:** <earlier labs>

## 1. The question
One sentence a customer or teammate could ask. Example: "Why does decode cost about the same for 1 or 32 users?"

## 2. Mental model
What we already know, stated plainly, with the worked arithmetic that lets you predict the result: formulas, a spec-sheet table of the
inputs (model dimensions, GPU bandwidth), and the one or two ideas the rest of the lab depends on. Define every term the first time.

## 3. Prediction
Written BEFORE running anything. Numbers with their arithmetic, and the claims that the experiments will test.
| Quantity | Prediction (with the arithmetic) | Real | Verdict |
|---|---|---|---|

## 4. Environment and setup
Hardware, driver, engine and library versions, model, flags, workload. Commands to reproduce, with a sentence on why each is needed.

## 5. Experiments
One subsection per experiment (5.1, 5.2, ...), each with:
- **Command** and a line on what it does and why it is in the lab.
- **Measured output**: a table or a log, copied from the raw file in `results/`. p50 and p99 where latency matters; say how many repeats agree.
- **Interpretation**: what the numbers say. Keep *measured* and *inferred* apart ("the clean conclusion is ..."), compute ratios inline so a reader can check them.

## 6. Limits
What this lab does NOT show: one GPU, one model, one workload, hypotheses not yet tested, effects not modelled.
Pitfalls that nearly produced a wrong answer, and the guard added.

## 7. Engineering takeaway
Two short paragraphs or a few bullets. Conditional recommendations (framed by the latency target or workload), not a single "best" number.
"If a customer ran this workload, I would ..." tied to tokens per GPU-hour, latency or cost.

## 8. Interview questions
Three to five questions on this lab, answered in your own words without notes.

## Further reading
The papers and documentation behind the techniques used.

---
Conventions: the README is the canonical lesson; raw outputs and data live in `results/`. Predictions are scored, not edited afterwards.
