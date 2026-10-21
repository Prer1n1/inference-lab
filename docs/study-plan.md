# Study plan: from here to interview-ready

Every lab follows the same loop: **read the concepts → write predictions → run → compare → explain → answer the interview questions aloud**.
A lab is *done* only when its README has real results, the predictions are scored, the pitfalls section lists what nearly fooled you,
and you can answer its interview questions without notes.

## Where you are
- Lab 00 concepts: written. Lab 05: measured (levels ≥128 need a re-run). Labs 01-04: scripts ready. Labs 06-16: lessons written, experiments not yet run.
- Budget: $30.41 of $40 left (see the budget log in the root README).

## Order of work

| Step | What | Where | Rough cost | You finish with |
|---|---|---|---|---|
| 0 | `git push` the repo | laptop | $0 | everything on GitHub |
| 1 | Read Lab 00, then the *Concepts* of Labs 01-04; fill every **Predictions** table (use the arithmetic in the concepts) | laptop | $0 | written predictions |
| 2 | **Session A**: run Labs 01-04 (`docs/session-a-labs-01-04.md`) | inexpensive 48 GB GPU, ~2 h | ~$1-3 | results for 4 labs, bugs in scripts found |
| 3 | Fill Results / Interpretation / Takeaway / Interview answers in Labs 01-04; commit | laptop | $0 | four finished labs |
| 4 | **Session B**: Lab 05 re-run (`labs/05_vllm_baseline/RUNBOOK.md`), then Lab 06 profiling | 1× H100, ~3 h | ~$12 | corrected Lab 05, profile of a step |
| 5 | Finish Labs 05-06 write-ups | laptop | $0 | two finished labs |
| 6 | **Session C**: Lab 07 (quantization; needs H100 for FP8) and Lab 08 (scheduling, caching) | 1× H100, ~4 h | ~$16 | two labs |
| 7 | Lab 09 speculative decoding; Lab 10 structured output + LoRA | inexpensive GPU or H100 | ~$6-10 | two labs |
| 8 | Lab 11 router (write and test the code on the laptop first); run on 2-3 inexpensive GPUs | laptop + cheap pods | ~$6-10 | router + results |
| 9 | Lab 12 Triton kernel (develop on any NVIDIA GPU) | cheap pod | ~$3 | kernel + plots |
| 10 | Lab 14 Kubernetes, autoscaling, observability | cheap GPU / small cluster | ~$5-10 | deploy + dashboards + post-mortem |
| 11 | Lab 13 bring-up; Lab 15 cost report; Lab 16 upstream PR (collect candidates from step 1 onward) | mixed | ~$12 | portfolio-grade finish |

**Money:** steps 2-6 fit in your balance; steps 7-11 will likely need a top-up of roughly $30-50, or cheaper GPUs plus smaller models where the lab
allows. Comparisons *within* a lab (config A vs B) are valid on any GPU; only absolute cost-per-token numbers need the H100 baseline from Lab 05.

## Rules that keep the work credible
1. Predictions are written before every run and scored after, even when wrong.
2. Check each result against a number you can compute by hand.
3. Two runs must agree before a number goes in a README; otherwise find out why.
4. Verify the load generator delivered the load you asked for.
5. Keep the mistakes: they are the evidence that you debug measurements.
6. Terminate the pod when the session ends and update the budget log.

## Interview preparation, in parallel
- After each lab, answer its interview questions aloud in two minutes each. Record yourself once; it exposes vague explanations.
- Keep a running list titled "things that fooled me" (cache contamination, client cap, clock resolution...). Interviewers love these.
- After Lab 08, you can tell the full story: *measure → find the tail-latency problem → explain it with a trace → fix it → quantify the cost.*
- Before applying, publish two or three write-ups (for example: the roofline explanation of decode; the Lab 05 benchmarking pitfalls; the quantization trade-off table) and a clean top-level README.
- When Labs 01-09 are done, do a mock interview: ask me to quiz you on any lab, one question at a time, without notes.

## Mapping to the roles you looked at
- **Inference Performance engineer**: Labs 01-13, 15, 16 are the core.
- **Infrastructure engineer**: Labs 11 and 14 (plus 05 for load testing).
- **Observability engineer**: Lab 14 (metrics, tracing, SLOs), plus the harness's telemetry and goodput work.
- **Engineering manager (the first posting)**: needs prior team leadership; the labs build the technical credibility, and leading or mentoring others (even in open source) builds the rest.
