# Lab 06: Profiling: where does a step's time go?

**Status:** Planned  
**Baseten posting items:** Profile end to end (kernel launch overhead, memory layout, scheduling); cross-layer investigations of tail latency  
**Needs:** 1x H100, ~2 h (~$8) · **Depends on:** Labs 04, 05

## 1. The question
Where does the time in one decode step and one prefill step actually go, how much is launch overhead, and what causes the p99 ITL spikes seen in Lab 05?

## 2. Plan
1. Capture a PyTorch-profiler trace of a prefill-heavy window and a decode-heavy window using vLLM's profiler endpoints (confirm the exact flags for the installed version).
2. If available, capture an Nsight Systems timeline and read the idle gaps between kernels.
3. Compare CUDA graphs on vs `--enforce-eager` at 1 user: the ITL difference is the launch overhead you can remove.
4. Break a step into attention, GEMMs, norms, sampling, communication and idle; report % of step time for each.
5. Cross-layer investigation: take one slow (p99) inter-token gap, tie it to the scheduler step that produced it (prompt chunks mixed into the step), then to the kernels.

## 3. Predictions (write before running)
| Quantity | Prediction (with arithmetic) | Real | Verdict |
|---|---|---|---|


## 4. Done when
- Table of top kernels with % of step time and the idle fraction
- Measured launch-overhead cost in ms and %
- A written explanation of the p99 ITL spikes, backed by a trace

## 5. Results, interpretation, production takeaway, pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`: raw data in `results/`, p50 and p99, repeats, and a check against a hand calculation._

## 6. Interview questions
1. What does CUDA graph capture remove, and what does it require of the model code?
2. A decode step is at 60% of the bandwidth floor. How do you find out why?
3. How would you trace one slow request from the HTTP request down to the kernel?
