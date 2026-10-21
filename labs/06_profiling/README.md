# Lab 06: profiling, where does a step's time go?

**Status:** Planned (exact commands are confirmed against the installed vLLM version when we run it)  
**Baseten posting items:** "Profile and optimize inference end to end, from kernel launch overhead and memory layout up to request scheduling"; "cross-layer investigations, such as tracing a tail-latency regression from request timing through routing and batching down to a kernel"  
**Needs:** 1× H100, about 2 hours (~$8) · **Depends on:** Labs 04, 05

## 1. The question
Where does the time in one decode step and one prefill step actually go, how much of it is launch overhead, and what causes the p99 ITL spikes seen in Lab 05?

## 2. Mental model

### 2.1 What a "step" looks like on the hardware
The CPU runs Python and queues **kernels** into a CUDA stream; the GPU executes them in order. Ideally the queue is always non-empty and the GPU is always busy. Two failure shapes:
- **GPU-bound**: kernels back to back, no gaps: the GPU is the limit (good; now optimize the kernels).
- **CPU/launch-bound**: gaps between kernels because the CPU cannot issue them fast enough: the GPU idles.
A decode step of an 8B model is several hundred small kernels, so launch overhead is a real fraction of a 6 ms step.

### 2.2 CUDA graphs
A CUDA graph records a fixed sequence of kernel launches once and replays it with a single launch, removing most per-kernel CPU cost. Requirement: **static shapes and memory addresses**. That is why vLLM captures graphs ahead of time for a set of batch sizes (the "Capturing CUDA graphs" lines in the Lab 05 startup log) and pads a batch up to the nearest captured size. The comparison `--enforce-eager` (no graphs) vs default measures what launch overhead costs.

### 2.3 Tools and what each one answers
| Tool | Shows | Use it to |
|---|---|---|
| `nvidia-smi` utilization | % of time *any* kernel was running (not efficiency: a poor kernel can show 100%) | sanity check only |
| PyTorch profiler (via vLLM's profiler endpoints) | per-operator and per-kernel time, Chrome/Perfetto trace | find which operations dominate |
| Nsight Systems | whole-timeline view: CPU threads, kernels, memcpys, gaps | see launch gaps and overlap |
| Nsight Compute | counters inside one kernel (memory throughput, occupancy, stalls) | explain *why* a kernel is slow |
Use them in that order: timeline first, then the top kernel, then its counters.

### 2.4 Kernel categories in a transformer step
GEMMs (the QKV/output/MLP matmuls; most of the time), attention (FlashAttention/FlashInfer-style kernels reading the KV cache), RMSNorm and rotary embedding (small, memory-bound), sampling (top-k/top-p, softmax), and, with multiple GPUs, all-reduce communication. A good profile table lists each as a share of step time plus **idle**.

### 2.5 MFU and MBU
`MFU = achieved FLOP/s ÷ peak FLOP/s` (the prefill yardstick); `MBU = achieved bytes/s ÷ peak bytes/s` (the decode yardstick). Lab 04 estimated them for eager PyTorch; here you estimate them for the real engine.

### 2.6 Tail-latency investigation method (cross-layer)
1. **Reproduce** the spike under a controlled load and capture per-request timestamps.
2. **Locate the slow step**: find the inter-token gap and the wall-clock time it happened.
3. **Explain it at the scheduler level**: what was in that step (a prompt chunk of N tokens joined 60 decoding sequences?). Engine logs/metrics expose running/waiting counts and token budgets.
4. **Explain it at the kernel level**: in the trace, which kernels got longer or what new work appeared?
5. **Test the explanation**: change one thing (chunk size, token budget) and see whether the spike moves as predicted (Lab 08).

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Share of a batch-1 decode step spent in GEMMs | | | |
| ITL difference, eager vs CUDA graphs, at 1 user (ms and %) | | | |
| Idle (gap) fraction of a decode step with graphs | | | |
| What a slow step at 64 users contains | | | |

## 4. Steps and why
1. **Start the server with the profiler enabled** (flags confirmed at run time) and warm it. *The profiler must be enabled at launch; profiling a cold server records warm-up noise.*
2. **Profile a decode-heavy window**: 1 user, long output. *Isolates the memory-bound regime.*
3. **Profile a prefill-heavy window**: long prompts, 1-token outputs. *Isolates the compute-bound regime.*
4. **Eager vs graphs**: relaunch with `--enforce-eager`, compare ITL at 1 user. *Measures launch overhead directly.*
5. **Reproduce a p99 spike** at 64 users with open-loop load and record per-request timestamps (harness), then match the slow gap to a trace window. *The cross-layer investigation.*
6. **Fill the kernel table** (top kernels, share, idle) and compute MBU/MFU.

## 5. Results, 6. Interpretation, 7. Engineering takeaway, 8. Pitfalls
_To fill in using `templates/LAB_TEMPLATE.md`._ Known pitfalls: profiling slows the run (compare shapes, not absolute times); traces get huge (profile a short window); the first iterations include compilation.

## 9. Done when
- A table of the top kernels with % of step time and the idle fraction.
- Launch overhead in ms and %, measured.
- A written explanation of the p99 ITL spikes backed by a trace.

## 10. Interview questions
1. What does CUDA graph capture remove, and what does it require of the model code?
2. A decode step is at 60% of the bandwidth floor. How do you find out why?
3. How would you trace one slow request from the HTTP request down to the kernel?
4. Why can `nvidia-smi` show 100% utilization while the GPU is badly underused?

## Further reading
- PyTorch Profiler documentation and tutorial.
- NVIDIA Nsight Systems and Nsight Compute user guides.
- CUDA C++ Programming Guide: the CUDA Graphs chapter.
- Dao et al., "FlashAttention" (2022): what an optimized attention kernel looks like in a profile.
