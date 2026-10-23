# Lab 01: GPU architecture and the roofline

**Status:** Done on an L40S (2026-10-23). The memory-accounting demo was fixed afterwards and needs a quick re-run.  
**Baseten posting items:** "Deep understanding of GPU architecture"; "reason from first principles about where time and memory go"  
**Needs:** any CUDA GPU (an inexpensive one is fine; the roofline scales with the card). About 15 minutes. · **Depends on:** Lab 00

## 1. The question
What can this GPU actually do per second, in bytes moved and in arithmetic done, and which of the two limits a given workload?

## 2. Mental model

### 2.1 What is inside a GPU
- **SM (streaming multiprocessor)**: the unit that runs work. An H100 SXM has 132 of them. Each SM has its own registers, a small fast **shared memory** (programmer-managed scratchpad), and execution units.
- **CUDA cores vs tensor cores**: CUDA cores do ordinary scalar math; **tensor cores** do small matrix multiply-accumulate operations in one step and are where nearly all LLM math happens. Low-precision formats (BF16, FP8) run much faster on tensor cores.
- **Warp**: 32 threads that execute together. Kernels launch thousands of threads; the hardware schedules warps across SMs and hides memory latency by switching to other warps while one waits.
- **Memory hierarchy** (fast and small → slow and large): registers → shared memory / L1 (per SM) → L2 cache (shared, about 50 MB on H100) → **HBM** (high-bandwidth memory, the 80 GB "VRAM"). Each step down is roughly an order of magnitude slower per byte. Good kernels keep data high in the hierarchy and touch HBM as little as possible.
- **Kernel launch**: the CPU queues a kernel (a function that runs on the GPU) into a **stream**; the GPU runs queued kernels in order. Launching costs microseconds of CPU time, which matters when a model step consists of hundreds of tiny kernels (Lab 06).

### 2.2 The two numbers that matter
- **Memory bandwidth** (bytes/s): how fast HBM can feed the SMs. H100 SXM spec: about 3.35 TB/s. A100 80GB: about 2.0 TB/s. Consumer and workstation cards using GDDR memory are far lower.
- **Compute throughput** (FLOP/s): how fast tensor cores can multiply. H100 SXM spec: about 989 TFLOP/s dense BF16 and about 1,979 TFLOP/s dense FP8.
- Spec sheets quote **peaks**; real kernels reach a fraction. That is why this lab *measures* both on your card, and why every later lab compares against the measured values.

### 2.3 The roofline model
A kernel moves some bytes and does some FLOPs. Its **arithmetic intensity** is `AI = FLOPs ÷ bytes moved`.

```
attainable FLOP/s = min( peak FLOP/s ,  AI × bandwidth )
ridge point        = peak FLOP/s ÷ bandwidth        (FLOP per byte)
```
- If `AI < ridge`: the kernel is **memory-bound**; faster math units would not help, only fewer bytes or more bandwidth.
- If `AI > ridge`: it is **compute-bound**; extra bandwidth would not help.
- H100 SXM: 989e12 ÷ 3.35e12 ≈ **295 FLOP/byte**.

### 2.4 Why LLM decode is far below the ridge
A matrix multiply `(M × K) · (K × N)` in BF16 does `2·M·K·N` FLOPs and moves about `2·(M·K + K·N + M·N)` bytes. During decode with batch size `B`, `M = B` (one new token per sequence) and the weight matrix `K × N` dominates the bytes:

```
AI ≈ 2·B·K·N / (2·K·N) = B   FLOP per byte
```
At batch 1, AI ≈ 1 against a ridge near 300: the GPU's math units sit idle waiting for weights. You need roughly `B ≈ ridge` tokens in the same step before decode becomes compute-bound. Prefill processes thousands of tokens at once (`M` in the thousands), so it is compute-bound. This single fact explains most of the rest of the repo (Labs 03-09).

### 2.5 The decode floor
Every decode step must read all the weights once: `step_time ≥ weight_bytes ÷ bandwidth`.
- 8B parameters in BF16 = about 16 GB; at 3.35 TB/s that is about **4.8 ms**; at a *measured* 90% of peak, about 5.3 ms.
- FP8 weights (8 GB): about 2.4 ms floor. INT4 (4 GB): about 1.2 ms. This is the entire motivation for quantization (Lab 07).

### 2.6 Why `nvidia-smi` and PyTorch disagree about memory
- A **CUDA context** (driver state, kernels, libraries) takes a few hundred MB before any tensor exists.
- PyTorch's allocator keeps freed blocks in a cache ("reserved") instead of returning them to the driver, so `nvidia-smi` stays high after `del tensor`. `memory_allocated` counts live tensors; `memory_reserved` counts what PyTorch holds.
- vLLM goes further and deliberately reserves most of the card at startup (Lab 05).

## 3. Predictions (written BEFORE running)

Predictions for an L40S (spec used: 864 GB/s and 362 TFLOP/s dense BF16, taken from NVIDIA's listing from memory and not re-checked here).

| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Achieved copy bandwidth as % of the spec for your GPU | 80% → 0.80 × 864 = 691.2 GB/s | **655.3 GB/s = 75.8%** | Close |
| Achieved BF16 matmul TFLOP/s as % of spec | 75% → 0.75 × 362 = 271.5 TFLOP/s | **195.7 TFLOP/s = 54.1%** | Too optimistic |
| Ridge point (FLOP/byte) from your two measured numbers | 271.5 TFLOP/s ÷ 691.2 GB/s = **393** (first written as 0.39: a 1,000× unit slip, caught and corrected) | **298.6** | Too high, because the matmul guess was too high |
| Decode step floor for an 8B BF16 model on this GPU (ms) | 16 GB ÷ 691.2 GB/s = 23.1 ms | **24.42 ms** | Close |
| After freeing a 1 GiB tensor, does the "free memory" figure return to its starting value? | No: PyTorch keeps freed blocks in its cache | Free memory **never moved** (39.947 GiB at all three points), `reserved` stayed 4.0 GiB | Right in spirit; see 6 for why the demo was flawed

## 4. Experiment, step by step
```bash
cd /root/inference-lab/labs/01_gpu_architecture
python gpu_info.py --params-b 8 --out results/gpu_info.json
nvidia-smi
```
**What the script does and why:**
1. Reads the card's properties (name, SM count, compute capability, memory). *Identifies what you are measuring.*
2. Copies a 2 GiB buffer on the GPU and times it with CUDA events. Bandwidth = bytes read + written ÷ time. *Gives the measured memory ceiling.*
3. Multiplies two 8192×8192 matrices in BF16 and FP16. TFLOP/s = `2·n³ ÷ time`. *Gives the measured compute ceiling.*
4. Prints the ridge point and the decode-step floors for BF16, FP8/INT8 and INT4 weights. *These are the targets Labs 04 and 07 aim at.*
5. Allocates and frees a 1 GiB tensor, printing free memory each time. *Shows the allocator cache.*

Time: under a minute. Cost: nothing noticeable.

## 5. Results
**Environment:** RunPod, 1× NVIDIA L40S (Ada, compute capability 8.9, 142 SMs, 48 GB listed = 44.39 GiB visible to PyTorch), driver 595.91.07, CUDA 13.2 driver / 12.8 runtime, PyTorch 2.8.0+cu128, Ubuntu 24.04, Python 3.12. Raw output: `results/gpu_info.json`.

| Quantity | Run 1 | Run 2 (kept) | % of spec |
|---|---|---|---|
| Copy bandwidth (read + write) | 655.6 GB/s | **655.3 GB/s** | 75.8% of 864 |
| BF16 matmul (8192³) | 194.4 TFLOP/s | **195.7 TFLOP/s** | 54.1% of 362 |
| FP16 matmul | 189.8 | 184.7 | |
| Ridge point (BF16 ÷ bandwidth) | 296.5 FLOP/byte | **298.6 FLOP/byte** | spec ratio would be 419 |

Decode-step floor for 8B parameters (weights ÷ measured bandwidth, run 2): **BF16 16 GB → 24.42 ms**, FP8/INT8 8 GB → 12.21 ms, INT4 4 GB → 6.10 ms. Two runs agree within 1%.

Memory accounting (run 1 and 2 identical): total 44.394 GiB; free 39.947 GiB before, with a 1 GiB tensor, and after `del`; `reserved` 4.0 GiB; `allocated` 0.008 GiB.

## 6. Interpretation
- **Ridge point 298.6 FLOP/byte** is far above decode's arithmetic intensity at batch 1 (about 1) and far below prefill's at a 2,048-token prompt (about 2,048). That is the whole reason the two phases behave differently (confirmed in Lab 04).
- **Bandwidth reached 75.8% of spec, matmul only 54.1%.** I do not know why the matmul is lower; plausible causes are power or clock limits and the matrix shape, but I did not test them. The practical lesson: measure, do not trust the datasheet.
- **The ridge point from measured numbers (298.6) is well below the spec ratio (362 ÷ 864 ≈ 419)** because the matmul fell further below its spec than the copy did. Decisions should use the measured one.
- **The 296.5 vs 298.6 difference between runs is under 1%**, so the measurement is stable.
- **The memory-accounting demo was flawed.** Free memory did not change when a 1 GiB tensor was allocated, because the earlier benchmarks had already left 4 GiB in PyTorch's cache (`reserved` 4.0 GiB) and the tensor came from that cache. The gap between total (44.39) and free (39.95) is that 4.0 GiB cache plus about 0.45 GiB, which is probably the CUDA context (not separately measured). `gpu_info.py` now empties the cache first and also reports free memory after `empty_cache()`; the clean result still needs a re-run on a GPU.

## 7. Engineering takeaway
Every later number is judged against two *measured* ceilings: 655 GB/s and 196 TFLOP/s on this L40S. They give a decode floor of about 24.4 ms per step for an 8B BF16 model (about 41 tokens/s for one user), and a ridge point near 300 FLOP/byte, which says roughly how large a batch must be before adding users stops being free. Quantization attacks the floor directly (FP8 halves it, INT4 quarters it).

## 8. Pitfalls to watch
- Timing GPU work without `torch.cuda.synchronize()` measures only the launch. The script uses CUDA events and synchronizes.
- Throttling: clocks and power can vary; repeat once and compare.
- Spec peaks for sparse vs dense tensor operations differ by 2×; compare against the dense figure.

## 9. Interview questions
1. What is the ridge point of a GPU, and why does decode sit far below it?
2. Why can `nvidia-smi` show 70 GB used when the model weights are 16 GB?
3. Given a GPU's bandwidth and a model's size, what is the fastest one user can possibly decode? What would change that?
4. Why do SMs switch between warps while one waits on memory?

### Worked answers
1. **Ridge point.** It is peak FLOP/s ÷ memory bandwidth, the arithmetic intensity at which a kernel stops being limited by memory and starts being limited by math. Measured on this L40S: 195.7 TFLOP/s ÷ 655.3 GB/s ≈ 299 FLOP/byte. Decode sits far below it: a BF16 weight is 2 bytes and does 2 FLOPs per sequence, so arithmetic intensity is about the batch size, 1 at batch 1. The math units are idle most of the time; only fewer bytes or more bandwidth helps.
2. **70 GB in `nvidia-smi` with 16 GB of weights.** `nvidia-smi` shows everything the process holds: weights, the CUDA context, PyTorch's cached blocks and, for vLLM, a KV-cache pool it reserves at startup (about 90% of the card by default). Most of it is empty room. What one step *reads* (the traffic) is only the weights plus the KV of running sequences.
3. **Fastest one user can decode.** Weight bytes ÷ bandwidth: 16 GB ÷ 655 GB/s ≈ 24.4 ms, about 41 tokens/s on this card (measured 24.9 ms in Lab 03, 98% of it). To go faster: fewer bytes (FP8/INT4), more bandwidth (a better GPU), split the bytes across GPUs (tensor parallelism), or emit more than one token per weight read (speculative decoding).
4. **Warp switching.** A memory access takes hundreds of cycles. While one warp waits, the SM runs another warp whose data is ready, so the execution units stay busy; this is latency hiding, and it is why GPUs want many threads in flight.

## Further reading
- Horace He, "Making Deep Learning Go Brrrr From First Principles" (memory-bound vs compute-bound reasoning).
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009).
- NVIDIA H100 datasheet and architecture whitepaper (SMs, tensor cores, HBM, NVLink).
- CUDA C++ Programming Guide: programming model and memory hierarchy chapters.
