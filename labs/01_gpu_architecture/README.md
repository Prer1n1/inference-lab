# Lab 01: GPU architecture and the roofline

**Status:** Scripts ready (not yet run)  
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

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Achieved copy bandwidth as % of the spec for your GPU | | | |
| Achieved BF16 matmul TFLOP/s as % of spec | | | |
| Ridge point (FLOP/byte) from your two measured numbers | | | |
| Decode step floor for an 8B BF16 model on this GPU (ms) | weights ÷ measured bandwidth | | |
| After freeing a 1 GiB tensor, does the "free memory" figure return to its starting value? | | | |

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
_To fill in. Record GPU name, driver, CUDA, torch version, the two measured ceilings, the ridge point and the floors._

## 6. Interpretation
_To fill in: is the ridge point far above decode's arithmetic intensity (about 1) and below prefill's (about the number of prompt tokens)? Which of your measured numbers is closer to its spec, and why might the other one not be?_

## 7. Engineering takeaway
_To fill in._ Hint: the floors computed here are the best decode speed a quantization or batching change can aim for on this GPU, and the ridge point tells you roughly how large a batch you need before adding users stops being free.

## 8. Pitfalls to watch
- Timing GPU work without `torch.cuda.synchronize()` measures only the launch. The script uses CUDA events and synchronizes.
- Throttling: clocks and power can vary; repeat once and compare.
- Spec peaks for sparse vs dense tensor operations differ by 2×; compare against the dense figure.

## 9. Interview questions
1. What is the ridge point of a GPU, and why does decode sit far below it?
2. Why can `nvidia-smi` show 70 GB used when the model weights are 16 GB?
3. Given a GPU's bandwidth and a model's size, what is the fastest one user can possibly decode? What would change that?
4. Why do SMs switch between warps while one waits on memory?

## Further reading
- Horace He, "Making Deep Learning Go Brrrr From First Principles" (memory-bound vs compute-bound reasoning).
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009).
- NVIDIA H100 datasheet and architecture whitepaper (SMs, tensor cores, HBM, NVLink).
- CUDA C++ Programming Guide: programming model and memory hierarchy chapters.
