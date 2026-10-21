# Lab 12: a Triton kernel against the roofline

**Status:** Planned  
**Baseten posting items:** "GPU kernels (CUDA, Triton, CUTLASS, or similar)"; deep understanding of GPU architecture; "Agentic Kernels in Production" (example initiative)  
**Needs:** an inexpensive NVIDIA GPU, about 1-2 hours (~$3) · **Depends on:** Labs 01, 04

## 1. The question
Can I write a kernel that matches or beats PyTorch on an inference operation, and explain the result with the roofline?

## 2. Concepts

### 2.1 Why write kernels at all
Most small operations in a transformer (normalization, residual add, rotary embedding, activation, sampling) are **memory-bound**: they read a tensor, do trivial math, and write a tensor. Run as separate PyTorch operations, each one makes a round trip to HBM. **Fusing** them into one kernel reads the data once, does all the math in registers, and writes once. The speedup is roughly the reduction in bytes moved.

### 2.2 The Triton programming model
Triton lets you write GPU kernels in Python at the level of **blocks** rather than individual threads:
- A kernel is launched over a **grid** of *programs*; each program handles one block of data and finds its block with `tl.program_id`.
- Blocks are sized by compile-time constants (`BLOCK_SIZE: tl.constexpr`); offsets are `tl.arange` vectors.
- `tl.load` and `tl.store` move data between HBM and registers, with a **mask** for the ragged edge.
- The compiler handles thread mapping, shared memory and vectorization. Autotuning (`@triton.autotune`) tries several block sizes and keeps the fastest.

### 2.3 Fused RMSNorm (the first kernel)
`y = x ÷ sqrt(mean(x²) + ε) × w`, optionally with a residual add first (`x = x + r`). Unfused PyTorch runs several kernels (square, mean, rsqrt, multiply, multiply by weight, add). Fused: each program loads one row, computes the sum of squares, normalizes and writes once. Bytes moved ≈ read `x` (+ read `r`) + write `y` (+ read `w`); at 8192 columns in BF16 that is about 48 KiB per row. Time lower bound = `bytes ÷ measured bandwidth` (Lab 01).

### 2.4 Quantized GEMV (the second kernel)
Decode's matmul is a **matrix-vector** product (batch 1). It is memory-bound, so reading INT8 or FP8 weights and dequantizing in registers moves half the bytes of BF16. A hand-written kernel shows exactly why weight-only quantization speeds decode (Lab 07).

### 2.5 How to judge a kernel
- **Correctness first**: compare against a reference with `torch.allclose` across shapes, dtypes and edge cases (non-multiples of the block size, tiny and huge rows). Tolerances depend on dtype; accumulate in FP32.
- **Achieved bandwidth** = bytes moved ÷ time, as a fraction of the Lab 01 measured ceiling. A memory-bound kernel near 80-90% of bandwidth is essentially done.
- **Benchmark honestly**: warm up, synchronize, sweep sizes, compare with PyTorch eager and `torch.compile` (which may already fuse these ops).
- Why a kernel stays slow: low occupancy, uncoalesced access, too-small blocks, register spills, launch overhead on tiny inputs.

## 3. Predictions (fill in BEFORE running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Fused vs unfused RMSNorm speedup (bytes-moved ratio) | | | |
| Fused kernel's achieved bandwidth as % of the Lab 01 ceiling | | | |
| Speedup of INT8-weight GEMV vs BF16 GEMV | | | |
| Size at which launch overhead dominates | | | |

## 4. Steps and why
1. **Write a reference** in plain PyTorch and a test harness over shapes and dtypes. *You cannot optimize what you cannot check.*
2. **Write the fused RMSNorm (+ residual) kernel** in Triton. *The simplest memory-bound win.*
3. **Benchmark** against eager and `torch.compile` across hidden sizes; plot achieved GB/s vs size with the roofline line. *Turns "faster" into "how close to the limit".*
4. **Autotune** block size and warps; record what changed.
5. **Second kernel**: weight-only INT8/FP8 dequantize-GEMV (or fused softmax), same procedure.
6. **Write one paragraph** explaining where each kernel sits relative to the roofline and what still limits it.

## 5. Results, 6. Interpretation, 7. Production takeaway, 8. Pitfalls
_To fill in._ Pitfalls: benchmarking without synchronization; testing only power-of-two sizes; comparing against an unfused baseline that `torch.compile` would already fuse; forgetting FP32 accumulation.

## 9. Done when
- Kernel code, tests and a bandwidth-vs-size plot.
- A paragraph explaining where the kernel is memory-bound and what limits it.

## 10. Interview questions
1. Why is fusing elementwise ops with a norm a win?
2. How do you know whether a kernel is memory- or compute-bound?
3. What does tile size trade off in a matmul kernel?
4. How would you decide between Triton, CUDA and CUTLASS for a new kernel?
