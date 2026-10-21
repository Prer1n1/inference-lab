# Lab 01: GPU architecture and the roofline

**Status:** Scripts ready (not yet run)  
**Baseten posting items:** "Deep understanding of GPU architecture"; "reason from first principles about where time and memory go"  
**Needs:** any CUDA GPU (an inexpensive one is fine; roofline numbers scale with the card). ~15 minutes. · **Depends on:** Lab 00

## 1. The question
What can this GPU actually do per second, in bytes moved and in math done, and which of the two limits a given workload?

## 2. Mental model
- A GPU has thousands of arithmetic units (grouped into **SMs**, streaming multiprocessors; tensor cores do the matrix math) and a large, comparatively slow memory (**HBM**).
- Every kernel does some FLOPs per byte it moves. If that ratio (the **arithmetic intensity**) is below the GPU's **ridge point** (peak FLOP/s ÷ peak bytes/s), the kernel waits on memory; above it, on math.
- Decode at batch 1 does about 1 FLOP per byte of weights, far below any ridge point (hundreds). Prefill on thousands of tokens is far above it.
- Spec-sheet numbers are peaks. You measure achievable ones, because those are what later labs compare against.
- `nvidia-smi` shows more memory than your tensors use: CUDA context, plus PyTorch's allocator caches freed blocks ("reserved" vs "allocated").

Reference for the H100 SXM used in Lab 05: bandwidth 3.35 TB/s (spec), ~990 dense BF16 TFLOP/s (spec), ridge ≈ 300 FLOP/byte.

## 3. Predictions (fill in before running)
| Quantity | Prediction | Real | Verdict |
|---|---|---|---|
| Achieved copy bandwidth, % of spec | 85-95% | | |
| Achieved BF16 matmul TFLOP/s, % of spec | 60-80% | | |
| Ridge point (FLOP/byte) | | | |
| Decode step floor for an 8B BF16 model (ms) | weights GB ÷ measured GB/s | | |
| PyTorch "reserved" memory after freeing a 1 GiB tensor | still ~1 GiB (cached) | | |

## 4. Experiment
```bash
pip install -r requirements-labs.txt          # torch is preinstalled on the RunPod PyTorch image
cd labs/01_gpu_architecture
python gpu_info.py --params-b 8 --out results/gpu_info.json
nvidia-smi                                    # compare with the memory-accounting numbers it prints
```
`gpu_info.py` copies a 2 GiB buffer (bandwidth), multiplies two 8192×8192 matrices (throughput), prints the ridge point,
the decode-step floors for BF16 / FP8 / INT4 weights, and a memory-accounting demo.

## 5. Results
_To fill in. Record GPU name, driver, CUDA, torch version from the JSON._

## 6. Interpretation
_To fill in: is the ridge point above or below the arithmetic intensity of decode (~1) and of prefill (~tokens)?_

## 7. Production takeaway
_To fill in. Hint: the floors computed here are the best decode speed a quantization or batching change can aim for, per GPU._

## 8. Pitfalls found
_To fill in._

## 9. Interview questions
1. What is the ridge point of a GPU and why does decode sit far below it?
2. Why can `nvidia-smi` show 70 GB used when the model weights are 16 GB?
3. Given a GPU's bandwidth and a model's size, what is the fastest one user can possibly decode? What would change it?
