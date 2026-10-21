# Lab 05 re-run runbook: measure the levels the old harness could not

Goal: replace the invalid c=128 / c=256 rows with real measurements at 128, 192, 256, 384 and 512 users, find where throughput
really saturates, and check whether vLLM itself limits concurrency. About 25 minutes of H100 time (~$1.70).

General pod procedure (RunPod, disks, tmux, venv, terminating): `docs/pod-workflow.md`.

## 1. On the pod, after installing vLLM and starting the server
```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000
```
Find the server's sequence limit in the startup log (look for `max_num_seqs`); a default below 512 caps concurrency on the server side:
```bash
tmux capture-pane -p -S -3000 -t server | grep -iE "max_num_seqs|max_num_batched_tokens|KV cache size"
```

## 2. Re-measure (client and server on the same machine)
```bash
cd /root/inference-lab && git pull
M=meta-llama/Llama-3.1-8B-Instruct
FLAGS="--max-model-len 8192 --gpu-memory-utilization 0.90"
python bench/bench.py --model $M --server-flags "$FLAGS" --label baseline --num-requests 1024 \
  --concurrency 64 128 192 256 384 512 --out labs/05_vllm_baseline/results/rerun.jsonl
```
Read the output: `server_peak_running` must be close to the level asked for. If it is not, the tool prints a WARNING and the
level reflects the lower number. Run it twice (second run to `rerun2.jsonl`) and compare.

## 3. Optional but valuable: open-loop load (real traffic arrives at a rate)
```bash
python bench/bench.py --model $M --label baseline-open --num-requests 600 --rate 30 45 60 75 \
  --slo-ttft-ms 500 --slo-tpot-ms 20 --out labs/05_vllm_baseline/results/open_loop.jsonl
```
Watch where `goodput_frac` collapses: that arrival rate is the real capacity under the latency target.

## 4. Save and stop
Copy results into the repo (`git add`, commit, push from the pod only if you have set up credentials; otherwise paste the rows
into the chat). Then Stop and Terminate the pod in the RunPod console. Update the table in `README.md` and the correction note.
