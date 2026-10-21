# GPU pod workflow (RunPod)

Written from the Lab 05 session, including the things that went wrong. Pods are disposable; GitHub is the source of truth.

## Before renting (laptop, free)
- Have the commands for the session written down and tested against the mock server where possible. Idle pod minutes cost money.
- Write the lab's predictions first.
- Push all code: the pod clones from GitHub.

## Choosing hardware
- Labs 01-04, 11, 16, 17, 19: an inexpensive GPU is enough (Lab 17 needs any NVIDIA GPU). Lab 07 (FP8) needs Hopper or Ada. Engine labs (05-10, 15, 18): H100. Lab 13 needs 2+ GPUs (ideally one run with NVLink, one without).
- Labs 09 (Part A), 12 (Part A), 14 (Part A), 20 and 21 need no GPU.
- Check the listing says **SXM** (3.35 TB/s) vs **PCIe** (about 2 TB/s); decode speed scales with memory bandwidth.
- Add storage before deploying: container disk 50 GB, plus a **global volume** mounted at `/workspace` (keeps the model download across pods).

## On the pod
Open Connect → Jupyter → File → New → Terminal. Every new terminal starts clean, so for each one:
```bash
source /root/venv/bin/activate
export HF_HOME=/workspace/hf
export HF_TOKEN=hf_xxx        # only in the terminal that starts the server; never write it to a file
```
First time on a new pod:
```bash
apt-get update && apt-get install -y tmux
git clone https://github.com/<you>/inference-lab.git /root/inference-lab       # NOT onto /workspace (see below)
python -m venv /root/venv && source /root/venv/bin/activate
pip install -r /root/inference-lab/requirements-vllm.txt                         # or requirements-labs.txt
nvidia-smi; vllm --version                                                       # record both in the lab README
```
Run long things (the server) inside `tmux new -s server`; detach with Ctrl+b then d. Do not click inside a tmux pane in the
browser terminal (clicks turn into garbage characters); scroll with Ctrl+b then `[`, leave with `q`, or search the log with
`tmux capture-pane -p -S -3000 -t server | grep ...` (tmux keeps only ~2,000 lines).

## Things that went wrong
| Symptom | Cause | Fix |
|---|---|---|
| `git clone` into `/workspace` fails: `chmod ... Operation not permitted` | the global volume does not support chmod | clone into `/root`; keep only the model cache on the volume |
| `vllm: command not found` | new shell without the venv | `source /root/venv/bin/activate` |
| 30 GB default disk too small | vLLM install plus a 16 GB model | set container disk to 50 GB before deploying |
| Benchmark numbers disagree between runs | repeated prompts hit the prefix cache | fresh prompts per level (harness default) |
| Asked for 256 users, server saw 100 | HTTP client connection cap | uncapped connector; harness checks the server gauge |

## Ending the session
1. Copy results into the repo (commit and push from the pod only if credentials are set up; otherwise paste rows into your notes).
2. In the RunPod console: **Stop**, then **Terminate**. Closing the browser tab does not stop billing.
3. Check the balance a few minutes later and update the budget log in the README.
