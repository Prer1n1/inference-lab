"""Run a matrix of (server configuration) x (workload) and collect every result in one JSONL file.

    python bench/run_matrix.py bench/matrix.example.json

For each server: start it, wait until it answers, run every workload with bench.py, stop it, move on.
This is how Labs 07-15 sweep quantization formats, scheduler flags, engines and models without hand-typing commands.
Every row carries label "<server>/<workload>" and the server's launch flags, so results are self-describing.

Config (JSON):
{
  "out": "results/matrix.jsonl",
  "servers": [
    {"name": "bf16", "cmd": ["vllm", "serve", "meta-llama/Llama-3.1-8B-Instruct", "--port", "8000"],
     "url": "http://localhost:8000", "model": "meta-llama/Llama-3.1-8B-Instruct", "ready_timeout_s": 900}
  ],
  "workloads": [
    {"name": "chat", "args": ["--num-requests", "512", "--concurrency", "16", "64", "--input-words", "256"]}
  ],
  "bench_args": ["--warmup", "8"]      // optional: added to every bench.py call
}
A server with no "cmd" is assumed to be running already (use it to sweep workloads against a server you started yourself).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def wait_ready(url, timeout_s, proc):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        for path in ("/health", "/version", "/v1/models"):
            try:
                with urllib.request.urlopen(url + path, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
        time.sleep(1)
    raise TimeoutError(f"server at {url} not ready after {timeout_s}s")


def start(cmd):
    kw = {"start_new_session": True} if os.name == "posix" else {}
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)   # engines spawn child processes
        else:
            proc.terminate()
        proc.wait(timeout=60)
    except Exception:
        proc.kill()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg = json.load(open(sys.argv[1]))
    out = cfg.get("out", "results/matrix.jsonl")
    extra = cfg.get("bench_args", [])
    failures = 0
    for srv in cfg["servers"]:
        proc = start(srv["cmd"]) if srv.get("cmd") else None
        try:
            wait_ready(srv["url"], srv.get("ready_timeout_s", 900), proc)
            flags = " ".join(srv["cmd"]) if srv.get("cmd") else srv.get("flags_note", "")
            for wl in cfg["workloads"]:
                label = f"{srv['name']}/{wl['name']}"
                cmd = [sys.executable, os.path.join(HERE, "bench.py"), "--url", srv["url"], "--model", srv["model"],
                       "--label", label, "--server-flags", flags, "--out", out, *extra, *wl["args"]]
                print(f"== {label}", flush=True)
                if subprocess.run(cmd).returncode != 0:
                    failures += 1
                    print(f"!! {label} failed", file=sys.stderr)
        except Exception as e:  # keep going: one bad config must not lose the rest of the matrix
            failures += 1
            print(f"!! server {srv['name']}: {e!r}", file=sys.stderr)
        finally:
            stop(proc)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
