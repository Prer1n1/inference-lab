"""Self-tests for bench/bench.py against the mock server. No GPU needed.

    python tests/test_harness.py

Each test guards a bug that cost us real measurements:
  - concurrency silently capped by the HTTP client (Lab 05: c=128/256 were really ~100)
  - one streamed chunk assumed to be one token (breaks under speculative decoding)
  - goodput and open-loop paths work
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


class Mock:
    def __init__(self, port, tokens_per_chunk=1):
        self.port = port
        self.proc = subprocess.Popen([PY, os.path.join(ROOT, "bench", "mock_server.py"), "--port", str(port),
                                      "--tokens-per-chunk", str(tokens_per_chunk)])
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/version", timeout=1)
                return
            except Exception:
                time.sleep(0.2)
        raise RuntimeError("mock server did not start")

    def peak(self):
        return json.load(urllib.request.urlopen(f"http://localhost:{self.port}/_peak"))["peak"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.proc.terminate()
        self.proc.wait()


def bench(port, *extra):
    out = os.path.join(tempfile.mkdtemp(), "r.jsonl")
    p = subprocess.run([PY, os.path.join(ROOT, "bench", "bench.py"), "--url", f"http://localhost:{port}",
                        "--warmup", "0", "--no-gpu-telemetry", "--out", out, *extra],
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    with open(out) as f:
        return [json.loads(line) for line in f], p.stderr


def test_basic_closed_loop():
    with Mock(8141) as _:
        rows, _err = bench(8141, "--num-requests", "8", "--concurrency", "4", "--output-tokens", "16",
                           "--input-words", "64")
    r = rows[0]
    assert r["ok"] == 8 and r["errors"] == 0, r
    assert r["chunk_mismatch_reqs"] == 0 and r["usage_missing_reqs"] == 0, r
    assert abs(r["out_tok_per_s"] * r["wall_s"] - 8 * 16) <= 0.05 * 8 * 16, r  # tokens counted exactly
    assert r["itl_ms_p50"] > 0 and r["ttft_ms_p50"] > 0, r


def test_concurrency_is_not_capped():
    n = 150  # above aiohttp's default connection limit of 100
    with Mock(8142) as m:
        rows, _err = bench(8142, "--num-requests", str(n), "--concurrency", str(n), "--output-tokens", "8",
                           "--input-words", "64")
        truth = m.peak()
    r = rows[0]
    assert truth >= 0.9 * n, f"server really saw only {truth} concurrent requests"
    assert r["client_peak_inflight"] >= 0.9 * n, r
    assert r["server_peak_running"] >= 0.8 * n, r  # reported through /metrics polling


def test_multi_token_chunks_counted_by_usage():
    with Mock(8143, tokens_per_chunk=4) as _:
        rows, _err = bench(8143, "--num-requests", "6", "--concurrency", "3", "--output-tokens", "16",
                           "--input-words", "64")
    r = rows[0]
    assert r["chunk_mismatch_reqs"] == 6, r  # 4 chunks vs 16 tokens, flagged
    assert abs(r["out_tok_per_s"] * r["wall_s"] - 6 * 16) <= 0.05 * 6 * 16, r  # still 16 tokens each


def test_open_loop_and_goodput():
    with Mock(8144) as _:
        rows, _err = bench(8144, "--num-requests", "20", "--rate", "20", "--output-tokens", "8",
                           "--input-words", "64", "--slo-ttft-ms", "0.001")
    r = rows[0]
    assert r["mode"] == "open" and r["ok"] == 20, r
    assert r["goodput_frac"] == 0.0, r  # impossible TTFT target, nothing is good


def test_matrix_runner_sweeps_servers_and_workloads():
    cfg = {"out": os.path.join(tempfile.mkdtemp(), "m.jsonl"),
           "servers": [
               {"name": "k1", "cmd": [PY, os.path.join(ROOT, "bench", "mock_server.py"), "--port", "8151"],
                "url": "http://localhost:8151", "model": "mock", "ready_timeout_s": 30},
               {"name": "k4", "cmd": [PY, os.path.join(ROOT, "bench", "mock_server.py"), "--port", "8152",
                                      "--tokens-per-chunk", "4"],
                "url": "http://localhost:8152", "model": "mock", "ready_timeout_s": 30}],
           "workloads": [{"name": "a", "args": ["--num-requests", "6", "--concurrency", "2", "--output-tokens", "8",
                                                 "--input-words", "32"]},
                         {"name": "b", "args": ["--num-requests", "6", "--concurrency", "3", "--output-tokens", "8",
                                                 "--input-words", "32"]}],
           "bench_args": ["--warmup", "0", "--no-gpu-telemetry"]}
    path = os.path.join(tempfile.mkdtemp(), "cfg.json")
    json.dump(cfg, open(path, "w"))
    p = subprocess.run([PY, os.path.join(ROOT, "bench", "run_matrix.py"), path], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    rows = [json.loads(line) for line in open(cfg["out"])]
    assert sorted(r["label"] for r in rows) == ["k1/a", "k1/b", "k4/a", "k4/b"], [r["label"] for r in rows]
    assert all(r["chunk_mismatch_reqs"] == (6 if r["label"].startswith("k4") else 0) for r in rows), rows
    for port in (8151, 8152):   # servers were stopped
        try:
            urllib.request.urlopen(f"http://localhost:{port}/version", timeout=1)
            raise AssertionError(f"server on {port} still running")
        except AssertionError:
            raise
        except Exception:
            pass


def test_warns_when_server_sees_less_load_than_requested():
    sys.path.insert(0, os.path.join(ROOT, "bench"))
    import bench as b
    row = {"errors": 0, "usage_missing_reqs": 0, "server_peak_running": 100.0}
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        b.check_row(row, "closed", 256)
    assert "fewer than the 256 requested" in buf.getvalue()


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_")):
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", repr(e))
    sys.exit(1 if failed else 0)
