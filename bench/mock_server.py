"""Fake OpenAI-compatible server that imitates LLM timing, so the harness can be tested without a GPU.

Timing model (deliberately simple; the real numbers are what you measure on a GPU):
  prefill : PREFILL_MS_PER_TOKEN * prompt_tokens * (requests in flight), once
  decode  : one step takes DECODE_BASE_MS + DECODE_MS_PER_SEQ * (requests in flight)
Like vLLM it also serves:
  stream_options.include_usage -> a final chunk with token counts
  GET /metrics                 -> vllm:num_requests_running (lets the harness verify real concurrency)
  GET /version
--tokens-per-chunk K emits K tokens per streamed chunk, imitating speculative decoding.
"""
import argparse
import asyncio
import json
import time

from aiohttp import web

PREFILL_MS_PER_TOKEN = 0.05
DECODE_BASE_MS = 8.0
DECODE_MS_PER_SEQ = 0.4

state = {"active": 0, "peak": 0, "tokens_per_chunk": 1}


async def precise_sleep(seconds):
    # asyncio.sleep can return early on Windows (~15 ms clock tick); finish with a short spin.
    end = time.perf_counter() + seconds
    await asyncio.sleep(max(0.0, seconds - 0.016))
    while time.perf_counter() < end:
        await asyncio.sleep(0)


async def completions(request):
    body = await request.json()
    prompt_tokens = int(len(body["prompt"].split()) * 1.3)
    max_tokens = int(body["max_tokens"])
    k = state["tokens_per_chunk"]
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    state["active"] += 1
    state["peak"] = max(state["peak"], state["active"])
    try:
        await precise_sleep(PREFILL_MS_PER_TOKEN * prompt_tokens / 1000 * state["active"])
        sent = 0
        while sent < max_tokens:
            await precise_sleep((DECODE_BASE_MS + DECODE_MS_PER_SEQ * state["active"]) / 1000)
            n = min(k, max_tokens - sent)
            text = "".join(f" t{sent + j}" for j in range(n))
            await resp.write(f"data: {json.dumps({'choices': [{'text': text}]})}\n\n".encode())
            sent += n
        if (body.get("stream_options") or {}).get("include_usage"):
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": max_tokens,
                     "total_tokens": prompt_tokens + max_tokens}
            await resp.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    finally:
        state["active"] -= 1
    return resp


async def metrics(request):
    text = (f'vllm:num_requests_running{{model_name="mock"}} {float(state["active"])}\n'
            f'vllm:num_requests_waiting{{model_name="mock"}} 0.0\n')
    return web.Response(text=text)


async def version(request):
    return web.json_response({"version": "mock"})


async def peak(request):
    """Test hook: highest number of requests ever in flight since start."""
    return web.json_response({"peak": state["peak"]})


def make_app():
    app = web.Application()
    app.add_routes([web.post("/v1/completions", completions), web.get("/metrics", metrics),
                    web.get("/version", version), web.get("/_peak", peak)])
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tokens-per-chunk", type=int, default=1)
    a = ap.parse_args()
    state["tokens_per_chunk"] = a.tokens_per_chunk
    web.run_app(make_app(), port=a.port, print=None)
