"""Fake OpenAI-compatible server that imitates LLM timing, so the harness can be tested without a GPU.

Model of the timing (deliberately simple, and it is the thing you'll learn to measure on a real GPU):
  prefill  : PREFILL_MS_PER_TOKEN * prompt_tokens, once
  decode   : one step takes DECODE_BASE_MS + DECODE_MS_PER_SEQ * (sequences in the batch)
All concurrent requests share steps, i.e. a crude continuous batch.
"""
import asyncio
import json
import time

from aiohttp import web

PREFILL_MS_PER_TOKEN = 0.05
DECODE_BASE_MS = 8.0
DECODE_MS_PER_SEQ = 0.4
active = 0


async def precise_sleep(seconds):
    # asyncio.sleep can return early on Windows (~15 ms clock tick); finish with a short spin.
    end = time.perf_counter() + seconds
    await asyncio.sleep(max(0.0, seconds - 0.016))
    while time.perf_counter() < end:
        await asyncio.sleep(0)


async def completions(request):
    global active
    body = await request.json()
    prompt_tokens = int(len(body["prompt"].split()) * 1.3)
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    active += 1
    try:
        await precise_sleep(PREFILL_MS_PER_TOKEN * prompt_tokens / 1000 * active)  # prefill contends too
        for i in range(body["max_tokens"]):
            await precise_sleep((DECODE_BASE_MS + DECODE_MS_PER_SEQ * active) / 1000)
            chunk = {"choices": [{"text": f" t{i}"}]}
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    finally:
        active -= 1
    return resp


app = web.Application()
app.add_routes([web.post("/v1/completions", completions)])

if __name__ == "__main__":
    web.run_app(app, port=8000)
