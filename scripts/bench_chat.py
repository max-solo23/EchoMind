"""Latency benchmark for POST /api/v1/chat with a zero-latency fake LLM.

It measures what EchoMind adds on top of the model: request overhead, time-to-first-byte,
event-loop blocking and the cost of database round trips (simulated, no real database needed).

    python scripts/bench_chat.py            # in-process: CPU overhead + event-loop lag probe
    python scripts/bench_chat.py --http     # real uvicorn over TCP: true time-to-first-byte

BENCH_DB_MS (default 20) is the simulated latency of one database round trip.
Reference numbers after the 2026-10-19 performance pass live in .claude/docs/DECISIONS.md (D014).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from config import Config  # noqa: E402
from core.chat import Chat  # noqa: E402
from core.llm.types import (  # noqa: E402
    CompletionMessage,
    CompletionResponse,
    StreamDelta,
    ToolCallDelta,
)
from core.persona import Persona  # noqa: E402
from tools.llm_tools import Tools  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
PORT = 8765
HEADERS = {"X-API-Key": "bench"}
BODY = {"message": "What are your main skills?", "history": []}
DB_ROUND_TRIP = int(os.environ.get("BENCH_DB_MS", "20")) / 1000
NOTIFIER_BLOCK_SECONDS = 0.3
STREAM_EVENTS = 41

CONFIG = Config(
    llm_provider="openai",
    llm_api_key="bench",
    llm_base_url=None,
    llm_model="bench",
    pushover_token=None,
    pushover_user=None,
    persona_name="Max",
    persona_file=str(REPO / "persona.yaml"),
    api_key="bench",
    allowed_origins=[],
    database_url=None,
)
DELTAS = [StreamDelta(content=f"token{i} ") for i in range(STREAM_EVENTS - 1)] + [
    StreamDelta(finish_reason="stop")
]


class FakeLLM:
    """Zero-latency provider. With tool_call=True the first stream round asks for a tool."""

    def __init__(self, tool_call: bool = False):
        self.tool_call = tool_call

    @property
    def capabilities(self):
        return {"tools": True, "streaming": True}

    async def complete(self, *, model, messages, tools=None):
        return CompletionResponse(
            finish_reason="stop",
            message=CompletionMessage(role="assistant", content="fake reply " * 40),
        )

    async def stream(self, *, model, messages, tools=None):
        if self.tool_call and not any(m.get("role") == "tool" for m in messages):
            tool_call = ToolCallDelta(
                index=0, id="c1", name="record_unknown_question", arguments='{"question": "x"}'
            )
            yield StreamDelta(tool_calls=[tool_call])
            yield StreamDelta(finish_reason="tool_calls")
            return
        for delta in DELTAS:
            yield delta


class SlowNotifier:
    """Stands in for PushOver: a synchronous network call that takes 300 ms."""

    def push(self, text):
        time.sleep(NOTIFIER_BLOCK_SECONDS)


def make_chat(llm) -> Chat:
    return Chat(Persona("Max", CONFIG.persona_file), llm, "bench", Tools(SlowNotifier()))


async def fake_db_lookup(*args, **kwargs):
    await asyncio.sleep(DB_ROUND_TRIP)
    return None


async def fake_db_write(*args, **kwargs):
    await asyncio.sleep(DB_ROUND_TRIP)


def configure_app(simulate_db: bool | None = None):
    """Return the real FastAPI app with auth, rate limiting and the LLM replaced by fakes.

    Also the uvicorn factory for --http mode: `uvicorn --factory scripts.bench_chat:configure_app`.
    """
    if simulate_db is None:
        simulate_db = bool(os.environ.get("BENCH_DB"))

    patch("api.dependencies.get_config", return_value=CONFIG).start()
    patch("api.main.get_config", return_value=CONFIG).start()

    from api.dependencies import get_chat_service
    from api.main import app
    from api.middleware.auth import verify_api_key
    from api.middleware.rate_limit import check_rate_limit
    from api.routes import chat as chat_routes

    plain_chat = make_chat(FakeLLM())
    app.dependency_overrides[verify_api_key] = lambda: None
    app.dependency_overrides[check_rate_limit] = lambda: None
    app.dependency_overrides[get_chat_service] = lambda: plain_chat

    if simulate_db:
        patch.object(chat_routes, "is_database_configured", return_value=True).start()
        patch.object(chat_routes, "get_config", return_value=CONFIG).start()
        patch.object(chat_routes, "_get_cached_answer", fake_db_lookup).start()
        patch.object(chat_routes, "_persist_reply", fake_db_write).start()

    return app


# --- measurement helpers ---------------------------------------------------------------------


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


def fmt(name, values):
    ms = [v * 1000 for v in values]
    return (
        f"{name:<46} p50 {pct(ms, 50):7.2f} ms   p95 {pct(ms, 95):7.2f} ms   max {max(ms):7.2f} ms"
    )


async def timed_json(client):
    t0 = time.perf_counter()
    response = await client.post("/api/v1/chat", json=BODY, headers=HEADERS)
    assert response.status_code == 200, response.text
    return time.perf_counter() - t0


async def timed_stream(client):
    t0 = time.perf_counter()
    first_byte = None
    async with client.stream(
        "POST", "/api/v1/chat?stream=true", json=BODY, headers=HEADERS
    ) as response:
        async for _chunk in response.aiter_raw():
            if first_byte is None:
                first_byte = time.perf_counter() - t0
    return first_byte, time.perf_counter() - t0


async def measure(client, title, concurrency_levels=(1, 50), samples=200):
    print(f"-- {title} --")
    for conc in concurrency_levels:
        json_lat, ttfb, total = [], [], []
        for _ in range(max(1, samples // conc)):
            json_lat += await asyncio.gather(*[timed_json(client) for _ in range(conc)])
            streams = await asyncio.gather(*[timed_stream(client) for _ in range(conc)])
            ttfb += [s[0] for s in streams]
            total += [s[1] for s in streams]
        print(fmt(f"json reply                    concurrency={conc}", json_lat))
        print(fmt(f"stream time-to-first-byte     concurrency={conc}", ttfb))
        print(fmt(f"stream total ({STREAM_EVENTS} events)      concurrency={conc}", total))


async def loop_lag_probe(duration: float):
    """Sleep 5 ms in a loop and record how late each wake-up is. Lag means a blocked event loop."""
    lags, end = [], time.perf_counter() + duration
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        await asyncio.sleep(0.005)
        lags.append(time.perf_counter() - t0 - 0.005)
    return lags


# --- modes -------------------------------------------------------------------------------------


async def run_inprocess():
    """httpx ASGITransport buffers whole responses, so TTFB here equals total; use --http for TTFB."""
    from api.dependencies import get_chat_service
    from api.routes import chat as chat_routes

    app = configure_app(simulate_db=False)
    plain_chat = app.dependency_overrides[get_chat_service]()
    tool_chat = make_chat(FakeLLM(tool_call=True))

    print("\n== in-process (ASGITransport) ==")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b") as c:
        for _ in range(20):
            await timed_json(c)
        await measure(c, "no database")

        with (
            patch.object(chat_routes, "is_database_configured", return_value=True),
            patch.object(chat_routes, "get_config", return_value=CONFIG),
            patch.object(chat_routes, "_get_cached_answer", fake_db_lookup),
            patch.object(chat_routes, "_persist_reply", fake_db_write),
        ):
            db_title = f"simulated database ({DB_ROUND_TRIP * 1000:.0f} ms per round trip)"
            await measure(c, db_title, concurrency_levels=(1,), samples=50)
            await chat_routes.drain_background_tasks()

        print(
            f"-- event loop while one stream runs a tool that blocks {NOTIFIER_BLOCK_SECONDS:.1f} s --"
        )
        probe = asyncio.create_task(loop_lag_probe(1.0))
        await asyncio.sleep(0.1)
        app.dependency_overrides[get_chat_service] = lambda: tool_chat
        await timed_stream(c)
        app.dependency_overrides[get_chat_service] = lambda: plain_chat
        print(fmt("event-loop lag (5 ms sleeps, 1 s window)", await probe))


async def wait_ready(client, proc):
    for _ in range(150):
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited with {proc.returncode}")
        try:
            if (await client.get("/health")).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(0.1)
    raise RuntimeError("server did not become ready")


async def run_http():
    print("\n== uvicorn over TCP ==")
    db_title = f"simulated database ({DB_ROUND_TRIP * 1000:.0f} ms per round trip)"
    for db_mode, title in (("", "no database"), ("1", db_title)):
        command = [
            sys.executable, "-m", "uvicorn", "--factory", "scripts.bench_chat:configure_app",
            "--port", str(PORT), "--log-level", "error",
        ]  # fmt: skip
        proc = subprocess.Popen(
            command,
            cwd=REPO,
            env={**os.environ, "BENCH_DB": db_mode},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as c:
                await wait_ready(c, proc)
                for _ in range(10):
                    await timed_json(c)
                await measure(c, title, concurrency_levels=(1, 20), samples=100)
        finally:
            proc.terminate()
            proc.wait(10)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--http", action="store_true", help="benchmark a real uvicorn server over TCP"
    )
    args = parser.parse_args()
    asyncio.run(run_http() if args.http else run_inprocess())


if __name__ == "__main__":
    main()
