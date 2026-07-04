"""
Live WebSocket server — the REAL pipeline.

Streams a wav file through the actual STT + LaBSE semantic matcher and pushes
the SAME `payload` shape that prototype_match.py prints, over a native
WebSocket. React cannot tell this apart from the fake server — identical JSON.

Design notes (the real-world lessons):
  1. Heavy models (LaBSE, Whisper) are loaded ONCE at startup, never per chunk.
  2. Inference is BLOCKING and slow, so each chunk is processed in a worker
     thread via `asyncio.to_thread` — otherwise it would freeze the WebSocket
     event loop and every client would stall.
  3. The matcher is STATEFUL (confirmed_pos only moves forward), so each client
     connection gets its OWN fresh matcher.

Run:
    virtual/bin/python src/live_ws_server.py
    virtual/bin/python src/live_ws_server.py "data/voices/Recording 6.wav" --chunk 3

Then point React at:  ws://localhost:8000/ws
"""

import argparse
import asyncio
import json
import os
import sys

import numpy as np
import websockets

_SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC)

# Reuse the EXACT helpers the prototype already uses — zero duplication.
# (Importing prototype_match also chdir's to the project root and sets the
#  Google credentials env var, so relative data/ paths resolve.)
from prototype_match import load_and_resample, chunk_audio, save_temp_wav
from trasncribe import transcribe, transcribe_whisper
from window_matcher import LiturgyWindowMatcher

HOST = "localhost"
PORT = 8000

# ── Lesson 1: load the heavy model ONCE, at startup ──────────────────────────
print("Loading LaBSE model (first run downloads weights ~1.8 GB)…")
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("LaBSE")
print("LaBSE ready.\n")


def process_chunk(chunk, matcher):
    """
    BLOCKING: STT + encode + semantic match for one audio chunk.
    Runs inside a worker thread (see stream_wav), so it must NOT touch the
    event loop. Returns a payload dict, or None if nothing was heard.
    """
    tmp = save_temp_wav(chunk)
    try:
        try:
            stt = transcribe(tmp)  # Google first, Whisper fallback
        except Exception as e:
            # Robustness: if the Google path errors (bad creds, network),
            # fall back to offline Whisper instead of killing the stream.
            print(f"[STT] Google path failed ({e}); using Whisper")
            stt = transcribe_whisper(tmp)
    finally:
        os.unlink(tmp)

    transcript = stt.get("transcript", "").strip()
    if not transcript:
        return None  # silence / no speech this chunk

    # spoken text → 768-dim LaBSE meaning vector → windowed match
    vec = MODEL.encode([transcript], normalize_embeddings=True)[0].astype(np.float32)
    result = matcher.match_vector(vec)
    m = result["meta"]

    # ── The SAME payload shape as prototype_match.py / fake_ws_server.py ──
    return {
        "id": m["id"],
        "score": result["score"],
        "confidence": stt.get("confidence", 0.0),
        "role": m["role"],
        "section": m["section"],
        "text": m["text"],
        "transcript": transcript,
        "backend": stt.get("backend"),
        "confirmed_pos": matcher.confirmed_pos,
        "window": list(result["window"]),
    }


async def stream_wav(websocket, wav_path, chunk_sec):
    # Lesson 3: fresh, per-connection matcher (it carries forward-only state).
    matcher = LiturgyWindowMatcher.from_files(page_size=10)
    audio = load_and_resample(wav_path)
    chunks = list(chunk_audio(audio, chunk_sec))
    print(f"{os.path.basename(wav_path)}: {len(chunks)} chunks of {chunk_sec}s")

    for idx, ts, chunk in chunks:
        # Lesson 2: offload blocking inference to a thread. `await` here yields
        # to the event loop so pings/other clients aren't starved.
        try:
            payload = await asyncio.to_thread(process_chunk, chunk, matcher)
        except Exception as e:
            print(f"[{ts:>5.1f}s] chunk {idx+1}: error {e}")
            continue

        if payload is None:
            print(f"[{ts:>5.1f}s] chunk {idx+1}: no speech")
            continue

        await websocket.send(json.dumps(payload, ensure_ascii=False))
        print(f"[{ts:>5.1f}s] → id {payload['id']:>2} [{payload['role']}] "
              f"score={payload['score']:.2f} \"{payload['transcript'][:30]}\"")


def make_handler(wav_path, chunk_sec):
    """Returns a handler bound to a specific wav file (websockets needs a
    1-arg coroutine; this closure carries the extra config)."""
    async def handler(websocket):
        print(f"client connected: {websocket.remote_address}")
        try:
            await stream_wav(websocket, wav_path, chunk_sec)
            print("stream finished")
        except websockets.ConnectionClosed:
            print("client disconnected")
    return handler


async def main(wav_path, chunk_sec):
    print(f"Live liturgy WS server on ws://{HOST}:{PORT}/ws")
    async with websockets.serve(make_handler(wav_path, chunk_sec), HOST, PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live liturgy WebSocket server (real pipeline)")
    p.add_argument("wav", nargs="?", default="data/voices/Recording 6.wav")
    p.add_argument("--chunk", type=float, default=3.0)
    args = p.parse_args()
    asyncio.run(main(args.wav, args.chunk))
