"""
Fake WebSocket server — replays liturgy `payload` objects so the React UI can be
built and tested WITHOUT running the heavy wav2vec2 / STT pipeline.

It emits exactly the JSON shape produced by prototype_match.py, walking forward
through the liturgy one line every couple of seconds.

Run:
    virtual/bin/python src/fake_ws_server.py

Then point the React app at:  ws://localhost:8000/ws
"""

import asyncio
import json
import os
import random

import websockets

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC, "..")

with open(os.path.join(_ROOT, "data", "geeze_liturgy.json"), encoding="utf-8") as f:
    LITURGY = json.load(f)

HOST = "localhost"
PORT = 8000
STEP_SECONDS = 2.0  # how fast to advance through the liturgy


def make_payload(line, confirmed_pos):
    """Build one message identical in shape to prototype_match.py's `payload`."""
    return {
        "id": line["id"],
        "score": round(random.uniform(0.62, 0.95), 3),       # semantic match
        "confidence": round(random.uniform(0.70, 0.98), 3),  # STT confidence
        "role": line["role"],
        "section": line["section"],
        "text": line["text"],
        "transcript": line["text"],   # fake: pretend we heard it perfectly
        "backend": "fake",
        "confirmed_pos": confirmed_pos,
        "window": [line["id"] - 1, line["id"], line["id"] + 1],
    }


async def handler(websocket):
    """Runs once per connected browser tab."""
    print(f"client connected: {websocket.remote_address}")
    try:
        for pos, line in enumerate(LITURGY):
            payload = make_payload(line, pos)
            await websocket.send(json.dumps(payload, ensure_ascii=False))
            print(f"  → sent line {line['id']:>2} [{line['role']}] {line['text']}")
            await asyncio.sleep(STEP_SECONDS)
        print("liturgy finished — closing")
    except websockets.ConnectionClosed:
        print("client disconnected")


async def main():
    print(f"Fake liturgy WS server on ws://{HOST}:{PORT}/ws  ({len(LITURGY)} lines)")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
