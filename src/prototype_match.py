"""
Prototype: match Recording 6.wav against the liturgy and PRINT, per chunk,
which line is being spoken and how confident we are.

Pipeline per chunk (STT + LaBSE — the runnable, confidence-producing path):
    audio chunk → resample 16 kHz → Google/Whisper STT → text + confidence
                → LaBSE encode → 768-dim vector
                → sliding-window semantic match → liturgy line

This is the backend half of the React display. Each `payload:` we print is
exactly the JSON the WebSocket will later push to React, where `id` moves the
highlight and `confidence`/`score` drive the confidence bar.

Run:
    virtual/bin/python src/prototype_match.py
    virtual/bin/python src/prototype_match.py "data/voices/Recording 6.wav" --chunk 3
"""

import argparse
import json
import os
import sys
import tempfile
import wave
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── paths / creds ────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC, "..")
os.chdir(_ROOT)
sys.path.insert(0, _SRC)
os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(_ROOT, "google-credentials.json"),
)

TARGET_SR = 16_000


# ── audio helpers ──────────────────────────────────────────────────────────────

def load_and_resample(wav_path: str) -> np.ndarray:
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        data = resample_poly(data, TARGET_SR // g, sr // g)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    return data


def chunk_audio(data: np.ndarray, chunk_sec: float):
    """Yield (index, timestamp_sec, audio_array). Drops final chunk if < 0.5s."""
    n = int(chunk_sec * TARGET_SR)
    for i, start in enumerate(range(0, len(data), n)):
        chunk = data[start: start + n]
        if len(chunk) < TARGET_SR * 0.5:
            break
        yield i, round(start / TARGET_SR, 2), chunk


def save_temp_wav(chunk: np.ndarray) -> str:
    pcm = (chunk * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())
    return tmp.name


def _bar(v, width=16):
    filled = round(max(0.0, min(1.0, v)) * width)
    return "█" * filled + "░" * (width - filled)


# ── main ─────────────────────────────────────────────────────────────────────

def run(wav_path, chunk_sec=3.0, use_fallback=True):
    from sentence_transformers import SentenceTransformer
    from trasncribe import transcribe, transcribe_google
    from window_matcher import LiturgyWindowMatcher

    print("Loading LaBSE model (first run downloads weights)…")
    model = SentenceTransformer("LaBSE")
    matcher = LiturgyWindowMatcher.from_files(page_size=10)
    print(f"Liturgy: {matcher.N} lines · {matcher.total_pages} pages\n")

    audio = load_and_resample(wav_path)
    chunks = list(chunk_audio(audio, chunk_sec))
    print(f"{os.path.basename(wav_path)}: {len(audio)/TARGET_SR:.1f}s "
          f"→ {len(chunks)} chunks of {chunk_sec}s\n")
    print("=" * 60)

    for idx, ts, chunk in chunks:
        tmp = save_temp_wav(chunk)
        try:
            stt = transcribe(tmp) if use_fallback else transcribe_google(tmp)
        finally:
            os.unlink(tmp)

        transcript = stt.get("transcript", "").strip()

        # nothing heard in this chunk — skip
        if not transcript:
            print(f"[{ts:>5.1f}s] chunk {idx+1:>2}  — no speech detected\n")
            continue

        # encode spoken text → 768-num meaning fingerprint, then match
        vec = model.encode([transcript], normalize_embeddings=True)[0].astype(np.float32)
        result = matcher.match_vector(vec)

        m = result["meta"]
        score = result["score"]                     # semantic-match cosine [0,1]
        conf = stt.get("confidence", 0.0)           # STT confidence [0,1]

        # ── human-readable print ──
        print(f"[{ts:>5.1f}s] chunk {idx+1:>2}   heard: \"{transcript}\"  ({stt.get('backend')})")
        print(f"          → LINE {m['id']:>2}  [{m['role']} · {m['section']}]")
        print(f"            {m['text']}")
        print(f"            STT confidence {conf:.2f} {_bar(conf)}   "
              f"match score {score:.2f} {_bar(score)}")

        # ── the exact object React will receive over the WebSocket ──
        payload = {
            "id": m["id"],
            "score": score,
            "confidence": conf,
            "role": m["role"],
            "section": m["section"],
            "text": m["text"],
            "transcript": transcript,
            "backend": stt.get("backend"),
            "confirmed_pos": matcher.confirmed_pos,
            "window": list(result["window"]),
        }
        print(f"            payload: {json.dumps(payload, ensure_ascii=False)}\n")

    print("=" * 60)
    cur = matcher.current_line()
    print(f"Ended on LINE {cur['id']} [{cur['role']}]: {cur['text']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prototype liturgy matcher (prints line + confidence)")
    p.add_argument("wav", nargs="?", default="data/voices/Recording 6.wav")
    p.add_argument("--chunk", type=float, default=3.0)
    p.add_argument("--no-fallback", action="store_true")
    args = p.parse_args()
    run(args.wav, chunk_sec=args.chunk, use_fallback=not args.no_fallback)
