"""
Real-time semantic matching simulation.

Slices a .wav file into fixed-duration chunks and feeds each chunk
through the full pipeline (resample → STT → LaBSE → sliding-window match),
simulating what the live microphone loop will do.

The LiturgyWindowMatcher state persists across chunks — the window
advances forward through the liturgy as speech is recognised.

Usage:
    python src/realtime_match.py                                 # default: Recording 6.wav, 3s chunks
    python src/realtime_match.py data/voices/deacon.wav          # different file
    python src/realtime_match.py data/voices/priest.wav --chunk 2
    python src/realtime_match.py data/voices/deacon.wav --chunk 2 --no-fallback
"""

import argparse
import os
import sys
import tempfile
import time
import wave
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# ── paths ──────────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC, "..")
os.chdir(_ROOT)
sys.path.insert(0, _SRC)

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(_ROOT, "google-credentials.json"),
)

TARGET_SR = 16_000
MIN_SCORE = 0.40   # below this we print the match but flag it as weak


# ── audio helpers ──────────────────────────────────────────────────────────────

def load_and_resample(wav_path: str) -> np.ndarray:
    """Load any wav, mono-mix, resample to TARGET_SR, normalise."""
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
    """
    Yield (chunk_index, timestamp_sec, audio_array) for each chunk.
    Drops the final chunk if it is shorter than 0.5 s.
    """
    chunk_samples = int(chunk_sec * TARGET_SR)
    for i, start in enumerate(range(0, len(data), chunk_samples)):
        chunk = data[start: start + chunk_samples]
        if len(chunk) < TARGET_SR * 0.5:
            break
        yield i, round(start / TARGET_SR, 2), chunk


def save_temp_wav(chunk: np.ndarray) -> str:
    """Write float32 mono array to a temporary 16-bit PCM wav."""
    pcm = (chunk * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())
    return tmp.name


# ── display helpers ────────────────────────────────────────────────────────────

def _bar(v: float, width: int = 16) -> str:
    filled = round(v * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _header(wav_path: str, chunk_sec: float, total_chunks: int):
    print(f"\n{'═' * 68}")
    print(f"  Real-Time Match Simulation")
    print(f"  File   : {os.path.basename(wav_path)}")
    print(f"  Chunks : {total_chunks}  ×  {chunk_sec}s  =  "
          f"{total_chunks * chunk_sec:.1f}s coverage")
    print(f"{'═' * 68}\n")


def _print_chunk(
    idx: int,
    ts: float,
    transcript: str,
    conf: float,
    backend: str,
    result: dict | None,
    skipped: bool = False,
):
    ts_label = f"[{ts:>5.1f}s]"
    chunk_label = f"chunk {idx + 1:>2}"

    if skipped:
        print(f"{ts_label}  {chunk_label}  — no speech detected")
        return

    tx_short = (transcript[:38] + "…") if len(transcript) > 40 else transcript
    conf_str = f"{conf:.2f} {_bar(conf)}"
    backend_tag = f"({backend})"

    print(f"{ts_label}  {chunk_label}  {backend_tag:<9}  conf={conf_str}")
    print(f"           transcript : {tx_short}")

    if result:
        m = result["meta"]
        score = result["score"]
        flag = "  ⚠ weak" if score < MIN_SCORE else ""
        match_text = m["text"]
        if len(match_text) > 35:
            match_text = match_text[:33] + "…"
        print(
            f"           match      : [id {m['id']:>2}] {match_text:<35} "
            f"score={score:.3f} {_bar(score)}{flag}"
        )
        print(f"           {result['window_summary']}")
    print()


# ── main pipeline ──────────────────────────────────────────────────────────────

def run(wav_path: str, chunk_sec: float = 3.0, use_fallback: bool = True):
    from sentence_transformers import SentenceTransformer
    from trasncribe import transcribe_google, transcribe
    from window_matcher import LiturgyWindowMatcher

    # Load models once
    print("Loading LaBSE model (first run may download weights)...")
    model = SentenceTransformer("LaBSE")
    print("LaBSE ready.\n")

    matcher = LiturgyWindowMatcher.from_files(page_size=10)
    print(f"Liturgy loaded: {matcher.N} lines across {matcher.total_pages} pages.\n")

    # Load and slice audio
    print(f"Loading audio: {wav_path}")
    audio = load_and_resample(wav_path)
    duration = len(audio) / TARGET_SR
    chunks = list(chunk_audio(audio, chunk_sec))
    print(f"Duration: {duration:.1f}s  →  {len(chunks)} chunks of {chunk_sec}s\n")

    _header(wav_path, chunk_sec, len(chunks))

    for idx, ts, chunk in chunks:
        tmp_path = save_temp_wav(chunk)

        try:
            if use_fallback:
                stt = transcribe(tmp_path)          # Google → Whisper fallback
            else:
                stt = transcribe_google(tmp_path)   # Google only
        finally:
            os.unlink(tmp_path)

        transcript = stt.get("transcript", "").strip()

        if not transcript:
            _print_chunk(idx, ts, "", 0.0, stt.get("backend", "?"), None, skipped=True)
            continue

        # Encode + match
        vec = model.encode([transcript], normalize_embeddings=True)[0].astype(np.float32)
        result = matcher.match_vector(vec)

        # Attach the window summary string so _print_chunk can display it
        result["window_summary"] = matcher.window_summary()

        _print_chunk(
            idx, ts,
            transcript,
            stt.get("confidence", 0.0),
            stt.get("backend", "?"),
            result,
        )

    # Final state
    print(f"{'─' * 68}")
    print(f"  Final position: {matcher.window_summary()}")
    current = matcher.current_line()
    print(f"  Last confirmed : [id {current['id']}] [{current['role']}] {current['text']}")
    print(f"{'─' * 68}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate real-time liturgy matching on a wav file"
    )
    parser.add_argument(
        "wav",
        nargs="?",
        default="data/voices/Recording 6.wav",
        help="Path to wav file (default: Recording 6.wav)",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=3.0,
        metavar="SEC",
        help="Chunk duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Use Google STT only, skip Whisper fallback",
    )
    args = parser.parse_args()

    run(args.wav, chunk_sec=args.chunk, use_fallback=not args.no_fallback)
