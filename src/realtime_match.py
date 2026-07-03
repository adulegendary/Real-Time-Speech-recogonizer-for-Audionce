"""
Real-time semantic matching — audio-only pipeline.

No transcription. No STT. Each audio chunk is encoded directly
into a content vector using wav2vec2 and matched against
pre-built reference embeddings of the liturgy lines.

Pipeline per chunk:
    audio chunk  →  wav2vec2 encoder  →  1024-dim vector
                →  cosine similarity against audio_line_vectors.npy
                →  sliding window match  →  liturgy line

Build reference embeddings first:
    python src/build_audio_line_embeddings.py

Then run:
    python src/realtime_match.py                              # default: Recording 6.wav
    python src/realtime_match.py data/voices/deacon.wav
    python src/realtime_match.py data/voices/priest.wav --chunk 2
"""

import argparse
import os
import sys
import tempfile
import wave
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ── paths ──────────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC, "..")
os.chdir(_ROOT)
sys.path.insert(0, _SRC)

TARGET_SR  = 16_000
MODEL_ID   = "facebook/wav2vec2-large-xlsr-53"
MIN_SCORE  = 0.40   # matches below this are flagged weak


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
    chunk_samples = int(chunk_sec * TARGET_SR)
    for i, start in enumerate(range(0, len(data), chunk_samples)):
        chunk = data[start: start + chunk_samples]
        if len(chunk) < TARGET_SR * 0.5:
            break
        yield i, round(start / TARGET_SR, 2), chunk


# ── wav2vec2 ───────────────────────────────────────────────────────────────────

def load_wav2vec2(model_id: str = MODEL_ID):
    print(f"Loading wav2vec2: {model_id}")
    print("(First run downloads ~1.2 GB — cached after that)\n")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id)
    model.eval()
    return extractor, model


def embed_chunk(audio: np.ndarray, extractor, model) -> np.ndarray:
    """
    Encode a 16 kHz mono float32 array.
    Mean-pools wav2vec2 hidden states → L2-normalised 1024-dim vector.
    """
    inputs = extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state.squeeze(0)  # (T, 1024)

    vec = hidden.mean(dim=0).numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ── display ────────────────────────────────────────────────────────────────────

def _bar(v: float, width: int = 16) -> str:
    filled = round(v * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _header(wav_path: str, chunk_sec: float, n_chunks: int, n_ref: int):
    print(f"\n{'═' * 68}")
    print(f"  Real-Time Audio Match  (no STT — pure audio → vector)")
    print(f"  File      : {os.path.basename(wav_path)}")
    print(f"  Chunks    : {n_chunks}  ×  {chunk_sec}s")
    print(f"  Ref lines : {n_ref}  liturgy lines with audio embeddings")
    print(f"{'═' * 68}\n")


def _print_match(idx, ts, result, score):
    m = result["meta"]
    flag = "  ⚠ weak" if score < MIN_SCORE else ""
    text = m["text"]
    if len(text) > 35:
        text = text[:33] + "…"
    ts_label = f"[{ts:>5.1f}s]"
    print(
        f"{ts_label}  chunk {idx+1:>2}  "
        f"[id {m['id']:>2}] [{m['role']:<13}] {text:<35}  "
        f"score={score:.3f} {_bar(score)}{flag}"
    )
    print(f"           {result['window_summary']}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def run(wav_path: str, chunk_sec: float = 3.0):
    from window_matcher import LiturgyWindowMatcher

    # Load reference audio embeddings (built by build_audio_line_embeddings.py)
    vec_path  = "data/embeddings/audio_line_vectors.npy"
    meta_path = "data/embeddings/audio_line_meta.json"

    if not os.path.exists(vec_path):
        print("ERROR: No audio line embeddings found.")
        print("Run first:  python src/build_audio_line_embeddings.py")
        sys.exit(1)

    print(f"Loading audio line embeddings from {vec_path} ...")
    matcher = LiturgyWindowMatcher.from_files(
        vectors_path=vec_path,
        meta_path=meta_path,
        page_size=10,
    )
    print(f"  {matcher.N} liturgy lines loaded.\n")

    # Load encoder
    extractor, model = load_wav2vec2()

    # Load and slice the input audio
    print(f"Loading: {wav_path}")
    audio  = load_and_resample(wav_path)
    dur    = len(audio) / TARGET_SR
    chunks = list(chunk_audio(audio, chunk_sec))
    print(f"Duration: {dur:.1f}s  →  {len(chunks)} chunks of {chunk_sec}s\n")

    _header(wav_path, chunk_sec, len(chunks), matcher.N)

    for idx, ts, chunk in chunks:
        vec    = embed_chunk(chunk, extractor, model)
        result = matcher.match_vector(vec)
        result["window_summary"] = matcher.window_summary()
        _print_match(idx, ts, result, result["score"])

    # Final state
    print(f"{'─' * 68}")
    print(f"  Final : {matcher.window_summary()}")
    cur = matcher.current_line()
    print(f"  Last confirmed line: [id {cur['id']}] [{cur['role']}] {cur['text']}")
    print(f"{'─' * 68}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time liturgy matching — audio only, no STT"
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
        help="Chunk size in seconds (default: 3.0)",
    )
    args = parser.parse_args()
    run(args.wav, chunk_sec=args.chunk)
