"""
Audio → Vector pipeline for liturgy matching.

Steps:
  1. Load audio file  (any sample rate)
  2. Resample to 16 kHz mono  (required by Google STT + Resemblyzer)
  3. Transcribe with Google STT  → text + confidence
  4. Encode text with LaBSE     → 768-dim normalised vector
  5. Match vector against liturgy using sliding window

Usage:
  python src/embed_audio.py "data/voices/Recording 6.wav"
  python src/embed_audio.py "data/voices/Recording 6.wav" --page-size 10
"""

import argparse
import io
import os
import sys
import tempfile
import wave

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(__file__), "..", "google-credentials.json"),
)

TARGET_SR = 16_000


# ── Step 1 + 2: Load and resample ─────────────────────────────────────────────

def load_and_resample(wav_path: str) -> tuple[np.ndarray, int]:
    """Load any wav file and return (mono_float32_array, TARGET_SR)."""
    data, sr = sf.read(wav_path)

    # Mix down to mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        up, down = TARGET_SR // g, sr // g
        data = resample_poly(data, up, down).astype(np.float32)
        print(f"  Resampled {sr} Hz → {TARGET_SR} Hz")
    else:
        data = data.astype(np.float32)

    # Normalise
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9

    return data, TARGET_SR


def save_temp_wav(data: np.ndarray, sr: int) -> str:
    """Write float32 mono array to a temporary 16-bit PCM wav file."""
    pcm = (data * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return tmp.name


# ── Step 3: Transcribe ─────────────────────────────────────────────────────────

def transcribe(wav_path: str) -> dict:
    """Run Google STT on a 16 kHz mono wav. Returns {transcript, confidence, language}."""
    from google.cloud import speech

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    client = speech.SpeechClient()
    audio_obj = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=TARGET_SR,
        audio_channel_count=1,
        language_code="ti-ET",
        alternative_language_codes=["am-ET", "en-US"],
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio_obj)

    if not response.results:
        return {"transcript": "", "confidence": 0.0, "language": "ti-ET"}

    best = response.results[0].alternatives[0]
    full_text = " ".join(r.alternatives[0].transcript for r in response.results).strip()
    confidence = round(best.confidence, 3) if best.confidence > 0 else 0.0
    language = response.results[0].language_code

    return {"transcript": full_text, "confidence": confidence, "language": language}


# ── Step 4: Encode text to LaBSE vector ───────────────────────────────────────

def encode_text(text: str, model) -> np.ndarray:
    """Return a normalised 768-dim LaBSE vector for the given text."""
    return model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


# ── Step 5: Match against liturgy ─────────────────────────────────────────────

def match(vec: np.ndarray, matcher) -> dict:
    return matcher.match_vector(vec)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def _bar(v: float, width: int = 20) -> str:
    filled = round(v * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def run(wav_path: str, page_size: int = 10, verbose: bool = True):
    from sentence_transformers import SentenceTransformer
    from window_matcher import LiturgyWindowMatcher

    print(f"\n{'=' * 60}")
    print(f"  File: {os.path.basename(wav_path)}")
    print(f"{'=' * 60}")

    # 1+2. Load and resample
    print("\n[1/4] Loading and resampling audio...")
    audio, sr = load_and_resample(wav_path)
    duration = len(audio) / sr
    print(f"  Duration : {duration:.2f}s  |  Sample rate : {sr} Hz")
    tmp_path = save_temp_wav(audio, sr)

    # 3. Transcribe
    print("\n[2/4] Transcribing with Google STT (ti-ET)...")
    stt = transcribe(tmp_path)
    os.unlink(tmp_path)

    transcript = stt["transcript"] or "(no speech detected)"
    conf = stt["confidence"]
    print(f"  Transcript : {transcript}")
    print(f"  Confidence : {conf:.2f}  {_bar(conf)}")
    print(f"  Language   : {stt['language']}")

    if not stt["transcript"]:
        print("\n  No speech detected — cannot match to liturgy.")
        return None

    # 4. Encode text
    print("\n[3/4] Encoding transcript with LaBSE...")
    model = SentenceTransformer("LaBSE")
    vec = encode_text(stt["transcript"], model)
    print(f"  Vector shape : {vec.shape}  (norm: {float(np.linalg.norm(vec)):.4f})")

    # 5. Match
    print("\n[4/4] Matching against liturgy window (page_size={})...".format(page_size))
    matcher = LiturgyWindowMatcher.from_files(page_size=page_size)
    result = match(vec, matcher)

    m = result["meta"]
    score = result["score"]
    print(f"\n  Best match  : [id {m['id']}] [{m['role']}] {m['text']}")
    print(f"  Section     : {m['section']}")
    print(f"  Match score : {score:.4f}  {_bar(score)}")
    print(f"  Page        : {result['page']}/{matcher.total_pages}")
    print(f"  Window      : [{result['window'][0]}, {result['window'][1]}]  "
          f"size={result['window_size']}/{matcher.N}")
    print(f"\n  {matcher.window_summary()}")

    return {
        "file": wav_path,
        "transcript": stt["transcript"],
        "stt_confidence": conf,
        "vector": vec,
        "match": result,
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio to vector and match liturgy")
    parser.add_argument("wav", nargs="?",
                        default="data/voices/Recording 6.wav",
                        help="Path to .wav file (default: Recording 6.wav)")
    parser.add_argument("--page-size", type=int, default=10,
                        help="Lines per page for sliding window (default: 10)")
    args = parser.parse_args()

    # Make paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..")
    os.chdir(project_root)
    sys.path.insert(0, script_dir)

    run(args.wav, page_size=args.page_size)
    print()
