import io
import math
import os
import threading
import wave

import numpy as np
import soundfile as sf
from google.cloud import speech

os.environ.setdefault(
    "GOOGLE_APPLICATION_CREDENTIALS",
    os.path.join(os.path.dirname(__file__), "..", "google-credentials.json"),
)

# Reject any transcription below this confidence score
MIN_CONFIDENCE = 0.50

_whisper_model = None
_whisper_lock = threading.Lock()


def _load_whisper(model_size="medium"):
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            import whisper
            print(f"[Whisper] Loading '{model_size}' model...")
            _whisper_model = whisper.load_model(model_size)
            print("[Whisper] Model ready.")
    return _whisper_model


def _to_wav_bytes(wav_path: str):
    """Read a wav file, mono-mix, normalize, return (pcm_bytes, sample_rate)."""
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    pcm = (data * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue(), sr


def _whisper_confidence(result: dict) -> float:
    """
    Estimate confidence from Whisper segment statistics.
    avg_logprob is typically in [-1, 0]; exp() converts to (0, 1].
    We subtract a penalty proportional to no_speech_prob.
    """
    segments = result.get("segments", [])
    if not segments:
        return 0.0

    avg_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
    no_speech_prob = sum(s["no_speech_prob"] for s in segments) / len(segments)

    raw = math.exp(avg_logprob)           # convert log-prob → probability
    confidence = raw * (1.0 - no_speech_prob)
    return round(min(max(confidence, 0.0), 1.0), 3)


def transcribe_google(wav_path: str,
                      language: str = "ti-ET",
                      fallbacks: tuple = ("am-ET", "en-US")) -> dict:
    """
    Transcribe with Google Cloud STT.
    Returns: {transcript, confidence, language, backend}
    """
    audio_bytes, sr = _to_wav_bytes(wav_path)
    client = speech.SpeechClient()
    audio_obj = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sr,
        audio_channel_count=1,
        language_code=language,
        alternative_language_codes=list(fallbacks),
        enable_automatic_punctuation=True,
    )
    response = client.recognize(config=config, audio=audio_obj)

    if not response.results:
        return {"transcript": "", "confidence": 0.0, "language": language, "backend": "google"}

    best_alt = response.results[0].alternatives[0]
    detected_lang = response.results[0].language_code
    confidence = round(best_alt.confidence, 3) if best_alt.confidence > 0 else 0.0

    # Merge all result segments into one transcript
    full_text = " ".join(
        r.alternatives[0].transcript for r in response.results
    ).strip()

    return {
        "transcript": full_text,
        "confidence": confidence,
        "language": detected_lang,
        "backend": "google",
    }


def transcribe_whisper(wav_path: str,
                       language: str = "am",
                       model_size: str = "medium") -> dict:
    """
    Transcribe with OpenAI Whisper.
    Confidence is estimated from avg_logprob and no_speech_prob per segment.
    Returns: {transcript, confidence, language, backend}
    """
    model = _load_whisper(model_size)
    result = model.transcribe(
        wav_path,
        language=language,
        temperature=0,
        beam_size=5,
        verbose=False,
    )
    confidence = _whisper_confidence(result)
    return {
        "transcript": result["text"].strip(),
        "confidence": confidence,
        "language": result.get("language", language),
        "backend": "whisper",
    }


def transcribe(wav_path: str,
               google_language: str = "ti-ET",
               whisper_language: str = "am",
               whisper_model: str = "medium",
               min_confidence: float = MIN_CONFIDENCE) -> dict:
    """
    Smart transcription: try Google STT first.
    If Google returns empty or confidence < min_confidence, fall back to Whisper.
    Always returns: {transcript, confidence, language, backend, accepted}
    """
    google = transcribe_google(wav_path, language=google_language)

    if google["confidence"] >= min_confidence and google["transcript"]:
        google["accepted"] = True
        return google

    # Fall back to Whisper
    whisper = transcribe_whisper(wav_path, language=whisper_language,
                                 model_size=whisper_model)
    whisper["accepted"] = whisper["confidence"] >= min_confidence
    whisper["fallback_reason"] = (
        f"Google confidence {google['confidence']:.2f} < {min_confidence}"
        if google["transcript"]
        else "Google returned no speech"
    )
    return whisper


def transcribe_both(wav_path: str,
                    google_language: str = "ti-ET",
                    whisper_language: str = "am",
                    whisper_model: str = "medium") -> dict:
    """Run both backends in parallel and return both results for comparison."""
    google_result, whisper_result = {}, {}
    errors = {}

    def run_google():
        try:
            google_result.update(transcribe_google(wav_path, language=google_language))
        except Exception as e:
            errors["google"] = str(e)

    def run_whisper():
        try:
            whisper_result.update(transcribe_whisper(wav_path,
                                                     language=whisper_language,
                                                     model_size=whisper_model))
        except Exception as e:
            errors["whisper"] = str(e)

    t1 = threading.Thread(target=run_google)
    t2 = threading.Thread(target=run_whisper)
    t1.start(); t2.start()
    t1.join(); t2.join()

    return {"google": google_result, "whisper": whisper_result, "errors": errors}


# ── CLI ────────────────────────────────────────────────────────────────────────

def _bar(confidence: float, width: int = 20) -> str:
    filled = round(confidence * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _print_result(result: dict, label: str = ""):
    sep = "─" * 50
    print(f"\n{sep}")
    if label:
        print(f"  {label}")
    accepted = result.get("accepted")
    status = ""
    if accepted is True:
        status = "  ACCEPTED"
    elif accepted is False:
        status = "  REJECTED (low confidence)"

    transcript = result.get("transcript") or "(no speech detected)"
    confidence = result.get("confidence", 0.0)
    language = result.get("language", "?")
    backend = result.get("backend", "?")
    fallback = result.get("fallback_reason", "")

    print(f"  Backend    : {backend}")
    print(f"  Language   : {language}")
    print(f"  Transcript : {transcript}")
    print(f"  Confidence : {confidence:.2f}  {_bar(confidence)}{status}")
    if fallback:
        print(f"  Note       : {fallback}")
    print(sep)


if __name__ == "__main__":
    import sys

    # Usage: python trasncribe.py [mode] [file1 file2 ...]
    # mode: smart (default) | google | whisper | both
    mode = sys.argv[1] if len(sys.argv) > 1 else "smart"
    wav_files = sys.argv[2:] if len(sys.argv) > 2 else [
        "data/voices/priest.wav",
        "data/voices/deacon.wav",
        "data/voices/audienc.wav",
    ]

    for wav in wav_files:
        speaker = os.path.splitext(os.path.basename(wav))[0].upper()
        print(f"\n{'#' * 50}")
        print(f"  FILE: {speaker}  —  {wav}")
        print(f"{'#' * 50}")

        if mode == "google":
            _print_result(transcribe_google(wav), "Google STT (ti-ET)")

        elif mode == "whisper":
            _print_result(transcribe_whisper(wav), "Whisper (am)")

        elif mode == "both":
            out = transcribe_both(wav)
            if out["errors"]:
                print("\n  Errors:", out["errors"])
            _print_result(out["google"], "Google STT (ti-ET)")
            _print_result(out["whisper"], "Whisper (am)")

        else:  # smart
            _print_result(transcribe(wav), f"Smart transcription (threshold={MIN_CONFIDENCE})")

    print()
