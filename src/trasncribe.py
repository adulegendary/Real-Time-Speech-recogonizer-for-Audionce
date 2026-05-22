import os
import io
import wave
import threading
import numpy as np
import soundfile as sf
from google.cloud import speech

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      os.path.join(os.path.dirname(__file__), "..", "google-credentials.json"))

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


def _to_wav_bytes(wav_path: str) -> bytes:
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


def transcribe_google(wav_path: str, language="ti-ET",
                      fallbacks=("am-ET", "en-US")) -> dict:
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
    results = []
    for r in response.results:
        b = r.alternatives[0]
        results.append({
            "language": r.language_code,
            "transcript": b.transcript,
            "confidence": round(b.confidence, 3) if b.confidence > 0 else None,
        })
    return {"backend": "google", "results": results}


def transcribe_whisper(wav_path: str, language="am",
                       model_size="medium") -> dict:
    # Whisper has no Tigrinya model; Amharic ("am") shares the same
    # Ethiopic script and gives the best results for Tigrinya audio.
    model = _load_whisper(model_size)
    result = model.transcribe(wav_path, language=language,
                              temperature=0, beam_size=5)
    return {
        "backend": "whisper",
        "results": [{
            "language": result.get("language", "unknown"),
            "transcript": result["text"].strip(),
            "confidence": None,
        }],
    }


def transcribe_both(wav_path: str, whisper_language="am",
                    google_language="ti-ET", whisper_model="medium") -> dict:
    google_result, whisper_result = {}, {}
    errors = {}

    def run_google():
        try:
            google_result.update(transcribe_google(wav_path, language=google_language))
        except Exception as e:
            errors["google"] = str(e)

    def run_whisper():
        try:
            whisper_result.update(transcribe_whisper(wav_path, language=whisper_language,
                                                     model_size=whisper_model))
        except Exception as e:
            errors["whisper"] = str(e)

    t1 = threading.Thread(target=run_google)
    t2 = threading.Thread(target=run_whisper)
    t1.start(); t2.start()
    t1.join(); t2.join()

    return {
        "google": google_result,
        "whisper": whisper_result,
        "errors": errors,
    }


def _print_result(label: str, result: dict):
    print(f"\n{'─'*40}")
    print(f"  {label}")
    print(f"{'─'*40}")
    if not result:
        print("  (no result)")
        return
    for r in result.get("results", []):
        transcript = r["transcript"] or "(empty)"
        conf = f"  conf: {r['confidence']}" if r["confidence"] else ""
        print(f"  [{r['language']}] {transcript}{conf}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"  # both | google | whisper
    voices = sys.argv[2:] if len(sys.argv) > 2 else ["data/voices/priest.wav",
                                                       "data/voices/deacon.wav"]

    for wav in voices:
        speaker = os.path.splitext(os.path.basename(wav))[0].upper()
        print(f"\n{'#'*48}")
        print(f"  SPEAKER: {speaker}  —  {wav}")
        print(f"{'#'*48}")

        if mode == "google":
            out = transcribe_google(wav)
            _print_result("Google STT (ti-ET)", out)

        elif mode == "whisper":
            out = transcribe_whisper(wav)
            _print_result("Whisper (am forced)", out)

        else:
            out = transcribe_both(wav)
            if out["errors"]:
                print("\n  Errors:", out["errors"])
            _print_result("Google STT  (ti-ET)", out["google"])
            _print_result("Whisper     (am forced)", out["whisper"])

    print()
