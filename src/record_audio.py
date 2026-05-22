import sounddevice as sd
import soundfile as sf
import os

SPEAKERS = ["deacon"]
VOICE_DIR = "data/voices"
DURATION = 8
SAMPLE_RATE = 16000

def record(filename, speaker_name, duration=DURATION):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sd.default.device = None  # uses system default mic

    input(f"\nReady to record '{speaker_name}'. Press Enter then speak for {duration} seconds...")
    print(f"🎤 Recording '{speaker_name}'...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved: {filename}")

if __name__ == "__main__":
    print("=== Speaker Voice Recording ===")
    print(f"Recording reference voices for: {', '.join(SPEAKERS)}")

    for speaker in SPEAKERS:
        path = os.path.join(VOICE_DIR, f"{speaker}.wav")
        record(path, speaker_name=speaker)

    print("\n All speakers recorded. Run build_embeddings_priest.py to generate embeddings.")
