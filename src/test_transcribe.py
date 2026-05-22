import os
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"

client = speech.SpeechClient()

voices = ["priest", "deacon"]

for speaker in voices:
    path = f"data/voices/{speaker}.wav"
    print(f"\n{'='*45}")
    print(f"  Speaker: {speaker.upper()}  —  {path}")
    print(f"{'='*45}")

    with open(path, "rb") as f:
        audio_content = f.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        audio_channel_count=1,
        language_code="ti-ET",
        alternative_language_codes=["am-ET", "en-US"],
        enable_automatic_punctuation=True,
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("  ❌ No speech detected (likely Ge'ez — needs Whisper)")
    else:
        for result in response.results:
            best = result.alternatives[0]
            confidence = f"{best.confidence:.2f}" if best.confidence > 0 else "N/A"
            print(f"  Language   : {result.language_code}")
            print(f"  Transcript : {best.transcript}")
            print(f"  Confidence : {confidence}")
