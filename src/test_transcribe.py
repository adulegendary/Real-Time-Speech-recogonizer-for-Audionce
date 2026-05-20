import os
from google.cloud import speech

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-credentials.json"

client = speech.SpeechClient()

with open("data/voices/priest.wav", "rb") as f:
    audio_content = f.read()

audio = speech.RecognitionAudio(content=audio_content)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    audio_channel_count=1,
    language_code="ti-ET",                          # Tigrinya (primary)
    alternative_language_codes=["am-ET", "en-US"],  # Amharic, English as fallbacks
    enable_automatic_punctuation=True,
)

print("Sending audio to Google Speech-to-Text...")
response = client.recognize(config=config, audio=audio)

if not response.results:
    print("No speech detected.")
else:
    for result in response.results:
        best = result.alternatives[0]
        confidence = f"{best.confidence:.2f}" if best.confidence > 0 else "N/A (Tigrinya not scored by Google)"
        print(f"Language detected : {result.language_code}")
        print(f"Transcript        : {best.transcript}")
        print(f"Confidence        : {confidence}")
