import sounddevice as sd
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.io.wavfile import write
import tempfile
import os

# Load voice embeddings
roles = ["priest", "deacon"]
embeddings = {}
for role in roles:
    embeddings[role] = np.load(f"data/embeddings/{role}.npy")

encoder = VoiceEncoder()

# Record & identify in a loop
print("🎤 Listening for speaker... (Ctrl+C to stop)")

try:
    while True:
        duration = 2  # seconds
        fs = 16000
        print("🎙️ Recording chunk...")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            write(tmp.name, fs, audio)
            wav = preprocess_wav(tmp.name)
            os.remove(tmp.name)

        # Get embedding
        live_embed = encoder.embed_utterance(wav)

        # Compare to known speakers
        scores = {}      
        for role in embeddings:
            # Calculate similarity
            similarity = np.dot(embeddings[role], live_embed)
            scores[role] = similarity
            
        
        best_match = max(scores, key=scores.get)
        confidence = scores[best_match]

        print(f"🧠 Detected: {best_match} (confidence: {confidence:.2f})\n")

except KeyboardInterrupt:
    print("🛑 Stopped.")
