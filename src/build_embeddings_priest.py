from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import os

# Define roles and voice files
roles = ["priest", "deacon"]
voice_dir = "data/voices"
embed_dir = "data/embeddings"

# Ensure embedding folder exists
os.makedirs(embed_dir, exist_ok=True)

# Load encoder once
encoder = VoiceEncoder()

for role in roles:
    path = os.path.join(voice_dir, f"{role}.wav")
    if not os.path.exists(path):
        print(f"❌ Voice file not found: {path}")
        continue

    wav = preprocess_wav(path)
    embedding = encoder.embed_utterance(wav)
    np.save(os.path.join(embed_dir, f"{role}.npy"), embedding)
    print(f"✅ Saved embedding for {role}")
