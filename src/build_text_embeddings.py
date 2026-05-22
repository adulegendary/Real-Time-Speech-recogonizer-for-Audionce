import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

SLIDE_FILE  = "data/slide_text.json"
EMBED_DIR   = "data/embeddings"
MODEL_NAME  = "LaBSE"

os.makedirs(EMBED_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME} model...")
model = SentenceTransformer(MODEL_NAME)
print("Model ready.\n")

# ── 1. Embed slide lines ────────────────────────────────────────────
with open(SLIDE_FILE, encoding="utf-8") as f:
    slides = json.load(f)

slide_texts = [s["text"] for s in slides]
slide_roles = [s["role"] for s in slides]

print(f"Embedding {len(slide_texts)} slide lines...")
slide_vectors = model.encode(slide_texts, normalize_embeddings=True)

np.save(os.path.join(EMBED_DIR, "slide_vectors.npy"), slide_vectors)
with open(os.path.join(EMBED_DIR, "slide_meta.json"), "w", encoding="utf-8") as f:
    json.dump(slides, f, ensure_ascii=False, indent=2)

print(f"Saved slide_vectors.npy  ({slide_vectors.shape})")

# ── 2. Embed known transcripts ──────────────────────────────────────
transcripts = {
    "priest": "ሰዓት ክንደይ ትምለስ ኢ ካብ መጽናዕት ደልየካ ነይረ",
    # deacon: add here once a Tigrinya recording is available
}

for speaker, text in transcripts.items():
    vec = model.encode([text], normalize_embeddings=True)[0]
    np.save(os.path.join(EMBED_DIR, f"{speaker}_text.npy"), vec)
    print(f"Saved {speaker}_text.npy")

# ── 3. Show similarity: transcript vs every slide line ─────────────
print("\n" + "="*55)
print("  MATCH TEST — Priest transcript vs slide lines")
print("="*55)

priest_vec = np.load(os.path.join(EMBED_DIR, "priest_text.npy"))
scores = [(np.dot(priest_vec, sv), slides[i]["role"], slides[i]["text"])
          for i, sv in enumerate(slide_vectors)]
scores.sort(reverse=True)

for score, role, text in scores:
    bar = "█" * int(score * 20)
    print(f"  {score:.3f} {bar:<20}  [{role}]  {text[:60]}")
