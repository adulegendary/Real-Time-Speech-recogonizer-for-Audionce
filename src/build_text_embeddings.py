import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

LITURGY_FILE = "data/geeze_liturgy.json"
EMBED_DIR    = "data/embeddings"
MODEL_NAME   = "LaBSE"

os.makedirs(EMBED_DIR, exist_ok=True)

print(f"Loading {MODEL_NAME} model...")
model = SentenceTransformer(MODEL_NAME)
print("Model ready.\n")

# ── 1. Embed every liturgy line ─────────────────────────────────────
with open(LITURGY_FILE, encoding="utf-8") as f:
    liturgy = json.load(f)

texts = [line["text"] for line in liturgy]

print(f"Embedding {len(texts)} liturgy lines...")
book_vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

np.save(os.path.join(EMBED_DIR, "book_vectors.npy"), book_vectors)
with open(os.path.join(EMBED_DIR, "book_meta.json"), "w", encoding="utf-8") as f:
    json.dump(liturgy, f, ensure_ascii=False, indent=2)

print(f"\nSaved book_vectors.npy  shape: {book_vectors.shape}")
print(f"Saved book_meta.json    ({len(liturgy)} lines)\n")

# ── 2. Quick match test using a sample Ge'ez phrase ─────────────────
test_phrases = [
    "ቅዱስ ቅዱስ ቅዱስ",
    "ሰላም ለኩሉ",
    "አቡነ ዘበሰማያት",
    "ሃሌ ሉያ",
]

print("=" * 60)
print("  MATCH TEST — sample phrases vs book")
print("=" * 60)

for phrase in test_phrases:
    query_vec = model.encode([phrase], normalize_embeddings=True)[0]
    scores = [(float(np.dot(query_vec, bv)), liturgy[i]) for i, bv in enumerate(book_vectors)]
    scores.sort(reverse=True)
    top = scores[0]
    print(f"\n  Query : {phrase}")
    print(f"  Match : [{top[1]['role']}] {top[1]['text']}  (score: {top[0]:.3f})")
    print(f"  Section: {top[1]['section']}")
