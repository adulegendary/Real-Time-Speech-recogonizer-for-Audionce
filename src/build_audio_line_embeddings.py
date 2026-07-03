"""
Build per-liturgy-line audio embeddings using wav2vec2.

For each line that has a reference recording, we embed the audio
into a 1024-dim content vector and save it alongside the line metadata.

Usage:
    python src/build_audio_line_embeddings.py

Add new lines to LINE_AUDIO_MAP as you record them.
The output file (audio_line_vectors.npy) only contains lines that have
a recording — audio_line_meta.json records which IDs are covered.
"""

import json
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# ── paths ──────────────────────────────────────────────────────────────────────
_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_SRC, "..")
os.chdir(_ROOT)

TARGET_SR = 16_000
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"

# ── Add a recording for each liturgy line you have ────────────────────────────
# Key = line id (from geeze_liturgy.json), Value = path to wav file
LINE_AUDIO_MAP = {
    2:  "data/voices/Recording 6.wav",   # ወቡሩክ ስሙ ለዓለመ ዓለም  (congregation)
    # Add more as you record them, e.g.:
    # 1:  "data/voices/line_01.wav",     # ቡረክ እግዚአብሔር አምላከ አበዊነ  (priest)
    # 3:  "data/voices/line_03.wav",     # ቁሙ ለጸሎት                 (deacon)
}


# ── audio helpers ──────────────────────────────────────────────────────────────

def load_and_resample(wav_path: str) -> np.ndarray:
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(TARGET_SR, sr)
        data = resample_poly(data, TARGET_SR // g, sr // g)
    data = data.astype(np.float32)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak * 0.9
    return data


# ── wav2vec2 embedding ─────────────────────────────────────────────────────────

def load_model(model_id: str = MODEL_ID):
    print(f"Loading wav2vec2 model: {model_id}")
    print("(First run downloads ~1.2 GB — subsequent runs use cache)")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
    model = Wav2Vec2Model.from_pretrained(model_id)
    model.eval()
    print("Model ready.\n")
    return extractor, model


def embed_audio(audio: np.ndarray, extractor, model) -> np.ndarray:
    """
    Encode a 16 kHz mono float32 array with wav2vec2.
    Returns a L2-normalised 1024-dim vector (mean-pooled over time frames).
    """
    inputs = extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean-pool over time dimension → (1024,)
    hidden = outputs.last_hidden_state.squeeze(0)   # (T, 1024)
    vec = hidden.mean(dim=0).numpy()                # (1024,)

    # L2 normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────────

def build(line_audio_map: dict):
    # Load the full liturgy to get metadata for covered lines
    with open("data/geeze_liturgy.json", encoding="utf-8") as f:
        all_lines = json.load(f)
    line_by_id = {item["id"]: item for item in all_lines}

    # Validate all files exist before loading the heavy model
    missing = [
        (lid, path)
        for lid, path in line_audio_map.items()
        if not os.path.exists(path)
    ]
    if missing:
        for lid, path in missing:
            print(f"  Missing file for line {lid}: {path}")
        sys.exit(1)

    processor, model = load_model()

    vectors = []
    meta = []

    for line_id in sorted(line_audio_map.keys()):
        wav_path = line_audio_map[line_id]
        line_meta = line_by_id.get(line_id)
        if not line_meta:
            print(f"  Line id {line_id} not found in geeze_liturgy.json — skipping")
            continue

        print(f"  Embedding line {line_id:>2}: [{line_meta['role']:<13}] {line_meta['text']}")
        audio = load_and_resample(wav_path)
        vec = embed_audio(audio, processor, model)
        vectors.append(vec)
        meta.append(line_meta)
        print(f"             vec shape={vec.shape}  norm={np.linalg.norm(vec):.4f}  file={wav_path}")

    # Save
    out_vec  = "data/embeddings/audio_line_vectors.npy"
    out_meta = "data/embeddings/audio_line_meta.json"

    np.save(out_vec, np.array(vectors))
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(vectors)} audio line embeddings → {out_vec}")
    print(f"Saved metadata                           → {out_meta}")
    print(f"\nCovered lines: {[m['id'] for m in meta]}")
    print("Add more recordings to LINE_AUDIO_MAP and re-run to expand coverage.")


if __name__ == "__main__":
    build(LINE_AUDIO_MAP)
