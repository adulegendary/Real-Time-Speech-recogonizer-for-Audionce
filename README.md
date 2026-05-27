# Real-Time Speech Recognizer for Audience

> A real-time speech recognition system that identifies speakers and transcribes spoken words during a Mass (or live presentation), synchronizing output with slides and subtitles for the audience.

---

## System Architecture

```text
 ┌────────────────────────┐
 │     Microphone Input   │
 │ (Priest / Deacon /     │
 │  Other Speakers)       │
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │   Audio Capture        │   ← record_audio.py / live_synch.py
 │   sounddevice @ 16kHz  │
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ Speaker Identification │   ← live_synch.py + build_embeddings_*.py
 │ Resemblyzer embeddings │
 │ (priest, deacon, ...)  │
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │   Speech-to-Text (STT) │   ← trasncribe.py
 │   Google STT (ti-ET)   │
 │   + Whisper fallback   │
 │   + confidence score   │
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ Semantic Matching Layer│   ← window_matcher.py + embed_audio.py
 │ LaBSE embeddings       │
 │ Page-aware sliding     │
 │ window (10 lines/page) │
 └───────────┬────────────┘
             │
      ┌──────┴────────┐
      │               │
      ▼               ▼
┌────────────────┐  ┌────────────────────────┐
│ Slide Highlight │  │   Translation Layer    │   [NOT YET BUILT]
│ (UI Sync)       │  │  Ge'ez → English       │
│ gui_display.py  │  │  Pre-aligned lookup or │
└────────┬────────┘  │  MarianMT / NLLB       │
         │            └──────────┬────────────┘
         └──────────┬────────────┘
                    ▼
         ┌────────────────────────┐
         │  Frontend Display      │   ← gui_display.py  [NOT YET BUILT]
         │  - Speaker label       │
         │  - Live subtitles      │
         │  - Highlighted slide   │
         └────────────────────────┘
```

---

## Tech Stack

| Component              | Tool / Framework                                        |
|------------------------|---------------------------------------------------------|
| Audio Capture          | `sounddevice`, `soundfile`, `scipy`                     |
| Audio Preprocessing    | `scipy.signal.resample_poly` (auto 48kHz → 16kHz)      |
| Speaker Identification | `resemblyzer` (VoiceEncoder, 256-dim embeddings)        |
| Speech-to-Text         | Google Cloud STT (`ti-ET`) + Whisper fallback (`am`)    |
| Confidence Scoring     | Google: native score; Whisper: `avg_logprob` + `no_speech_prob` |
| Text Embeddings        | `sentence-transformers` LaBSE (768-dim, multilingual)   |
| Semantic Matching      | Cosine similarity + page-aware sliding window           |
| Slide Parsing          | `python-pptx` / PDF.js  *(planned)*                     |
| GUI / Subtitles        | PyGame / Tkinter / React  *(planned)*                   |
| Translation            | Pre-aligned lookup → MarianMT / NLLB  *(planned)*       |

---

## Project Structure

```
src/
├── record_audio.py            # Capture mic input and save as .wav
├── build_embeddings_priest.py # Build Resemblyzer voice embeddings per speaker
├── build_text_embeddings.py   # Build LaBSE text embeddings for liturgy lines
├── live_synch.py              # Real-time speaker identification loop
├── matcher.py                 # One-shot voice match test
├── trasncribe.py              # Dual-backend STT: Google (ti-ET) + Whisper with confidence
├── window_matcher.py          # Page-aware sliding window semantic matcher
├── embed_audio.py             # Full pipeline: audio → resample → STT → LaBSE → match
├── speaker_id.py              # [TODO] Full speaker ID module
└── gui_display.py             # [TODO] GUI subtitle/slide display

data/
├── voices/                    # Reference + test .wav files per speaker
├── embeddings/
│   ├── priest.npy             # Resemblyzer voice embedding (256-dim)
│   ├── deacon.npy             # Resemblyzer voice embedding (256-dim)
│   ├── book_vectors.npy       # LaBSE text embeddings (38 × 768)
│   ├── book_meta.json         # Liturgy line metadata (id, role, section, text)
│   └── slide_vectors.npy      # Slide text embeddings (stub — needs real content)
├── geeze_liturgy.json         # 38-line Ge'ez liturgy dataset (full Mass structure)
└── slide_text.json            # Slide text source (stub — needs real slide content)
```

---

## Roadmap

### Phase 1 — Speaker Identification (Done)
> Know WHO is speaking before transcribing what they say.

- [x] Record reference audio for a speaker (`record_audio.py`)
- [x] Build voice embedding for priest (`build_embeddings_priest.py`)
- [x] Real-time speaker identification loop (`live_synch.py`)
- [x] One-shot speaker matching test (`matcher.py`)
- [x] Upload deacon voice and build deacon embedding
- [x] Resample deacon audio from 48kHz → 16kHz for pipeline consistency
- [x] Verify priest vs deacon voice similarity score (0.47 — well separated)
- [ ] Support N speakers (priest, deacon, lector, etc.)
- [ ] Set a confidence threshold — reject unknown speakers

---

### Phase 2 — Speech-to-Text Transcription (Done)
> Convert the identified speaker's voice into text.

- [x] Build dual-backend transcriber (`trasncribe.py`) — Google STT + Whisper side by side
- [x] Confirm Google STT (`ti-ET`) works for Tigrinya speech
- [x] Confirm Whisper `medium` does NOT support Tigrinya (hallucinates; use Amharic `am` as proxy)
- [x] Identified that deacon recording is Ge'ez liturgical — handled via semantic matching
- [x] Add real confidence scoring: Google returns 0–1 score; Whisper estimated from `avg_logprob` + `no_speech_prob`
- [x] `MIN_CONFIDENCE = 0.50` threshold — results below this are flagged `accepted: False`
- [x] Smart `transcribe()` function: tries Google first, falls back to Whisper automatically
- [x] Visual confidence bar output for CLI testing
- [ ] Stream audio chunks in real time
- [ ] Output transcription with timestamps per speaker

> **Language findings:**
> - Google STT `ti-ET` → works for Tigrinya conversational speech ✅
> - Whisper `medium` → no Tigrinya support; use `am` (Amharic) as closest script match ⚠️
> - Ge'ez liturgical → no cloud STT support; handled by semantic matching instead ✅
> - Google `ti-ET` confidence often returns `0.0` even when transcript is correct — do not gate on score alone

---

### Phase 3 — Semantic Matching (In Progress)
> Match speech audio to the correct line in the liturgy book — no perfect transcription needed.

- [x] Add 38-line Ge'ez liturgy dataset (`data/geeze_liturgy.json`) covering full Mass structure
- [x] Embed all liturgy lines with LaBSE multilingual model (`build_text_embeddings.py`)
- [x] Save vector database to `data/embeddings/book_vectors.npy` (38 × 768)
- [x] Verified partial phrase matching works (`ቅዱስ ቅዱስ ቅዱስ` → 0.881, `ሰላም ለኩሉ` → 1.000)
- [x] Build page-aware sliding window matcher (`window_matcher.py`)
  - Liturgy split into pages of 10 lines (4 pages for 38 lines)
  - Window locked to current page; left shrinks as we advance, right peeks into next page near boundary
  - `confirmed_pos` only moves forward — liturgy never goes backward
- [x] Build full audio → vector → match pipeline (`embed_audio.py`)
  - Auto-resamples any input (48kHz → 16kHz tested)
  - Runs Google STT → encodes with LaBSE → matches via sliding window
  - Tested on `Recording 6.wav`: matched `[id 2] ወቡሩክ ስሙ ለዓለመ ዓለም` (score 0.625) ✅
- [ ] Test across all 4 pages with real recordings
- [ ] Load real liturgy book from PDF/PPTX when available

---

### Phase 4 — GUI Display
> Show the audience what is being said, by whom, and where on the slide.

- [ ] Build subtitle overlay in `gui_display.py`
- [ ] Display current speaker label (e.g., "Priest", "Deacon")
- [ ] Highlight the active slide line in sync with speech
- [ ] Low-latency rendering (< 1s delay target)

---

### Phase 5 — Translation Layer
> Support bilingual or multilingual audiences.

- [ ] Pre-aligned Ge'ez → English lookup table for liturgical text
- [ ] Live neural machine translation (MarianMT / NLLB) for free-form speech
- [ ] Toggle translation on/off in the UI

---

### Phase 6 — Polish & Deployment
> Make it reliable and easy to run at a real Mass.

- [ ] Single-command startup script
- [ ] Config file for speaker names, mic device, language
- [ ] Handle mic silence / background noise gracefully
- [ ] Package as a standalone desktop app

---

## Current Status

| Module                    | File                          | Status                                          |
|---------------------------|-------------------------------|-------------------------------------------------|
| Audio recording           | `record_audio.py`             | Done                                            |
| Voice embedding           | `build_embeddings_priest.py`  | Done (priest + deacon, 256-dim Resemblyzer)     |
| Live speaker ID           | `live_synch.py`               | Done (priest + deacon real-time loop)           |
| Speaker match test        | `matcher.py`                  | Done (basic one-shot test)                      |
| Dual STT + confidence     | `trasncribe.py`               | Done (Google + Whisper, MIN_CONFIDENCE=0.50)    |
| Liturgy text dataset      | `data/geeze_liturgy.json`     | Done (38 lines, full Mass, 4 sections/pages)    |
| Text embedding build      | `build_text_embeddings.py`    | Done (LaBSE, 38 × 768 vectors saved)           |
| Sliding window matcher    | `window_matcher.py`           | Done (page-aware, forward-only, auto-expand)    |
| Audio → vector pipeline   | `embed_audio.py`              | Done (resample → STT → LaBSE → window match)   |
| Live integrated pipeline  | `main.py`                     | Not started (next)                              |
| Speaker ID module         | `speaker_id.py`               | Not started                                     |
| GUI / subtitles           | `gui_display.py`              | Not started                                     |
| Slide text content        | `data/slide_text.json`        | Stub only — needs real slide content            |

---

## Getting Started

```bash
# 1. Activate virtual environment
source virtual/bin/activate

# 2. Build voice embeddings (priest + deacon)
python src/build_embeddings_priest.py

# 3. Build liturgy text embeddings (LaBSE, 38 lines)
python src/build_text_embeddings.py

# 4. Run live speaker identification
python src/live_synch.py

# 5. Test STT + confidence on a recorded file
python src/trasncribe.py smart                        # smart mode (Google → Whisper fallback)
python src/trasncribe.py both                         # side-by-side comparison

# 6. Test the full audio → vector → liturgy match pipeline
python src/embed_audio.py "data/voices/Recording 6.wav"

# 7. Test the sliding window matcher alone
python src/window_matcher.py
```
