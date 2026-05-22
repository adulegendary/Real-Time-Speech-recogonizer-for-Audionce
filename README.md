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
 │   Speech-to-Text (STT) │   ← trasncribe.py  [NOT YET BUILT]
 │   Whisper              │
 │   Output: Ge'ez /      │
 │   Tigrinya + timestamps│
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │ Semantic Matching Layer│   ← matcher.py  [PARTIAL]
 │ Sentence-BERT / TF-IDF │
 │ Matches speech → slide │
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

| Component              | Tool / Framework                          |
|------------------------|-------------------------------------------|
| Audio Capture          | `sounddevice`, `soundfile`, `scipy`       |
| Speaker Identification | `resemblyzer` (VoiceEncoder + embeddings) |
| Speech-to-Text         | `whisper` (OpenAI)                        |
| Text Matching          | Sentence-BERT, TF-IDF cosine similarity   |
| Slide Parsing          | `python-pptx` / PDF.js                    |
| GUI / Subtitles        | PyGame / Tkinter / React                  |
| Translation            | Pre-aligned lookup → MarianMT / NLLB      |

---

## Project Structure

```
src/
├── record_audio.py           # Capture mic input and save as .wav
├── build_embeddings_priest.py # Build voice embedding for known speakers
├── live_synch.py             # Real-time speaker identification loop
├── matcher.py                # One-shot voice match test
├── trasncribe.py             # [TODO] Whisper speech-to-text
├── speaker_id.py             # [TODO] Full speaker ID module
└── gui_display.py            # [TODO] GUI subtitle/slide display

data/
├── voices/                   # Reference .wav files per speaker
├── embeddings/               # Saved .npy voice embeddings
└── sample_voices/            # Recorded test samples
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

### Phase 2 — Speech-to-Text Transcription (In Progress)
> Convert the identified speaker's voice into text.

- [x] Build dual-backend transcriber (`trasncribe.py`) — Google STT + Whisper side by side
- [x] Confirm Google STT (`ti-ET`) works for conversational Tigrinya speech
- [x] Confirm Whisper `medium` does NOT support Tigrinya (no `ti` in language list)
- [x] Identified that deacon recording is Ge'ez liturgical — outside current STT support
- [ ] Wire Google STT into live speaker identification loop
- [ ] Stream audio chunks in real time
- [ ] Output transcription with timestamps per speaker

> **Language findings:**
> - Google STT `ti-ET` → works for Tigrinya conversational speech ✅
> - Whisper `medium` → no Tigrinya support, hallucinates other languages ❌
> - Ge'ez liturgical → no cloud STT support yet; tackled via semantic matching instead

---

### Phase 3 — Semantic Matching (In Progress)
> Match speech audio to the correct line in the liturgy book — no perfect transcription needed.

- [x] Add 38-line Ge'ez liturgy dataset (`data/geeze_liturgy.json`) covering full Mass structure
- [x] Embed all liturgy lines with LaBSE multilingual model (`build_text_embeddings.py`)
- [x] Save vector database to `data/embeddings/book_vectors.npy` (38 × 768)
- [x] Verified partial phrase matching works (e.g. `ቅዱስ ቅዱስ ቅዱስ` → score 0.881, `ሰላም ለኩሉ` → score 1.000)
- [ ] Wire live audio → Google STT → LaBSE → find closest liturgy line
- [ ] Track position — move forward, never backward
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

| Module               | File                          | Status                        |
|----------------------|-------------------------------|-------------------------------|
| Audio recording      | `record_audio.py`             | Done                          |
| Voice embedding      | `build_embeddings_priest.py`  | Done (priest + deacon)        |
| Live speaker ID      | `live_synch.py`               | Done (priest + deacon)        |
| Speaker match test   | `matcher.py`                  | Done (basic)                  |
| Dual STT transcriber | `trasncribe.py`               | Done (Google STT + Whisper)   |
| Liturgy text dataset | `data/geeze_liturgy.json`     | Done (38 lines, full Mass)    |
| Text embedding build | `build_text_embeddings.py`    | Done (LaBSE, 38 × 768 vectors)|
| Live semantic match  | `matcher.py`                  | In Progress                   |
| Speaker ID module    | `speaker_id.py`               | Not started                   |
| GUI / subtitles      | `gui_display.py`              | Not started                   |

---

## Getting Started

```bash
# 1. Install dependencies
pip install sounddevice soundfile resemblyzer scipy numpy

# 2. Record reference voice for a speaker
python src/record_audio.py

# 3. Build the voice embedding
python src/build_embeddings_priest.py

# 4. Run live speaker identification
python src/live_synch.py
```
