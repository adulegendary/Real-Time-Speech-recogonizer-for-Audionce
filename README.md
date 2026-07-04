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
 │ Semantic Matching Layer│   ← window_matcher.py
 │ TWO interchangeable     │
 │ encoders:               │
 │  • LaBSE text (768-d)   │   ← prototype_match.py (STT → text → vector)
 │  • wav2vec2 audio       │   ← realtime_match.py  (audio → vector, no STT)
 │ Page-aware sliding      │
 │ window (10 lines/page)  │
 └───────────┬────────────┘
             │  payload JSON {id, score, confidence, role, text, window, …}
             ▼
 ┌────────────────────────┐
 │   WebSocket Server      │   ← live_ws_server.py  (real pipeline)
 │   ws://localhost:8000    │   ← fake_ws_server.py  (replay, no models)
 └───────────┬────────────┘
             │  ws.send(payload)
             ▼
 ┌────────────────────────┐
 │  React Frontend (Vite) │   ← frontend/  (App.jsx + useLiturgySocket.js)
 │  - Speaker label/role  │
 │  - Live transcript     │
 │  - Match + STT bars    │
 │  - Auto-scroll highlight│
 └───────────┬────────────┘
             │
             ▼
 ┌────────────────────────┐
 │   Translation Layer    │   [NOT YET BUILT]
 │  Ge'ez → English       │
 │  Pre-aligned lookup or │
 │  MarianMT / NLLB       │
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
| Audio Embeddings       | `transformers` wav2vec2-large-xlsr-53 (1024-dim, no STT)|
| Semantic Matching      | Cosine similarity + page-aware sliding window           |
| Real-Time Transport    | `websockets` (native WS) — `ws://localhost:8000/ws`     |
| Frontend               | React 19 + Vite (live liturgy display, auto-scroll)     |
| Slide Parsing          | `python-pptx` / PDF.js  *(planned)*                     |
| Translation            | Pre-aligned lookup → MarianMT / NLLB  *(planned)*       |

---

## Project Structure

```
src/
├── record_audio.py               # Capture mic input and save as .wav
├── build_embeddings_priest.py    # Build Resemblyzer voice embeddings per speaker
├── build_text_embeddings.py      # Build LaBSE text embeddings for liturgy lines
├── build_audio_line_embeddings.py# Build wav2vec2 audio embeddings per liturgy line
├── live_synch.py                 # Real-time speaker identification loop
├── matcher.py                    # One-shot voice match test
├── trasncribe.py                 # Dual-backend STT: Google (ti-ET) + Whisper with confidence
├── window_matcher.py             # Page-aware sliding window semantic matcher
├── embed_audio.py                # Full pipeline: audio → resample → STT → LaBSE → match
├── prototype_match.py            # Chunked STT + LaBSE matcher; prints the WebSocket payload
├── realtime_match.py             # Chunked audio-only matcher (wav2vec2, no STT)
├── live_ws_server.py             # WebSocket server — REAL pipeline (STT + LaBSE → ws)
├── fake_ws_server.py             # WebSocket server — replays payloads (no models, UI dev)
├── speaker_id.py                 # [TODO] Full speaker ID module
└── gui_display.py                # [legacy stub — replaced by React frontend/]

frontend/                         # React 19 + Vite live display
├── src/
│   ├── App.jsx                   # Liturgy list, active-line highlight, confidence bars
│   ├── useLiturgySocket.js       # WebSocket hook (auto-reconnect) → latest match
│   └── liturgy.json              # Liturgy lines rendered in the UI
├── index.html
└── package.json

data/
├── voices/                       # Reference + test .wav files per speaker
├── audio_samples/                # Additional audio clips
├── embeddings/
│   ├── priest.npy                # Resemblyzer voice embedding (256-dim)
│   ├── deacon.npy                # Resemblyzer voice embedding (256-dim)
│   ├── book_vectors.npy          # LaBSE text embeddings (38 × 768)
│   ├── book_meta.json            # Liturgy line metadata (id, role, section, text)
│   ├── audio_line_vectors.npy    # wav2vec2 audio embeddings (covered lines × 1024)
│   ├── audio_line_meta.json      # Metadata for lines that have an audio embedding
│   └── slide_vectors.npy         # Slide text embeddings (stub — needs real content)
├── geeze_liturgy.json            # 38-line Ge'ez liturgy dataset (full Mass structure)
└── slide_text.json               # Slide text source (stub — needs real slide content)
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
- [x] Add chunked, per-second prototype matcher (`prototype_match.py`)
  - Slices the wav into N-second chunks → STT → LaBSE → window match per chunk
  - Prints the exact `payload` JSON the WebSocket later pushes to React
- [x] Add **audio-only** matching path (`realtime_match.py` + `build_audio_line_embeddings.py`)
  - wav2vec2-large-xlsr-53 encodes audio → 1024-dim vector, **no STT needed**
  - Same sliding-window matcher, different encoder — a fallback when STT fails
- [ ] Expand `LINE_AUDIO_MAP` — record reference audio for all 38 lines (only id 2 so far)
- [ ] Test across all 4 pages with real recordings
- [ ] Load real liturgy book from PDF/PPTX when available

---

### Phase 4 — Live Display (React + WebSocket) (In Progress)
> Show the audience what is being said, by whom, and where in the liturgy.

- [x] Native WebSocket transport (`websockets`) at `ws://localhost:8000/ws`
- [x] Fake server (`fake_ws_server.py`) replays payloads so the UI can be built without models
- [x] Real server (`live_ws_server.py`) streams the actual STT + LaBSE pipeline
  - Heavy models loaded once at startup; blocking inference offloaded via `asyncio.to_thread`
  - Fresh per-connection matcher (forward-only state isn't shared between clients)
  - Emits the **same** payload shape as the fake server — React can't tell them apart
- [x] React 19 + Vite frontend (`frontend/`)
  - `useLiturgySocket.js` — WebSocket hook with auto-reconnect and live/offline status
  - Full liturgy list with the active line highlighted and auto-scrolled into view
  - Live transcript + match-score and STT-confidence bars
- [ ] Confirm the real pipeline end-to-end in the browser (built, not yet verified live)
- [ ] Display current speaker label (e.g., "Priest", "Deacon") from speaker ID
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
| Chunked STT matcher       | `prototype_match.py`          | Done (prints WebSocket payload per chunk)       |
| Audio-only matcher        | `realtime_match.py`           | Done (wav2vec2, no STT — 1024-dim)              |
| Audio line embeddings     | `build_audio_line_embeddings.py` | Done (only line id 2 recorded so far)        |
| Fake WebSocket server     | `fake_ws_server.py`           | Done (payload replay for UI dev)                |
| Live WebSocket server     | `live_ws_server.py`           | Done — built, not yet verified live in browser  |
| React frontend            | `frontend/`                   | Done — live display, auto-scroll, confidence bars|
| Speaker ID module         | `speaker_id.py`               | Not started                                     |
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

# 8. Chunked matchers (print per-chunk line + confidence)
python src/prototype_match.py "data/voices/Recording 6.wav" --chunk 3   # STT + LaBSE
python src/build_audio_line_embeddings.py                                # build wav2vec2 refs first
python src/realtime_match.py "data/voices/Recording 6.wav" --chunk 3    # audio-only (wav2vec2)
```

### Run the live display (backend + frontend)

```bash
# Terminal 1 — WebSocket backend (ws://localhost:8000/ws)
python src/fake_ws_server.py                       # replay mode, no models (fast UI dev)
# or the real pipeline:
python src/live_ws_server.py "data/voices/Recording 6.wav" --chunk 3

# Terminal 2 — React frontend
cd frontend
npm install
npm run dev                                        # open the printed localhost URL
```
