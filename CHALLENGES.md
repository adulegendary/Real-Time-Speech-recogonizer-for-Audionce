# Project Challenges & Limitations

---

## 1. WSL Has No Audio Device Access
**Problem:** WSL (Windows Subsystem for Linux) cannot access the Windows microphone.  
Running `sd.query_devices()` returns empty — no input devices found.  
**Workaround:** Record audio on Windows (Voice Recorder app) and copy the `.wav` file into `data/voices/`, or set up PulseAudio for WSL2.

---

## 2. Wrong Audio Device Index
**Problem:** `sd.default.device = 2` was hardcoded — crashes with `PortAudioError: Error querying device 2` if that index doesn't exist on the machine.  
**Fix:** Changed to `sd.default.device = None` so sounddevice picks the system default.

---

## 3. Virtual Environment Not Activated
**Problem:** Running scripts under `(base)` conda env instead of the project's `virtual/` env causes `ModuleNotFoundError` for packages like `sounddevice`, `resemblyzer`, etc.  
**Fix:** Always run `source virtual/bin/activate` before running any project script.

---

## 4. Only One Speaker Embedded (Priest Only)
**Problem:** Initial setup only built an embedding for the priest. The deacon was not recognized.  
**Fix:** Added `"deacon"` to the `roles` list in `build_embeddings_priest.py` and `live_synch.py`. Still need to record the deacon's voice and build the embedding.

---

## 5. Transcription Not Yet Implemented
**Problem:** `trasncribe.py` and `speaker_id.py` are empty — no speech-to-text is running yet.  
**Status:** Pending. Next step is integrating OpenAI Whisper.

---

## 6. No Confidence Threshold for Unknown Speakers
**Problem:** `live_synch.py` always picks the best match even if the speaker is unknown (e.g., congregation noise, background sound). Low-confidence matches get labeled incorrectly.  
**Status:** Need to add a minimum similarity threshold (e.g., reject if score < 0.75).
