# GUI Interface — Implementation Plan

**Status:** Planning — no code changes made yet.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Framework Choice](#2-framework-choice)
3. [UI Layout and UX Design](#3-ui-layout-and-ux-design)
4. [Architecture](#4-architecture)
5. [Phase 1 — Skeleton and Recording Tab](#5-phase-1--skeleton-and-recording-tab)
6. [Phase 2 — Transcription Tab](#6-phase-2--transcription-tab)
7. [Phase 3 — Analysis Tab](#7-phase-3--analysis-tab)
8. [Phase 4 — Pipeline Connectivity](#8-phase-4--pipeline-connectivity)
9. [Phase 5 — Settings and Polish](#9-phase-5--settings-and-polish)
10. [Cross-Cutting Technical Notes](#10-cross-cutting-technical-notes)
11. [Summary of New Files and Changes](#11-summary-of-new-files-and-changes)

---

## 1. Overview

The GUI is a desktop application that wraps the existing three-stage pipeline in a visual interface accessible to non-technical users. Each stage of the pipeline maps to a dedicated tab:

| Tab | Maps to | Core actions |
|---|---|---|
| **Recording** | `record_meeting.sh` | Start / stop recording, name the output WAV |
| **Transcription** | `lib/transcription.py` | Pick a WAV, transcribe, view and save the transcript |
| **Analysis** | `lib/analysis.py` + `lib/report.py` | Generate the Markdown report, preview and save it |

The CLI entry point (`transcribe.py`) is left completely intact. The GUI is an additive layer — users can continue using the CLI if they prefer.

---

## 2. Framework Choice

**Recommended: PySide6** (the official Qt 6 Python bindings, LGPL-licensed)

| Criterion | Why PySide6 wins |
|---|---|
| Cross-platform | First-class support on Linux and macOS, which matches the current roadmap |
| Background threads | `QThread` + signals/slots is the standard pattern for non-blocking long operations; transcription and analysis both take minutes on CPU |
| macOS integration | Renders native macOS window chrome, title bar, and menus automatically |
| Pop!_OS / COSMIC | Qt 6 integrates cleanly with GTK-based and Wayland compositors; no known issues on COSMIC |
| Maturity | Qt has been the standard cross-platform GUI toolkit for 30 years; PySide6 is actively maintained by The Qt Company |

**Alternative considered: Tkinter** — ships with Python so no extra dependency, but the widget set is dated and background thread integration is awkward (`after()` polling rather than signals).

**Alternative considered: Gradio / Streamlit** — would produce a browser-based UI, which avoids the native dependency entirely. Rejected because the user expects a desktop application feel, and browser UIs add latency and a web server dependency.

**New dependency to add to `requirements.txt`:**
```
PySide6>=6.7
```

---

## 3. UI Layout and UX Design

### Main window

A single window with a `QTabWidget` containing three tabs. Tabs are the natural fit because the three stages are sequential but independent — a user may open the app just to run analysis on an existing transcript without ever using the recording tab.

```
┌────────────────────────────────────────────────────────┐
│  🎙️ Transcriber                            [─][□][×]  │
├──────────────────────────────────────────────────────── │
│  [ 🔴 Recording ]  [ 📝 Transcription ]  [ 📋 Analysis ]│
├────────────────────────────────────────────────────────┤
│                                                        │
│                  (active tab content)                  │
│                                                        │
├────────────────────────────────────────────────────────┤
│  Status bar: Ready                                     │
└────────────────────────────────────────────────────────┘
```

### Recording tab

```
┌────────────────────────────────────────────────────────┐
│  Audio Devices                                         │
│  🎙️  Microphone:    Built-in Audio Analog Stereo       │
│  🔊  System audio:  ...monitor                         │
│                                                        │
│              ╔══════════════════╗                      │
│              ║  ● Start         ║                      │
│              ║  Recording       ║  ← large button      │
│              ╚══════════════════╝                      │
│                                                        │
│  (while recording)                                     │
│  ● Recording...  00:04:32   ████████░░  (live meter)   │
│                                                        │
│  (after stop)                                          │
│  Save as: [ meeting_20260527_114300.wav  ] [💾 Save]   │
└────────────────────────────────────────────────────────┘
```

### Transcription tab

```
┌────────────────────────────────────────────────────────┐
│  Audio file: [ path/to/meeting.wav          ] [Browse] │
│                                                        │
│              ╔══════════════════╗                      │
│              ║  ▶ Transcribe    ║                      │
│              ╚══════════════════╝                      │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ [00:00:04.50 -> 00:00:12.30]  Hey everyone...    │  │
│  │ [00:00:12.80 -> 00:00:21.10]  We need to make... │  │
│  │ [00:00:21.50 -> 00:00:29.00]  Agreed, let's...   │  │
│  │                              ← lines appear live  │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Save as: [ meeting_20260527_transcript.txt ] [💾 Save]│
│                              [→ Send to Analysis]      │
└────────────────────────────────────────────────────────┘
```

### Analysis tab

```
┌────────────────────────────────────────────────────────┐
│  Transcript: [ meeting_transcript.txt       ] [Browse] │
│                                                        │
│              ╔══════════════════════╗                  │
│              ║  ⚙️  Generate Report  ║                  │
│              ╚══════════════════════╝                  │
│                                                        │
│  Sections:                                             │
│  ✅ Executive Summary    ✅ Detailed Summary            │
│  ⏳ Action Items         ○  Key Decisions               │
│  ○  Topics Discussed                                   │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ # Meeting Report                                 │  │
│  │                                                  │  │
│  │ ## Executive Summary                             │  │
│  │ The team agreed to...                            │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Save as: [ meeting_20260527_report.md      ] [💾 Save]│
└────────────────────────────────────────────────────────┘
```

---

## 4. Architecture

### File structure

```
transcriber/
├── gui.py                       ← GUI entry point  (new)
├── transcribe.py                ← CLI entry point  (unchanged)
├── lib/
│   ├── gui/                     ← GUI package      (new)
│   │   ├── __init__.py
│   │   ├── app.py               ← QApplication setup and global config
│   │   ├── window.py            ← Main window, tab container, status bar
│   │   ├── workers.py           ← QThread subclasses for background tasks
│   │   ├── widgets.py           ← Reusable custom widgets
│   │   └── tabs/
│   │       ├── __init__.py
│   │       ├── recording.py     ← Recording tab
│   │       ├── transcription.py ← Transcription tab
│   │       └── analysis.py      ← Analysis tab
│   ├── transcription.py         ← (unchanged)
│   ├── analysis.py              ← (unchanged)
│   ├── report.py                ← (unchanged)
│   └── progress.py              ← (unchanged)
```

### Background worker pattern

All three pipeline stages are long-running operations that must not block the UI thread. Each maps to a `QThread` subclass in `lib/gui/workers.py`:

```
RecordingWorker(QThread)
  signals:
    elapsed_tick(seconds: int)     ← emitted every second during recording
    level_tick(level: float)       ← audio level for the meter (0.0–1.0)
    finished(wav_path: str)        ← emitted when FFmpeg exits cleanly
    error(message: str)

TranscriptionWorker(QThread)
  signals:
    line_ready(line: str)          ← emitted per Whisper segment (live streaming)
    language_detected(lang: str, prob: float)
    finished(lines: list[str])
    error(message: str)

AnalysisWorker(QThread)
  signals:
    section_ready(heading: str, text: str)  ← one per completed section
    finished(sections: list[tuple[str,str]])
    error(message: str)
```

The tab widgets connect to worker signals on the main thread — all UI updates happen via signal/slot, which Qt guarantees is thread-safe.

### Relationship to `lib/progress.py`

`lib/progress.py`'s `ProgressTimer` writes to stdout using `\r` cursor tricks. When the library modules are called from a `QThread`, this output still goes to the terminal (if the app was launched from one) but does not affect the GUI. This is acceptable for early phases. A later refactor could make the progress mechanism pluggable — the library emits progress events, and either `ProgressTimer` (CLI) or a worker signal (GUI) handles them.

---

## 5. Phase 1 — Skeleton and Recording Tab

**Goal:** A working window with all three tabs visible. Only the Recording tab is functional. The other tabs show a "coming soon" placeholder.

### Tasks

**5.1 — Project wiring**

- Add `PySide6>=6.7` to `requirements.txt`
- Create `gui.py`:
  ```python
  from lib.gui.app import run_app
  if __name__ == "__main__":
      run_app()
  ```
- Create `lib/gui/__init__.py` (empty)
- Create `lib/gui/tabs/__init__.py` (empty)

**5.2 — `lib/gui/app.py`**

Sets up `QApplication`, applies the system font, detects dark/light mode from the platform palette, and calls `window.show()`. Entry point is `run_app()`.

**5.3 — `lib/gui/window.py`**

`MainWindow(QMainWindow)`:
- Title: `Transcriber`
- Central widget: `QTabWidget` with three tabs
- Tab 0: `RecordingTab`
- Tab 1: `TranscriptionTab` (disabled placeholder in Phase 1)
- Tab 2: `AnalysisTab` (disabled placeholder in Phase 1)
- Bottom status bar: `QStatusBar` — message persists for 5 seconds then resets to "Ready"
- Minimum size: 700 × 520 px

**5.4 — `lib/gui/workers.py` — `RecordingWorker`**

Replaces the shell script FFmpeg call with a Python-managed subprocess:

```python
class RecordingWorker(QThread):
    elapsed_tick = Signal(int)
    finished     = Signal(str)   # wav_path
    error        = Signal(str)

    def __init__(self, output_path: str, desktop_sink: str, mic_source: str):
        ...

    def run(self) -> None:
        # Launch FFmpeg subprocess with the same amix pipeline as record_meeting.sh
        # Emit elapsed_tick every second via a QTimer running on this thread
        # On self.stop() being called: send SIGTERM to FFmpeg, wait for clean exit
        # Emit finished(output_path) when FFmpeg exits with code 0

    def stop(self) -> None:
        # Signal FFmpeg to stop cleanly
```

The device names (`desktop_sink`, `mic_source`) are detected using the same `pactl info` logic from the shell script, now called via `subprocess.run` in Python.

**5.5 — `lib/gui/tabs/recording.py`**

`RecordingTab(QWidget)`:

- **Device info panel** (top): Two read-only labels showing detected microphone and system audio sink. If detection fails, shows an error message with a "Retry" button. Populated on tab construction.
- **Main action button** (centre): `QPushButton`, large (minimum 160 × 60 px). Text alternates between `● Start Recording` (green) and `■ Stop Recording` (red) based on state.
- **Status area** (shown only while recording): elapsed time label (`00:00:00`) updated via `elapsed_tick` signal, and a simple `QProgressBar` set to `setRange(0, 0)` (indeterminate/pulsing) as a visual recording indicator.
- **Save panel** (shown only after recording stops): `QLineEdit` pre-populated with the auto-timestamped filename, and a `💾 Save` `QPushButton`. Clicking Save opens a `QFileDialog.getSaveFileName()` pre-seeded with the suggested name, writes the WAV to the chosen path, and emits a tab-level signal `recording_saved(wav_path: str)` for use in Phase 4.

**State machine:**

```
IDLE → (Start pressed)     → DETECTING_DEVICES
DETECTING_DEVICES → (ok)   → RECORDING
DETECTING_DEVICES → (fail) → IDLE (show error)
RECORDING → (Stop pressed) → STOPPING
STOPPING → (FFmpeg exited) → SAVE_PENDING
SAVE_PENDING → (Save)      → IDLE
```

---

## 6. Phase 2 — Transcription Tab

**Goal:** A user can pick any WAV file (or the one just recorded), transcribe it, watch the transcript lines appear live as Whisper processes the audio, and save the result.

### Tasks

**6.1 — `lib/gui/widgets.py` — `FilePickerWidget`**

A reusable compound widget: `QLineEdit` (path display) + `Browse…` `QPushButton`. Emits `path_changed(path: str)` when a valid file is selected. Used in both the Transcription and Analysis tabs.

**6.2 — `lib/gui/workers.py` — `TranscriptionWorker`**

Calls `transcription.detect_device()` and `transcription.transcribe_audio()`. The challenge is that `transcribe_audio()` currently collects all segments before returning. For live streaming, the worker needs to iterate the Whisper segment generator directly:

```python
def run(self) -> None:
    device, compute_type = transcription.detect_device()
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device, compute_type=compute_type)
    segments, info = model.transcribe(self.audio_path, beam_size=5)
    self.language_detected.emit(info.language, info.language_probability)
    lines = []
    for seg in segments:
        line = f"[{fmt_ts(seg.start)} -> {fmt_ts(seg.end)}]  {seg.text.strip()}"
        lines.append(line)
        self.line_ready.emit(line)   # ← live streaming to UI
    self.finished.emit(lines)
```

This means the worker re-implements the inner loop of `transcribe_audio()` rather than calling it, to gain per-segment signal emission. A future refactor of `transcription.py` could add an optional callback parameter to avoid this duplication.

**6.3 — `lib/gui/tabs/transcription.py`**

`TranscriptionTab(QWidget)`:

- **File picker** (top): `FilePickerWidget` labelled "Audio file". Initially empty; populated automatically when `recording_saved` signal arrives from Recording tab (Phase 4), or selected manually.
- **Language badge**: small read-only label, hidden until transcription starts, then shows e.g. `Detected: English (97%)`.
- **Action button**: `▶ Transcribe`, disabled until a valid WAV path is set.
- **Transcript viewer**: `QPlainTextEdit` (read-only, monospace font — `QFontDatabase.systemFont(QFontDatabase.FixedFont)`). Each `line_ready` signal appends a line and auto-scrolls to the bottom. An `appendLine(line: str)` method ensures the scroll follows output.
- **Progress indicator**: `QProgressBar` in indeterminate mode while transcription runs. Hidden when idle or complete.
- **Save panel**: `FilePickerWidget`-style row with a `QLineEdit` pre-filled with the suggested `_transcript.txt` name + `💾 Save` button. Enabled only after `finished` is received. Clicking Save writes the file; emits `transcript_saved(path: str)`.
- **"→ Send to Analysis" button**: Enabled after save, carries the transcript path to the Analysis tab (Phase 4).

---

## 7. Phase 3 — Analysis Tab

**Goal:** A user can load a transcript (from the Transcription tab or by picking a `.txt` file), generate the full Markdown report, watch each section appear as it completes, and save the result.

### Tasks

**7.1 — `lib/gui/workers.py` — `AnalysisWorker`**

Calls `analysis.generate_summaries()`. Because the current implementation generates all sections in a single model pass and then parses them, the worker receives the full `list[tuple[str, str]]` from `finished` and emits each `section_ready` sequentially:

```python
def run(self) -> None:
    meta = report.parse_recording_meta(self.audio_path or "unknown.wav")
    meta["duration"] = report.estimate_duration(self.transcript_lines)
    transcript_body = report.build_transcript_body(self.transcript_lines)
    sections = analysis.generate_summaries(transcript_body, meta)
    for heading, text in sections:
        self.section_ready.emit(heading, text)
    self.finished.emit(sections)
```

The UI therefore shows sections appearing one by one as the signal loop runs, which gives the user visual feedback that something is happening even though the model generates all sections in one pass.

**7.2 — Section progress indicators**

Five `QLabel` pills arranged horizontally, one per section. Each starts with a `○` prefix (pending) and updates to `⏳` (in progress — shown for the current section) or `✅` (done) as `section_ready` signals arrive. This is purely cosmetic but gives the user a clear sense of how much is left.

**7.3 — Markdown preview**

`QTextBrowser` (read-only, supports basic HTML rendering). When each section arrives, the raw Markdown is converted to HTML using the `markdown` Python package (new dependency) and the HTML content of the `QTextBrowser` is updated. This gives a rendered preview with headers and bold text rather than raw `##` markers.

If the `markdown` package is not desired as a dependency, a fallback `QPlainTextEdit` showing raw Markdown is acceptable and simpler to implement.

**7.4 — `lib/gui/tabs/analysis.py`**

`AnalysisTab(QWidget)`:

- **Transcript source** (top): `FilePickerWidget` labelled "Transcript file". Pre-populated when `transcript_saved` arrives from the Transcription tab (Phase 4), or picked manually.
- **Action button**: `⚙️ Generate Report`, disabled until a transcript path is set.
- **Section progress bar**: Five pill labels (see 7.2), hidden until generation starts.
- **Markdown preview panel**: `QTextBrowser`, hidden until the first section arrives. Scrollable.
- **Copy to clipboard button**: `📋 Copy`, enabled after `finished`.
- **Save panel**: `FilePickerWidget`-style row with `QLineEdit` pre-filled with the suggested `_report.md` name + `💾 Save` button. Enabled after `finished`.

---

## 8. Phase 4 — Pipeline Connectivity

**Goal:** After completing one stage, the app naturally guides the user to the next without requiring them to manually locate the file they just created.

### Tasks

**8.1 — Cross-tab signalling**

`MainWindow` acts as the message bus between tabs. Tab-level signals are connected in `window.py`:

```python
self.recording_tab.recording_saved.connect(self.transcription_tab.set_audio_path)
self.transcription_tab.transcript_saved.connect(self.analysis_tab.set_transcript_path)
```

When `recording_saved(wav_path)` fires, the Transcription tab's file picker is populated and the tab becomes enabled. When `transcript_saved(txt_path)` fires, the Analysis tab's transcript picker is populated and the Analysis tab becomes enabled.

**8.2 — "Continue to next step" prompts**

After a save event, a non-blocking inline banner appears at the bottom of the current tab:

```
✅ Saved to meeting_20260527_transcript.txt
   [→ Go to Analysis tab]
```

This is a `QFrame` with a label and a button. Clicking the button calls `self.parent().setCurrentIndex(next_tab_index)`. The banner auto-dismisses after 10 seconds.

**8.3 — Tab enable/disable logic**

Tabs 1 and 2 start disabled (greyed out). Tab 1 (Transcription) is enabled when either:
- The user picks a WAV file manually in the Transcription tab, OR
- `recording_saved` fires

Tab 2 (Analysis) is enabled when either:
- The user picks a `.txt` file manually in the Analysis tab, OR
- `transcript_saved` fires

**8.4 — Status bar integration**

`MainWindow` connects all worker signals to the status bar:
- `RecordingWorker.elapsed_tick` → `"Recording... 00:04:32"`
- `TranscriptionWorker.line_ready` → `"Transcribing... (segment N)"`
- `AnalysisWorker.section_ready` → `"Generating: Action Items..."`
- Any `error` signal → `"❌ Error: <message>"` (persists until next action)

---

## 9. Phase 5 — Settings and Polish

**Goal:** The app feels complete and production-ready. Configuration is persisted. Keyboard users are fully supported. Error messages are actionable.

### Tasks

**9.1 — Settings dialog**

`QDialog` accessible from a `⚙️ Settings` button in `MainWindow` or from a menu bar. Settings are persisted via `QSettings` (writes to the platform-appropriate location: `~/.config/transcriber/` on Linux, `~/Library/Preferences/` on macOS).

Settings to expose:

| Setting | Widget | Default |
|---|---|---|
| Whisper model size | `QComboBox` (tiny / base / small / medium / large-v3) | base |
| Output directory | `FilePickerWidget` (directory mode) | Same folder as source file |
| Auto-continue pipeline | `QCheckBox` | Off |
| Markdown preview rendering | `QComboBox` (Rendered HTML / Raw Markdown) | Rendered HTML |

**9.2 — Keyboard shortcuts**

| Shortcut | Action |
|---|---|
| `Ctrl+R` (`Cmd+R` on macOS) | Start / stop recording |
| `Ctrl+T` (`Cmd+T` on macOS) | Start transcription |
| `Ctrl+G` (`Cmd+G` on macOS) | Generate report |
| `Ctrl+S` (`Cmd+S` on macOS) | Save current tab's output |
| `Ctrl+,` (`Cmd+,` on macOS) | Open settings |
| `Ctrl+1/2/3` | Switch to Recording / Transcription / Analysis tab |

**9.3 — Recent files**

A `QMenu` on a `📂 Recent` button in the Transcription tab, listing up to 10 recently used WAV files stored in `QSettings`. Clicking an entry populates the file picker.

**9.4 — Error handling improvements**

Currently errors appear in the status bar. Phase 5 adds `QMessageBox` dialogs for fatal errors (model load failure, FFmpeg not found, disk write error) with:
- Clear description of what went wrong
- Suggested action (e.g. "Install FFmpeg: `sudo apt install ffmpeg`")
- No stack traces shown to the user

**9.5 — Platform-specific polish**

- **macOS**: Add a `QMenuBar` with standard File / Edit / Help menus (expected on macOS; optional on Linux)
- **macOS**: Dock icon badge showing current operation state
- **Linux**: App icon set via `QApplication.setWindowIcon()` using an `.svg` bundled in `lib/gui/assets/`
- **Both**: Respect the system dark/light mode preference by detecting `QPalette` at startup and applying appropriate stylesheet tweaks

---

## 10. Cross-Cutting Technical Notes

### `lib/progress.py` in a GUI context

`ProgressTimer` writes to stdout using `\r` cursor tricks. When `TranscriptionWorker` calls `transcription.transcribe_audio()`, the `ProgressTimer` context managers inside that function will write to the terminal (if the GUI was launched from one). This does not break the GUI but produces noisy terminal output.

A clean fix (Phase 5 candidate): make `ProgressTimer` optional by accepting a progress callback:

```python
# lib/transcription.py — future refactor
def transcribe_audio(audio_path, device, compute_type, on_progress=None):
    ...
```

Until then, the GUI workers can suppress the noise by redirecting stdout to `os.devnull` for the duration of the library call, or simply accept the terminal output as a debug log.

### Thread safety of library modules

`lib/transcription.py`, `lib/analysis.py`, and `lib/report.py` contain no shared mutable state. Each call to `transcribe_audio()` or `generate_summaries()` loads models and operates entirely on local variables. Running them inside `QThread.run()` is safe. Do not call them from multiple workers simultaneously — the models would compete for VRAM/RAM.

### Model loading time

`WhisperModel` and `LFM2` each take several seconds to load even from disk cache. The `QThread` approach means the UI remains interactive during loading. A spinner or progress indicator should be shown as soon as a button is pressed, not only when the actual processing begins.

### `AnalysisGroundingError`

`lib/analysis.py` raises `AnalysisGroundingError` if the generated report appears ungrounded. `AnalysisWorker.run()` must catch this and emit it via the `error` signal rather than letting it crash the thread:

```python
try:
    sections = analysis.generate_summaries(transcript_body, meta)
except analysis.AnalysisGroundingError as exc:
    self.error.emit(str(exc))
    return
```

### macOS audio recording (Phase 1 of Apple Silicon plan)

When the Apple Silicon support plan (Phase 1) is implemented, `record_meeting.sh` will gain a macOS path. `RecordingWorker` in `workers.py` should mirror that — it will need the same OS detection and AVFoundation device enumeration logic. The worker should be written with an abstract `_build_ffmpeg_args()` method so the Linux and macOS paths are cleanly separated within the same class.

---

## 11. Summary of New Files and Changes

### New files

| File | Phase | Description |
|---|---|---|
| `gui.py` | 1 | GUI entry point |
| `lib/gui/__init__.py` | 1 | Package marker |
| `lib/gui/app.py` | 1 | QApplication setup |
| `lib/gui/window.py` | 1 | Main window and tab container |
| `lib/gui/workers.py` | 1–3 | QThread workers for all three pipeline stages |
| `lib/gui/widgets.py` | 2 | Reusable custom widgets (FilePickerWidget, etc.) |
| `lib/gui/tabs/__init__.py` | 1 | Package marker |
| `lib/gui/tabs/recording.py` | 1 | Recording tab |
| `lib/gui/tabs/transcription.py` | 2 | Transcription tab |
| `lib/gui/tabs/analysis.py` | 3 | Analysis tab |

### Modified files

| File | Phase | Change |
|---|---|---|
| `requirements.txt` | 1 | Add `PySide6>=6.7` |
| `requirements.txt` | 3 | Add `markdown` (for rendered Markdown preview) |
| `README.md` | 1 | Add "Running the GUI" section alongside existing CLI instructions |
| `README.md` | 5 | Add settings and keyboard shortcuts reference |

### Unchanged files

| File | Reason |
|---|---|
| `transcribe.py` | CLI entry point — untouched |
| `lib/transcription.py` | Called from workers; no changes needed until Phase 5 progress refactor |
| `lib/analysis.py` | Called from workers; no changes needed |
| `lib/report.py` | Called from workers; no changes needed |
| `lib/progress.py` | Still used by CLI; GUI workers tolerate its stdout output |
| `record_meeting.sh` | Still used as the CLI recording tool; replaced internally by `RecordingWorker` for GUI |
