# GUI Integration Plan

**File:** `docs/gui-integration.md`

---

## Overview

The Transcriber project is currently a CLI‑only tool that records audio, runs Faster‑Whisper transcription, and generates a summary report. To make the solution accessible to a broader audience we will add a desktop graphical user interface (GUI). The GUI will be a thin client that calls the existing Python library functions, keeping all heavy processing (model inference, file I/O) on the same machine and preserving the privacy‑first design.

## Goals
1. **Expose core functionality as callable APIs** – refactor the CLI entry points into library‑level functions that can be invoked programmatically.
2. **Provide a lightweight desktop front‑end** – choose a cross‑platform UI framework (e.g., PySide6 / Qt, Tauri + Rust, or Electron) and build a simple window with “Record”, “Transcribe”, and “Summarize” actions.
3. **Maintain Docker support** – the GUI should be able to run either natively (via the virtual environment) or inside the existing Docker image for advanced deployments.
4. **Keep privacy intact** – no data leaves the user’s machine; the GUI launches subprocesses only on the local host.

## Architecture Changes

### 1. Refactor CLI → Library
| Existing CLI | New Library API | Description |
|--------------|----------------|-------------|
| `record_meeting.sh <file.wav>` | `record_meeting(output_path: Path) -> Path` | Wrapper that invokes the FFmpeg command and returns the created WAV file. |
| `python transcribe.py <audio.wav>` | `transcribe(audio_path: Path, *, model: str = "base", device: str = "cpu") -> Path` | Runs Faster‑Whisper, writes `<base>_transcript.txt`, and returns the transcript file path. |
| `transcribe` → `analysis` → `report` pipeline (inside `transcribe.py`) | `generate_report(transcript_path: Path, *, model: str = "gemma-4", max_chars: int = 20000) -> Path` | Calls `lib/analysis.py` and `lib/report.py` to produce `<base>_report.md`. |
| `docker run …` | `run_in_docker(image: str, cmd: List[str]) -> subprocess.CompletedProcess` | Helper that can be used by the GUI to launch Docker containers when needed. |

*Implementation notes*
- Move the argument parsing from `transcribe.py` into a `main()` that simply forwards to the new functions.
- Export the three functions (`record_meeting`, `transcribe`, `generate_report`) from a new module `transcriber/api.py`.
- Add type hints and docstrings for IDE auto‑completion.

### 2. Add a Thin “Service” Layer (optional)
If we later want to support a background daemon or web‑socket API, introduce `transcriber/service.py` that exposes the same functions via `FastAPI` (running on `localhost` only). The GUI can then talk HTTP‑JSON instead of direct imports – this decouples the UI language (e.g., a Rust/Tauri front‑end).

### 3. Choose GUI Toolkit
| Toolkit | Pros | Cons | Recommendation |
|---------|------|------|----------------|
| **PySide6 / Qt** | Pure Python, easy to call the library directly, cross‑platform, native look | Larger binary size, Qt licenses (LGPL) | Good for a quick MVP |
| **Tauri (Rust + Web)** | Very small bundle, modern UI (HTML/CSS/JS), can call Python via `tauri::invoke` + a small server | Requires Rust toolchain, extra glue code | Best for a polished product |
| **Electron** | Familiar web stack, abundant UI components | Heavy (≥100 MB) | Overkill for this project |

We will start with **PySide6** because the existing code is Python‑only; later we can swap to Tauri if size becomes a concern.

### 4. GUI Wireframe
```
+---------------------------------------------------+
|  Transcriber GUI                                   |
|---------------------------------------------------|
|  [Record]   [Select file …]  (filepath display)   |
|                                                   |
|  [Transcribe]   (progress bar)                    |
|                                                   |
|  Transcript preview (scrollable textbox)          |
|                                                   |
|  [Summarize]   (progress bar)                    |
|                                                   |
|  Report preview (markdown viewer)                |
|                                                   |
|  [Export]  [Settings]  [Help]                    |
+---------------------------------------------------+
```
*Buttons* trigger the corresponding library calls via a background thread to keep the UI responsive.
*Settings* allow the user to pick:
- Whisper model (tiny / base / large)
- Device (CPU / CUDA)
- Summary model (Gemma‑4, etc.)
- Output directory

### 5. Docker Integration
- The GUI will detect if the host has Docker installed.
- When “Run in Docker” is enabled (Settings → Advanced), the GUI calls `run_in_docker()` with the same arguments as the native flow, mounting the host directory containing audio files.
- UI shows a spinner while the container runs and streams the log output to a console view.

### 6. Build & Distribution
| Platform | Tool | Output |
|----------|------|--------|
| macOS (Intel/Apple) | `pyinstaller --onefile --windowed` | `Transcriber.app` |
| Windows | `pyinstaller` | `Transcriber.exe` |
| Linux | `pyinstaller` or native package (deb/rpm) | `transcriber` binary |

All builds will bundle the virtual‑environment `whisper_env` packages (PySide6, Faster‑Whisper, llama.cpp, etc.) to avoid users having to run `pip install`.

## Milestones
| Milestone | Tasks | Owner | Due |
|-----------|-------|-------|-----|
| **M0 – Repo Prep** | Add `api.py`, move core logic, write unit tests for new API | Dev A | Week 1 |
| **M1 – GUI Prototype** | Scaffold PySide6 UI, connect buttons to API, test on Linux | Dev B | Week 2 |
| **M2 – Settings & Docker** | Implement Settings dialog, Docker helper, error handling | Dev A/B | Week 3 |
| **M3 – Packaging** | PyInstaller configs for macOS/Windows/Linux, CI pipelines | Dev C | Week 4 |
| **M4 – Documentation** | Write user guide, update README, add screenshots | Docs | Week 4 |
| **M5 – Beta Release** | Internal testing, collect feedback, fix bugs | All | Week 5 |

## Testing Strategy
- **Unit tests** for `api.py` (mock FFmpeg, Whisper, llama.cpp).
- **Integration tests**: end‑to‑end CLI → GUI call path using temporary audio fixtures.
- **UI tests**: use `pytest-qt` to simulate button clicks and verify state transitions.
- **Docker tests**: mock Docker CLI to ensure command strings are correct.

All tests must pass under `whisper_env/bin/python -m pytest` before each release.

## Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|------------|
| Heavy UI bundle size (Qt) | Users may avoid download | Provide “lite” mode that launches the GUI without bundling Qt (requires local PySide6 install). |
| GPU detection failures inside Docker | Transcription may fall back to CPU | Add explicit `--device` flag in Docker run and surface clear error messages. |
| Platform‑specific FFmpeg issues | Record button fails on some OSes | Ship a small FFmpeg binary per platform or fallback to system FFmpeg if present. |
| Future UI framework switch | Need to rewrite UI code | Keep the API stable; UI can be swapped without touching core library. |

## Next Steps
1. Create `transcriber/api.py` with the three public functions.
2. Add unit tests under `tests/test_api.py`.
3. Scaffold a minimal PySide6 window (`gui/main_window.py`) that calls the API.
4. Iterate through the milestones listed above.

---

*Prepared by the development team – ready for implementation.*