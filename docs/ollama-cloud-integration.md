# Ollama Cloud Integration Plan

## Goal
Enable the analysis/summarization step to run via a locally hosted Ollama server (which pulls models from Ollama Cloud). This allows users with limited local hardware to leverage powerful LLMs without needing to run them locally, while keeping transcription (Whisper) local. **Note: The connection is strictly to a local Ollama instance; no data leaves the user's machine except for the initial model download from Ollama Cloud (if the model is not already cached).**

## Overview
- Transcription remains local (Faster-Whisper) – uses CPU/GPU as before.
- Analysis step will call an Ollama REST API endpoint (`http://localhost:11434/api/generate`) to generate summaries.
- User must have Ollama installed and running locally, with the desired model pulled (e.g., `ollama pull gemma4` then `ollama serve`).
- Integration will be optional, controlled via environment variables.
- The existing local backends (CUDA, ROCm, CPU, llama.cpp) remain unchanged.

### Suggested Models for Analysis
For summarization and analysis tasks, we recommend models with a large context window to accommodate full transcripts. Examples:
- **gemma4:31b-cloud** (Google Gemma 4 Cloud) - official Ollama library model for Gemma 4 Cloud, 131k token context window.
- **gemma4:e4b** (Google Gemma 4 Expert 4B) - smaller, edge-optimized variant (~4.5B parameters) with 128K token context window, uses fewer resources.
- **gemma4** (Google Gemma 4) - 131k token context window, excellent for long documents.
- **llama3.1:8b-instruct** or **llama3.1:70b-instruct** (Meta Llama 3.1) - 128k token context window.
- **mistral-nemo:12b-instruct** (Mistral Nemo) - 128k token context window.

These models are available on Ollama Cloud (where applicable) and can be pulled via `ollama pull <model>` (e.g., `ollama pull gemma4:31b-cloud` for the cloud variant, `ollama pull gemma4:e4b` for the smaller E4B variant).


## Implementation Details

### 1. Backend Selection
Add a new backend engine `"ollama"` in `lib/analysis/backend.py`.

- New environment variables:
  - `TRANSCRIBER_ANALYSIS_BACKEND`: set to `"ollama"` to select Ollama backend.
  - `TRANSCRIBER_OLLAMA_HOST`: base URL of Ollama server (default `http://localhost:11434`).
  - `TRANSCRIBER_OLLAMA_MODEL`: model name to use (e.g., `"gemma4"`). Required when backend is ollama.
  - Optional: `TRANSCRIBER_OLLAMA_TIMEOUT`: HTTP timeout in seconds (default `120`).

- `detect_analysis_backend()` will check the backend preference; if `"ollama"` is requested, it returns an `AnalysisBackend` with:
  - `name`: `"ollama"`
  - `engine`: `"ollama"`
  - `model_id`: the model name from `TRANSCRIBER_OLLAMA_MODEL`
  - `model_kwargs`: dict containing `"host"`, `"timeout"`, etc.
  - `device_name`: `"Ollama"`
  - `max_new_tokens`: maybe derived from `TRANSCRIBER_MAX_TRANSCRIPT_CHARS` or set to a sensible default (e.g., 4096).

### 2. Model Invocation
Add a new function in `lib/analysis/model.py`:

```python
def _generate_report_with_ollama(backend: AnalysisBackend, transcript_body: str, meta: dict[str, str]) -> str:
    import json
    import urllib.request
    import urllib.error

    host = backend.model_kwargs.get("host", "http://localhost:11434")
    model = backend.model_id
    timeout = float(backend.model_kwargs.get("timeout", 120.0))

    # Build prompt using existing prompt helpers
    from .prompt import _build_prompt_for_backend
    prompt = _build_prompt_for_backend(backend, transcript_body, meta)

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            # num_predict corresponds to max_new_tokens
            "num_predict": backend.max_new_tokens,
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            return resp_data.get("response", "").strip()
    except urllib.error.URLError as e:
        raise AnalysisModelError(f"Failed to connect to Ollama at {host}: {e}")
    except ValueError as e:
        raise AnalysisModelError(f"Invalid JSON response from Ollama: {e}")
```

### 3. Dispatcher Update
In `lib/analysis/__init__.py`, inside `generate_summaries`, add a branch:

```python
if backend.engine == "ollama":
    generated_report = _generate_report_with_ollama(backend, transcript_body, meta)
elif backend.engine == LLAMA_CPP_BACKEND_NAME:
    generated_report = _generate_report_with_llama_cpp(backend, transcript_body, meta)
else:
    generated_report = _generate_report_with_transformers(backend, transcript_body, meta)
```

### 4. Dependencies
- No new Python packages required if using `urllib` (standard library).
- Ensure `requests` is not required; but if preferred, we could add `requests` to `requirements.txt` optional.

### 5. Configuration & Documentation
- Update `README.md` or a new section in docs explaining how to set up Ollama locally.
- Example:
  ```bash
  # Install Ollama (see https://ollama.com)
  ollama pull gemma4:31b-cloud   # pull the Gemma 4 Cloud model from Ollama Cloud (first time only)
  ollama serve &       # start the Ollama server on localhost:11434 in background
  # Then run transcriber with Ollama backend:
  TRANSCRIBER_ANALYSIS_BACKEND=ollama \
  TRANSCRIBER_OLLAMA_MODEL=gemma4:31b-cloud \
  python transcribe.py meeting.wav
  ```
  **Note:** For lower resource usage, you can instead use the smaller `gemma4:e4b` model:
  ```bash
  ollama pull gemma4:e4b
  TRANSCRIBER_ANALYSIS_BACKEND=ollama \
  TRANSCRIBER_OLLAMA_MODEL=gemma4:e4b \
  python transcribe.py meeting.wav
  ```

### 6. Testing
- Add unit tests that mock `urllib.request.urlopen` to verify the Ollama call.
- Ensure existing tests still pass (mocking Ollama backend not required unless we want to test the integration path).

### 7. Considerations
- **Context Window**: Ollama models have their own context limits. We should ensure the prompt size fits. Use existing `transcript_char_budget()` from `report.py` to limit the transcript size sent to Ollama, similar to how llama.cpp does it.
- **Streaming**: We disable streaming for simplicity; could be enabled later for faster perceived response.
- **Error Handling**: Provide clear messages if Ollama server is not reachable or model not found.
- **Security**: Since this is local-only, no extra concerns.

## Open Questions
- Should we allow custom Ollama parameters (e.g., `top_p`, `repeat_penalty`) via env vars? Could be added later.
- Note: This integration is designed for connecting to a local Ollama instance only. While the `TRANSCRIBER_OLLAMA_HOST` variable could theoretically be set to a remote endpoint, doing so would send transcript data outside the local machine, violating the project's privacy boundary. We explicitly discourage and do not support remote Ollama endpoints for this reason.

## Timeline
1. Implement backend detection and model_kwargs (backend.py) – 1h
2. Implement `_generate_report_with_ollama` (model.py) – 1h
3. Update dispatcher (__init__.py) – 0.5h
4. Update documentation (this doc + README) – 0.5h
5. Manual testing – 1h
Total ~4 hours.
