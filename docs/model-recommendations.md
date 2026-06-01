# Model Recommendations

This project has two separate model stages:

1. **Transcription**: speech-to-text, currently handled by Faster-Whisper in `lib/transcription.py`.
2. **Analysis and summarization**: transcript-to-report generation, handled by llama.cpp with GGUF models for all hardware targets (CPU, NVIDIA, Intel, and AMD ROCm).

## Transcription Model Recommendations

The current transcription backend is Faster-Whisper, with:

```python
WHISPER_MODEL_SIZE = "small"
```

Recommended Faster-Whisper model choices:

| Model             | Best For                              | Recommendation                                     |
| ----------------- | ------------------------------------- | -------------------------------------------------- |
| `tiny`            | Very fast drafts                      | Only use when speed matters more than accuracy     |
| `base`            | Fast local transcription              | Good speed/accuracy balance                        |
| `small`           | Better accuracy while still practical | Best first upgrade from `base`                     |
| `medium`          | Higher-quality meeting transcripts    | Good if CPU time is acceptable or GPU is available |
| `large-v3`        | Best Whisper accuracy                 | Best with NVIDIA CUDA; likely slow on CPU          |
| `distil-large-v3` | Near-large quality with better speed  | Strong option on NVIDIA CUDA                       |

Practical recommendation:

```python
WHISPER_MODEL_SIZE = "small"
```

For an NVIDIA CUDA system:

```python
WHISPER_MODEL_SIZE = "distil-large-v3"
```

or:

```python
WHISPER_MODEL_SIZE = "large-v3"
```

Notes:

- Faster-Whisper uses CTranslate2 for inference.
- In this project, CUDA acceleration is available for Faster-Whisper when CTranslate2 detects an NVIDIA GPU.
- AMD ROCm is currently useful for the summarization step through the Docker llama.cpp/GGUF backend, but Faster-Whisper falls back to CPU in the current implementation.

## Analysis and Summarization Model Recommendations

The default analysis model is `google/gemma-4-E4B-it`. For all backends, this model is run through the llama.cpp engine using GGUF quantizations, which allow the model to be split across GPU VRAM and system RAM.

For a native Python installation, a GGUF version of the model is downloaded automatically. For Docker runs, the images come with the necessary build tools, and the wrapper handles the GGUF cache.

To compare another local GGUF model without editing source, point to it explicitly:

```bash
TRANSCRIBER_LLAMA_CPP_MODEL_PATH="/path/to/model.gguf" \
	python transcribe.py meeting_20260527_114300.wav
```

The default download source is `ggml-org/gemma-4-E4B-it-GGUF`. If you want the same filename from a different Hugging Face GGUF repository, set `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`.

The automatic layer split defaults to 42 model layers, matching the Gemma 4 E4B text configuration. Override `TRANSCRIBER_LLAMA_CPP_LAYER_COUNT` only when using a different GGUF architecture.

The llama.cpp context window is sized automatically to hold the whole transcript prompt plus the generated report (derived from `TRANSCRIBER_MAX_TRANSCRIPT_CHARS`, capped at the model's trained 131072-token window). Set `TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE` only to pin a fixed window.

Advanced llama.cpp tuning is available through `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`, `TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE`, `TRANSCRIBER_LLAMA_CPP_BATCH_SIZE`, `TRANSCRIBER_LLAMA_CPP_GPU_LAYERS`, `TRANSCRIBER_LLAMA_CPP_GPU_HEADROOM_GIB`, and `TRANSCRIBER_LLAMA_CPP_LAYER_COUNT`. The defaults are intended to be conservative.

Transcript prompt size is capped dynamically from available RAM. For repeatable model comparisons, use a fixed cap:

```bash
TRANSCRIBER_MAX_TRANSCRIPT_CHARS=80000 \
	python transcribe.py meeting_20260527_114300.wav
```

Recommended alternatives (convert these to GGUF format for use):

| Model                                  | Why Consider It                                            | Fit                                             |
| -------------------------------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| `google/gemma-4-E4B-it`                | Current default; strong reasoning and concise output        | Best first option for local runs                |
| `Qwen/Qwen2.5-3B-Instruct`             | Fast, strong instruction following for its size             | Strong lightweight alternative                  |
| `mistralai/Mistral-7B-Instruct-v0.3`   | Efficient, strong summarization, widely supported          | Strong comparison model                         |
| `meta-llama/Meta-Llama-3-8B-Instruct`  | Strong instruction following and meeting analysis          | Good quality upgrade if hardware allows         |
| `Qwen/Qwen2.5-7B-Instruct`             | Better quality than the 3B models, but heavier            | Good upgrade if hardware allows                 |
| `google/gemma-2-9b-it`                 | Strong summarization and reasoning                         | Good quality option, but check hardware support |
| `microsoft/Phi-3-mini-4k-instruct`     | Small and fast                                             | Good low-resource option, but limited context   |
| `microsoft/Phi-3-small-8k-instruct`    | Compact with better context than Phi mini                  | Good middle-ground local model                  |
| `NousResearch/Hermes-2-Pro-Mistral-7B` | Often good at structured response formats                  | Useful for action items and decisions           |
| `CohereForAI/c4ai-command-r-v01`       | Strong long-context summarization                          | Excellent but much heavier                      |

Shortlist to test:

1. `google/gemma-4-E4B-it`
2. `Qwen/Qwen2.5-7B-Instruct`
3. `mistralai/Mistral-7B-Instruct-v0.3`
4. `google/gemma-2-9b-it`

For smaller machines or CPU-heavy workflows:

1. `microsoft/Phi-3-mini-4k-instruct`
2. `microsoft/Phi-3-small-8k-instruct`
3. `Qwen/Qwen2.5-3B-Instruct`

GGUF-specific recommendations (ready to use with llama.cpp):

| GGUF Model File                         | Why Consider It                                  | Fit                                      |
| --------------------------------------- | ------------------------------------------------ | ---------------------------------------- |
| `gemma-4-E4B-it-Q4_K_M.gguf`            | Current default target; strong summaries through llama.cpp | Best first option for all backends         |
| `Qwen2.5-3B-Instruct-Q4_K_M.gguf`       | Compact and capable if Gemma is too heavy        | Good lower-resource fallback             |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf`       | Better summaries if VRAM/RAM budget allows       | Good quality upgrade for larger systems  |
| `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`  | Widely used instruct model with strong summaries | Strong comparison model                  |
| `Phi-3-mini-4k-instruct-Q4_K_M.gguf`    | Small and fast                                   | Good low-resource option                 |

## Recommended First Comparison

The first smaller model worth comparing against the Gemma 4 E4B default is:

```bash
TRANSCRIBER_LLAMA_CPP_MODEL_PATH="/path/to/Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
	python transcribe.py meeting_20260527_114300.wav
```

Compare outputs using the same transcript and evaluate:

- Missed action items
- Hallucinated decisions
- Summary verbosity
- Handling of messy ASR text
- Runtime
- Memory usage
- Ability to follow the five report sections consistently

## Important Caveat

The analysis backend now uses llama.cpp and GGUF models. If you wish to use a model from Hugging Face that is not already in GGUF format, you must first convert it using the tools provided by the [llama.cpp project](https://github.com/ggerganov/llama.cpp).

## Overall Recommendation

Use `google/gemma-4-E4B-it` as the baseline for all backends.

For better transcription, first try:

```python
WHISPER_MODEL_SIZE = "small"
```

For a lighter summarization comparison, first try:

```bash
TRANSCRIBER_LLAMA_CPP_MODEL_PATH="/path/to/Qwen2.5-3B-Instruct-Q4_K_M.gguf" \
	python transcribe.py meeting_20260527_114300.wav
```

Then compare both models against the same transcript before making a permanent switch.
