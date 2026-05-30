# Model Recommendations

This project has two separate model stages:

1. **Transcription**: speech-to-text, currently handled by Faster-Whisper in `lib/transcription.py`.
2. **Analysis and summarization**: transcript-to-report generation, handled by a Hugging Face/PyTorch model on CPU, NVIDIA, and Intel, and by llama.cpp with GGUF models for the default AMD ROCm Docker path.

## Transcription Model Recommendations

The current transcription backend is Faster-Whisper, with:

```python
WHISPER_MODEL_SIZE = "base"
```

Recommended Faster-Whisper model choices:

| Model             | Best For                              | Recommendation                                     |
| ----------------- | ------------------------------------- | -------------------------------------------------- |
| `tiny`            | Very fast drafts                      | Only use when speed matters more than accuracy     |
| `base`            | Fast local transcription              | Current default; reasonable for clear speech       |
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

The default Hugging Face analysis model is:

```python
DEFAULT_ANALYSIS_MODEL_ID = "google/gemma-4-E4B-it"
```

For one-off comparisons, override it without editing source:

```bash
TRANSCRIBER_ANALYSIS_MODEL="mistralai/Mistral-7B-Instruct-v0.3" \
	python transcribe.py meeting_20260527_114300.wav
```

This is the current default for CPU, NVIDIA CUDA, and Intel paths because it provides excellent reasoning and instruction-following while remaining practical for local deployment.

AMD ROCm Docker runs use a separate llama.cpp/GGUF backend by default. The default GGUF filename is:

```text
gemma-4-E4B-it-Q4_K_M.gguf
```

It downloads on first use into the GGUF cache used by the wrapper:

```text
~/.cache/transcriber/gguf/gemma-4-E4B-it-Q4_K_M.gguf
```

or point directly at another local GGUF file:

```bash
TRANSCRIBER_LLAMA_CPP_MODEL_PATH="/path/to/model.gguf" \
	./docker-run-transcribe.sh meeting_20260527_114300.wav
```

The default download source is `ggml-org/gemma-4-E4B-it-GGUF`. If you want the same filename from a different Hugging Face GGUF repository, set `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`.

The ROCm llama.cpp backend computes a conservative `n_gpu_layers` value at runtime from available VRAM, model file size, context size, configured headroom, and the default 42 Gemma 4 E4B text layers. It then leaves the remaining layers in system RAM, which is safer on consumer AMD cards than relying on Transformers device offload.

Transcript prompt size is capped dynamically from available RAM. For repeatable model comparisons, use a fixed cap:

```bash
TRANSCRIBER_MAX_TRANSCRIPT_CHARS=80000 \
	TRANSCRIBER_ANALYSIS_MODEL="Qwen/Qwen2.5-7B-Instruct" \
	python transcribe.py meeting_20260527_114300.wav
```

Recommended alternatives:

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

For AMD ROCm Docker workflows, start with GGUF models instead of Hugging Face IDs:

| GGUF Model File                         | Why Consider It                                  | Fit                                      |
| --------------------------------------- | ------------------------------------------------ | ---------------------------------------- |
| `gemma-4-E4B-it-Q4_K_M.gguf`            | Current ROCm default target; strong summaries through llama.cpp | Best first AMD ROCm option               |
| `Qwen2.5-3B-Instruct-Q4_K_M.gguf`       | Compact and capable if Gemma is too heavy        | Good lower-resource fallback             |
| `Qwen2.5-7B-Instruct-Q4_K_M.gguf`       | Better summaries if VRAM/RAM budget allows       | Good quality upgrade for larger systems  |
| `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf`  | Widely used instruct model with strong summaries | Strong comparison model                  |
| `Phi-3-mini-4k-instruct-Q4_K_M.gguf`    | Small and fast                                   | Good low-resource option                 |

Useful ROCm llama.cpp tuning variables:

| Variable                              | Purpose                                                   |
| ------------------------------------- | --------------------------------------------------------- |
| `TRANSCRIBER_LLAMA_CPP_MODEL_PATH`    | Absolute path to a GGUF file                              |
| `TRANSCRIBER_LLAMA_CPP_MODEL_REPO`    | Hugging Face repo used to download a missing GGUF file    |
| `TRANSCRIBER_GGUF_CACHE_DIR`          | Directory containing the default GGUF model filename      |
| `TRANSCRIBER_LLAMA_CPP_CONTEXT_SIZE`  | llama.cpp context window                                  |
| `TRANSCRIBER_LLAMA_CPP_BATCH_SIZE`    | llama.cpp batch size                                      |
| `TRANSCRIBER_LLAMA_CPP_GPU_LAYERS`    | Manual `n_gpu_layers` override                            |
| `TRANSCRIBER_LLAMA_CPP_GPU_HEADROOM_GIB` | Extra VRAM headroom to reserve before offloading layers |
| `TRANSCRIBER_LLAMA_CPP_LAYER_COUNT`    | Model layer count used by the automatic GPU/RAM split estimator |

Use `TRANSCRIBER_ANALYSIS_MODEL` only for Hugging Face model IDs. Use `TRANSCRIBER_LLAMA_CPP_MODEL_PATH` for GGUF files.

## Recommended First Comparison

The first smaller model worth comparing against the Gemma 4 E4B default is:

```bash
TRANSCRIBER_ANALYSIS_MODEL="Qwen/Qwen2.5-3B-Instruct" \
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

The Hugging Face summarization backend assumes that the model can be loaded with:

```python
AutoModelForCausalLM.from_pretrained(...)
AutoTokenizer.from_pretrained(...)
```

and that the tokenizer supports:

```python
tokenizer.apply_chat_template(...)
```

Most modern instruct models support this, but some may need adjusted tokenizer/model arguments or prompt formatting. The AMD ROCm Docker backend does not use this loading path; it uses llama.cpp and local GGUF files instead.

## Overall Recommendation

Use `google/gemma-4-E4B-it` as the CPU/NVIDIA/Intel baseline. Use `gemma-4-E4B-it-Q4_K_M.gguf` as the first AMD ROCm Docker baseline.

For better transcription, first try:

```python
WHISPER_MODEL_SIZE = "small"
```

For a lighter summarization comparison, first try:

```bash
TRANSCRIBER_ANALYSIS_MODEL="Qwen/Qwen2.5-3B-Instruct" \
	python transcribe.py meeting_20260527_114300.wav
```

Then compare both models against the same transcript before making a permanent switch.
