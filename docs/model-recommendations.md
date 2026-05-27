# Model Recommendations

This project has two separate model stages:

1. **Transcription**: speech-to-text, currently handled by Faster-Whisper in `lib/transcription.py`.
2. **Analysis and summarization**: transcript-to-report generation, currently handled by `LiquidAI/LFM2-2.6B-Transcript` in `lib/analysis.py`.

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
- AMD ROCm is currently useful for the summarization step through PyTorch, but Faster-Whisper falls back to CPU in the current implementation.

## Analysis and Summarization Model Recommendations

The current analysis model is:

```python
LFM2_MODEL_ID = "LiquidAI/LFM2-2.6B-Transcript"
```

This is a sensible default because it is local, relatively small, and designed for transcript summarization.

Recommended alternatives:

| Model                                  | Why Consider It                                   | Fit                                             |
| -------------------------------------- | ------------------------------------------------- | ----------------------------------------------- |
| `mistralai/Mistral-7B-Instruct-v0.3`   | Efficient, strong summarization, widely supported | Best first serious comparison                   |
| `meta-llama/Meta-Llama-3-8B-Instruct`  | Strong instruction following and meeting analysis | Good quality upgrade if hardware allows         |
| `Qwen/Qwen2-7B-Instruct`               | Good structured output and multilingual handling  | Good for technical or mixed-language meetings   |
| `google/gemma-2-9b-it`                 | Strong summarization and reasoning                | Good quality option, but check hardware support |
| `microsoft/Phi-3-mini-4k-instruct`     | Small and fast                                    | Good low-resource option, but limited context   |
| `microsoft/Phi-3-small-8k-instruct`    | Compact with better context than Phi mini         | Good middle-ground local model                  |
| `NousResearch/Hermes-2-Pro-Mistral-7B` | Often good at structured response formats         | Useful for action items and decisions           |
| `CohereForAI/c4ai-command-r-v01`       | Strong long-context summarization                 | Excellent but much heavier                      |

Shortlist to test:

1. `mistralai/Mistral-7B-Instruct-v0.3`
2. `meta-llama/Meta-Llama-3-8B-Instruct`
3. `Qwen/Qwen2-7B-Instruct`
4. `google/gemma-2-9b-it`

For smaller machines or CPU-heavy workflows:

1. `microsoft/Phi-3-mini-4k-instruct`
2. `microsoft/Phi-3-small-8k-instruct`
3. `LiquidAI/LFM2-2.6B-Transcript`

## Recommended First Comparison

The first model worth comparing against LFM2 is:

```python
LFM2_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
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

The current summarization code assumes that the model can be loaded with:

```python
AutoModelForCausalLM.from_pretrained(...)
AutoTokenizer.from_pretrained(...)
```

and that the tokenizer supports:

```python
tokenizer.apply_chat_template(...)
```

Most modern instruct models support this, but some may need adjusted tokenizer/model arguments or prompt formatting.

## Overall Recommendation

Keep `LiquidAI/LFM2-2.6B-Transcript` as the baseline.

For better transcription, first try:

```python
WHISPER_MODEL_SIZE = "small"
```

For better summarization, first try:

```python
LFM2_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
```

Then compare both models against the same transcript before making a permanent switch.
