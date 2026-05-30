# Multi-language Support Plan

## Purpose

This document outlines a plan for extending the app from its current English-centered meeting workflow to support:

- English
- Portuguese
- Afrikaans
- Meetings that mix any of the above languages

This is a planning document only. It does not implement the feature.

---

## Goals

### Primary goals

- Support transcription of meetings spoken in English, Portuguese, Afrikaans, or a mixture of those languages.
- Preserve privacy and local-only execution.
- Keep the transcript faithful to the spoken language(s).
- Allow report generation in a single chosen output language.
- Avoid regressions for existing English-only workflows.

### Non-goals for the first iteration

- Cloud-based translation or summarization
- Speaker diarization
- Real-time live translation
- Perfect per-segment language labeling
- Broad support for all Whisper-supported languages

---

## Current state

The current app architecture is:

- `record_meeting.sh` records local audio to WAV
- `transcribe.py` orchestrates transcription and report generation
- `lib/transcription.py` runs Faster-Whisper and returns timestamped transcript lines
- `lib/analysis.py` prompts a local LLM to generate report sections
- `lib/report.py` compiles the final Markdown report

### What already helps

- The app already uses Faster-Whisper, which supports multilingual transcription.
- The current Whisper model is `small`, which is multilingual rather than English-only.
- The transcript stores language metadata plus timestamped text, which is a good foundation for multilingual support.
- The CLI supports forced single-language transcription with `-l` / `--language` for Whisper language codes.

### Current English-centric limitations

#### `lib/transcription.py`

- No explicit mixed-language transcription mode.
- Only one top-level detected language is surfaced.
- The current default model size (`small`) may still need benchmarking for code-switching and lower-resource language accuracy.

#### `lib/analysis.py`

- Prompts are written only in English.
- Section headings are hard-coded in English.
- The report parser expects English headings.
- Grounding validation assumes mostly English ASCII words.
- Stop-word logic is English-only.

#### `lib/report.py`

- Report title and section headings are English-only.
- There is no concept of report language.
- Requested and detected transcription language metadata is surfaced, but report output language is not configurable yet.

#### Documentation

- The README and current framing center on English meeting usage.

---

## Product design proposal

The key design choice is to separate:

1. **Spoken language handling**
2. **Report output language**

These should not be treated as the same setting.

### Proposed input language modes

- `auto`
- `english`
- `portuguese`
- `afrikaans`
- `mixed`

### Proposed report language options

- `english`
- `portuguese`
- `afrikaans`
- optional later: `dominant_meeting_language`

### Recommended default behavior

- Spoken language mode: `auto`
- Report language: `english`

### Recommended output behavior

- **Transcript:** preserve the original spoken language(s)
- **Report:** generate in one chosen output language

This is the cleanest user experience for mixed meetings.

---

## Key architectural decisions

## 1. Keep transcript language separate from report language

For multilingual meetings, the transcript should remain faithful to what was spoken. The report can then normalize the content into one chosen language.

This avoids:

- losing original wording too early
- coupling transcription and summarization unnecessarily
- forcing a translation pass before summarization

## 2. Treat mixed-language meetings as a first-class use case

Mixed-language meetings should not be treated as an edge case. The design should assume that:

- different speakers may use different languages
- a speaker may switch languages mid-meeting
- meeting jargon, names, and product terms will remain mixed

## 3. Replace markdown parsing with structured analysis output

The current analysis flow asks the model to emit Markdown with exact headings and then parses that Markdown back into sections. That is fragile once headings vary by language.

Instead, the analysis layer should eventually return structured fields such as:

- `executive_summary`
- `detailed_summary`
- `action_items`
- `key_decisions`
- `topics_discussed`

The report layer should then localize those fields when rendering Markdown.

## 4. Make grounding validation language-aware

Current grounding checks are too English-specific. Any real multilingual release should include Unicode-aware, language-neutral validation.

---

## Detailed implementation plan

## A. Transcription planning

### Objectives

- Improve accuracy for Portuguese and Afrikaans
- Improve handling of code-switching
- Expose clearer language controls
- Preserve backward compatibility for English meetings

### Planned changes

#### Add a transcription configuration layer

`transcribe.py` and `lib/transcription.py` should be extended to support a language-aware transcription configuration.

This configuration should describe at least:

- spoken language mode
- requested report language
- transcription model size
- whether mixed-language handling is enabled

#### Support forced single-language mode

Status: implemented for Whisper language codes through `-l` / `--language`.

If the user knows the meeting is entirely in one language, the app should be able to explicitly transcribe as:

- English
- Portuguese
- Afrikaans

This should improve stability over auto-detection for known single-language meetings.

#### Support auto mode

For unknown audio, the app should:

- detect the dominant language
- use the detected language when confidence is high
- fall back to mixed-language handling when confidence is weak or ambiguous

#### Support mixed-language mode

For meetings that mix English, Portuguese, and Afrikaans, the app should use Faster-Whisper in a way that best preserves multilingual speech rather than forcing a single language.

### Model evaluation plan

The current default is `base`. Before choosing defaults for multilingual support, benchmark:

- `base`
- `small`
- `medium`
- `large-v3` or `distil-large-v3` on NVIDIA systems

### Recommendation

For multilingual support, `small` is a likely better baseline than `base`, pending benchmarks.

Reason:

- code-switched and lower-resource language audio usually exposes model weaknesses faster than clear English audio

---

## B. Transcription metadata planning

### Objectives

Return richer transcription metadata so later stages can make better decisions and the app can explain what happened.

### Metadata to add

Plan for transcription results to include metadata such as:

- dominant detected language
- detection confidence
- alternative language probabilities, if available
- transcription mode used (`forced`, `auto`, `mixed`)
- Whisper model size used

### Optional later enhancement

- per-segment language labels for mixed-language meetings

This should be treated as a later improvement rather than a requirement for the first release.

---

## C. Analysis and summarization planning

### Objectives

- Accept transcripts in English, Portuguese, Afrikaans, or mixtures
- Generate a report in a chosen single output language
- Maintain grounding to the transcript
- Avoid hallucination and format fragility

### Prompting strategy

The analysis prompts should eventually specify:

- the transcript may contain multiple languages
- the report must use only facts present in the transcript
- the output language must be the chosen report language
- if the transcript mixes languages, the report should normalize into the chosen output language

### Structured output strategy

The analysis model should target structured fields internally rather than exact localized Markdown headings.

This makes it easier to:

- change output language
- localize section headings
- test parsing reliably
- reduce formatting failures

### Model evaluation plan

The current analysis default is Gemma 4 E4B, with ROCm Docker runs using the GGUF llama.cpp variant by default.

This should remain the baseline, but multilingual summarization quality should be tested against at least:

- `google/gemma-4-E4B-it`
- `Qwen/Qwen2.5-3B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- at least one multilingual-first local model

### Important consideration

Portuguese is likely to be better supported by many multilingual models than Afrikaans. Afrikaans quality should be treated as the more important stress test when selecting a summarization model.

---

## D. Grounding and validation planning

### Why this matters

The current grounding logic in `lib/analysis.py` assumes English-like ASCII tokens and English stop words. That can cause valid Portuguese or Afrikaans summaries to be incorrectly flagged as unrelated.

### Planned redesign

Grounding validation should become:

- Unicode-aware
- less dependent on English stop words
- less dependent on ASCII-only tokenization
- more robust for multilingual transcripts and localized summaries

### Possible validation directions

Potential approaches include:

- normalized Unicode token overlap
- overlap on named entities, dates, numbers, and repeated keywords
- section-level grounding checks rather than only one overall ratio

This should be considered a required part of any true multilingual release.

---

## E. Report rendering and localization planning

### Objectives

Allow the app to render reports in English, Portuguese, or Afrikaans while keeping the internal analysis representation stable.

### Planned changes

`lib/report.py` should eventually localize:

- report title
- section headings
- metadata labels such as date and duration
- optional metadata such as detected language or report language

### Recommended design

Keep internal section identifiers stable in code, and perform localization only during report rendering.

This minimizes complexity and keeps future expansion manageable.

---

## F. Translation strategy decision

There are two broad strategies for multilingual summarization.

### Strategy 1: Direct multilingual summarization

- transcript remains in original language(s)
- analysis model reads the multilingual transcript directly
- report is generated in the chosen output language

### Strategy 2: Normalize transcript first

- transcript is translated or normalized into one language first
- summarization is then performed on the normalized transcript

### Recommendation

Start with **Strategy 1**.

Reason:

- simpler architecture
- fewer failure points
- better preservation of original transcript fidelity
- no extra translation pass required

A transcript-normalization stage should only be considered later if testing shows that direct multilingual summarization is not reliable enough, especially for Afrikaans or heavily code-switched meetings.

---

## Evaluation and testing plan

## Create a multilingual evaluation set

Before changing defaults, assemble a small privacy-safe set of meeting recordings or synthetic samples covering:

- English-only
- Portuguese-only
- Afrikaans-only
- English + Portuguese
- English + Afrikaans
- Portuguese + Afrikaans
- English + Portuguese + Afrikaans

### For each sample, evaluate transcription quality

- word accuracy by manual review
- handling of language switches
- preservation of names and jargon
- timestamp usefulness
- behavior under forced, auto, and mixed modes

### For each sample, evaluate report quality

- factual correctness
- action item extraction
- key decision extraction
- consistency of chosen output language
- resistance to hallucination
- whether grounding validation falsely rejects good summaries

### Acceptance criteria

A first multilingual release should aim for:

- no meaningful regression for English-only meetings
- usable transcript quality in Portuguese and Afrikaans
- coherent report generation for mixed-language meetings
- no strong bias toward false grounding failures on non-English reports

---

## Risks and constraints

## 1. Mixed-language detection can be unstable

Auto-detection can become less reliable when speakers switch languages frequently or when the audio is short.

Mitigation:

- allow explicit language modes
- support a dedicated mixed-language mode
- surface detection metadata for transparency

## 2. Afrikaans may be the weakest point

Afrikaans is more likely than Portuguese to expose weak multilingual coverage in summarization models.

Mitigation:

- benchmark models with Afrikaans-specific samples
- avoid assuming English/Portuguese results generalize

## 3. Larger multilingual models increase local resource usage

Better multilingual quality may require heavier models.

Mitigation:

- benchmark before changing defaults
- keep a practical small-machine path
- document trade-offs clearly

## 4. Grounding logic may fail before the model does

Even if the model produces a good Portuguese or Afrikaans report, current validation may incorrectly reject it.

Mitigation:

- prioritize validator redesign early in the implementation

---

## Phased implementation plan

## Phase 0 — Product and UX definition

### Deliverables

- Finalize supported spoken languages for v1
- Finalize supported report output languages for v1
- Decide exact user-facing configuration names
- Decide defaults for spoken language mode and report language

### Suggested decisions

- spoken language mode default: `auto`
- report language default: `english`
- transcript remains in original spoken language(s)
- report is generated in one chosen language

---

## Phase 1 — Transcription configuration and language controls

### Scope

- Add a transcription language mode concept
- Support explicit single-language transcription
- Support mixed-language transcription mode
- Preserve current English behavior by default

### Likely files

- `transcribe.py`
- `lib/transcription.py`
- `README.md`

### Outcome

The app can intentionally run in English, Portuguese, Afrikaans, auto-detect, or mixed-language mode.

---

## Phase 2 — Richer transcription metadata

### Scope

- Return and propagate transcription metadata
- Record detected language, confidence, and mode used
- Make this metadata available to report generation

### Likely files

- `lib/transcription.py`
- `transcribe.py`
- `lib/report.py`

### Outcome

The app can explain what language it thought it heard and which transcription strategy it used.

---

## Phase 3 — Summarization contract redesign

### Scope

- Replace markdown-heading parsing with structured output
- Keep internal section identifiers language-neutral
- Separate analysis output structure from report rendering

### Likely files

- `lib/analysis.py`
- `lib/report.py`

### Outcome

The analysis stage becomes more robust and easier to localize.

---

## Phase 4 — Multilingual prompting and output-language control

### Scope

- Allow analysis prompts to accept multilingual transcripts
- Add explicit report language selection
- Ensure mixed-language transcripts can produce a coherent single-language report

### Likely files

- `transcribe.py`
- `lib/analysis.py`
- `lib/report.py`

### Outcome

The app can summarize multilingual transcripts into English, Portuguese, or Afrikaans.

---

## Phase 5 — Grounding and validation redesign

### Scope

- Replace English-only token assumptions
- Implement Unicode-aware, language-neutral validation
- Reduce false positives on Portuguese and Afrikaans summaries

### Likely files

- `lib/analysis.py`

### Outcome

The app can safely validate multilingual summaries without rejecting valid output due to English-specific heuristics.

---

## Phase 6 — Report localization

### Scope

- Localize report title and headings
- Localize metadata labels
- Optionally include detected language and report language metadata

### Likely files

- `lib/report.py`

### Outcome

The final Markdown report can be rendered cleanly in English, Portuguese, or Afrikaans.

---

## Phase 7 — Model benchmarking and default selection

### Scope

- Benchmark Whisper model sizes for multilingual transcription
- Benchmark analysis models for multilingual summarization quality
- Choose practical defaults for multilingual use

### Likely artifacts

- updates to `docs/model-recommendations.md`
- documented benchmark notes

### Outcome

The project selects defaults based on observed multilingual quality rather than assumptions.

---

## Phase 8 — Documentation and release preparation

### Scope

- Update README and relevant docs
- Document supported languages and known caveats
- Add usage guidance for auto vs forced vs mixed modes
- Explain transcript language versus report language

### Likely files

- `README.md`
- `docs/model-recommendations.md`
- this document, if needed

### Outcome

Users can understand how multilingual support works and what trade-offs to expect.

---

## Suggested task breakdown by file

### `transcribe.py`

Planned responsibilities:

- accept language-related configuration
- pass transcription mode into transcription layer
- pass report language into analysis layer
- surface clearer runtime messages

### `lib/transcription.py`

Planned responsibilities:

- support forced language, auto mode, and mixed mode
- return richer transcription metadata
- expose model and detection details

### `lib/analysis.py`

Planned responsibilities:

- support multilingual transcripts
- support chosen report language
- emit structured section data
- validate grounding in a language-neutral way

### `lib/report.py`

Planned responsibilities:

- localize headings and metadata labels
- render from structured analysis data
- optionally include detected language/report language metadata

### `README.md`

Planned updates:

- explain multilingual support
- document configuration choices
- explain the difference between transcript language and report language

### `docs/model-recommendations.md`

Planned updates:

- add multilingual transcription recommendations
- add multilingual summarization model evaluation notes
- record any new default recommendations

---

## Recommended order of work

1. Finalize UX and configuration design
2. Add transcription language modes
3. Return richer transcription metadata
4. Redesign summarization output structure
5. Add report-language control and multilingual prompts
6. Redesign grounding validation
7. Localize report rendering
8. Benchmark models and finalize defaults
9. Update documentation

---

## Summary

The app is already close to supporting multilingual transcription because the transcription backend is multilingual-capable. The main work is in making language handling explicit, treating mixed-language meetings as a first-class case, and redesigning the summarization/report pipeline so it is not tightly coupled to English prompts and English markdown headings.

The most important planning decisions are:

- separate spoken language handling from report output language
- preserve transcripts in the original spoken language(s)
- generate reports in one chosen language
- redesign analysis around structured output rather than localized markdown parsing
- replace English-specific grounding logic before declaring multilingual support

If implemented in phases, this can add English, Portuguese, Afrikaans, and mixed-language support without disrupting the current local-first architecture.
