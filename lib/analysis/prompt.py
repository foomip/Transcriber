"""
prompt — prompt construction and summary task definitions.

Responsibilities:
    - Define the summary sections to generate (SUMMARY_TASKS)
    - Define the system prompt for the analysis model
    - Build user messages for both full and compact prompt styles
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .backend import AnalysisBackend

ANALYSIS_SYSTEM_PROMPT = (
    "You are an expert meeting analyst. Analyze the transcript carefully "
    "and provide clear, accurate information based only on the transcript. "
    "Do not invent facts, names, dates, events, decisions, or background details. "
    "If the transcript does not contain information for a requested section, say so explicitly."
)

# Each tuple: (markdown heading, analysis instruction)
SUMMARY_TASKS: list[tuple[str, str]] = [
    (
        "## Executive Summary",
        "Provide a brief executive summary (2-3 sentences) of the key outcomes and decisions from this transcript.",
    ),
    (
        "## Detailed Summary",
        "Provide a detailed summary of the transcript, covering all major topics, discussions, and outcomes in paragraph form.",
    ),
    (
        "## Action Items",
        "List the specific action items that were assigned during this meeting. Include who is responsible for each item when mentioned.",
    ),
    (
        "## Key Decisions",
        "List the key decisions that were made during this meeting. Focus on concrete decisions and outcomes.",
    ),
    (
        "## Topics Discussed",
        "List the main topics and subjects that were discussed in this meeting.",
    ),
]


def _build_user_message(
    transcript_body: str,
    meta: dict[str, str],
) -> str:
    """
    Assemble a user message for the analysis model:

        <instructions>

        Title: …
        Date:  …
        …
        ----------

        <transcript text>
    """
    section_instructions = "\n".join(
        f"{heading}\n{instruction}" for heading, instruction in SUMMARY_TASKS
    )

    return (
        "Create a Markdown meeting report using exactly the sections listed below. "
        "Use only facts present in the transcript. Do not add external context, examples, "
        "biographical details, sports results, historical background, or generic explanations. "
        "If there are no action items or no key decisions, write that none were explicitly stated. "
        "Return only the requested Markdown sections and their content.\n\n"
        f"{section_instructions}\n\n"
        f"Title: {meta['title']}\n"
        f"Date: {meta['date']}\n"
        f"Time: {meta['time']}\n"
        f"Duration: {meta['duration']}\n"
        f"Requested transcription language: {meta.get('requested_language', 'Auto-detect')}\n"
        f"Detected transcription language: {meta.get('detected_language', 'Unknown')}\n"
        f"Language detection confidence: {meta.get('language_probability', 'Unknown')}\n"
        f"Transcription model: {meta.get('transcription_model', 'Unknown')}\n"
        f"Participants: Unknown (no speaker diarization)\n\n"
        "----------\n\n"
        f"{transcript_body}"
    )


def _build_compact_user_message(
    transcript_body: str,
    _meta: dict[str, str],
) -> str:
    return (
        "You are an expert meeting analyst. Use only the transcript. "
        "Do not invent facts. Return a Markdown report with these exact headings:\n"
        "## Executive Summary\n"
        "## Detailed Summary\n"
        "## Action Items\n"
        "## Key Decisions\n"
        "## Topics Discussed\n"
        "If a section has no explicit information, say none was explicitly stated.\n\n"
        "Transcript:\n"
        f"{transcript_body}"
    )


def _build_prompt_for_backend(
    backend: "AnalysisBackend",
    transcript_body: str,
    meta: dict[str, str],
) -> str:
    if backend.use_plain_prompt:
        return _build_compact_user_message(transcript_body, meta)
    return _build_user_message(transcript_body, meta)
