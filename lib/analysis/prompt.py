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
    "You are an expert meeting analyst. Analyze the transcript carefully and extract every detail that matters for decision-making, project direction, or follow-up action.\n\n"
    "- Capture specific numbers, dates, deadlines, and dependencies wherever they appear.\n"
    "- Note trade-offs discussed, risks identified, blockers raised, and open questions left unresolved.\n"
    "- Distinguish clearly between what was decided versus what is still being debated or deferred.\n"
    "- When multiple speakers contribute to a point, synthesize the consensus (or disagreement).\n\n"
    "IMPORTANT: This transcript has no speaker diarization — it is hard to tell who said what. Only attribute a statement to a specific person if the transcript makes it unambiguous (e.g., someone introduces themselves, or explicitly says 'I will do X'). Otherwise, describe the content without naming who said it.\n\n"
    "Use ONLY facts present in the transcript. Do not invent names, dates, events, decisions, background details, or external context. If the transcript does not contain information for a requested section, say so explicitly."
)

# Each tuple: (markdown heading, analysis instruction)
SUMMARY_TASKS: list[tuple[str, str]] = [
    (
        "## Executive Summary",
        "Write 3-5 sentences capturing the key outcomes, decisions, and strategic implications of this meeting. Focus on what was resolved or advanced, not just what was discussed.",
    ),
    (
        "## Action Items",
        "List every specific action item assigned during this meeting. For each: who owns it if clear, what exactly needs to be done, any deadline given, and the context/reason for the action. Note that speaker diarization was not used so ownership may be uncertain — describe the task itself even when the owner is unclear. If no action items were explicitly stated, write that none were assigned.",
    ),
    (
        "## Key Decisions",
        "List every concrete decision made during this meeting. For each: state the decision clearly, include relevant specifics (numbers, dates, scope), and note any conditions or dependencies attached to it. If no decisions were explicitly stated, write that none were made.",
    ),
    (
        "## Risks & Open Questions",
        "List risks, blockers, concerns, or unresolved questions raised during this meeting. For each: describe the issue specifically and any proposed mitigations or next steps discussed. Note that speaker diarization was not used — attribute to a person only if clearly indicated in the transcript. If none were raised, write that none were identified.",
    ),
    (
        "## Detailed Summary",
        "Provide a thorough paragraph-by-paragraph summary covering all major topics discussed — including background context shared, arguments debated, data presented, and the resolution of each discussion point. Capture specifics: numbers cited, dates mentioned, trade-offs weighed, and how each topic was resolved or deferred. Do not attribute statements to specific speakers unless clearly indicated in the transcript.",
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
    headings = "\n".join(heading for heading, _instruction in SUMMARY_TASKS)
    return (
        f"{ANALYSIS_SYSTEM_PROMPT}\n\n"
        "Return a Markdown report with these exact headings:\n"
        f"{headings}\n"
        "If a section has no explicit information, say none was explicitly stated.\n\n"
        "Transcript:\n"
        f"{transcript_body}"
    )


def _build_prompt_for_backend(
    backend: "AnalysisBackend",
    transcript_body: str,
    meta: dict[str, str],
) -> str:
    return _build_compact_user_message(transcript_body, meta)
