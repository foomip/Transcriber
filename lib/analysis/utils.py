"""
utils — shared exceptions, grounding validation, and report parsing.

Responsibilities:
    - Define AnalysisGroundingError and AnalysisModelError exceptions
    - Extract content words from text (filtering stop words)
    - Validate that generated analysis is grounded in the transcript
    - Parse generated Markdown into structured (heading, text) sections
"""

import re
from collections import Counter

GROUNDING_MIN_RATIO = 0.16
GROUNDING_MIN_TOP_TERM_OVERLAP = 2

_STOP_WORDS = {
    "about", "above", "after", "again", "against", "also", "another", "because",
    "before", "being", "between", "could", "during", "every", "first", "from",
    "have", "having", "into", "itself", "just", "like", "major", "meeting",
    "more", "most", "other", "over", "really", "should", "some", "specific",
    "still", "their", "there", "these", "thing", "things", "think", "this",
    "those", "through", "today", "under", "using", "where", "which", "while",
    "with", "would", "youre",
}


class AnalysisGroundingError(RuntimeError):
    """Raised when generated analysis is not sufficiently grounded in the transcript."""


class AnalysisModelError(RuntimeError):
    """Raised when the configured analysis model cannot be loaded or initialized."""


def _content_words(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9']+", text.lower())
        if len(token) >= 4 and token not in _STOP_WORDS
    ]


def _validate_grounding(generated_report: str, transcript_body: str) -> None:
    transcript_words = _content_words(transcript_body)
    generated_words = _content_words(generated_report)
    if not transcript_words or not generated_words:
        return

    transcript_terms = set(transcript_words)
    grounded_count = sum(1 for word in generated_words if word in transcript_terms)
    grounded_ratio = grounded_count / len(generated_words)

    top_transcript_terms = {
        term for term, _ in Counter(transcript_words).most_common(20)
    }
    top_term_overlap = top_transcript_terms.intersection(generated_words)

    if (
        grounded_ratio < GROUNDING_MIN_RATIO
        or len(top_term_overlap) < GROUNDING_MIN_TOP_TERM_OVERLAP
    ):
        raise AnalysisGroundingError(
            "Generated report appears unrelated to the transcript "
            f"(grounded word ratio {grounded_ratio:.0%}, "
            f"top-term overlap {len(top_term_overlap)}). "
            "Refusing to save hallucinated analysis."
        )


def _parse_report_sections(generated_report: str) -> list[tuple[str, str]]:
    # Deferred import: prompt -> (nothing), utils -> prompt is a one-way chain
    # with no cycle, but deferring keeps load order explicit.
    from .prompt import SUMMARY_TASKS

    headings = [heading for heading, _ in SUMMARY_TASKS]
    pattern = re.compile(
        r"(?m)^\s*(## (?:"
        + "|".join(re.escape(heading.removeprefix("## ")) for heading in headings)
        + r"))\s*$"
    )
    matches = list(pattern.finditer(generated_report))
    found: dict[str, str] = {}

    for index, match in enumerate(matches):
        heading = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(generated_report)
        text = generated_report[start:end].strip()
        if heading not in found:
            found[heading] = text

    return [
        (heading, found.get(heading, "No information was generated for this section."))
        for heading in headings
    ]
