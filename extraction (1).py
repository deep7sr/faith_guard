"""
Extraction: pull the user's question and the retrieved context out of a chat
request, without asking developers to send anything in a special shape.

This is the load-bearing part of the guardrail. Every downstream score is only
as good as the query/context split produced here, so the design goal is: high
precision when we DO claim a split, and an honest "I couldn't find context"
when we can't, rather than a confident wrong guess.

Two tiers, tried in order:

  Tier 1  MARKER   - look for the labels developers (and the LLMs writing their
                     code) actually use to introduce retrieved context:
                     "Context:", "Use the following retrieved context", the
                     <context>...</context> XML style Anthropic recommends,
                     dash-wrapped headers like "--- Retrieved Evidence ---"
                     (including when they prefix an assistant-role message -
                     some teams inject context that way instead of via
                     system/user), etc. Deterministic. When it matches it's
                     almost always right. For multi-turn conversations where
                     more than one marker-prefixed assistant message exists,
                     the MOST RECENT one is used - not all of them, and not
                     the first - so a stale prior turn's context never leaks
                     into scoring the current turn's answer.

  Tier 2  LLM      - no marker found. Hand the messages to the judge model and
                     ask it to quote the question and context verbatim, then
                     verify each returned span is a real substring. This module
                     only defines the interface (LLMExtractor protocol); the
                     guardrail wires in the actual judge client so we don't keep
                     two model clients around.

  Tier 3  NONE     - even the LLM found nothing context-shaped. Almost certainly
                     not a RAG call. Caller should SKIP the faithfulness check
                     rather than score against empty context.

The tier that produced a result travels with it (ExtractionResult.tier) so a
score computed off a Tier-1 marker can be trusted more than one off a Tier-2
LLM-assisted guess when someone audits a decision later.

A structural "SHAPE" tier (guessing from an unlabeled block's length/position)
previously sat between these two. It was removed once a company-standard,
marker-based context format was adopted - it was already the lowest-confidence
tier by design, and skipping straight to the verified LLM fallback (or NONE)
is safer than guessing from structure alone.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Protocol, Tuple


class Tier(IntEnum):
    MARKER = 1
    LLM = 2
    NONE = 3


@dataclass
class ExtractionResult:
    query: Optional[str]
    context_chunks: List[str]
    tier: Tier
    # Free-form note for logs/debugging: which marker hit, why the LLM fell
    # back, etc.
    detail: str = ""
    # Confidence is coarse on purpose - it maps to tier, not a calibrated number.
    # MARKER -> high, LLM -> medium, NONE -> n/a.
    confidence: str = ""

    @property
    def has_context(self) -> bool:
        return bool(self.context_chunks) and bool(self.query)


class LLMExtractor(Protocol):
    """Tier-2 hook. The guardrail supplies a concrete impl backed by the judge
    model. Kept as a protocol so extraction has zero LLM-client dependency and
    stays unit-testable with plain data."""

    def extract(self, messages: List[dict]) -> Optional[Tuple[str, List[str]]]:
        """Return (query, context_chunks) or None if nothing found."""
        ...


# ---------------------------------------------------------------------------
# Tier 1 markers
# ---------------------------------------------------------------------------
# Ordered roughly by how strong a signal each is. These come from the prompt
# conventions that actually show up in the wild: LangChain's rag-prompt hub
# template ("Use the following pieces of retrieved context"), LlamaIndex QA
# templates ("Context information is below" / "Context:"), Anthropic's XML
# delimiter guidance (<context>, <document>), and the hand-rolled
# "Context:\n...\nQuestion:\n..." pattern that f-string RAG code produces.
#
# Two kinds:
#   PAIRED   - a context opener AND a question opener both present. Strongest,
#              because we can bound the context precisely between them.
#   XML      - <context>...</context> style wrappers. Also strong; explicit
#              close tag bounds the span.
#   OPENER   - only a context label. We take everything after it (minus a
#              trailing question line, if any) as context.

# XML-style wrappers: tag name -> matched via a single regex below.
_XML_CONTEXT_TAGS = [
    "context",
    "contexts",
    "document",
    "documents",
    "retrieved_context",
    "retrieved_documents",
    "reference",
    "references",
    "source",
    "sources",
    "knowledge",
    "passage",
    "passages",
]

# Line-label openers that introduce retrieved context. Case-insensitive, matched
# at the start of a line, optional markdown bold/heading punctuation allowed.
# NOTE: keep these specific. "information:" alone is too generic and would
# misfire on ordinary prose, so we require the retrieval-flavored phrasings.
_CONTEXT_OPENERS = [
    r"use the following pieces of retrieved context",
    r"use the following pieces of context",
    r"use the following retrieved context",
    r"use the following context",
    r"use the following information to answer",
    r"use the following information",
    r"answer the question based on the context below",
    r"answer the question based only on the following context",
    r"answer based only on the (?:provided|following) context",
    r"based on the context below",
    r"based on the following context",
    r"given the context information",
    r"context information is below",
    r"retrieved context",
    r"retrieved documents",
    r"retrieved chunks",
    r"context documents",
    r"reference material",
    r"relevant documents",
    r"relevant context",
    r"here is the context",
    r"here are the retrieved",
    r"context",          # bare "Context:" label - broad, so it's LAST
    r"contexts",
]

# Question openers, used to locate where context ends / query begins in the
# PAIRED case and to peel a trailing question off an OPENER-only block.
_QUESTION_OPENERS = [
    r"question",
    r"query",
    r"user question",
    r"user query",
    r"user input",
    r"the question is",
]


def _compile_line_label(labels: List[str]) -> re.Pattern:
    # Matches a label at line start, optional leading markdown (#, *, -, spaces),
    # optional surrounding ** for bold, followed by ':' or newline.
    alt = "|".join(labels)
    return re.compile(
        r"(?im)^[ \t>*#\-]*\**\s*(?:" + alt + r")\**\s*:?[ \t]*",
    )


_CONTEXT_OPENER_RE = _compile_line_label(_CONTEXT_OPENERS)
_QUESTION_OPENER_RE = _compile_line_label(_QUESTION_OPENERS)

# Inline "Context: ...." form (label and value on the same line), which the
# line-anchored regex above intentionally doesn't fully capture.
_CONTEXT_INLINE_RE = re.compile(
    r"(?im)^[ \t>*#\-]*\**\s*"
    r"(?:context|retrieved context|reference material|"
    r"use the following (?:pieces of )?(?:retrieved )?context)"
    r"\**\s*:[ \t]*(?P<val>\S.*)$"
)
_QUESTION_INLINE_RE = re.compile(
    r"(?im)^[ \t>*#\-]*\**\s*"
    r"(?:question|query|user question|user query)"
    r"\**\s*:[ \t]*(?P<val>\S.*)$"
)


def _xml_context_regex() -> re.Pattern:
    tags = "|".join(_XML_CONTEXT_TAGS)
    # Non-greedy body, tolerant of attributes and case. DOTALL so context can
    # span multiple lines.
    return re.compile(
        r"(?is)<(" + tags + r")\b[^>]*>(?P<body>.*?)</\1\s*>",
    )


_XML_CONTEXT_RE = _xml_context_regex()

# Dash-wrapped header pattern, e.g. "--- Retrieved Evidence ---",
# "-----Context-----". Generic rather than tied to one exact phrase, since
# different teams will word this differently while keeping the dash-wrapped
# shape. Used only for the marker-prefixed-assistant-message check below.
_DASH_HEADER_RE = re.compile(r"^\s*-{2,}\s*[A-Za-z][\w\s]{0,40}?\s*-{2,}")

# What a "question" looks like for the shape heuristic: a shortish line, usually
# ending in '?' or starting with an interrogative/imperative.
_INTERROGATIVE_START = re.compile(
    r"(?i)^\s*(who|what|when|where|why|how|which|whom|whose|is|are|was|were|do|"
    r"does|did|can|could|should|would|will|explain|describe|summarize|list|"
    r"tell me|give me)\b"
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def extract(
    messages: List[dict],
    llm_extractor: Optional[LLMExtractor] = None,
) -> ExtractionResult:
    """Extract (query, context) from an OpenAI-style messages list.

    Args:
        messages: the request's messages array.
        llm_extractor: optional Tier-2 backend (the judge). If None, we stop at
            Tier 1 and return NONE rather than calling any model.

    Tiers: MARKER (explicit labels/wrappers, incl. marker-prefixed
    assistant-role messages) -> LLM (verbatim-quote-verified fallback) ->
    NONE (skip the check rather than guess). The former structural SHAPE
    heuristic (unlabeled-but-long block) was removed - it was the lowest-
    confidence tier by design, and with a company-standard marker format now
    in place, skipping straight to the verified LLM fallback (or NONE) is
    safer than guessing from structure alone.
    """
    query = _last_user_question(messages)
    searchable = _searchable_text(messages)

    # -- Tier 1: markers ----------------------------------------------------
    marker = _try_markers(messages, searchable, fallback_query=query)
    if marker is not None:
        return marker

    # -- Tier 2: LLM --------------------------------------------------------
    if llm_extractor is not None:
        llm = _try_llm(messages, llm_extractor)
        if llm is not None:
            return llm

    # -- Tier 3: nothing ----------------------------------------------------
    return ExtractionResult(
        query=query,
        context_chunks=[],
        tier=Tier.NONE,
        detail="no context markers, no context-shaped block, LLM found nothing",
        confidence="n/a",
    )


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------
def _last_marker_prefixed_assistant_context(messages: List[dict]) -> Optional[str]:
    """Scan assistant-role messages for ones that START WITH a recognized
    context marker (dash-wrapped header, or one of the standard opener
    phrases). Returns the MOST RECENT match, not all matches and not the
    first - a real generated answer never starts with these markers, so a
    non-matching assistant message (real chat history) is correctly skipped
    without resetting what was found so far."""
    last_match: Optional[str] = None
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = _content_to_text(msg.get("content"))
        if not content:
            continue
        stripped = content.lstrip()
        dash_match = _DASH_HEADER_RE.match(stripped)
        if dash_match:
            last_match = stripped[dash_match.end():].strip()
            continue
        opener_match = _CONTEXT_OPENER_RE.match(stripped)
        if opener_match:
            last_match = stripped[opener_match.end():].lstrip(":").strip()
    return last_match if last_match else None


def _try_markers(
    messages: List[dict],
    text: str,
    fallback_query: Optional[str],
) -> Optional[ExtractionResult]:
    # 1a-pre. Marker-prefixed ASSISTANT-role messages. Some teams inject fresh
    # retrieved context via role="assistant" (e.g. content starting with
    # "--- Retrieved Evidence ---") rather than system/user. We deliberately do
    # NOT scan assistant content in general (that's real generation history,
    # not context - see _searchable_text), but a message that STARTS WITH a
    # recognized context marker is an unambiguous exception: a real generated
    # answer essentially never begins with that literal header text.
    #
    # Multi-turn correctness: if earlier turns also injected context this way,
    # multiple assistant messages may match. We take the LAST (most recent)
    # one relative to the current question, not all of them and not the
    # first - otherwise stale context from a previous turn's question would
    # leak into scoring the current turn's answer.
    assistant_ctx = _last_marker_prefixed_assistant_context(messages)
    if assistant_ctx is not None:
        return ExtractionResult(
            query=fallback_query,
            context_chunks=[assistant_ctx],
            tier=Tier.MARKER,
            detail="marker-prefixed assistant-role message (most recent)",
            confidence="high",
        )

    # 1a. XML wrappers first - most explicit, cleanest bounds.
    xml_bodies = [m.group("body").strip() for m in _XML_CONTEXT_RE.finditer(text)]
    xml_bodies = [b for b in xml_bodies if b]
    if xml_bodies:
        q = _query_outside_xml(text) or fallback_query
        return ExtractionResult(
            query=q,
            context_chunks=xml_bodies,
            tier=Tier.MARKER,
            detail=f"xml wrapper x{len(xml_bodies)}",
            confidence="high",
        )

    # 1b. PAIRED inline labels: "Context: ..." and "Question: ..." same-line.
    ctx_inline = _CONTEXT_INLINE_RE.search(text)
    q_inline = _QUESTION_INLINE_RE.search(text)
    if ctx_inline and q_inline:
        ctx_val, q_val = ctx_inline.group("val").strip(), q_inline.group("val").strip()
        if ctx_val:
            return ExtractionResult(
                query=q_val or fallback_query,
                context_chunks=[ctx_val],
                tier=Tier.MARKER,
                detail="inline paired Context:/Question:",
                confidence="high",
            )

    # 1c. Block opener, handled PER MESSAGE so we don't blur message boundaries.
    #     Two sub-cases matter:
    #       (i)  opener has real content after it in the SAME message -> context
    #            is that content (up to a Question opener if present).
    #       (ii) opener is instruction-only (e.g. a system msg that just says
    #            "Use the following retrieved context to answer") -> the actual
    #            chunks live in a LATER message; use that message as context and
    #            the last user turn as the query.
    for idx, msg in enumerate(messages):
        if msg.get("role") not in ("system", "user"):
            continue
        content = _content_to_text(msg.get("content"))
        if not content:
            continue
        ctx_open = _CONTEXT_OPENER_RE.search(content)
        if not ctx_open:
            continue

        after = content[ctx_open.end():].lstrip(":").strip()
        q_open = _QUESTION_OPENER_RE.search(content, pos=ctx_open.end())

        if q_open:
            # Question label in the same message -> bound context between them.
            context_block = content[ctx_open.end():q_open.start()].lstrip(":").strip()
            query_block = content[q_open.end():].strip()
            if context_block:
                return ExtractionResult(
                    query=query_block or fallback_query,
                    context_chunks=[context_block],
                    tier=Tier.MARKER,
                    detail="block opener + question opener (same message)",
                    confidence="high",
                )

        if _is_meaningful_context(after):
            # (i) real content after the opener in this same message.
            context_block, peeled_q = _peel_trailing_question(after)
            context_block = context_block.strip()
            if context_block:
                return ExtractionResult(
                    query=peeled_q or fallback_query,
                    context_chunks=[context_block],
                    tier=Tier.MARKER,
                    detail="block opener (inline content)",
                    confidence="high",
                )

        # (ii) opener is instruction-only -> look for context in later messages.
        later = _context_from_later_messages(messages, idx, fallback_query)
        if later is not None:
            return later

    # 1d. Inline context only (no explicit question label): take the inline value
    #     as context, use the last user turn as the query.
    if ctx_inline:
        ctx_val = ctx_inline.group("val").strip()
        if ctx_val:
            return ExtractionResult(
                query=fallback_query,
                context_chunks=[ctx_val],
                tier=Tier.MARKER,
                detail="inline Context: only",
                confidence="high",
            )

    return None


def _is_meaningful_context(text: str) -> bool:
    """Distinguish 'real retrieved content follows the opener' from 'the opener
    was just a trailing instruction fragment' (e.g. the word 'to answer.' left
    after 'Use the following retrieved context'). Instruction tails are short and
    lack substance; real context is longer."""
    if not text:
        return False
    stripped = text.strip()
    # Common instruction tails that follow the opener phrase but aren't context.
    instruction_tails = {"to answer.", "to answer", "below.", "below", "to answer the question."}
    if stripped.lower() in instruction_tails:
        return False
    # Require some substance - a real chunk is more than a few words.
    return len(stripped) >= 40


def _context_from_later_messages(
    messages: List[dict],
    opener_idx: int,
    fallback_query: Optional[str],
) -> Optional[ExtractionResult]:
    """The context opener was instruction-only; the chunks are in a subsequent
    message. Use the next non-empty user/system message as context, and the last
    user turn as the query (peeling the question off if context and question
    share that message)."""
    for j in range(opener_idx + 1, len(messages)):
        msg = messages[j]
        if msg.get("role") not in ("system", "user"):
            continue
        content = _content_to_text(msg.get("content")).strip()
        if not content:
            continue

        # If this later message itself carries Context:/Question: labels, the
        # inline paths (1b/1d) would normally catch them; re-check here since we
        # short-circuit into this branch.
        c_inline = _CONTEXT_INLINE_RE.search(content)
        q_inline = _QUESTION_INLINE_RE.search(content)
        if c_inline:
            ctx_val = c_inline.group("val").strip()
            q_val = q_inline.group("val").strip() if q_inline else None
            if ctx_val:
                return ExtractionResult(
                    query=q_val or fallback_query,
                    context_chunks=[ctx_val],
                    tier=Tier.MARKER,
                    detail="opener in earlier message, labeled context in later",
                    confidence="high",
                )

        # Otherwise treat the message as context, peeling a trailing question.
        context_block, peeled_q = _peel_trailing_question(content)
        context_block = context_block.strip()
        if context_block and _is_meaningful_context(context_block):
            return ExtractionResult(
                query=peeled_q or fallback_query,
                context_chunks=[context_block],
                tier=Tier.MARKER,
                detail="opener in earlier message, context in later message",
                confidence="high",
            )
    return None


def _query_outside_xml(text: str) -> Optional[str]:
    """After removing <context>...</context> spans, look for a Question: label or
    a trailing interrogative line to use as the query."""
    stripped = _XML_CONTEXT_RE.sub(" ", text)
    q_inline = _QUESTION_INLINE_RE.search(stripped)
    if q_inline and q_inline.group("val").strip():
        return q_inline.group("val").strip()
    # last non-empty line that looks like a question
    for line in reversed([l.strip() for l in stripped.splitlines() if l.strip()]):
        if line.endswith("?") or _INTERROGATIVE_START.match(line):
            return line
    return None


def _peel_trailing_question(text: str) -> Tuple[str, Optional[str]]:
    lines = [l for l in text.splitlines()]
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.endswith("?") or _INTERROGATIVE_START.match(stripped):
            return "\n".join(lines[:i]).strip(), stripped
        break
    return text, None


# ---------------------------------------------------------------------------
# Tier 2
# ---------------------------------------------------------------------------
def _try_llm(messages: List[dict], extractor: LLMExtractor) -> Optional[ExtractionResult]:
    """Ask the judge to identify query + context. The extractor impl is
    responsible for the verbatim-substring containment check (rejecting spans
    the model paraphrased instead of quoting); it returns None if verification
    fails, and we treat that as 'nothing found'."""
    try:
        out = extractor.extract(messages)
    except Exception:
        # A failing judge call must not take down the whole request - degrade to
        # "no context found" and let the caller skip the check.
        return None
    if not out:
        return None
    q, chunks = out
    chunks = [c for c in (chunks or []) if c and c.strip()]
    if not chunks:
        return None
    return ExtractionResult(
        query=q,
        context_chunks=chunks,
        tier=Tier.LLM,
        detail="llm-assisted, spans verified verbatim",
        confidence="medium",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _content_to_text(content) -> str:
    """OpenAI content can be a plain string or a list of content blocks
    ({'type': 'text', 'text': ...}). Normalize to a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    parts.append(block["content"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _last_user_question(messages: List[dict]) -> Optional[str]:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            text = _content_to_text(msg.get("content")).strip()
            if text:
                return text
    return None


def _searchable_text(messages: List[dict]) -> str:
    """Concatenate system + user content (excluding assistant turns, which are
    generation history, not retrieved context) for marker scanning."""
    parts = []
    for msg in messages:
        if msg.get("role") in ("system", "user"):
            t = _content_to_text(msg.get("content"))
            if t:
                parts.append(t)
    return "\n\n".join(parts)
