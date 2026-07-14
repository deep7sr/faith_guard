"""
Judge model plumbing.

One place that owns "how do we talk to the judge model", so the faithfulness
scorer and the Tier-3 extractor share a single client instead of each holding
their own. Model identity comes from env vars so swapping Groq (testing) for a
local vLLM/Ollama endpoint (production, no data egress) is config, not a code
change.

Contains:
  - JudgeLLM        : a deepeval-compatible LLM backed by litellm + instructor
                      (instructor gives us the schema-constrained JSON that
                      deepeval's FaithfulnessMetric needs; without it these
                      metrics fail with a confusing AttributeError).
  - LLMContextExtractor : Tier-3 extraction. Asks the judge to quote the query
                      and context verbatim, then verifies each span is a real
                      substring before trusting it - a paraphrase is a hallucination
                      signal, so we drop it.

deepeval/instructor are real (not deferred) imports here - this file is only
ever loaded lazily by faithfulness_guardrail.py when scoring is actually
needed, so there's no benefit to deferring again inside this file, and
deferring the DeepEvalBaseLLM import specifically broke real inheritance
(see JudgeLLM below - it must inherit at class-definition time, a runtime
"pretend" subclass does not satisfy deepeval's isinstance check).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import List, Optional, Tuple

import instructor
from litellm import acompletion, completion
from deepeval.models.base_model import DeepEvalBaseLLM

# Env-driven model config. Keep judge and agent as SEPARATE settings even if they
# point at the same model during testing, so they can diverge later.
# Default matches the 'judge-model' alias's underlying string in config.yaml.
# Called directly via litellm (not routed back through the proxy's own model
# alias) - simpler and more reliable for now. Revisit routing judge calls
# through the proxy itself later, once this baseline is proven, for unified
# logging of judge traffic alongside agent traffic.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "groq/openai/gpt-oss-20b")
# gpt-oss models are reasoning models; capping effort matters for latency since
# the retry loop can call the judge several times per request. Empty string
# means "don't pass the param" (e.g. for non-reasoning judge models).
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "low")


def _extra_judge_kwargs() -> dict:
    return {"reasoning_effort": JUDGE_REASONING_EFFORT} if JUDGE_REASONING_EFFORT else {}
JUDGE_MAX_RETRIES = int(os.getenv("JUDGE_MAX_RETRIES", "4"))
JUDGE_BASE_DELAY = float(os.getenv("JUDGE_BASE_DELAY", "1.5"))


def _is_rate_limit(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    if "ratelimit" in name:
        return True
    # litellm sometimes wraps provider 429s; check message/status as a fallback.
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


class JudgeLLM(DeepEvalBaseLLM):
    """deepeval-compatible LLM backed by litellm + instructor. Must genuinely
    inherit DeepEvalBaseLLM - deepeval's metrics isinstance-check the model
    argument, so anything short of real inheritance is rejected at metric
    construction with 'Unsupported type for model: ...'."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or JUDGE_MODEL
        self._acompletion = acompletion
        self._completion = completion
        # mode=JSON, not the default TOOLS: Groq's own docs recommend this -
        # forced tool-calling is unreliable on several Groq-hosted models
        # (including reasoning models like gpt-oss), producing correct JSON
        # content but failing the "did the model call a tool" check.
        self._sync_client = instructor.from_litellm(completion, mode=instructor.Mode.JSON)
        self._async_client = instructor.from_litellm(acompletion, mode=instructor.Mode.JSON)

    # --- DeepEvalBaseLLM interface -------------------------------------------
    def load_model(self):
        return self.model

    def get_model_name(self) -> str:
        return f"JudgeLLM({self.model})"

    def generate(self, prompt: str, schema=None):
        return self._with_backoff_sync(prompt, schema)

    async def a_generate(self, prompt: str, schema=None):
        return await self._with_backoff_async(prompt, schema)

    # --- internals -----------------------------------------------------------
    def _with_backoff_sync(self, prompt: str, schema):
        last = None
        for attempt in range(JUDGE_MAX_RETRIES):
            try:
                if schema is not None:
                    return self._sync_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema,
                        **_extra_judge_kwargs(),
                    )
                resp = self._completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **_extra_judge_kwargs(),
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa
                last = e
                if _is_rate_limit(e) and attempt < JUDGE_MAX_RETRIES - 1:
                    time.sleep(JUDGE_BASE_DELAY * (2 ** attempt))
                    continue
                raise
        raise last  # pragma: no cover

    async def _with_backoff_async(self, prompt: str, schema):
        last = None
        for attempt in range(JUDGE_MAX_RETRIES):
            try:
                if schema is not None:
                    return await self._async_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        response_model=schema,
                        **_extra_judge_kwargs(),
                    )
                resp = await self._acompletion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    **_extra_judge_kwargs(),
                )
                return resp.choices[0].message.content
            except Exception as e:  # noqa
                last = e
                if _is_rate_limit(e) and attempt < JUDGE_MAX_RETRIES - 1:
                    await asyncio.sleep(JUDGE_BASE_DELAY * (2 ** attempt))
                    continue
                raise
        raise last  # pragma: no cover


# ---------------------------------------------------------------------------
# Tier-3 extractor
# ---------------------------------------------------------------------------
_EXTRACT_PROMPT = """You are a precise text extraction tool for a RAG system.

Below is a chat request's messages. Identify:
1. The user's actual QUESTION (the thing they want answered).
2. Any RETRIEVED CONTEXT / reference material that was inserted for the model to
   ground its answer on (documents, passages, facts). This is NOT the system
   persona or instructions.

Rules:
- Quote EXACTLY. Copy spans verbatim from the text. Do not paraphrase, summarize,
  reword, or add anything.
- If there is no retrieved context (this is just a normal chat), return empty
  context.
- Return ONLY valid JSON, no prose, in this exact shape:
  {"question": "<verbatim question>", "context": ["<verbatim chunk>", ...]}

MESSAGES:
---
{messages_text}
---
JSON:"""


class LLMContextExtractor:
    """Implements the extraction.LLMExtractor protocol. Returns None when the
    judge's spans can't be verified as verbatim substrings (a signal it
    paraphrased/hallucinated rather than extracted)."""

    def __init__(self, judge: Optional[JudgeLLM] = None):
        self.judge = judge or JudgeLLM()

    def extract(self, messages: List[dict]) -> Optional[Tuple[str, List[str]]]:
        text = _messages_to_text(messages)
        prompt = _EXTRACT_PROMPT.replace("{messages_text}", text)
        raw = self.judge.generate(prompt, schema=None)  # plain JSON string
        parsed = _safe_json(raw)
        if not parsed:
            return None

        question = (parsed.get("question") or "").strip()
        chunks = [c.strip() for c in (parsed.get("context") or []) if c and c.strip()]

        # Verify verbatim: every returned chunk must actually appear in the source.
        # Normalize whitespace for the containment check so minor reflow doesn't
        # cause false rejects, but reject genuine paraphrases.
        haystack = _normalize_ws(text)
        verified = [c for c in chunks if _normalize_ws(c) in haystack]
        if not verified:
            return None

        if question and _normalize_ws(question) not in haystack:
            # Question wasn't quoted verbatim; fall back to the last user turn by
            # returning it as-is is unsafe, so signal "found context, no reliable
            # question" by letting the caller's fallback_query fill in. We return
            # an empty question string; extraction.py keeps its own fallback.
            question = ""

        return (question, verified)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _messages_to_text(messages: List[dict]) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        lines.append(f"[{role}]\n{content}")
    return "\n\n".join(lines)


def _safe_json(raw) -> Optional[dict]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    # Strip ```json fences if the model added them.
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    try:
        return json.loads(s)
    except Exception:
        # Last resort: grab the first {...} block.
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()
