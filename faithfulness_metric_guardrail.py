"""
Faithfulness guardrail - DeepEval's built-in FaithfulnessMetric.

Standalone, single-purpose guardrail. Registers as its own guardrail_name in
LiteLLM config, independently selectable per virtual key. Does not import from
or reference the GEval-based guardrail in any way - the only files shared with
that guardrail are extraction.py and judge.py, which are generic infrastructure
(query/context extraction, judge-model client), not scoring logic.

Mechanism: FaithfulnessMetric decomposes the answer into discrete claims, then
checks each claim for CONTRADICTION against the retrieved context. A claim that
the context is simply silent on (neither confirms nor denies) is NOT penalized.
See geval_groundedness_guardrail.py for the alternate implementation, which
penalizes unsupported claims as well as contradictions.

Pipeline, all inside async_post_call_success_hook:
    extract(query, context) from the original request
        -> no context found: SKIP (not a RAG call), return answer as-is
    score the answer against the context with DeepEval FaithfulnessMetric
        -> pass: return answer as-is
        -> fail: retry loop (up to N):
               regenerate with the previous answer + judge's reason as
               corrective feedback, re-score; deliver the first grounded answer
        -> still failing after N retries: block, return a sanitized fallback
           message; log full detail internally.

async_pre_call_hook rejects stream=true (a post-call guardrail cannot act on
tokens already delivered to the client).
"""
from __future__ import annotations

import os
from typing import List, Literal, Optional, Protocol

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth

from extraction import extract, Tier, ExtractionResult

# ---------------------------------------------------------------------------
# Config - fully namespaced to this guardrail. Deliberately NOT shared with
# the GEval guardrail's config, so tuning one can never accidentally affect
# the other.
# ---------------------------------------------------------------------------
FAITHFULNESS_METRIC_THRESHOLD = float(os.getenv("FAITHFULNESS_METRIC_THRESHOLD", "0.7"))
FAITHFULNESS_METRIC_MAX_RETRIES = int(os.getenv("FAITHFULNESS_METRIC_MAX_RETRIES", "3"))
FAITHFULNESS_METRIC_FALLBACK_MESSAGE = os.getenv(
    "FAITHFULNESS_METRIC_FALLBACK_MESSAGE",
    "I'm not able to provide a reliable answer grounded in the available "
    "information right now. Please rephrase your question or try again.",
)
# The real underlying litellm model string for regeneration calls - NOT a
# proxy model_list alias. The retry loop calls litellm.acompletion() directly,
# bypassing the proxy's own router, so it has no way to resolve a proxy alias.
AGENT_MODEL = os.getenv("AGENT_MODEL", "groq/llama-3.1-8b-instant")
# Metadata flag unique to THIS guardrail's regeneration calls, so they skip
# only this guardrail on themselves - never interacts with the GEval guardrail
# even if both happen to be attached to the same key.
SKIP_FLAG = "__faithfulness_metric_guard_skip__"


# ---------------------------------------------------------------------------
# Injectable interfaces
# ---------------------------------------------------------------------------
class ScoreResult:
    def __init__(self, score: float, passed: bool, reason: str = ""):
        self.score = score
        self.passed = passed
        self.reason = reason


class Scorer(Protocol):
    async def score(self, query: str, context: List[str], answer: str) -> ScoreResult: ...


class Regenerator(Protocol):
    async def regenerate(
        self, data: dict, query: str, context: List[str], answer: str, reason: str
    ) -> str: ...


# ---------------------------------------------------------------------------
# Scorer: DeepEval FaithfulnessMetric (deferred imports)
# ---------------------------------------------------------------------------
class DeepEvalFaithfulnessMetricScorer:
    def __init__(self, threshold: float = FAITHFULNESS_METRIC_THRESHOLD, judge=None):
        self.threshold = threshold
        from deepeval.metrics import FaithfulnessMetric  # deferred
        from judge import JudgeLLM

        self._judge = judge or JudgeLLM()
        # Build once, reuse. include_reason=True gives us the corrective text.
        self._metric = FaithfulnessMetric(
            threshold=threshold, model=self._judge, include_reason=True
        )

    async def score(self, query: str, context: List[str], answer: str) -> ScoreResult:
        from deepeval.test_case import LLMTestCase

        tc = LLMTestCase(input=query, actual_output=answer, retrieval_context=context)
        await self._metric.a_measure(tc, _show_indicator=False)
        score = float(self._metric.score or 0.0)
        return ScoreResult(
            score=score,
            passed=score >= self.threshold,
            reason=self._metric.reason or "",
        )


# ---------------------------------------------------------------------------
# Regenerator: call the agent model directly via litellm, with a corrective
# instruction and this guardrail's skip flag set.
# ---------------------------------------------------------------------------
class LiteLLMRegenerator:
    async def regenerate(
        self, data: dict, query: str, context: List[str], answer: str, reason: str
    ) -> str:
        from litellm import acompletion

        corrective = _build_corrective_messages(data.get("messages", []), context, answer, reason)
        params = {
            k: v
            for k, v in data.items()
            if k in ("model", "temperature", "top_p", "max_tokens", "user")
        }
        # Override with the real model string - data["model"] is the proxy's
        # alias, which acompletion() (bypassing the proxy's router) can't resolve.
        params["model"] = AGENT_MODEL
        params["messages"] = corrective
        params["stream"] = False
        meta = dict(params.get("metadata") or {})
        meta[SKIP_FLAG] = True
        params["metadata"] = meta

        resp = await acompletion(**params)
        return _answer_text(resp)


def _build_corrective_messages(
    original_messages: List[dict], context: List[str], answer: str, reason: str
) -> List[dict]:
    ctx_block = "\n\n".join(context)
    corrective = {
        "role": "system",
        "content": (
            "You previously gave this answer:\n"
            f'"{answer}"\n\n'
            "That answer was WRONG - it was not grounded in the provided context "
            f"and contained unsupported claims. Specifically: {reason}\n\n"
            "Regenerate a corrected answer using ONLY the information in the "
            "context below. Do not repeat the unsupported claims from your "
            "previous answer. If the context does not contain the information "
            "needed to answer, say so explicitly rather than guessing.\n\n"
            f"Context:\n{ctx_block}"
        ),
    }
    # Prepend the corrective system message; keep the original conversation.
    return [corrective] + list(original_messages)


# ---------------------------------------------------------------------------
# The guardrail
# ---------------------------------------------------------------------------
class GuardDecision:
    """What happened, for logging/audit."""
    def __init__(self, verdict: str, tier: str, attempts: int, scores=None, detail: str = ""):
        self.verdict = verdict            # "skipped" | "passed" | "passed_after_retry" | "blocked"
        self.tier = tier
        self.attempts = attempts
        self.scores = scores if scores is not None else []
        self.detail = detail


class FaithfulnessMetricGuardrail(CustomGuardrail):
    def __init__(
        self,
        scorer: Optional[Scorer] = None,
        regenerator: Optional[Regenerator] = None,
        llm_extractor=None,
        threshold: float = FAITHFULNESS_METRIC_THRESHOLD,
        max_retries: int = FAITHFULNESS_METRIC_MAX_RETRIES,
        **kwargs,
    ):
        self.threshold = threshold
        self.max_retries = max_retries
        self._scorer = scorer
        self._regenerator = regenerator
        self._llm_extractor = llm_extractor  # Tier-3; optional
        self.last_decision: Optional[GuardDecision] = None
        super().__init__(**kwargs)

    # --- lazy default wiring ------------------------------------------------
    def _get_scorer(self) -> Scorer:
        if self._scorer is None:
            self._scorer = DeepEvalFaithfulnessMetricScorer(threshold=self.threshold)
        return self._scorer

    def _get_regenerator(self) -> Regenerator:
        if self._regenerator is None:
            self._regenerator = LiteLLMRegenerator()
        return self._regenerator

    def _get_llm_extractor(self):
        if self._llm_extractor is None and os.getenv("FAITHFULNESS_METRIC_LLM_EXTRACT", "1") == "1":
            try:
                from judge import LLMContextExtractor
                self._llm_extractor = LLMContextExtractor()
            except Exception:
                self._llm_extractor = None
        return self._llm_extractor

    # --- pre-call: block streaming -------------------------------------------
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion", "text_completion", "embeddings", "image_generation",
            "moderation", "audio_transcription", "pass_through_endpoint", "rerank",
        ],
    ):
        if _is_skip(data):
            return data
        if data.get("stream") is True:
            raise ValueError(
                "This key has the faithfulness-metric guardrail enabled, which "
                "requires stream=false. Set stream to false, or contact your "
                "admin if you need streaming."
            )
        return data

    # --- post-call: the pipeline ---------------------------------------------
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response,
    ):
        if _is_skip(data):
            return response

        messages = data.get("messages", [])
        answer = _answer_text(response)
        if not answer:
            return response

        ex: ExtractionResult = extract(messages, llm_extractor=self._get_llm_extractor())
        if not ex.has_context:
            return self._finish(response, GuardDecision(
                verdict="skipped", tier=ex.tier.name, attempts=0,
                detail=f"no extractable context ({ex.detail})",
            ))

        scorer = self._get_scorer()
        scores: List[float] = []

        result = await scorer.score(ex.query, ex.context_chunks, answer)
        scores.append(result.score)
        if result.passed:
            return self._finish(response, GuardDecision(
                verdict="passed", tier=ex.tier.name, attempts=0, scores=scores,
                detail=f"score={result.score:.3f} reason={result.reason!r}",
            ))

        regen = self._get_regenerator()
        current_answer = answer
        current_reason = result.reason
        for attempt in range(1, self.max_retries + 1):
            new_answer = await regen.regenerate(
                data, ex.query, ex.context_chunks, current_answer, current_reason
            )
            result = await scorer.score(ex.query, ex.context_chunks, new_answer)
            scores.append(result.score)
            if result.passed:
                _set_answer_text(response, new_answer)
                return self._finish(response, GuardDecision(
                    verdict="passed_after_retry", tier=ex.tier.name, attempts=attempt,
                    scores=scores, detail=f"passed on retry {attempt}, score={result.score:.3f}",
                ))
            current_answer = new_answer
            current_reason = result.reason

        _set_answer_text(response, FAITHFULNESS_METRIC_FALLBACK_MESSAGE)
        return self._finish(response, GuardDecision(
            verdict="blocked", tier=ex.tier.name, attempts=self.max_retries,
            scores=scores,
            detail=(
                f"failed after {self.max_retries} retries; "
                f"scores={['%.3f' % s for s in scores]}; last_reason={current_reason!r}"
            ),
        ))

    def _finish(self, response, decision: "GuardDecision"):
        self.last_decision = decision
        print(
            f"[faithfulness-metric-guard] verdict={decision.verdict} tier={decision.tier} "
            f"attempts={decision.attempts} scores={['%.3f' % s for s in decision.scores]} "
            f"detail={decision.detail}",
            flush=True,
        )
        return response


# ---------------------------------------------------------------------------
# response helpers (litellm ModelResponse is ChatCompletion-shaped)
# ---------------------------------------------------------------------------
def _answer_text(response) -> str:
    try:
        choice = response.choices[0]
        msg = getattr(choice, "message", None) or choice.get("message")
        content = getattr(msg, "content", None) if msg is not None else None
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        return content or ""
    except Exception:
        return ""


def _set_answer_text(response, text: str) -> None:
    choice = response.choices[0]
    msg = getattr(choice, "message", None)
    if msg is not None and hasattr(msg, "content"):
        msg.content = text
    elif isinstance(choice, dict):
        choice.setdefault("message", {})["content"] = text
    else:
        try:
            choice["message"]["content"] = text
        except Exception:
            pass


def _is_skip(data: dict) -> bool:
    meta = data.get("metadata") or {}
    return bool(meta.get(SKIP_FLAG))
