"""Regression tests for PromptAlignmentMetric when the judge omits verdicts.

The verdicts prompt asks for one verdict per prompt instruction, but the
schema accepts a list of any length. A judge that returned fewer verdicts
(or none) used to be scored only on what it returned, so an empty list scored
1.0 and passed even in strict mode. Missing verdicts now count as unaligned.

These tests drive the metric with a scripted stub model, so they run without
an LLM provider API key.
"""

import json

import pytest

from deepeval.metrics import PromptAlignmentMetric
from deepeval.metrics.utils import score_qag_verdicts
from deepeval.metrics.base_metric import Verdict
from deepeval.metrics.prompt_alignment.schema import PromptAlignmentVerdict
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase

INSTRUCTIONS = [
    "Reply in all uppercase",
    "Mention the refund window",
    "End with a question",
]


class _ScriptedJudge(DeepEvalBaseLLM):
    """Returns a fixed verdicts payload and records the reason prompt."""

    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.reason_prompts = []
        super().__init__(model="scripted-judge")

    def load_model(self, *args, **kwargs):
        return None

    def _reply(self, prompt: str) -> str:
        if "Reasons for unalignment" in prompt:
            self.reason_prompts.append(prompt)
            return json.dumps({"reason": "scripted reason"})
        return json.dumps({"verdicts": self.verdicts})

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return self._reply(prompt)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self._reply(prompt)

    def get_model_name(self, *args, **kwargs) -> str:
        return "scripted-judge"


def _test_case() -> LLMTestCase:
    return LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="We offer a 30-day full refund at no extra cost.",
    )


YES = {"verdict": "yes"}
NO = {"verdict": "no", "reason": "Not uppercase."}


@pytest.mark.parametrize("async_mode", [True, False])
@pytest.mark.parametrize(
    "verdicts, expected_score",
    [
        ([], 0.0),
        ([YES], 1 / 3),
        ([YES, YES], 2 / 3),
        ([YES, NO, YES], 2 / 3),
        ([YES, YES, YES], 1.0),
    ],
)
def test_missing_verdicts_count_as_unaligned(
    verdicts, expected_score, async_mode
):
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS,
        model=_ScriptedJudge(verdicts),
        async_mode=async_mode,
    )
    metric.measure(_test_case())

    assert metric.score == pytest.approx(expected_score)


@pytest.mark.parametrize("verdicts", [[], [YES], [YES, YES]])
def test_strict_mode_fails_when_verdicts_are_missing(verdicts):
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS,
        model=_ScriptedJudge(verdicts),
        strict_mode=True,
    )
    metric.measure(_test_case())

    assert metric.score == 0
    assert metric.is_successful() is False


def test_reason_prompt_mentions_missing_verdicts():
    judge = _ScriptedJudge([YES])
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS, model=judge
    )
    metric.measure(_test_case())

    assert len(judge.reason_prompts) == 1
    assert "no verdict for 2 of 3" in judge.reason_prompts[0]


def test_complete_verdicts_do_not_add_missing_reason():
    judge = _ScriptedJudge([YES, YES, YES])
    metric = PromptAlignmentMetric(
        prompt_instructions=INSTRUCTIONS, model=judge
    )
    metric.measure(_test_case())

    assert "no verdict for" not in judge.reason_prompts[0]


class _Metric:
    strict_mode = False
    threshold = 0.5


def _verdicts(*values):
    return [PromptAlignmentVerdict(verdict=v) for v in values]


def test_score_qag_verdicts_without_expected_is_unchanged():
    assert score_qag_verdicts(_Metric(), [], passing=(Verdict.YES,)) == 1
    assert (
        score_qag_verdicts(
            _Metric(), _verdicts("yes", "no"), passing=(Verdict.YES,)
        )
        == 0.5
    )


def test_score_qag_verdicts_expected_counts_missing_as_failing():
    passing = (Verdict.YES,)
    assert score_qag_verdicts(_Metric(), [], passing=passing, expected=3) == 0
    assert score_qag_verdicts(
        _Metric(), _verdicts("yes"), passing=passing, expected=3
    ) == pytest.approx(1 / 3)
    # A judge returning extra verdicts is scored over what it returned.
    assert score_qag_verdicts(
        _Metric(), _verdicts("yes", "yes", "no"), passing=passing, expected=2
    ) == pytest.approx(2 / 3)
