from dataclasses import fields, replace

import pytest

from evidence_rag.evaluation import evaluate, replay
from evidence_rag.metrics import answer_scores
from evidence_rag.policies import StopPolicy
from evidence_rag.schema import Cost, PublicQuestion, PublicState, sum_cost


def test_first_stop_and_continuation_harm(dataset, traces):
    example, trace = dataset.examples[0], traces[0]
    confidence = replay(example, trace, StopPolicy("confidence", 3))
    sufficient = replay(example, trace, StopPolicy("sufficiency", 3))
    fixed = replay(example, trace, StopPolicy("fixed", 3))
    assert (confidence["stopped_round"], confidence["em"]) == (1, 0)
    assert (sufficient["stopped_round"], sufficient["em"]) == (2, 1)
    assert (fixed["stopped_round"], fixed["em"]) == (3, 0)
    assert confidence["future_improvement_available_diagnostic"]
    assert sufficient["future_harm_available_diagnostic"]
    assert sufficient["cost"]["reader_calls"] == 2


def test_no_gold_or_future_in_runtime_types():
    assert {f.name for f in fields(PublicQuestion)} == {"id", "text", "candidate_ids"}
    assert {f.name for f in fields(PublicState)} == {
        "question",
        "round",
        "evidence_ids",
        "proposals",
    }


def test_changed_gold_does_not_change_stop(dataset, traces):
    example = dataset.examples[0]
    policy = StopPolicy("sufficiency", 3)
    first = replay(example, traces[0], policy)
    second = replay(
        replace(example, answers=("impossible",), gold_passage_ids=()), traces[0], policy
    )
    assert first["stopped_round"] == second["stopped_round"]
    assert first["answer"] == second["answer"]


@pytest.mark.parametrize("mode", ["fixed", "confidence", "sufficiency", "stability"])
def test_hard_round_budget(dataset, traces, mode):
    for example, trace in zip(dataset.examples, traces, strict=True):
        result = replay(example, trace, StopPolicy(mode, 1))
        assert result["stopped_round"] == 1
        assert result["cost"]["reader_calls"] == 1


def test_failures_abstentions_stay_in_denominator(dataset, traces):
    summary, _ = evaluate(dataset, traces, "train", StopPolicy("sufficiency", 3))
    assert summary["questions"] == 5
    assert summary["answer_em"] == pytest.approx(3 / 5)
    assert summary["answered_coverage"] == pytest.approx(3 / 5)
    assert summary["selected_state_failure_rate"] == pytest.approx(1 / 5)
    assert summary["deployed_prefix_cost_estimate"]["input_tokens"] is None
    assert summary["full_trajectory_collection_cost"]["reader_calls"] == 12
    assert summary["result_kind"] == "software_smoke"


def test_partial_scope_is_explicit(dataset, traces):
    summary, _ = evaluate(dataset, traces[:1], "train", StopPolicy("fixed", 3))
    assert summary["questions"] == 1
    assert summary["missing_trace_questions"] == 4


def test_empty_partition_rejected(dataset, traces):
    with pytest.raises(ValueError, match="no traces"):
        evaluate(dataset, traces, "test", StopPolicy("fixed", 3))


def test_unknown_usage_not_zero():
    assert sum_cost([Cost(reader_calls=1)])["input_tokens"] is None
    assert sum_cost([Cost(1, 1, 20, 5), Cost(1, 1, 10, 2)])["input_tokens"] == 30


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reader_calls": -1},
        {"reader_calls": True},
        {"input_tokens": 1.5},
        {"reader_ms": float("nan")},
        {"retrieval_ms": -1},
    ],
)
def test_invalid_cost_rejected(kwargs):
    with pytest.raises(ValueError):
        Cost(**kwargs)


@pytest.mark.parametrize(
    "answer,gold,expected",
    [
        ("The Westland!", ("Westland",), (1, 1)),
        ("Westland city", ("Westland",), (0, 2 / 3)),
        (None, ("Westland",), (0, 0)),
        ("yes", ("yes please",), (0, 0)),
        ("Bellhaven", ("Westland", "Bellhaven"), (1, 1)),
    ],
)
def test_answer_metrics(answer, gold, expected):
    assert answer_scores(answer, gold) == expected
