from __future__ import annotations

from time import perf_counter

from .metrics import answer_scores, percentile
from .policies import StopPolicy
from .schema import Dataset, Example, PublicState, Trace, sum_cost


def replay(example: Example, trace: Trace, policy: StopPolicy) -> dict:
    evidence: list[str] = []
    proposals = []
    controller_ms = 0.0
    for step in trace.steps:
        evidence.extend(step.retrieved_ids)
        proposals.append(step.proposal)
        state = PublicState(example.question, step.round, tuple(evidence), tuple(proposals))
        started = perf_counter()
        stop = policy.stop(state)
        controller_ms += (perf_counter() - started) * 1000
        if stop:
            break
    em, f1 = answer_scores(step.proposal.answer, example.answers)
    support = set(example.gold_passage_ids)
    future = [
        answer_scores(s.proposal.answer, example.answers)[1] for s in trace.steps[step.round :]
    ]
    current_cost = sum_cost([s.cost for s in trace.steps[: step.round]])
    return {
        "question_id": example.question.id,
        "group_id": example.group_id,
        "stopped_round": step.round,
        "stop_reason": (
            "budget" if step.round >= policy.max_rounds else "policy" if stop else "trace_exhausted"
        ),
        "answer": step.proposal.answer,
        "citations": list(step.proposal.citations),
        "em": em,
        "f1": f1,
        "answered": step.proposal.answer is not None,
        "selected_state_failed": step.status != "ok",
        "prefix_had_error": any(s.status != "ok" for s in trace.steps[: step.round]),
        "gold_passage_recall": len(support & set(evidence)) / len(support) if support else None,
        "future_improvement_available_diagnostic": any(value > f1 for value in future),
        "future_harm_available_diagnostic": any(value < f1 for value in future),
        "cost": current_cost,
        "controller_calls": step.round,
        "controller_llm_calls": 0,
        "controller_cpu_ms": controller_ms,
        "index_ms": trace.index_ms,
        "prefix_latency_ms_estimate": (
            trace.index_ms
            + current_cost["retrieval_ms"]
            + current_cost["reader_ms"]
            + controller_ms
        ),
    }


def evaluate(
    dataset: Dataset, traces: tuple[Trace, ...], split: str, policy: StopPolicy
) -> tuple[dict, list[dict]]:
    examples = {e.question.id: e for e in dataset.examples if e.split == split}
    selected = [t for t in traces if t.question_id in examples]
    if not selected:
        raise ValueError("no traces for requested split")
    if len({t.question_id for t in selected}) != len(selected):
        raise ValueError("duplicate trajectories")
    rows = [replay(examples[t.question_id], t, policy) for t in selected]
    size = len(rows)
    answered = sum(r["answered"] for r in rows)
    supports = [r["gold_passage_recall"] for r in rows if r["gold_passage_recall"] is not None]
    prefix_costs = [
        s.cost
        for t, row in zip(selected, rows, strict=True)
        for s in t.steps[: row["stopped_round"]]
    ]
    latency = [r["prefix_latency_ms_estimate"] for r in rows]
    summary = {
        "policy": policy.name,
        "split": split,
        "result_kind": (
            "software_smoke"
            if dataset.provenance.get("kind") == "synthetic_fixture"
            else "fixed_path_prefix_replay_not_online_experiment"
        ),
        "questions": size,
        "split_total_questions": len(examples),
        "missing_trace_questions": len(examples) - size,
        "scope": "traced questions only; missing/unstarted questions reported separately",
        "answer_em": sum(r["em"] for r in rows) / size,
        "answer_f1": sum(r["f1"] for r in rows) / size,
        "answered_coverage": answered / size,
        "selective_em_error": (
            sum(r["answered"] and not r["em"] for r in rows) / answered if answered else None
        ),
        "gold_passage_recall": sum(supports) / len(supports) if supports else None,
        "selected_state_failure_rate": sum(r["selected_state_failed"] for r in rows) / size,
        "prefix_error_rate": sum(r["prefix_had_error"] for r in rows) / size,
        "deployed_prefix_cost_estimate": sum_cost(prefix_costs),
        "full_trajectory_collection_cost": sum_cost([s.cost for t in selected for s in t.steps]),
        "controller_calls": sum(r["controller_calls"] for r in rows),
        "controller_llm_calls": 0,
        "controller_cpu_ms": sum(r["controller_cpu_ms"] for r in rows),
        "index_ms": sum(t.index_ms for t in selected),
        "prefix_latency_ms_estimate_p50": percentile(latency, 0.5),
        "prefix_latency_ms_estimate_p95": percentile(latency, 0.95),
        "future_improvement_available_rate_diagnostic": (
            sum(r["future_improvement_available_diagnostic"] for r in rows) / size
        ),
        "future_harm_available_rate_diagnostic": (
            sum(r["future_harm_available_diagnostic"] for r in rows) / size
        ),
        "confidence_intervals": None,
    }
    return summary, rows
