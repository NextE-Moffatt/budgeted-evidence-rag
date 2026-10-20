from __future__ import annotations

from time import perf_counter

from .retrieval import BM25Retriever
from .schema import Dataset


def audit_retrieval(dataset: Dataset, split: str, ks: tuple[int, ...]) -> dict:
    if split not in ("train", "calibration", "validation"):
        raise ValueError("retrieval feasibility audit excludes the confirmatory test partition")
    if not ks or any(type(k) is not int or k < 1 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError("retrieval cutoffs must be unique positive integers")
    examples = sorted(
        (e for e in dataset.examples if e.split == split), key=lambda e: e.question.id
    )
    if not examples:
        raise ValueError("empty requested audit partition")
    rows = []
    started = perf_counter()
    for example in examples:
        question = example.question
        retriever = BM25Retriever(tuple(dataset.passages[pid] for pid in question.candidate_ids))
        # No labels enter retrieval: gold is read only after ranking has completed.
        ranked = retriever.search(question.text, max(ks), set())
        gold = set(example.gold_passage_ids)
        metrics = {}
        for k in sorted(ks):
            observed = {p.id for p in ranked[:k]}
            metrics[str(k)] = {
                "passage_recall": len(observed & gold) / len(gold) if gold else None,
                "all_support_retrieved": gold <= observed if gold else None,
                "retrieved_count": len(observed),
            }
        rows.append(
            {
                "question_id": question.id,
                "group_id": example.group_id,
                "candidate_count": len(question.candidate_ids),
                "gold_support_count": len(gold),
                "cutoffs": metrics,
            }
        )
    cutoffs = {}
    for k in sorted(ks):
        eligible = [r["cutoffs"][str(k)] for r in rows if r["gold_support_count"]]
        cutoffs[str(k)] = {
            "eligible_questions": len(eligible),
            "mean_passage_recall": (
                sum(r["passage_recall"] for r in eligible) / len(eligible) if eligible else None
            ),
            "all_support_retrieved_rate": (
                sum(r["all_support_retrieved"] for r in eligible) / len(eligible)
                if eligible
                else None
            ),
        }
    return {
        "result_kind": "retrieval_only_feasibility_not_answer_quality_or_stopping_improvement",
        "split": split,
        "questions": len(rows),
        "llm_calls": 0,
        "query_policy": "original_question_once_no_reformulation",
        "retriever": "bm25s-lucene-k1=1.5-b=0.75-stopwords=en-no-stemmer-positive-scores-only",
        "setting": dataset.provenance.get("setting", "synthetic_fixture"),
        "wall_seconds": perf_counter() - started,
        "cutoffs": cutoffs,
        "rows": rows,
        "limitations": [
            "Passage support recall is not sentence support/joint F1 or answer F1.",
            "Distractor candidate sets include benchmark-selected supporting paragraphs.",
            "This is single-query retrieval, not the future multi-round query policy.",
            "No continuation benefit/harm is measured without reader answers.",
        ],
    }
