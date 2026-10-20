from dataclasses import replace

import pytest

from evidence_rag.retrieval_audit import audit_retrieval


def test_retrieval_audit_labels_and_runtime_separated(dataset):
    report = audit_retrieval(dataset, "train", (1, 2, 3))
    assert report["llm_calls"] == 0
    assert report["questions"] == 5
    assert report["cutoffs"]["3"]["eligible_questions"] == 4
    assert "not_answer_quality" in report["result_kind"]
    assert all(r["cutoffs"]["1"]["retrieved_count"] <= 1 for r in report["rows"])
    assert (
        report["cutoffs"]["3"]["mean_passage_recall"]
        >= (report["cutoffs"]["1"]["mean_passage_recall"])
    )
    altered = replace(
        dataset, examples=tuple(replace(e, answers=("different",)) for e in dataset.examples)
    )
    assert audit_retrieval(altered, "train", (1, 2, 3))["cutoffs"] == report["cutoffs"]


@pytest.mark.parametrize(
    "split,ks", [("test", (1,)), ("train", (0,)), ("train", (1, 1)), ("train", ())]
)
def test_invalid_audit_scope_rejected(dataset, split, ks):
    with pytest.raises(ValueError):
        audit_retrieval(dataset, split, ks)
