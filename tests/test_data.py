import json
from dataclasses import replace

import pytest

from evidence_rag.data import dataset_dict, load_dataset, load_traces, trace_dict, write_json
from evidence_rag.hotpot import convert_hotpot
from evidence_rag.splits import audit_split_leakage, grouped_split


def test_dataset_round_trip(dataset, tmp_path):
    path = tmp_path / "dataset.json"
    write_json(path, dataset_dict(dataset))
    assert load_dataset(path) == dataset


def test_trace_round_trip(dataset, traces, tmp_path):
    path = tmp_path / "roundtrip.json"
    write_json(path, trace_dict(traces, dataset))
    assert load_traces(path, dataset) == traces


@pytest.mark.parametrize("mutation", ["duplicate", "repeat", "future_citation", "fingerprint"])
def test_invalid_trace_rejected(dataset, traces, tmp_path, mutation):
    raw = trace_dict(traces, dataset)
    if mutation == "duplicate":
        raw["traces"].append(raw["traces"][0])
    elif mutation == "repeat":
        raw["traces"][0]["steps"][1]["retrieved_ids"] = ["p1"]
    elif mutation == "future_citation":
        raw["traces"][0]["steps"][0]["proposal"]["citations"] = ["p2"]
    else:
        raw["dataset_sha256"] = "wrong"
    path = tmp_path / "bad.json"
    write_json(path, raw)
    with pytest.raises(ValueError):
        load_traces(path, dataset)


def test_shared_sources_grouped_and_deterministic(dataset):
    first, audit = grouped_split(dataset, "seed")
    second, _ = grouped_split(replace(dataset, examples=dataset.examples[::-1]), "seed")
    assert {e.question.id: (e.group_id, e.split) for e in first.examples} == {
        e.question.id: (e.group_id, e.split) for e in second.examples
    }
    assert audit["components"] == 1
    assert audit["largest_component"] == 5
    assert len(audit["empty_splits"]) == 3
    audit_split_leakage(first)


def test_leakage_rejected(dataset):
    examples = (replace(dataset.examples[0], split="test"), *dataset.examples[1:])
    with pytest.raises(ValueError, match="leakage"):
        audit_split_leakage(replace(dataset, examples=examples))


def test_hotpot_conversion_without_gold_source_selection(tmp_path):
    rows = [
        {
            "_id": f"q{n}",
            "question": "Where is the fictional institute?",
            "answer": "Westland",
            "context": [
                ["Shared", ["A fictional institute."]],
                [f"Distractor{n}", ["Unrelated invented passage."]],
            ],
            "supporting_facts": [["Shared", 0]],
        }
        for n in range(4)
    ]
    path = tmp_path / "hotpot.json"
    path.write_text(json.dumps(rows))
    dataset, audit = convert_hotpot(path, 4, "seed", "train")
    assert len(dataset.examples) == 4
    assert all(len(e.question.candidate_ids) == 2 for e in dataset.examples)
    assert audit["components"] == 1
    assert dataset.provenance["setting"] == "distractor_candidates"
    assert dataset.provenance["original_split"] == "train"
    rows[0]["supporting_facts"] = [["Missing", 0]]
    path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="supporting fact"):
        convert_hotpot(path, 4, "seed")


def test_generated_results_cannot_overwrite(tmp_path):
    path = tmp_path / "result.json"
    write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"original": False})
