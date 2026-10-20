import json

import pytest

from evidence_rag.hotpot import convert_hotpot, parquet_row

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def records():
    return [
        {
            "id": f"fiction-{i}",
            "question": "Where is the fictional institute?",
            "answer": "Westland",
            "context": {
                "title": [f"Institute-{i}", f"District-{i}"],
                "sentences": [["An invented institute."], ["An invented district."]],
            },
            "supporting_facts": {"title": [f"Institute-{i}"], "sent_id": [0]},
        }
        for i in range(8)
    ]


def test_parquet_and_json_match_and_sample_across_all_shards(tmp_path):
    rows = records()
    source = tmp_path / "snapshot"
    source.mkdir()
    pq.write_table(pa.Table.from_pylist(rows[:4]), source / "train-0.parquet")
    pq.write_table(pa.Table.from_pylist(rows[4:]), source / "train-1.parquet")
    raw = tmp_path / "source.json"
    raw.write_text(json.dumps([parquet_row(row) for row in rows]))
    parquet, _ = convert_hotpot(source, 3, "seed", "train")
    original, _ = convert_hotpot(raw, 3, "seed", "train")
    assert parquet.examples == original.examples
    assert parquet.passages == original.passages
    assert parquet.provenance["source_questions_scanned"] == 8
    assert len(parquet.provenance["input_files"]) == 2
    assert parquet.provenance["input_format"] == "parquet"


def test_duplicate_questions_across_shards_rejected(tmp_path):
    rows = records()
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "first.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:1]), tmp_path / "duplicate.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        convert_hotpot(tmp_path, 3, "seed")


def test_parquet_parallel_arrays_are_not_silently_truncated():
    row = records()[0]
    row["context"]["title"].append("Unmatched")
    with pytest.raises(ValueError):
        parquet_row(row)
