import importlib.metadata
import json

import httpx
import pytest

from evidence_rag.cli import manifest, parser, run
from evidence_rag.data import dataset_dict, load_traces, write_json
from evidence_rag.reader import HTTPReader
from evidence_rag.schema import ABSTENTION


def test_smoke_has_no_network_and_is_repeatable(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline smoke attempted network")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        report = run(parser().parse_args(["smoke", "--output", str(output)]))
        assert report["network_requests"] == 0
        assert report["questions"] == 5
        assert report["policies"] == 6
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (first, second)]
    assert manifests[0]["run_fingerprint"] == manifests[1]["run_fingerprint"]
    summaries = [json.loads((p / "summary.json").read_text()) for p in (first, second)]
    assert [s["answer_em"] for s in summaries[0]] == [s["answer_em"] for s in summaries[1]]
    with pytest.raises(FileExistsError):
        run(parser().parse_args(["smoke", "--output", str(first)]))


def test_confirmatory_guard_before_dataset_access(tmp_path):
    args = parser().parse_args(
        [
            "evaluate",
            "--dataset",
            "missing.json",
            "--traces",
            "missing.json",
            "--split",
            "test",
            "--output",
            str(tmp_path / "test"),
        ]
    )
    with pytest.raises(ValueError, match="confirmatory"):
        run(args)


def test_manifest_does_not_copy_environment(monkeypatch):
    monkeypatch.setenv("RAG_LLM_API_KEY", "sentinel-secret")
    assert "sentinel-secret" not in json.dumps(manifest({"command": "test"}, {}))


def test_dependency_versions_participate_in_run_fingerprint(monkeypatch):
    first = manifest({"command": "test"}, {})
    original = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "changed" if name == "numpy" else original(name)
    )
    second = manifest({"command": "test"}, {})
    assert first["run_fingerprint"] != second["run_fingerprint"]


def test_collection_evaluation_pipeline_with_mock_reader(dataset, tmp_path, monkeypatch):
    class FakeReader:
        model = "offline-fixture-not-a-model"

        def read(self, question, evidence):
            return ABSTENTION, None, None

        def close(self):
            pass

    monkeypatch.setattr(HTTPReader, "from_env", lambda: FakeReader())
    path = tmp_path / "dataset.json"
    write_json(path, dataset_dict(dataset))
    collection = tmp_path / "collection"
    base = [
        "collect",
        "--dataset",
        str(path),
        "--split",
        "train",
        "--limit",
        "2",
        "--max-rounds",
        "2",
        "--output",
        str(collection),
    ]
    with pytest.raises(ValueError, match="allow-llm"):
        run(parser().parse_args(base))
    assert not collection.exists()
    report = run(parser().parse_args([*base, "--allow-llm"]))
    assert report["questions"] == 2
    assert report["collection_cost"]["reader_calls"] <= 4
    traces = load_traces(collection / "traces.json", dataset)
    assert len(traces) == 2
    assert len((collection / "traces.jsonl").read_text().splitlines()) == 2
    evaluation = tmp_path / "evaluation"
    report = run(
        parser().parse_args(
            [
                "evaluate",
                "--dataset",
                str(path),
                "--split",
                "train",
                "--traces",
                str(collection / "traces.json"),
                "--output",
                str(evaluation),
            ]
        )
    )
    assert report["questions"] == 2
    audit = json.loads((evaluation / "continuation_audit.json").read_text())
    assert audit["beneficial"] == audit["harmful"] == 0


def test_negative_offset_rejected_before_dataset_access(tmp_path):
    args = parser().parse_args(
        [
            "collect",
            "--dataset",
            "missing.json",
            "--split",
            "train",
            "--offset",
            "-1",
            "--output",
            str(tmp_path / "output"),
        ]
    )
    with pytest.raises(ValueError, match="offset"):
        run(args)
