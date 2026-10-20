import hashlib

import httpx
import pytest

import evidence_rag.download as downloads
from evidence_rag.cli import parser, run
from evidence_rag.download import HOTPOT_REVISION, download_file, fetch_pilot, pilot_offsets
from evidence_rag.hotpot import convert_hotpot


def test_download_requires_explicit_opt_in(tmp_path):
    args = parser().parse_args(["fetch-hotpot", "--output", str(tmp_path / "raw")])
    with pytest.raises(ValueError, match="allow-download"):
        run(args)
    assert not args.output.exists()


def test_download_verification_and_reuse(tmp_path):
    payload = b"parquet-like-fixture"
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=payload)

    target = tmp_path / "train.parquet"
    checksum = hashlib.sha256(payload).hexdigest()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        download_file(client, "https://fixture.invalid/file", target, len(payload), checksum)
        download_file(client, "https://fixture.invalid/file", target, len(payload), checksum)
    assert target.read_bytes() == payload
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload,status,checksum",
    [
        (b"bad", 200, "0" * 64),
        (b"too-large-payload", 200, "0" * 64),
        (b"", 403, "0" * 64),
    ],
)
def test_failed_download_not_accepted_or_left_as_complete(tmp_path, payload, status, checksum):
    target = tmp_path / "train.parquet"
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(status, content=payload))
        ) as client,
        pytest.raises(ValueError),
    ):
        download_file(client, "https://fixture.invalid/file", target, 3, checksum)
    assert not target.exists()
    assert not target.with_name(target.name + ".part").exists()


def test_existing_bad_file_not_overwritten(tmp_path):
    target = tmp_path / "train.parquet"
    target.write_bytes(b"existing")
    with httpx.Client() as client, pytest.raises(ValueError, match="existing"):
        download_file(client, "https://fixture.invalid/file", target, 3, "0" * 64)
    assert target.read_bytes() == b"existing"


def test_preexisting_partial_not_deleted(tmp_path):
    target = tmp_path / "train.parquet"
    partial = target.with_name(target.name + ".part")
    partial.write_bytes(b"another run")
    with httpx.Client() as client, pytest.raises(FileExistsError):
        download_file(client, "https://fixture.invalid/file", target, 3, "0" * 64)
    assert partial.read_bytes() == b"another run"


def test_stratified_offsets_deterministic_nonoverlapping():
    offsets = pilot_offsets(90447, 200, 10, "pilot-v1")
    assert offsets == pilot_offsets(90447, 200, 10, "pilot-v1")
    assert len(offsets) == 20
    assert all(a + 10 <= b for a, b in zip(offsets, offsets[1:], strict=False))
    assert offsets[-1] + 10 <= 90447


def mock_row_client(monkeypatch, *, truncate=False, revision=HOTPOT_REVISION):
    def handler(request):
        if request.url.path.startswith("/api/datasets/"):
            return httpx.Response(200, json={"sha": revision})
        offset, length = int(request.url.params["offset"]), int(request.url.params["length"])
        rows = [
            {
                "row_idx": i,
                "truncated_cells": ["context"] if truncate else [],
                "row": {
                    "id": f"fiction-{i}",
                    "question": "Where is this fictional institute?",
                    "answer": "Westland",
                    "context": {
                        "title": [f"Institute-{i}"],
                        "sentences": [["An invented institute in Westland."]],
                    },
                    "supporting_facts": {"title": [f"Institute-{i}"], "sent_id": [0]},
                },
            }
            for i in range(offset, offset + length)
        ]
        return httpx.Response(200, json={"partial": False, "num_rows_total": 1000, "rows": rows})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(downloads.httpx, "Client", lambda **kwargs: client)


def test_row_api_pilot_integrity_metadata_and_conversion(tmp_path, monkeypatch):
    mock_row_client(monkeypatch)
    output = tmp_path / "pilot"
    report = fetch_pilot(output, 20, "seed")
    assert report["questions"] == 20
    assert not report["final_benchmark_eligible"]
    dataset, _ = convert_hotpot(output / "rows.json", 20, "seed", "train")
    assert dataset.provenance["source_questions_scanned"] == 20
    metadata = dataset.provenance["source_metadata"]
    assert metadata["source_total_rows"] == 1000
    assert metadata["source_revision_before_and_after"] == HOTPOT_REVISION
    assert len(metadata["blocks"]) == 2
    assert metadata["llm_calls"] == 0


@pytest.mark.parametrize("truncate,revision", [(True, HOTPOT_REVISION), (False, "changed")])
def test_incomplete_or_changed_row_pilot_not_accepted(tmp_path, monkeypatch, truncate, revision):
    mock_row_client(monkeypatch, truncate=truncate, revision=revision)
    with pytest.raises(ValueError):
        fetch_pilot(tmp_path / "pilot", 20, "seed")
    assert not (tmp_path / "pilot" / "rows.json").exists()
