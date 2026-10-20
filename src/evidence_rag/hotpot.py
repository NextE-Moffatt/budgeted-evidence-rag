from __future__ import annotations

import json
from heapq import nsmallest
from pathlib import Path
from typing import Any

from .data import digest, file_sha
from .schema import Dataset, Example, Passage, PublicQuestion
from .splits import grouped_split


def input_snapshot(path: Path) -> tuple[list[Path], str]:
    paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not paths:
        raise ValueError("no parquet files in the snapshot directory")
    checksums = [{"path": p.name, "sha256": file_sha(p)} for p in paths]
    return paths, checksums[0]["sha256"] if len(paths) == 1 else digest(checksums)


def parquet_row(raw: dict[str, Any]) -> dict[str, Any]:
    context = raw["context"]
    support = raw["supporting_facts"]
    return {
        "_id": raw["id"],
        "question": raw["question"],
        "answer": raw["answer"],
        "context": list(zip(context["title"], context["sentences"], strict=True)),
        "supporting_facts": list(zip(support["title"], support["sent_id"], strict=True)),
    }


def input_rows(paths: list[Path]):
    for path in paths:
        if path.suffix == ".json":
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows:
                raise ValueError("expected labelled HotpotQA array")
            yield from rows
        elif path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise ValueError("install the data extra to read parquet snapshots") from None
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield parquet_row(row)
        else:
            raise ValueError("supported HotpotQA inputs are JSON or Parquet")


def convert_hotpot(
    path: Path, limit: int, seed: str, source_split: str = "unspecified"
) -> tuple[Dataset, dict]:
    if limit < 1:
        raise ValueError("limit must be positive")
    paths, checksum = input_snapshot(path)
    seen = set()

    def checked_rows():
        for row in input_rows(paths):
            qid = row["_id"]
            if not isinstance(qid, str) or not qid or qid in seen:
                raise ValueError("invalid/duplicate HotpotQA question id")
            seen.add(qid)
            yield row

    selected = nsmallest(limit, checked_rows(), key=lambda r: digest([seed, r["_id"]]))
    if not selected:
        raise ValueError("empty HotpotQA snapshot")
    passages = {}
    examples = []
    for row in selected:
        title_ids = {}
        lengths = {}
        for title, sentences in row["context"]:
            if title in title_ids or not title or not isinstance(sentences, list):
                raise ValueError("invalid/duplicate context title")
            if not sentences or any(not isinstance(s, str) for s in sentences):
                raise ValueError("invalid context sentences")
            text = "".join(sentences).strip()
            if not text:
                raise ValueError("empty context")
            pid = digest([title, text])
            passages[pid] = Passage(pid, title, text, title)
            title_ids[title] = pid
            lengths[title] = len(sentences)
        gold = set()
        for title, sentence_index in row["supporting_facts"]:
            if title not in title_ids or not 0 <= sentence_index < lengths[title]:
                raise ValueError("supporting fact not found in context")
            gold.add(title_ids[title])
        if not gold or not isinstance(row["answer"], str) or not row["answer"].strip():
            raise ValueError("missing gold labels")
        examples.append(
            Example(
                PublicQuestion(row["_id"], row["question"], tuple(title_ids.values())),
                (row["answer"],),
                tuple(sorted(gold)),
                row["_id"],
                "train",
            )
        )
    dataset = Dataset(
        passages,
        tuple(examples),
        {
            "kind": "benchmark",
            "dataset": "HotpotQA",
            "setting": "distractor_candidates",
            "source_url": "https://hotpotqa.github.io/",
            "license": "CC-BY-SA-4.0",
            "input_sha256": checksum,
            "input_files": [{"path": p.name, "sha256": file_sha(p)} for p in paths],
            "input_format": paths[0].suffix.removeprefix("."),
            "source_questions_scanned": len(seen),
            "sample_method": "smallest seeded ID hashes over all supplied files",
            "sample_seed": seed,
            "sample_limit": limit,
            "original_split": source_split,
            "support_metric": "passage recall only; sentence support/joint metrics not implemented",
        },
    )
    sidecar = path.with_name(path.name + ".source.json")
    if path.is_file() and sidecar.exists():
        dataset.provenance["source_metadata"] = json.loads(sidecar.read_text())
        dataset.provenance["source_metadata_sha256"] = file_sha(sidecar)
    return grouped_split(dataset, seed)
