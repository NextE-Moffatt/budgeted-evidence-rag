from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from random import Random

import httpx

from .data import digest, file_sha, write_json

HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_FILES = (
    (
        "train-00000-of-00002.parquet",
        165624177,
        "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    ),
    (
        "train-00001-of-00002.parquet",
        166162479,
        "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
    ),
)
HOTPOT_BASE = f"https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/{HOTPOT_REVISION}"


def download_file(client: httpx.Client, url: str, target: Path, size: int, sha256: str) -> None:
    if target.exists():
        if target.stat().st_size != size or file_sha(target) != sha256:
            raise ValueError("existing dataset file failed integrity verification")
        return
    partial = target.with_name(target.name + ".part")
    owned = False
    try:
        with partial.open("xb") as stream:
            owned = True
            count = 0
            try:
                with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise ValueError("dataset download returned a non-success status")
                    for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                        count += len(chunk)
                        if count > size:
                            raise ValueError("dataset download exceeded the pinned size")
                        stream.write(chunk)
            except httpx.HTTPError:
                raise ValueError(
                    "dataset transport failure; no credentials or model calls used"
                ) from None
        if partial.stat().st_size != size or file_sha(partial) != sha256:
            raise ValueError("downloaded dataset failed size/SHA256 verification")
        partial.rename(target)
    finally:
        if owned and partial.exists():
            partial.unlink()


def download_hotpot(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": "hotpotqa/hotpot_qa",
        "config": "distractor",
        "split": "train",
        "revision": HOTPOT_REVISION,
        "license": "CC-BY-SA-4.0",
        "source_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "integrity": "pinned size and SHA256 from upstream Git LFS pointers",
        "files": [],
    }
    with httpx.Client(timeout=httpx.Timeout(90, connect=10), follow_redirects=True) as client:
        for name, size, checksum in HOTPOT_FILES:
            url = f"{HOTPOT_BASE}/distractor/{name}"
            download_file(client, url, output / name, size, checksum)
            manifest["files"].append({"path": name, "bytes": size, "sha256": checksum, "url": url})
    path = output / "download_manifest.json"
    if not path.exists():
        write_json(path, manifest)
    return {
        "state": "verified",
        "split": "train",
        "files": len(HOTPOT_FILES),
        "bytes": sum(size for _, size, _ in HOTPOT_FILES),
        "revision": HOTPOT_REVISION,
    }


def pilot_offsets(total: int, count: int, block_size: int, seed: str) -> list[int]:
    if (
        type(count) is not int
        or type(block_size) is not int
        or count < 1
        or count > 200
        or block_size < 1
        or count % block_size
        or total < count
    ):
        raise ValueError("pilot requires 1-200 questions and complete positive-sized blocks")
    blocks = count // block_size
    rng = Random(seed)
    return [
        rng.randrange(i * total // blocks, (i + 1) * total // blocks - block_size + 1)
        for i in range(blocks)
    ]


def fetch_pilot(output: Path, count: int = 200, seed: str = "pilot-v1") -> dict:
    from .hotpot import parquet_row

    if count < 1 or count > 200 or count % 10:
        raise ValueError("row-API pilot count must be a positive multiple of 10, at most 200")
    output.mkdir(parents=True, exist_ok=False)
    repo_url = "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa"
    rows_url = "https://datasets-server.huggingface.co/rows"
    params = {"dataset": "hotpotqa/hotpot_qa", "config": "distractor", "split": "train"}
    rows, snapshots = [], []
    try:
        with httpx.Client(timeout=httpx.Timeout(60, connect=10), follow_redirects=True) as client:
            before = client.get(repo_url)
            before.raise_for_status()
            if before.json()["sha"] != HOTPOT_REVISION:
                raise ValueError("upstream revision changed; review snapshot before sampling")
            probe = client.get(rows_url, params={**params, "offset": 0, "length": 1})
            probe.raise_for_status()
            total = probe.json()["num_rows_total"]
            offsets = pilot_offsets(total, count, 10, seed)
            for index, offset in enumerate(offsets):
                response = client.get(rows_url, params={**params, "offset": offset, "length": 10})
                response.raise_for_status()
                body = response.json()
                if (
                    body["partial"]
                    or body["num_rows_total"] != total
                    or [r["row_idx"] for r in body["rows"]] != list(range(offset, offset + 10))
                    or any(r["truncated_cells"] for r in body["rows"])
                ):
                    raise ValueError("row API returned a partial, truncated or misaligned block")
                snapshot = output / f"block-{index:02d}.json"
                write_json(snapshot, body)
                snapshots.append(
                    {
                        "offset": offset,
                        "length": 10,
                        "path": snapshot.name,
                        "sha256": file_sha(snapshot),
                        "payload_digest": digest(body),
                    }
                )
                rows.extend(parquet_row(r["row"]) for r in body["rows"])
            after = client.get(repo_url)
            after.raise_for_status()
            if after.json()["sha"] != HOTPOT_REVISION:
                raise ValueError("upstream changed during row acquisition; reject this pilot")
    except httpx.HTTPError:
        raise ValueError(
            "pilot row API transport failure; incomplete acquisition is not accepted"
        ) from None
    if len(rows) != count or len({r["_id"] for r in rows}) != count:
        raise ValueError("incorrect or duplicate row API sample")
    write_json(output / "rows.json", rows)
    metadata = {
        "dataset": params["dataset"],
        "config": "distractor",
        "split": "train",
        "source_url": rows_url,
        "source_revision_before_and_after": HOTPOT_REVISION,
        "license": "CC-BY-SA-4.0",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "source_total_rows": total,
        "seed": seed,
        "questions": count,
        "sampling": "one random contiguous 10-row block per equal-width row-index stratum",
        "sampling_caveat": "exploratory block sample, not iid or global ID-hash sampling",
        "integrity_caveat": "payload hashes/revision checks, not full-shard verification",
        "final_benchmark_eligible": False,
        "llm_calls": 0,
        "blocks": snapshots,
    }
    write_json(output / "rows.json.source.json", metadata)
    return {
        "state": "exploratory_pilot_acquired",
        "questions": count,
        "source_total_rows": total,
        "llm_calls": 0,
        "final_benchmark_eligible": False,
    }
