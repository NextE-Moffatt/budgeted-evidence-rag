from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from time import perf_counter

from . import __version__
from .collection import collect
from .data import SPLITS, dataset_dict, digest, file_sha, load_dataset, load_traces, write_json
from .download import download_hotpot, fetch_pilot
from .evaluation import evaluate
from .hotpot import convert_hotpot
from .metrics import answer_scores
from .policies import StopPolicy
from .reader import SYSTEM_PROMPT, HTTPReader
from .retrieval_audit import audit_retrieval
from .schema import Dataset, Trace, sum_cost
from .splits import audit_split_leakage


def new_run(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def manifest(config: dict, input_hashes: dict) -> dict:
    root = Path(__file__).parent
    source_hash = digest(
        {
            str(p.relative_to(root)): file_sha(p)
            for p in sorted(root.rglob("*"))
            if p.suffix in (".py", ".json")
        }
    )
    dependencies = {
        p: importlib.metadata.version(p)
        for p in ("bm25s", "httpx", "numpy", "scipy", "httpcore", "anyio", "h11", "certifi", "idna")
    }
    try:
        dependencies["pyarrow"] = importlib.metadata.version("pyarrow")
    except importlib.metadata.PackageNotFoundError:
        dependencies["pyarrow"] = None
    environment = {"python_version": sys.version.split()[0], "dependencies": dependencies}
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "package_version": __version__,
        "source_sha256": source_hash,
        "python_version": sys.version.split()[0],
        "config": config,
        "inputs": input_hashes,
        "run_fingerprint": digest([config, input_hashes, source_hash, environment]),
        "dependencies": dependencies,
    }


def continuation_audit(dataset: Dataset, traces: tuple[Trace, ...], split: str) -> dict:
    examples = {e.question.id: e for e in dataset.examples if e.split == split}
    rows = []
    for trace in traces:
        if trace.question_id not in examples:
            continue
        golds = examples[trace.question_id].answers
        for before, after in zip(trace.steps, trace.steps[1:], strict=False):
            delta = (
                answer_scores(after.proposal.answer, golds)[1]
                - answer_scores(before.proposal.answer, golds)[1]
            )
            rows.append(
                {
                    "question_id": trace.question_id,
                    "round": before.round,
                    "group_id": examples[trace.question_id].group_id,
                    "confidence_before": before.proposal.confidence,
                    "sufficiency_before": before.proposal.sufficiency,
                    "f1_delta_observed": delta,
                    "benefit": delta > 0,
                    "harm": delta < 0,
                    "continuation_failed": after.status != "ok",
                }
            )
    return {
        "interpretation": "observed adjacent continuation, not a causal or counterfactual effect",
        "independence_warning": "states from the same question/group are correlated",
        "transitions": len(rows),
        "beneficial": sum(r["benefit"] for r in rows),
        "harmful": sum(r["harm"] for r in rows),
        "neutral": sum(r["f1_delta_observed"] == 0 for r in rows),
        "rows": rows,
    }


def evaluate_all(
    dataset: Dataset,
    traces: tuple[Trace, ...],
    split: str,
    rounds: int,
    threshold: float,
    output: Path,
) -> list[dict]:
    summaries = []
    policies = [StopPolicy("fixed", n, threshold) for n in range(1, rounds + 1)] + [
        StopPolicy(mode, rounds, threshold) for mode in ("confidence", "sufficiency", "stability")
    ]
    for policy in policies:
        summary, rows = evaluate(dataset, traces, split, policy)
        write_json(output / f"{policy.name}.json", {"summary": summary, "questions": rows})
        summaries.append(summary)
    write_json(output / "summary.json", summaries)
    write_json(output / "continuation_audit.json", continuation_audit(dataset, traces, split))
    return summaries


def validate_args(args: argparse.Namespace) -> None:
    for name in ("max_rounds", "limit", "top_k"):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if getattr(args, "offset", 0) < 0:
        raise ValueError("offset must be nonnegative")
    if hasattr(args, "threshold"):
        StopPolicy("fixed", args.max_rounds, args.threshold)
    if getattr(args, "split", None) == "test" and not args.confirmatory:
        raise ValueError("test requires --confirmatory and a previously frozen protocol/config")


def run(args: argparse.Namespace) -> dict:
    validate_args(args)
    config = {
        k: v for k, v in vars(args).items() if k not in ("output", "dataset", "traces", "input")
    }
    if args.command in ("fetch-hotpot", "fetch-hotpot-pilot"):
        if not args.allow_download:
            raise ValueError("dataset network access requires --allow-download")
        if args.command == "fetch-hotpot-pilot":
            return fetch_pilot(args.output, args.limit, args.seed)
        return download_hotpot(args.output)
    if args.command == "prepare-hotpot":
        dataset, audit = convert_hotpot(args.input, args.limit, args.seed, args.source_split)
        audit_split_leakage(dataset)
        new_run(args.output)
        write_json(args.output / "dataset.json", dataset_dict(dataset))
        write_json(args.output / "split_audit.json", audit)
        write_json(
            args.output / "manifest.json",
            manifest(config, {"input": dataset.provenance["input_sha256"]}),
        )
        return audit
    if args.command == "smoke":
        fixture_dir = files("evidence_rag").joinpath("fixtures")
        dataset = load_dataset(Path(str(fixture_dir.joinpath("dataset.json"))))
        audit_split_leakage(dataset)
        raw_traces = {
            "schema_version": 1,
            "dataset_sha256": file_sha(dataset=dataset),
            "traces": json.loads(fixture_dir.joinpath("scenarios.json").read_text()),
        }
        new_run(args.output)
        write_json(args.output / "traces.json", raw_traces)
        traces = load_traces(args.output / "traces.json", dataset)
        summaries = evaluate_all(dataset, traces, "train", 3, 0.8, args.output)
        write_json(
            args.output / "manifest.json",
            manifest(
                {"command": "smoke", "split": "train", "max_rounds": 3, "threshold": 0.8},
                {"dataset": file_sha(dataset=dataset), "traces": digest(raw_traces)},
            ),
        )
        return {
            "result_kind": "software_smoke_not_benchmark",
            "questions": len(traces),
            "policies": len(summaries),
            "network_requests": 0,
        }
    dataset = load_dataset(args.dataset)
    audit_split_leakage(dataset)
    inputs = {"dataset": file_sha(dataset=dataset)}
    if args.command == "audit-retrieval":
        report = audit_retrieval(dataset, args.split, tuple(args.ks))
        new_run(args.output)
        write_json(args.output / "retrieval_audit.json", report)
        write_json(args.output / "manifest.json", manifest(config, inputs))
        return {k: v for k, v in report.items() if k not in ("rows", "limitations")}
    if args.command == "evaluate":
        traces = load_traces(args.traces, dataset)
        inputs["traces"] = file_sha(args.traces)
        if not any(
            e.split == args.split and e.question.id in {t.question_id for t in traces}
            for e in dataset.examples
        ):
            raise ValueError("no trace questions in requested split")
        new_run(args.output)
        summaries = evaluate_all(
            dataset, traces, args.split, args.max_rounds, args.threshold, args.output
        )
        write_json(args.output / "manifest.json", manifest(config, inputs))
        return {
            "result_kind": summaries[0]["result_kind"],
            "policies": len(summaries),
            "questions": summaries[0]["questions"],
        }
    if not args.allow_llm:
        raise ValueError("model calls require explicit --allow-llm")
    if args.limit > 50 or args.max_rounds > 8:
        raise ValueError("pilot cap is 50 questions and 8 rounds; review protocol before scaling")
    examples = sorted(
        (e for e in dataset.examples if e.split == args.split), key=lambda e: e.question.id
    )[args.offset : args.offset + args.limit]
    if not examples:
        raise ValueError("requested partition is empty; inspect split_audit.json")
    reader = HTTPReader.from_env()
    traces = []
    try:
        new_run(args.output)
        config.update(
            {
                "reader_model": reader.model,
                "temperature": 0,
                "max_output_tokens": 512,
                "prompt_sha256": digest(SYSTEM_PROMPT),
                "retries": 0,
                "timeout_seconds": 60,
                "chat_template_kwargs": getattr(reader, "chat_template_kwargs", None),
                "private_http_opt_in": getattr(reader, "allow_private_http", False),
                "query_policy": "reader_next_query_or_question",
                "retriever": "bm25s-lucene-k1=1.5-b=0.75-stopwords=en-no-stemmer",
                "question_ids": [e.question.id for e in examples],
            }
        )
        write_json(args.output / "manifest.json", manifest(config, inputs))
        started = perf_counter()
        with (args.output / "traces.jsonl").open("x") as stream:
            for example in examples:
                trace = collect(
                    example.question, dataset.passages, reader, args.max_rounds, args.top_k
                )
                traces.append(trace)
                stream.write(json.dumps(asdict(trace), allow_nan=False) + "\n")
                stream.flush()
        write_json(
            args.output / "traces.json",
            {
                "schema_version": 1,
                "dataset_sha256": inputs["dataset"],
                "traces": [asdict(t) for t in traces],
            },
        )
        report = {
            "state": "complete",
            "questions": len(traces),
            "reader_error_states": sum(s.status != "ok" for t in traces for s in t.steps),
            "collection_wall_seconds": perf_counter() - started,
            "collection_cost": sum_cost([s.cost for t in traces for s in t.steps]),
            "index_ms": sum(t.index_ms for t in traces),
        }
        write_json(args.output / "collection_summary.json", report)
        return report
    finally:
        reader.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Research scaffold; no paper results implied")
    sub = result.add_subparsers(dest="command", required=True)
    smoke = sub.add_parser("smoke", help="offline fictional replay; no network or model calls")
    smoke.add_argument("--output", type=Path, required=True)
    fetch = sub.add_parser(
        "fetch-hotpot", help="download and verify the pinned public training snapshot"
    )
    fetch.add_argument("--allow-download", action="store_true")
    fetch.add_argument("--output", type=Path, required=True)
    pilot = sub.add_parser("fetch-hotpot-pilot", help="explicit exploratory row-API block sample")
    pilot.add_argument("--allow-download", action="store_true")
    pilot.add_argument("--limit", type=int, default=200)
    pilot.add_argument("--seed", default="pilot-v1")
    pilot.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare-hotpot", help="convert a local labelled HotpotQA snapshot")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--source-split", choices=("train", "dev"), required=True)
    prepare.add_argument("--limit", type=int, default=200)
    prepare.add_argument("--seed", default="pilot-v1")
    prepare.add_argument("--output", type=Path, required=True)
    audit = sub.add_parser(
        "audit-retrieval", help="original-query BM25 feasibility; zero LLM calls"
    )
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--split", choices=("train", "calibration", "validation"), required=True)
    audit.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    audit.add_argument("--output", type=Path, required=True)
    for name in ("collect", "evaluate"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", type=Path, required=True)
        command.add_argument("--split", choices=SPLITS, required=True)
        command.add_argument("--max-rounds", type=int, default=3)
        command.add_argument("--confirmatory", action="store_true")
        command.add_argument("--output", type=Path, required=True)
        if name == "collect":
            command.add_argument("--limit", type=int, default=5)
            command.add_argument("--offset", type=int, default=0)
            command.add_argument("--top-k", type=int, default=1)
            command.add_argument("--allow-llm", action="store_true")
        else:
            command.add_argument("--traces", type=Path, required=True)
            command.add_argument("--threshold", type=float, default=0.8)
    return result


def main() -> None:
    args = parser().parse_args()
    try:
        report = run(args)
    except (ValueError, OSError, KeyError, TypeError):
        # Avoid raw paths, HTTP bodies, or caller-controlled input in logs.
        print(
            "Command failed: invalid input/configuration or unavailable output. "
            "Check input schemas, split, environment and a new output path.",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    print(json.dumps({"output": str(args.output), **report}, indent=2, allow_nan=False))
