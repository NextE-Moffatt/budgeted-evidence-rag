import json
from dataclasses import fields

import httpx
import pytest

from evidence_rag.collection import collect
from evidence_rag.reader import HTTPReader, ReaderError
from evidence_rag.retrieval import BM25Retriever
from evidence_rag.schema import ABSTENTION, Proposal, PublicQuestion


def test_real_bm25_dedup_and_empty_query(dataset):
    retriever = BM25Retriever(tuple(dataset.passages.values()))
    result = retriever.search("Meridian Institute", 2, set())
    assert result[0].id == "p1"
    assert all(p.id != "p1" for p in retriever.search("Meridian", 10, {"p1"}))
    assert retriever.search("", 2, set()) == ()
    assert retriever.search("unfindablexyz", 2, set()) == ()


def test_collector_bounded_and_gold_free(dataset):
    calls = []

    class FakeReader:
        def read(self, question, evidence):
            calls.append(question)
            assert {f.name for f in fields(question)} == {"id", "text", "candidate_ids"}
            assert evidence
            return Proposal(None, (), 0.0, 0.0, "Northmere"), 20, 5

    trace = collect(dataset.examples[0].question, dataset.passages, FakeReader(), 2, 1)
    assert len(calls) == len(trace.steps) == 2
    assert sum(s.cost.reader_calls for s in trace.steps) == 2


def test_failed_attempt_charged(dataset):
    class BrokenReader:
        def read(self, question, evidence):
            raise ReaderError("invalid_reader_output", 30, 10)

    trace = collect(dataset.examples[0].question, dataset.passages, BrokenReader(), 3, 1)
    assert len(trace.steps) == 1
    assert trace.steps[0].status == "reader_error"
    assert trace.steps[0].proposal == ABSTENTION
    assert trace.steps[0].cost.reader_calls == 1
    assert trace.steps[0].cost.input_tokens == 30


@pytest.mark.parametrize(
    "change",
    [
        {"citations": ["future"]},
        {"citations": []},
        {"confidence": float("nan")},
        {"sufficiency": 1.1},
        {"confidence": True},
        {"answer": ""},
        {"answer": None},
        {"next_query": 3},
    ],
)
def test_invalid_proposal_fails_closed(change):
    raw = {
        "answer": "Westland",
        "citations": ["p1"],
        "confidence": 0.9,
        "sufficiency": 0.9,
        "next_query": "",
    }
    with pytest.raises(ValueError):
        Proposal.parse({**raw, **change}, {"p1"})


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/v1",
        "https://user:secret@example.invalid/v1",
        "https://example.invalid/v1?api_key=secret",
    ],
)
def test_credentials_url_guard(url):
    with pytest.raises(ValueError):
        HTTPReader(url, "fixture-model", "sentinel-secret")


@pytest.mark.parametrize(
    "url,allowed",
    [
        ("http://10.23.45.67/v1", True),
        ("http://172.20.1.2/v1", True),
        ("http://192.168.12.34/v1", True),
        ("http://8.8.8.8/v1", False),
        ("http://169.254.169.254/v1", False),
        ("http://example.invalid/v1", False),
        ("http://0.0.0.0/v1", False),
        ("http://172.15.255.255/v1", False),
        ("http://172.32.0.1/v1", False),
        ("http://192.169.0.1/v1", False),
    ],
)
def test_private_http_opt_in_does_not_allow_public_or_metadata_endpoints(url, allowed):
    if allowed:
        with pytest.raises(ValueError):
            HTTPReader(url, "fixture", "sentinel-secret")
        reader = HTTPReader(url, "fixture", "sentinel-secret", allow_private_http=True)
        reader.close()
    else:
        with pytest.raises(ValueError):
            HTTPReader(url, "fixture", "sentinel-secret", allow_private_http=True)


def test_explicit_thinking_payload_and_unknown_usage(dataset):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": None,
                                    "citations": [],
                                    "confidence": 0,
                                    "sufficiency": 0,
                                    "next_query": "",
                                }
                            )
                        }
                    }
                ]
            },
        )

    reader = mocked_reader(handler)
    reader.chat_template_kwargs = {"enable_thinking": False}
    try:
        _, inputs, outputs = reader.read(dataset.examples[0].question, (dataset.passages["p1"],))
        assert inputs is outputs is None
        assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        reader.close()


def test_reader_env_flags_are_explicit(monkeypatch):
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://10.23.45.67/v1")
    monkeypatch.setenv("RAG_LLM_MODEL", "fixture")
    monkeypatch.setenv("RAG_LLM_API_KEY", "sentinel-secret")
    monkeypatch.setenv("RAG_LLM_ALLOW_PRIVATE_HTTP", "1")
    monkeypatch.setenv("RAG_LLM_ENABLE_THINKING", "0")
    reader = HTTPReader.from_env()
    assert reader.allow_private_http
    assert reader.chat_template_kwargs == {"enable_thinking": False}
    reader.close()
    monkeypatch.setenv("RAG_LLM_ALLOW_PRIVATE_HTTP", "yes")
    with pytest.raises(ValueError):
        HTTPReader.from_env()


@pytest.mark.parametrize("flag", ["false", 1, None])
def test_private_http_constructor_requires_boolean(flag):
    with pytest.raises(ValueError, match="explicit boolean"):
        HTTPReader("http://10.23.45.67/v1", "fixture", "sentinel-secret", allow_private_http=flag)


def mocked_reader(handler):
    reader = HTTPReader("https://reader.invalid/v1", "fixture-model", "sentinel-secret")
    reader.client.close()
    reader.client = httpx.Client(
        base_url="https://reader.invalid/v1/", transport=httpx.MockTransport(handler)
    )
    return reader


def test_http_error_is_redacted():
    reader = mocked_reader(lambda request: httpx.Response(401, text="sentinel-secret"))
    try:
        with pytest.raises(ReaderError, match="http_failure") as caught:
            reader.read(PublicQuestion("q", "question", ()), ())
        assert "sentinel-secret" not in str(caught.value)
    finally:
        reader.close()


@pytest.mark.parametrize("body", ["not-json", "[]", '{"usage": 7}', '{"choices": []}'])
def test_malformed_response_is_safe(body):
    reader = mocked_reader(lambda request: httpx.Response(200, text=body))
    try:
        with pytest.raises(ReaderError, match="invalid_reader_output"):
            reader.read(PublicQuestion("q", "question", ()), ())
    finally:
        reader.close()


def test_prompt_no_gold_and_usage_preserved_on_invalid_answer(dataset):
    def handler(request):
        payload = json.loads(request.content)
        assert set(json.loads(payload["messages"][1]["content"])) == {"question", "evidence"}
        assert payload["temperature"] == 0
        raw = {
            "answer": "Westland",
            "citations": ["future"],
            "confidence": 0.9,
            "sufficiency": 0.9,
            "next_query": "",
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(raw)}}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 10},
            },
        )

    reader = mocked_reader(handler)
    try:
        with pytest.raises(ReaderError, match="invalid_reader_output") as caught:
            reader.read(dataset.examples[0].question, (dataset.passages["p1"],))
        assert caught.value.input_tokens == 30
        assert caught.value.output_tokens == 10
    finally:
        reader.close()
