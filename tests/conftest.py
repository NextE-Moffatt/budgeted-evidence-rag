import json
from importlib.resources import files
from pathlib import Path

import pytest

from evidence_rag.data import file_sha, load_dataset, load_traces, write_json


@pytest.fixture
def dataset():
    return load_dataset(Path(str(files("evidence_rag").joinpath("fixtures/dataset.json"))))


@pytest.fixture
def traces(dataset, tmp_path):
    scenarios = files("evidence_rag").joinpath("fixtures/scenarios.json").read_text()
    path = tmp_path / "traces.json"
    write_json(
        path,
        {
            "schema_version": 1,
            "dataset_sha256": file_sha(dataset=dataset),
            "traces": json.loads(scenarios),
        },
    )
    return load_traces(path, dataset)
