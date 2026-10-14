import json

import pytest

from epf_spf.config import build_config
from epf_spf.results import completed_cells, provenance

SETTINGS = {"benchmark": "mmar", "generator_model": "g", "sampling_temperature": 0.8}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_fingerprint_changes_with_any_result_relevant_setting():
    base = provenance(build_config("epf-project"), SETTINGS, {})["run_fingerprint"]
    other_t = provenance(build_config("epf-project", weight_temperature=1.0), SETTINGS, {})
    other_sampling = provenance(
        build_config("epf-project"), SETTINGS | {"sampling_temperature": 0.7}, {}
    )
    assert base != other_t["run_fingerprint"]
    assert base != other_sampling["run_fingerprint"]


def test_fingerprint_ignores_the_code_version():
    a = provenance(build_config("spf"), SETTINGS, {"git_commit": "aaa"})["run_fingerprint"]
    b = provenance(build_config("spf"), SETTINGS, {"git_commit": "bbb"})["run_fingerprint"]
    assert a == b


def test_resume_skips_finished_cells_and_retries_errors(tmp_path):
    output = tmp_path / "run.jsonl"
    write_rows(
        output,
        [
            {"unique_id": "a", "budget": 8, "error": None, "run_fingerprint": "f1"},
            {"unique_id": "b", "budget": 8, "error": "Timeout", "run_fingerprint": "f1"},
        ],
    )
    assert completed_cells(output, "f1") == {("a", 8)}


def test_a_file_never_mixes_configurations(tmp_path):
    output = tmp_path / "run.jsonl"
    write_rows(output, [{"unique_id": "a", "budget": 8, "error": None, "run_fingerprint": "t3"}])
    with pytest.raises(SystemExit, match="different configuration"):
        completed_cells(output, "t1")
