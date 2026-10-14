import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
# The previous implementation is used only when asked for explicitly, so the suite never
# depends on anything outside this repo by default.
LEGACY_ROOT = os.environ.get("LEGACY_EPF_ROOT")
DATA_ROOT = Path(os.environ.get("EPF_DATA_ROOT", PROJECT / "epf_data"))
EXTENDED_DIR = Path(
    os.environ.get("EPF_EXTENDED_DIR", PROJECT / "extended_dataset" / "distractor_results")
)


@pytest.fixture(scope="session")
def legacy():
    """The previous implementation, imported for parity checks only (opt-in)."""
    if LEGACY_ROOT is None:
        pytest.skip("parity checks are opt-in: set LEGACY_EPF_ROOT")
    root = Path(LEGACY_ROOT)
    if not (root / "babel-ship" / "probe_epf.py").exists():
        pytest.skip(f"previous implementation not found at {root}")
    for path in (str(root), str(root / "babel-ship")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import judge_client
    import probe_epf
    from benchmarking.mmau_pro import prompt, scoring
    from its_hub.core.algorithms import particle_filtering

    return {
        "probe_epf": probe_epf,
        "judge_client": judge_client,
        "prompt": prompt,
        "scoring": scoring,
        "particle_filtering": particle_filtering,
    }


@pytest.fixture(scope="session")
def data_paths():
    from epf_spf.data import DataPaths

    if not (DATA_ROOT / "mmar" / "MMAR-meta.json").exists():
        pytest.skip(f"benchmark data not found at {DATA_ROOT}")
    return DataPaths.from_layout(DATA_ROOT, EXTENDED_DIR)
