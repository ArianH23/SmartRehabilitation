import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11+

ROOT = Path(__file__).resolve().parents[1]


def _normalise(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement).lower()


def test_requirements_txt_matches_pyproject():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {_normalise(r) for r in pyproject["project"]["dependencies"]}

    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    listed = {_normalise(line) for line in lines if line.strip() and not line.lstrip().startswith("#")}

    assert listed == declared, "requirements.txt and pyproject.toml dependencies differ"
