from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture
def support_dataset():
    from evalforge.dataset import load_jsonl

    return load_jsonl(ROOT / "datasets" / "support-qa.jsonl")


@pytest.fixture
def redteam_dataset():
    from evalforge.dataset import load_jsonl

    return load_jsonl(ROOT / "datasets" / "redteam.jsonl")
