"""Keep pinned runtime versions synchronized with the repository requirements."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _normalize(requirement: str) -> str:
    return requirement.lower().replace("_", "-")


def test_runtime_requirements_are_synchronized() -> None:
    """Assert that root requirements and project metadata pin the same runtime."""
    project_root = Path(__file__).parents[2]
    repository_root = project_root.parent
    with (project_root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    project_dependencies = {_normalize(item) for item in project["project"]["dependencies"]}
    root_dependencies = {
        _normalize(line.strip())
        for line in (repository_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#") and not line.startswith("setuptools")
    }
    assert project_dependencies == root_dependencies
