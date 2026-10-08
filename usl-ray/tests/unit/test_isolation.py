"""Tests that keep the new implementation independent from the legacy package."""

from __future__ import annotations

import ast
from pathlib import Path


def test_new_package_does_not_import_legacy_package() -> None:
    """Ensure no source module imports the legacy ``usl`` package."""
    source_root = Path(__file__).parents[2] / "src" / "usl_ray"
    offenders: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "usl" or alias.name.startswith("usl.") for alias in node.names
            ):
                offenders.append(str(path))
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == "usl" or node.module.startswith("usl."))
            ):
                offenders.append(str(path))
    assert not offenders, f"Legacy imports found: {sorted(set(offenders))}"
