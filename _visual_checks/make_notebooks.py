"""Converts the ``NN_*.py`` sources (``# %%`` percent format) into Jupyter notebooks.

The ``.py`` files are the sources; the notebooks are generated next to them and not tracked.
Cells start at a line ``# %%`` (code) or ``# %% [markdown]`` (markdown, whose lines are
comments). No dependency beyond the standard library.

Usage:
    python make_notebooks.py [source.py ...]     (default: every NN_*.py in this directory)
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
CELL_MARKER = re.compile(r"^# %%( \[markdown\])?\s*$")


def _markdown_line(line: str) -> str:
    """A commented markdown line without its comment mark."""
    if line.startswith("# "):
        return line[2:]
    return line[1:] if line.startswith("#") else line


def _cell(kind: str, lines: list[str]) -> dict | None:
    """A notebook cell from its source lines (None if empty)."""
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return None
    if kind == "markdown":
        lines = [_markdown_line(line) for line in lines]
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def to_notebook(source_path: Path) -> dict:
    """Notebook (nbformat 4) of a percent-format source."""
    cells, kind, lines = [], None, []
    for line in source_path.read_text().splitlines():
        match = CELL_MARKER.match(line)
        if match:
            if kind is not None and (cell := _cell(kind, lines)):
                cells.append(cell)
            kind, lines = ("markdown" if match.group(1) else "code"), []
        elif kind is not None:
            lines.append(line)
    if kind is not None and (cell := _cell(kind, lines)):
        cells.append(cell)
    return {"cells": cells, "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python",
                               "name": "python3"},
                "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def main(paths: list[str]) -> None:
    sources = [Path(path) for path in paths] or sorted(HERE.glob("[0-9][0-9]_*.py"))
    for source in sources:
        notebook_path = source.with_suffix(".ipynb")
        notebook_path.write_text(json.dumps(to_notebook(source), indent=1) + "\n")
        print(f"{source.name} -> {notebook_path.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
