"""Runs the samples of a notebook, shows or saves their figures and collects their counts.

In a Jupyter kernel the figures are shown inline. Run as a script (``python 01_spur_reach.py``)
they are only written to ``output/<notebook>/``. In both cases ``Report.finish`` writes
``output/<notebook>/summary.csv`` with the counts of every graph and the comparison with the
expected counts of the samples.
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from labels import Sample
from pipeline import (
    DEFAULT_KNOBS,
    GraphResult,
    build_graph,
    knob_sweep,
    leaf_notes,
    trace_steps,
)
from plots import slice_viewer, sweep_figure

OUTPUT_DIR = Path(__file__).parent / "output"


def in_notebook() -> bool:
    """Whether the code runs in a Jupyter kernel."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and "IPKernelApp" in shell.config


def _show(obj):
    """Displays an object in a notebook, or prints it in a script."""
    if in_notebook():
        from IPython.display import display
        display(obj)
    else:
        print(obj.to_string() if isinstance(obj, pd.DataFrame) else obj)


class Report:
    """The samples of one notebook, their figures and their counts.

    Args:
        notebook: Name of the notebook, used for the output directory.
    """

    def __init__(self, notebook: str):
        self.notebook = notebook
        self.out_dir = OUTPUT_DIR / notebook
        shutil.rmtree(self.out_dir, ignore_errors=True)     # no figures of older runs
        self.out_dir.mkdir(parents=True)
        self.rows = []
        # last results of each sample and variant, for the viewer
        self.results: dict[tuple[str, str], dict[str, GraphResult]] = {}

    def sweep(self, sample: Sample, knob: str, values, annotate_border: bool = False,
              annotate_leaves: bool = False, **fixed) -> dict[str, dict[str, GraphResult]]:
        """Graphs of every variant of a sample for several values of one knob.

        The other knobs keep their defaults (or the values in ``fixed``). The counts at the
        default value of the knob are compared with the expected counts of the sample.
        ``annotate_leaves`` annotates the leaves of the first column (``pipeline.leaf_notes``;
        meant for a first value of 0, which keeps every leaf).

        Returns:
            {variant: {column name: GraphResult}}.
        """
        default = DEFAULT_KNOBS[knob]
        by_variant = {}
        for variant, label in sample.labels().items():
            sweep = knob_sweep(label, knob, values, **fixed)
            columns = {self._column_name(knob, value, default): result
                       for value, result in sweep.items()}
            self._record(sample, variant, columns,
                         expected_column=self._column_name(knob, default, default)
                         if default in sweep else None)
            first = next(iter(columns))
            notes = {first: leaf_notes(columns[first])} if annotate_leaves else None
            self._figure(sample, variant, columns, f"{knob} sweep", annotate_border, notes)
            by_variant[variant] = columns
        self._show_counts(sample.name)
        return by_variant

    def compare(self, sample: Sample, settings: dict[str, dict], expected_column: str,
                annotate_border: bool = False) -> dict[str, dict[str, GraphResult]]:
        """Graphs of every variant of a sample for several named settings.

        Args:
            sample: The sample.
            settings: {column name: keyword arguments of ``build_graph``}.
            expected_column: Column whose counts are compared with the expected ones.
            annotate_border: Write the ``border_fraction`` of the edges.

        Returns:
            {variant: {column name: GraphResult}}.
        """
        by_variant = {}
        for variant, label in sample.labels().items():
            skeleton = None
            columns = {}
            for name, kwargs in settings.items():
                columns[name] = build_graph(label, skeleton, **kwargs)
                skeleton = columns[name].skeleton
            self._record(sample, variant, columns, expected_column)
            self._figure(sample, variant, columns, "", annotate_border)
            by_variant[variant] = columns
        self._show_counts(sample.name)
        return by_variant

    def trace(self, sample: Sample, variant: str, **knobs) -> pd.DataFrame:
        """Table of the counts after each refine step that changed the graph."""
        table = pd.DataFrame(trace_steps(sample.labels()[variant], **knobs))
        table.to_csv(self.out_dir / f"{sample.name}_{_slug(variant)}_trace.csv", index=False)
        return table

    def viewer(self, sample: Sample, variant: str):
        """Interactive slice viewer of the last graphs of a sample variant (notebook only)."""
        if not in_notebook():
            return None
        return slice_viewer(self.results[(sample.name, variant)], f"{sample.name}, {variant}")

    def finish(self) -> pd.DataFrame:
        """Writes ``summary.csv`` and returns the rows whose counts differ from the expected
        ones."""
        table = pd.DataFrame(self.rows)
        table.to_csv(self.out_dir / "summary.csv", index=False)
        mismatches = table[table["expected_ok"] == False]   # noqa: E712 (None means no expectation)
        print(f"{len(table)} graphs, {table['expected_ok'].notna().sum()} with expected counts, "
              f"{len(mismatches)} mismatches; summary in {self.out_dir / 'summary.csv'}")
        return mismatches

    # -------------------------------------------------------------------------------- helpers

    @staticmethod
    def _column_name(knob: str, value, default) -> str:
        name = f"{knob} = {value:g}"
        if value == default:
            return f"{name} (default)"
        return f"{name} (off)" if value == 0 else name

    def _record(self, sample: Sample, variant: str, columns: dict[str, GraphResult],
                expected_column: str | None):
        self.results[(sample.name, variant)] = columns
        expected = sample.expected_for(variant)
        for name, result in columns.items():
            row = {"sample": sample.name, "variant": variant, "graph": name, **result.counts,
                   "problems": len(result.problems), "expected": "", "expected_ok": None}
            if name == expected_column and expected:
                differences = {key: (value, result.counts[key])
                               for key, value in expected.items() if result.counts[key] != value}
                row["expected"] = " ".join(f"{key}={value}" for key, value in expected.items())
                row["expected_ok"] = not differences and not result.problems
            self.rows.append(row)

    def _figure(self, sample: Sample, variant: str, columns: dict[str, GraphResult],
                what: str, annotate_border: bool, notes: dict | None = None):
        title = f"{sample.name}, {variant}" + (f", {what}" if what else "")
        fig = sweep_figure(columns, title, annotate_border, notes)
        fig.savefig(self.out_dir / f"{sample.name}_{_slug(variant)}.png", dpi=110)
        if in_notebook():
            plt.show()
        else:
            plt.close(fig)

    def _show_counts(self, sample_name: str):
        table = pd.DataFrame([row for row in self.rows if row["sample"] == sample_name])
        columns = ["variant", "graph", "branch_points", "leaves", "border_leaves",
                   "multi_pairs", "self_loops", "length", "problems", "expected", "expected_ok"]
        _show(table[columns].set_index(["variant", "graph"]))
        for variant_results in [self.results[key] for key in self.results
                                if key[0] == sample_name]:
            for name, result in variant_results.items():
                for problem in result.problems:
                    print(f"check_graph, {name}: {problem}")


def _slug(text: str) -> str:
    return text.replace(" ", "_").replace("(", "").replace(")", "")
