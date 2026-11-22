# Visual checks of the refine heuristics

Small synthetic labels, built to isolate one situation each, run through
`create_graph(method="refine")` to check by eye that each heuristic does what it should
and to spot its side effects. These are development checks, not examples: the labels are not
realistic vessels.

| Notebook | What it checks |
|---|---|
| `01_spur_reach` | leaf pruning: branch lengths, projection guide, branches near a face, branches along z |
| `02_junction_reach` | merging of split junctions: branch pairs, crossings, thick vessels, crossings at different depths |
| `03_tunnel_width` | paths around holes: holes along z and y, real loops, rings |
| `04_border` | vessels cut by or touching the XY faces: `border`, `border_fraction` |
| `05_fixed_steps` | false anastomoses (thin bridges) and `keep_rings` |

Every sample is run in three variants: `2D` (the middle slice), `3D round` and `3D elongated`.
The elongated variant has cross-sections 4 times taller than wide, like the confocal labels.
It also has small bumps on the top and bottom surfaces, which grow z-spurs. The vessels stay
at the same positions, so vessels at different depths can touch.

## Running

- **VS Code**: open an `NN_*.py` file and run its `# %%` cells, or generate notebooks with
  `python make_notebooks.py` and open the `.ipynb` files. The `.py` files are the sources;
  the notebooks are not tracked.
- **Script (or an agent)**: `python 01_spur_reach.py` writes the figures to
  `output/01_spur_reach/` and prints the count tables. Run it from any directory with the
  pyvane environment.
- The interactive viewer needs `ipywidgets` and only appears in a notebook.

`output/<notebook>/summary.csv` has the counts of every graph. Some samples have expected
counts at the default knobs, set from the design of the label (not from pyvane's output):
`expected_ok` is True or False on those rows. A ✓ only means that the counts match, not
that the right branches were kept. Look at the figure. A ✗ is something to look at in
pyvane, not in the check.

## Files

- `labels.py`: scenes of tubes (`Scene`), their three variants, `Sample`.
- `pipeline.py`: `build_graph` (corrected thinning + `create_graph(..., full_output=True)`),
  `knob_sweep`, `trace_steps` (counts after each refine step), `leaf_notes` (what the leaf
  rule measures).
- `plots.py`: the static figure (`sweep_figure`) and the interactive viewer (`slice_viewer`).
- `report.py`: runs the samples of a notebook, shows or saves the figures, writes
  `summary.csv`.
- `make_notebooks.py`: `.py` sources to `.ipynb`.

## Figure legend

- Columns: the label with its skeleton, then one graph per knob value (or setting).
- 3D: top row, maximum projection along z (XY); bottom row, along y (XZ, z upward).
- Label: dark gray. Skeleton and graph paths: colored by z (colorbar). In a projection, the
  voxel of largest z is drawn. In 2D they are blue.
- Removed paths (outside the junction spheres): light gray.
- Nodes: circle, degree 1; diamond, degree 2 (ring); triangle, degree 3; square, degree 4+.
  Filled with their z color. A magenta outline means `border=True`.
- Panel title: knob value; branch points; leaves (of which border leaves); multi-edge pairs;
  self-loops; total length. Red if `check_graph` reports problems (listed under the table).
- `04_border`: numbers on the edges are `border_fraction`.
- Leaf notes (first column of `01_spur_reach` and of the along-face sample):
  - each leaf on a junction is labeled with its in-plane reach in parent radii; the rule
    removes it below `spur_reach`;
  - `G`: matched to a tip of the projection guide (kept in 3D);
  - `B`: border exemption;
  - the white crosses are the guide tips.

## Interactive viewer

`report.viewer(sample, variant)` shows the label slice by slice. Its controls:
- plane: XY or XZ;
- slice: the slice index;
- graph: one of the columns of the figure;
- checkboxes: skeleton (orange), graph paths (blue), nodes (red, only those within one slice;
  magenta outline if `border=True`), removed paths (gray).
