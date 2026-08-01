# Intelligent Robotic Sorting System

An automated sorting system for a conveyor line that combines computer vision
and robotics. It detects items, classifies them by size and shape, and routes
them to the right sorting zone.

## Features

- **3D model classification (STL/STEP)** — extracts geometric features: overall
  dimensions, volume, and the roundness coefficient of the cross-section.
- **Sorting cell simulation** — conveyor belt plus sorting zones A, B, C, D.
- **Kinematic robot model** — pick-and-place with checks for reach, payload,
  grip width, and clamping force.
- **PyBullet physics simulation** — runs headless or with a GUI window.
- **Centralized configuration** — everything lives in `config.yaml`.
- **Logging** — the system logs to a file instead of spamming the console.

### Classification rules

| Category | Rule | Zone |
|----------|------|------|
| "Suitable for Sorting" | Dimensions within 10x10x10 to 450x320x320 mm, no round cross-section | B |
| "Oversized/Undersized" | Smaller than 10x10x10 mm OR larger than 450x320x320 mm | C |
| "Needs Repackaging" | Dimensions OK, but has a round cross-section (roundness >= 0.8) | D |

Rules are applied in a fixed order: **dimensions first**, then shape.

Roundness coefficient = r_inscribed / r_circumscribed of a section. If ANY
section of the object has a coefficient >= 0.8, the object is treated as round.

Sections are taken through the object's centroid:

- along the three principal axes (X/Y/Z);
- plus a rotational sweep of the section plane around each axis
  (`classifier.cross_section_angles` steps). This catches round sections in
  objects that are rotated relative to the axes. As soon as a section with
  roundness >= the threshold is found, the analysis stops early.

## Installation

Requires Python 3.10+.

```bash
# Base install
pip install -r requirements.txt

# With dev tooling (pytest, ruff)
pip install -r requirements-dev.txt
```

> Note: STEP (`.stp`/`.step`) support needs the `cascadio` backend (already in
> `requirements.txt`). STEP geometry is expressed in metres by the CAD
> standard; the system automatically rescales it to millimetres.

## Sample 3D models

Ready-made models for testing live in two directories next to the project:

| Directory | Format | Contents |
|-----------|--------|----------|
| `zip1/Stl/`  | STL | Boxes, cylinders, a bottle, a helmet, a handle and more |
| `zip2/Step/` | STEP | The same items exported to the STEP format |

Use them like this:

```bash
python main.py --mode classify --input ../zip1/Stl/Helmet.stl
python main.py --mode classify --input ../zip2/Step/Box_400x400x300.stp
```

## Usage

All modes are run from the `sorting-system/` directory:

```bash
# Classify the built-in test items
python main.py --mode classify

# Classify a specific 3D model
python main.py --mode classify --input path/to/model.stl

# Run the sorting cell simulation (20 items)
python main.py --mode simulate --items 20

# Demonstrate the robot manipulator
python main.py --mode robot

# Full pipeline (classify + robot pick-and-place)
python main.py --mode full --items 10

# PyBullet physics simulation (headless)
python main.py --mode pybullet --items 10

# PyBullet physics simulation with a GUI window
python main.py --mode pybullet --items 10 --gui

# Save full-cycle results to JSON
python main.py --mode full --items 10 --output results.json

# Use an alternative config file
python main.py --mode full --config my_config.yaml

# Run the PyBullet simulation directly
python arm/pybullet_sim.py
```

### Running the tests

```bash
python -m pytest tests/ -v
python -m ruff check .
```

## Configuration

All parameters live in [`config.yaml`](config.yaml) and are loaded via
`config.py` (by default it picks up the file next to the module; you can
override the path with the `--config` flag).

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `robot` | `base_position` | `[3000, 5000, 0]` mm | Robot base position |
| `robot` | `reach` | `3500` mm | Horizontal reach |
| `robot` | `height` | `1500` mm | Maximum lift height |
| `robot` | `max_payload` | `20` kg | Maximum payload |
| `robot` | `move_speed` | `2000` mm/s | Travel speed |
| `robot` | `item_density` | `5e-7` kg/mm³ | Item density (~500 kg/m³) |
| `robot.gripper` | `jaw_width` | `500` mm | Gripper jaw opening |
| `robot.gripper` | `max_force` | `300` N | Maximum clamping force |
| `classifier` | `roundness_threshold` | `0.8` | Roundness threshold |
| `classifier` | `cross_section_angles` | `4` | Sweep steps for section planes |
| `classifier` | `cache_enabled` | `true` | Cache extracted features |
| `simulation` | `conveyor.*` | — | Conveyor parameters |
| `simulation` | `zones.A/B/C/D` | — | Zone positions and sizes |
| `logging` | `level` | `INFO` | Log level |

The system is deliberately not "optimistic": it reports an error whenever an
item is out of reach, exceeds the payload, doesn't fit the gripper, or isn't
processed within the allotted time.

## Reliability and performance

- **Feature caching** — extraction results are cached by file path, size, and
  modification time, so reclassifying the same file never re-parses the mesh.
- **Fast roundness computation** — the minimum enclosing circle uses the exact
  Welzl algorithm (expected O(n)); the inscribed circle is estimated from the
  centroid-to-edge distance plus a short Monte Carlo refinement.
- **Early exit** — section analysis stops as soon as a round section is found.
- **Resilient to messy models** — non-watertight meshes don't crash the system:
  the volume is estimated from the convex hull with a warning in the log, and
  corrupted/empty files produce a clear error.
- **Robot checks** — besides reach and payload, the robot verifies item width
  against the jaw opening and the required clamping force.

## Project layout

```
sorting-system/
├── main.py                    # CLI entry point
├── config.py                  # Config loading and typing
├── config.yaml                # System parameters
├── pyproject.toml             # Packaging, pytest/ruff settings
├── requirements.txt           # Dependencies
├── requirements-dev.txt       # Dev dependencies (pytest, ruff)
├── classifier/                # Classifier
│   ├── feature_extractor.py   # Geometric feature extraction (STL/STEP)
│   └── sorter.py              # Classification logic
├── simulation/                # Simulation
│   ├── items.py               # Items and conveyor belt
│   └── sorting_cell.py        # Sorting cell logic
├── arm/                       # Robot
│   ├── manipulator.py         # Kinematic manipulator model
│   └── pybullet_sim.py        # PyBullet physics simulation
└── tests/                     # Unit and integration tests
    ├── conftest.py
    ├── test_config.py
    ├── test_sorter.py
    ├── test_feature_extractor.py
    ├── test_items.py
    ├── test_sorting_cell.py
    ├── test_manipulator.py
    └── test_integration.py
```

## Zone layout

```
Working area: 6000 x 10000 mm

   A (input)                       B (main sorter)
   [conveyor 500x700]              [conveyor 500x700]
      |                               |
      +-------------------------------+
      |
======================================
|          Working area               |
======================================
      |                               |
  C (oversized)                 D (repackaging)
  [roll-cage                  [roll-cage
   1200x800x800]               1200x800x800]
```

## Tech stack

- **Python 3.10+** — primary language
- **trimesh** — 3D model processing (STL/STEP)
- **cascadio** — STEP (ISO 10303) geometry backend for trimesh
- **PyBullet** — physics simulation
- **NumPy/SciPy** — numerical computing
- **PyYAML** — configuration
- **pytest** — testing
- **ruff** — linting
