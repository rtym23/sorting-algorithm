# Intelligent Robotic Sorting System

An automated sorting system for a conveyor line that combines computer vision
and robotics. It detects items, classifies them by size and shape, and routes
them to the right sorting zone.

## What it does

- **3D model classification (STL/STEP)** — extracts geometric features: overall
  dimensions, volume, and the roundness coefficient of the cross-section.
- **Sorting cell simulation** — conveyor belt plus sorting zones A, B, C, D.
- **Kinematic robot model** — pick-and-place with checks for reach, payload,
  grip width, and clamping force.
- **PyBullet physics simulation** — runs headless or with a GUI window.
- **Centralized configuration** — everything lives in `config.yaml`.

## Quick start

```bash
cd sorting-system
pip install -r requirements.txt

# Classify the built-in test items
python main.py --mode classify

# Classify a specific 3D model
python main.py --mode classify --input ../zip1/Stl/Helmet.stl

# Run the sorting cell simulation (20 items)
python main.py --mode simulate --items 20

# Full pipeline (classify + robot pick-and-place)
python main.py --mode full --items 10

# PyBullet physics simulation (headless)
python main.py --mode pybullet --items 10
```

Requires Python 3.10+. STEP support needs the `cascadio` backend (already in
`requirements.txt`).

## Sample 3D models

| Directory | Format | Contents |
|-----------|--------|----------|
| `zip1/Stl/`  | STL | Boxes, cylinders, a bottle, a helmet, a handle and more |
| `zip2/Step/` | STEP | The same items exported to the STEP format |

## Project layout

```
sorting-system/
├── main.py                    # CLI entry point
├── config.py                  # Config loading and typing
├── config.yaml                # System parameters
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
```

## Documentation

- [Detailed README](sorting-system/README.md) — usage, configuration,
  reliability and performance notes.
- [Report](REPORT.md) — project report (in Russian).

## Tech stack

- **Python 3.10+**, **trimesh**, **cascadio**, **PyBullet**, **NumPy/SciPy**,
  **PyYAML**, **pytest**, **ruff**

## License

[MIT](LICENSE)
