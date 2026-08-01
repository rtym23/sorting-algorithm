# Intelligent Robotic Sorting System

## Description

An automated product sorting system for conveyor lines using computer vision and robotics technologies. The system detects, classifies, and physically routes items on a production line based on their dimensions and shape characteristics.

### Classification Rules

| Category | Rule | Zone |
|----------|------|------|
| "Suitable for Sorting" | Size 10x10x10 — 450x320x320 mm, no circle in cross-section | B |
| "Oversized/Undersized" | < 10x10x10 mm OR > 450x320x320 mm | C |
| "Needs Repackaging" | Size within range but has circular cross-section (roundness coeff >= 0.8) | D |

### Circle-in-Cross-Section Criterion

Roundness coefficient = r_inscribed / r_circumscribed

If the roundness coefficient is >= 0.8 in ANY cross-section of the object, the item is classified as "circular".

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Classifier Demo
```bash
python main.py --mode classify
```

### Sorting Simulation
```bash
python main.py --mode simulate --items 20
```

### Robot Manipulator Demo
```bash
python main.py --mode robot
```

### Full System Integration
```bash
python main.py --mode full --items 10
```

### Classify a Single File
```bash
python main.py --mode classify --input path/to/model.stl
```

## Project Structure

```
sorting_system/
├── main.py                 # Entry point
├── requirements.txt        # Dependencies
├── classifier/             # CV-based classifier
│   ├── __init__.py
│   ├── feature_extractor.py  # Geometric feature extraction
│   └── sorter.py            # Classification logic
├── simulation/             # Conveyor belt simulation
│   ├── __init__.py
│   ├── items.py             # Items and conveyor
│   └── sorting_cell.py      # Sorting cell logic
├── robot/                  # Robot manipulator
│   ├── __init__.py
│   ├── manipulator.py       # Manipulator logic
│   └── pybullet_sim.py      # PyBullet physics simulation
├── models/                 # 3D models (STL/STEP)
└── assets/                 # Additional resources
```

## Layout Diagram

```
Work Area: 6000 x 10000 mm

     A (feeding)                    B (main sorter)
     [conveyor 500x700]            [conveyor 500x700]
         |                               |
         +-------------------------------+
         |
    =======================================
    |                                     |
    |          Work Area                  |
    |                                     |
    =======================================
         |                               |
     C (oversized)                D (repackaging)
     [roller cage                  [roller cage
      1200x800x800]                 1200x800x800]
```

## Technology Stack

- **Python 3.12** — primary language
- **trimesh** — 3D model processing (STL)
- **PyBullet** — physics simulation
- **NumPy/SciPy** — numerical computations
- **OpenCV** — computer vision (for extensions)

## Classification Rules (Priority Order)

1. **Check dimensions first** (strict bounds: 10-450 x 10-320 x 10-320 mm)
2. **Then check shape** (circle in cross-section)

An item with circular shape AND out-of-dimension size → classified as "Oversized/Undersized"
