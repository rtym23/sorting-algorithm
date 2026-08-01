# Project Report: Intelligent Robotic Sorting System

## 1. What the project is about

This is a software model of an automated sorting line. The system "recognizes"
items from their 3D models (STL/STEP), works out their size and shape, and sends
them to the right zone of the sorting cell, while a kinematic robot model does
the pick-and-place.

It's built from four parts:

| Module | What it does |
|--------|--------------|
| `classifier/` | Pulls geometric features out of 3D models and applies the classification rules |
| `simulation/` | Simulates the sorting cell: conveyor, zones A–D, routing |
| `arm/` | Kinematic robot model and the PyBullet physics simulation |
| `config.py` / `config.yaml` | Central configuration (units: mm, mm/s, kg, s) |

Classification follows fixed rules, applied in a strict order: **dimensions
first, then shape**.

| Category | Condition | Zone |
|----------|-----------|------|
| Suitable for sorting | Dimensions 10×10×10 … 450×320×320 mm, no round cross-section | B |
| Oversized / undersized | Smaller than 10×10×10 mm **or** larger than 450×320×320 mm | C |
| Needs repackaging | Dimensions are fine, but has a round cross-section (roundness ≥ 0.8) | D |

The roundness coefficient is r_inscribed / r_circumscribed of a section. The
system takes sections through the object's centroid along the three axes, and
then does a rotational "sweep" of the section planes around each axis — so a
round section is found no matter how the object is oriented.

## 2. How it was tested

Testing ran on Python 3.12 (Windows) with all dependencies from
`requirements-dev.txt`.

### 2.1. Automated tests and the linter

- `python -m pytest tests/` — **64 tests, all pass**.
- `python -m ruff check .` — clean, no warnings.

### 2.2. Every CLI mode checked

| Mode | Command | Result |
|------|---------|--------|
| classify (built-in items) | `python main.py --mode classify` | OK, 15 items sorted into 3 categories |
| classify (STL file) | `--input ../zip1/Stl/Box_400x400x300.stl` | OK |
| classify (STEP file) | `--input ../zip2/Step/Box_400x400x300.stp` | OK after fixes |
| simulate | `python main.py --mode simulate --items 20` | OK, 20/20 processed |
| robot | `python main.py --mode robot` | OK, 2/3 scenarios succeed (3rd fails on purpose) |
| full | `python main.py --mode full --items 10 --output results.json` | OK, 10/10, JSON saved |
| pybullet (headless) | `python main.py --mode pybullet --items 10` | OK, 10/10 |

### 2.3. Real 3D models (zip1/zip2)

All 11 STL and 11 STEP models were classified without a single crash.

## 3. Bugs found and fixed

### 3.1. The critical ones

1. **Stack overflow in the Welzl algorithm.**
   The minimum enclosing circle was computed with a recursive implementation of
   the Welzl algorithm, and recursion depth grew with the number of points in a
   section. On models with thousands of vertices (Helmet.stl, Handle.stl,
   Cleaning_Solution.STL, Helmet.stp) the system died with
   `RecursionError: maximum recursion depth exceeded`.
   **Fix:** swapped the recursion for an equivalent iterative incremental
   algorithm with a constant stack depth. Regression tests added.

2. **Wrong units for STEP files.**
   The cascadio/trimesh backend returns STEP geometry in metres, but the whole
   system works in millimetres. So `Box_400x400x300.stp` (400×300×400 mm) was
   classified as 0.4×0.3×0.4 mm and silently dumped into the "undersized"
   category.
   **Fix:** STEP/STP models are now automatically rescaled from metres to
   millimetres (×1000) on load; scaling tests added.

### 3.2. Dependencies and environment

3. **The STEP backend (cascadio) was missing.** The STEP support promised in the
   README didn't actually work — `trimesh.load` crashed with
   `ModuleNotFoundError`.
   **Fix:** `cascadio>=0.1.2` added to `requirements.txt` and `pyproject.toml`;
   if the backend is missing, the user gets a clear error instead of a crash.

4. **Heavy PyBullet import on every launch.** PyBullet was imported the moment
   the `arm` package was imported — it printed a build banner to stderr and
   slowed down every mode.
   **Fix:** made the import lazy (PEP 562, `__getattr__`); PyBullet only loads
   in `--mode pybullet`.

### 3.3. Performance

5. **Slow per-element Python loops.** "Distance to nearest edge" and the
   "point in polygon" test ran as loops over edges. On sections with ~2700 edges
   that meant dozens of seconds per model.
   **Fix:** both functions vectorized with NumPy. Speed-up:

   | Model | Before | After |
   |-------|--------|-------|
   | Helmet.stl | 6.9 s | 0.7 s |
   | Cleaning_Solution.STL | 9.5 s | 1.1 s |
   | Handle.stl | 8.4 s | 0.7 s |
   | Lunchbox.stl | 4.7 s | 0.4 s |
   | Helmet.stp | 30.5 s | 10.3 s* |

   *`Helmet.stp` includes ~6 s for the STEP decoding in cascadio itself.

### 3.4. Output quality and packaging

6. **Mojibake in the Windows console:** the em-dash "—" in the robot output
   rendered as "�". Replaced with a regular hyphen.
7. **Packaging:** `pyproject.toml` didn't list the root modules `config` and
   `main`, so `pip install .` produced a broken package and no console command.
   `py-modules` added.
8. **`.gitignore`:** `.coverage` and the root `.pytest_cache` weren't ignored;
   added.
9. **No license:** added `LICENSE` (MIT).

## 4. System results (after the fixes)

- Simulation of 20 random items: **20/20 processed, 0 failures**, throughput
  ~3.4 items/s (simulation time 5.95 s).
- Full cycle of 10 items: **10/10 successful**, average cycle 6.54 s.
- PyBullet (headless), 10 items: **10/10 successful**, zones B/C/D: 6/1/3.
- Robot: 2 of 3 demo scenarios succeed; the zone C scenario is deliberately
  rejected (a 40 kg item exceeds the 20 kg payload — the system is not
  "optimistic" and honestly reports the error).

## 5. Notes and limitations (deliberate trade-offs)

- **Hollow cylinders** (like `Cylinder.stl`, a pipe) may not reach roundness
  ≥ 0.8, because the inscribed circle is constrained by the inner hole. That's
  a consequence of how the coefficient is defined, not a bug.
- **STEP unit assumption:** we assume cascadio returns geometry in metres after
  conversion, and bring it to millimetres.
- The robot in PyBullet is a simplified visual model; the reach and payload
  checks are done by the analytic model (`arm/manipulator.py`).

## 6. Recommendations

1. Add CI (GitHub Actions): run `pytest` and `ruff` on every push.
2. Wire up a coverage report (`pytest-cov`, already present in the environment).
3. Move the shared and config fixtures into `conftest.py`.
4. For large models, add a disk-based feature cache (right now the cache only
   lives in the process memory).

## 7. Bottom line

The project is now in working shape: two critical bugs fixed (crash on large
models, wrong STEP units), dependencies, packaging and repo hygiene cleaned up,
four regression tests added. Final numbers: **64 tests pass, ruff is clean, all
5 CLI modes work, and the classifier handles all 22 bundled models.**
