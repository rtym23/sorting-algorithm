#!/usr/bin/env python3
"""Intelligent Robotic Sorting System — command-line entry point.

The system detects, classifies and routes items on a conveyor line using
computer vision and a robot manipulator model. Three operation modes are
available:

* ``classify``  — classify built-in test items or a single 3D model file;
* ``simulate``  — run a sorting-cell simulation for a batch of items;
* ``robot``     — demonstrate a single pick-and-place cycle;
* ``pybullet``  — run the PyBullet physics simulation for a batch of items;
* ``full``      — classify + route + pick-and-place for a batch of items.

Example::

    python main.py --mode simulate --items 20
    python main.py --mode full --items 10 --output results.json
    python main.py --mode pybullet --items 10 --gui
    python main.py --mode classify --input path/to/model.stl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Make sibling packages importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arm.manipulator import RobotManipulator
from classifier.sorter import Category, ItemClassifier
from config import AppConfig, get_config
from simulation.items import ItemGenerator
from simulation.sorting_cell import SortingCell

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------- #
def setup_logging(config: AppConfig | None = None) -> None:
    """Configure the root logger from ``config.yaml`` (or sensible defaults)."""
    cfg = (config or get_config()).logging
    handlers = [logging.StreamHandler(sys.stdout)]
    if cfg.file:
        try:
            handlers.append(logging.FileHandler(cfg.file, encoding="utf-8"))
        except OSError as exc:
            logger.warning("Cannot open log file %r: %s", cfg.file, exc)

    logging.basicConfig(
        level=getattr(logging, cfg.level.upper(), logging.INFO),
        format=cfg.format,
        handlers=handlers,
    )


# --------------------------------------------------------------------- #
# Demo modes
# --------------------------------------------------------------------- #
def run_classifier_demo() -> None:
    """Classify the built-in test items and print a summary."""
    header = "=" * 60
    print(header)
    print("CLASSIFIER DEMO")
    print(header)

    classifier = ItemClassifier()
    generator = ItemGenerator()

    results: dict[Category, list] = {cat: [] for cat in Category}
    for item in generator.get_all_test_items():
        result = classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )
        results[result.category].append((item, result))

    for category, items in results.items():
        print(f"\n{'='*40}")
        print(f"Category: {category.value}")
        print(f"{'='*40}")
        for item, result in items:
            dims = item.dimensions
            print(f"\n  {item.name}")
            print(f"    Dimensions: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm")
            print(f"    Roundness: {item.roundness:.2f}")
            print(f"    Reason: {result.reason}")

    print(f"\n{header}")
    print("SUMMARY:")
    for cat, items in results.items():
        print(f"  {cat.value}: {len(items)} items")
    print(header)


def run_sorting_simulation(num_items: int = 20) -> dict:
    """Run the sorting-cell simulation and return its report."""
    print("=" * 60)
    print(f"SORTING CELL SIMULATION ({num_items} items)")
    print("=" * 60)

    cell = SortingCell()
    generator = ItemGenerator()

    items = [
        generator.get_test_item(i)
        if i < len(generator.TEST_ITEMS)
        else generator.generate_random(seed=i)
        for i in range(num_items)
    ]
    print(f"\nGenerated {len(items)} items for sorting")

    start = time.perf_counter()
    logger.info("Starting batch simulation with %d items", num_items)
    cell.run_batch(items, dt=0.01)
    sim_time = time.perf_counter() - start

    report = cell.get_report()
    print("\nSimulation results:")
    print(f"  Simulation time: {report['total_time']:.2f} sec")
    print(f"  Real time: {sim_time:.2f} sec")
    print(f"  Processed: {report['total_items']} items")
    print(f"  Successful: {report['successful']}")
    print(f"  Failed: {report['failed']}")
    print(f"  Timeouts: {report['timeout_items']}")
    print(f"  Throughput: {report['items_per_second']:.2f} items/sec")

    print("\nDistribution by zone:")
    for zone, count in report["zone_distribution"].items():
        print(f"  Zone {zone}: {count} items")
    print("\nDistribution by category:")
    for cat, count in report["category_distribution"].items():
        print(f"  {cat}: {count} items")
    return report


def run_robot_demo() -> None:
    """Demonstrate pick-and-place for a few representative scenarios."""
    print("=" * 60)
    print("ROBOT MANIPULATOR DEMO")
    print("=" * 60)

    robot = RobotManipulator()
    scenarios = [
        ("Transfer to Zone B (main sorter)",
         np.array([1000, 5000, 800]), np.array([200, 150, 100]),
         np.array([2000, 5000, 800])),
        ("Transfer to Zone C (oversized)",
         np.array([1000, 5000, 800]), np.array([500, 400, 400]),
         np.array([4000, 2000, 400])),
        ("Transfer to Zone D (repackaging)",
         np.array([1000, 5000, 800]), np.array([150, 150, 200]),
         np.array([4000, 8000, 400])),
    ]

    for name, pick_pos, dims, place_pos in scenarios:
        print(f"\n{'='*40}")
        print(f"Scenario: {name}")
        print(f"{'='*40}")
        result = robot.pick_and_place(pick_pos, dims, place_pos)
        if result["success"]:
            print(f"  Result: Success (cycle {result['cycle_time']:.2f} sec)")
        else:
            print(f"  Result: Failed — {result.get('reason', 'Unknown')}")

    print(f"\n{'='*60}")
    print("ROBOT STATISTICS:")
    stats = robot.get_stats()
    print(f"  Picks completed: {stats['picks_completed']}")
    print(f"  Total move time: {stats['total_move_time']:.2f} sec")
    print(f"  Average cycle time: {stats['avg_cycle_time']:.2f} sec")
    print("=" * 60)


def run_full_system(num_items: int = 10) -> list[dict]:
    """Run the full pipeline (classify -> route -> pick-and-place) per item."""
    print("=" * 60)
    print(f"FULL SYSTEM INTEGRATION ({num_items} items)")
    print("=" * 60)

    classifier = ItemClassifier()
    generator = ItemGenerator()
    robot = RobotManipulator()

    zones = get_config().simulation.zones
    zone_positions = {
        "B": zones["B"].position,
        "C": zones["C"].position,
        "D": zones["D"].position,
    }
    pickup_position = zones["A"].position + np.array([200, 0, 800])

    items = [
        generator.get_test_item(i)
        if i < len(generator.TEST_ITEMS)
        else generator.generate_random(seed=i)
        for i in range(num_items)
    ]
    print(f"\nGenerated {len(items)} items")

    results: list[dict] = []
    total_cycle_time = 0.0
    failed_count = 0

    for i, item in enumerate(items):
        print(f"\n--- Item {i + 1}: {item.name} ---")
        dims = item.dimensions
        print(f"    Dimensions: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm")

        classification = classifier.classify_from_dimensions(
            dimensions=dims,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )
        print(f"    Category: {classification.category.value}")
        print(f"    Reason: {classification.reason}")

        target_zone = {
            Category.SUITABLE: "B",
            Category.OVERSIZED: "C",
            Category.NEEDS_REPACKAGING: "D",
        }[classification.category]

        timing = robot.pick_and_place(
            item_position=pickup_position,
            item_dimensions=dims,
            target_position=zone_positions[target_zone],
        )

        if timing["success"]:
            print(f"    Zone: {target_zone}")
            print(f"    Cycle time: {timing['cycle_time']:.2f} sec")
            total_cycle_time += timing["cycle_time"]
        else:
            failed_count += 1
            print(f"    FAILED: {timing.get('reason', 'Unknown error')}")

        results.append({
            "item": item.name,
            "category": classification.category.value,
            "zone": target_zone,
            "cycle_time": timing.get("cycle_time", 0.0),
            "success": timing["success"],
            "reason": timing.get("reason", ""),
        })

    # Summary
    print(f"\n{'='*60}")
    print("INTEGRATION SUMMARY:")
    print(f"  Total processed: {len(items)} items")
    print(f"  Successful: {len(items) - failed_count}")
    print(f"  Failed: {failed_count}")
    if total_cycle_time > 0:
        print(f"  Total time: {total_cycle_time:.2f} sec")
        print(f"  Average cycle: {total_cycle_time / len(items):.2f} sec")
        print(f"  Throughput: {len(items) / total_cycle_time:.2f} items/sec")
    else:
        print("  No successful operations completed")

    counts: dict[str, int] = {}
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print("\nDistribution:")
    for cat, count in counts.items():
        print(f"  {cat}: {count}")
    print("=" * 60)
    return results


def run_pybullet_demo(num_items: int = 10, gui: bool = False) -> dict:
    """Run the PyBullet physics simulation and print its report."""
    from arm.pybullet_sim import PyBulletSimulation, SimulationConfig

    print("=" * 60)
    print(f"PYBULLET PHYSICS SIMULATION ({num_items} items)")
    print("=" * 60)

    sim = PyBulletSimulation(SimulationConfig(gui=gui, items_to_process=num_items))
    try:
        report = sim.run_simulation(num_items)
        print("\nSimulation results:")
        print(f"  Total items: {report['total_items']}")
        print(f"  Successful: {report['successful']}")
        print(f"  Failed: {report['failed']}")
        print(f"  Sorted to zones: {report['sorted_items']}")

        for r in report["results"]:
            status = "OK" if r["success"] else "FAILED"
            print(f"\n  [{status}] {r['category']} -> Zone {r['target_zone']}")
            if r["success"]:
                print(f"    Cycle time: {r['cycle_time']:.2f} sec")
            else:
                print(f"    Reason: {r['reason']}")
        print("=" * 60)
        return report
    finally:
        sim.cleanup()


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intelligent Robotic Sorting System")
    parser.add_argument(
        "--mode",
        choices=["classify", "simulate", "robot", "pybullet", "full"],
        default="full",
        help="Operation mode (default: full)",
    )
    parser.add_argument("--items", type=int, default=10,
                        help="Number of items to process")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=disabled)")
    parser.add_argument("--input", type=str,
                        help="Path to a 3D model file to classify")
    parser.add_argument("--output", type=str,
                        help="Save results to this JSON file")
    parser.add_argument("--config", type=str,
                        help="Path to an alternative config.yaml")
    parser.add_argument("--gui", action="store_true",
                        help="Run the PyBullet simulation with a GUI window")
    return parser


def _save_results(results: list[dict], output: str) -> None:
    out_path = Path(output)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    logger.info("Results saved to %s", out_path)


def validate_args(args: argparse.Namespace) -> str | None:
    """Validate input parameters; returns an error message or None if valid."""
    if args.items is not None and args.items <= 0:
        return "Number of items must be positive"

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            return f"Input file not found: {args.input}"
        if input_path.suffix.lower() not in (".stl", ".step", ".stp"):
            return f"Unsupported file format: {input_path.suffix}"

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            return f"Configuration file not found: {args.config}"

    if args.output:
        output_path = Path(args.output)
        parent = output_path.parent
        if str(parent) and not parent.exists():
            return f"Output directory does not exist: {parent}"

    return None


def run_full_system_parallel(num_items: int = 10, workers: int = 4) -> list[dict]:
    """Run the full pipeline with parallel processing for larger batches."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("=" * 60)
    print(f"FULL SYSTEM (PARALLEL) ({num_items} items, {workers} workers)")
    print("=" * 60)

    classifier = ItemClassifier()
    generator = ItemGenerator()
    robot = RobotManipulator()

    zones = get_config().simulation.zones
    zone_positions = {
        "B": zones["B"].position,
        "C": zones["C"].position,
        "D": zones["D"].position,
    }
    pickup_position = zones["A"].position + np.array([200, 0, 800])

    items = [
        generator.get_test_item(i)
        if i < len(generator.TEST_ITEMS)
        else generator.generate_random(seed=i)
        for i in range(num_items)
    ]
    logger.info("Parallel processing: %d items with %d workers", num_items, workers)

    def process_single_item(i: int, item) -> dict:
        logger.debug("Processing item %d: %s", i + 1, item.name)
        dims = item.dimensions
        classification = classifier.classify_from_dimensions(
            dimensions=dims,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )
        target_zone = {
            Category.SUITABLE: "B",
            Category.OVERSIZED: "C",
            Category.NEEDS_REPACKAGING: "D",
        }[classification.category]
        timing = robot.pick_and_place(
            item_position=pickup_position,
            item_dimensions=dims,
            target_position=zone_positions[target_zone],
        )
        return {
            "item": item.name,
            "category": classification.category.value,
            "zone": target_zone,
            "cycle_time": timing.get("cycle_time", 0.0),
            "success": timing["success"],
            "reason": timing.get("reason", ""),
        }

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_item = {
            executor.submit(process_single_item, i, item): item
            for i, item in enumerate(items)
        }
        for future in as_completed(future_to_item):
            results.append(future.result())

    total_cycle_time = sum(r["cycle_time"] for r in results if r["success"])
    failed_count = sum(1 for r in results if not r["success"])

    print(f"\n{'='*60}")
    print("PARALLEL SUMMARY:")
    print(f"  Total processed: {len(results)} items")
    print(f"  Successful: {len(results) - failed_count}")
    print(f"  Failed: {failed_count}")
    if total_cycle_time > 0:
        print(f"  Total cycle time: {total_cycle_time:.2f} sec")
        print(f"  Throughput: {len(results) / total_cycle_time:.2f} items/sec")
    print("=" * 60)
    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = build_parser().parse_args(argv)

    # Input validation
    error = validate_args(args)
    if error:
        print(f"Error: {error}")
        return 1

    # Load custom configuration if provided
    import config as cfg_mod
    if args.config:
        try:
            cfg_mod._config = cfg_mod.AppConfig.load(args.config)
            logger.info("Loaded custom config from %s", args.config)
        except Exception as exc:
            logger.error("Failed to load config %s: %s", args.config, exc)
            return 1
    setup_logging()
    logger.info("Starting system in '%s' mode", args.mode)

    if args.mode == "classify":
        if args.input:
            classifier = ItemClassifier()
            logger.info("Classifying file: %s", args.input)
            try:
                result = classifier.classify_from_file(args.input)
            except (FileNotFoundError, ValueError) as exc:
                logger.error("%s", exc)
                return 1
            print(f"Category: {result.category.value}")
            print(f"Reason: {result.reason}")
        else:
            run_classifier_demo()

    elif args.mode == "simulate":
        report = run_sorting_simulation(args.items)

    elif args.mode == "robot":
        run_robot_demo()

    elif args.mode == "pybullet":
        report = run_pybullet_demo(args.items, gui=args.gui)
        if args.output:
            _save_results(report["results"], args.output)

    elif args.mode == "full":
        workers = getattr(args, "workers", 0) or 0
        if workers and args.items and args.items > 20:
            results = run_full_system_parallel(args.items, workers=workers)
        else:
            results = run_full_system(args.items)
        if args.output:
            _save_results(results, args.output)

    logger.info("System completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
