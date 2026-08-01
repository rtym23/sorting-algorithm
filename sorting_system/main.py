#!/usr/bin/env python3
"""
Intelligent Robotic Sorting System
==================================

Main entry point for the sorting system simulation.

This system:
1. Detects and classifies items on a conveyor belt
2. Routes items to appropriate zones based on:
   - Size (10-450 x 10-320 x 10-320 mm)
   - Shape (circle in cross-section check)
3. Physically moves items using a robot manipulator

Usage:
    python main.py --mode classify --input path/to/model.stl
    python main.py --mode simulate --items 20
    python main.py --mode full --items 10
"""

import argparse
import json
import time
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.sorter import ItemClassifier, Category
from classifier.feature_extractor import FeatureExtractor
from simulation.items import Item, ItemGenerator, ConveyorBelt
from simulation.sorting_cell import SortingCell
from robot.manipulator import RobotManipulator, RobotConfig


def run_classifier_demo():
    """Run the classifier demonstration."""
    print("=" * 60)
    print("CLASSIFIER DEMO")
    print("=" * 60)

    classifier = ItemClassifier()
    generator = ItemGenerator()

    items = generator.get_all_test_items()

    results = {
        Category.SUITABLE: [],
        Category.OVERSIZED: [],
        Category.NEEDS_REPACKAGING: [],
    }

    for item in items:
        result = classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )
        results[result.category].append((item, result))

    for category, items_list in results.items():
        print(f"\n{'='*40}")
        print(f"Category: {category.value}")
        print(f"{'='*40}")

        for item, result in items_list:
            dims = item.dimensions
            print(f"\n  {item.name}")
            print(f"    Dimensions: {dims[0]:.0f} x {dims[1]:.0f} x {dims[2]:.0f} mm")
            print(f"    Roundness: {item.roundness:.2f}")
            print(f"    Reason: {result.reason}")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    for cat, items_list in results.items():
        print(f"  {cat.value}: {len(items_list)} items")
    print("=" * 60)


def run_sorting_simulation(num_items: int = 20):
    """Run the sorting cell simulation."""
    print("=" * 60)
    print(f"SORTING CELL SIMULATION ({num_items} items)")
    print("=" * 60)

    cell = SortingCell()
    generator = ItemGenerator()

    # Generate items
    items = []
    for i in range(num_items):
        if i < len(generator.TEST_ITEMS):
            items.append(generator.get_test_item(i))
        else:
            items.append(generator.generate_random(seed=i))

    print(f"\nGenerated {len(items)} items for sorting")

    # Run simulation
    start_time = time.time()
    events = cell.run_batch(items, dt=0.01)
    sim_time = time.time() - start_time

    # Get report
    report = cell.get_report()

    print(f"\nSimulation results:")
    print(f"  Simulation time: {report['total_time']:.2f} sec")
    print(f"  Real time: {sim_time:.2f} sec")
    print(f"  Processed: {report['total_items']} items")
    print(f"  Throughput: {report['items_per_second']:.1f} items/sec")

    print(f"\nDistribution by zone:")
    for zone, count in report["zone_distribution"].items():
        print(f"  Zone {zone}: {count} items")

    print(f"\nDistribution by category:")
    for cat, count in report["category_distribution"].items():
        print(f"  {cat}: {count} items")

    return report


def run_robot_demo():
    """Run robot manipulator demonstration."""
    print("=" * 60)
    print("ROBOT MANIPULATOR DEMO")
    print("=" * 60)

    config = RobotConfig(
        base_position=np.array([3000, 5000, 0]),
        reach=1200,
        height=1500,
    )

    robot = RobotManipulator(config)

    # Simulate pick-and-place operations
    test_scenarios = [
        {
            "name": "Transfer to Zone B (main sorter)",
            "item_pos": np.array([1000, 5000, 800]),
            "item_dims": np.array([200, 150, 100]),
            "target_pos": np.array([2000, 5000, 800]),
        },
        {
            "name": "Transfer to Zone C (oversized)",
            "item_pos": np.array([1000, 5000, 800]),
            "item_dims": np.array([500, 400, 400]),
            "target_pos": np.array([4000, 2000, 400]),
        },
        {
            "name": "Transfer to Zone D (repackaging)",
            "item_pos": np.array([1000, 5000, 800]),
            "item_dims": np.array([150, 150, 200]),
            "target_pos": np.array([4000, 8000, 400]),
        },
    ]

    for scenario in test_scenarios:
        print(f"\n{'='*40}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*40}")

        result = robot.pick_and_place(
            item_position=scenario["item_pos"],
            item_dimensions=scenario["item_dims"],
            target_position=scenario["target_pos"],
        )

        print(f"  Result: {'Success' if result['success'] else 'Failed'}")
        if result["success"]:
            print(f"  Cycle time: {result['cycle_time']:.2f} sec")
        else:
            print(f"  Reason: {result.get('reason', 'Unknown')}")

    print(f"\n{'='*60}")
    print("ROBOT STATISTICS:")
    stats = robot.get_stats()
        print(f"  Picks completed: {stats['picks_completed']}")
        print(f"  Total move time: {stats['total_move_time']:.2f} sec")
        print(f"  Average cycle time: {stats['avg_cycle_time']:.2f} sec")
    print("=" * 60)


def run_full_system(num_items: int = 10):
    """Run the complete integrated system."""
    print("=" * 60)
    print(f"FULL SYSTEM INTEGRATION ({num_items} items)")
    print("=" * 60)

    # Initialize components
    classifier = ItemClassifier()
    generator = ItemGenerator()
    robot = RobotManipulator()

    # Zone positions (mm)
    zone_positions = {
        "B": np.array([2000, 5000, 800]),
        "C": np.array([4000, 2000, 400]),
        "D": np.array([4000, 8000, 400]),
    }

    # Generate items
    items = []
    for i in range(num_items):
        if i < len(generator.TEST_ITEMS):
            items.append(generator.get_test_item(i))
        else:
            items.append(generator.generate_random(seed=i))

    print(f"\nGenerated {len(items)} items")

    # Process each item
    results = []
    total_cycle_time = 0

    for i, item in enumerate(items):
        print(f"\n--- Item {i+1}: {item.name} ---")
        print(f"    Dimensions: {item.dimensions[0]:.0f} x {item.dimensions[1]:.0f} x {item.dimensions[2]:.0f} mm")

        # Classify
        classification = classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )

        print(f"    Category: {classification.category.value}")
        print(f"    Reason: {classification.reason}")

        # Determine target zone
        if classification.category == Category.SUITABLE:
            target_zone = "B"
        elif classification.category == Category.OVERSIZED:
            target_zone = "C"
        else:
            target_zone = "D"

        # Simulate pick-and-place
        item_pos = np.array([500, 5000, 800])
        target_pos = zone_positions[target_zone]

        timing = robot.pick_and_place(
            item_position=item_pos,
            item_dimensions=item.dimensions,
            target_position=target_pos,
        )

        if timing["success"]:
            print(f"    Zone: {target_zone}")
            print(f"    Cycle time: {timing['cycle_time']:.2f} sec")
            total_cycle_time += timing["cycle_time"]

        results.append({
            "item": item.name,
            "category": classification.category.value,
            "zone": target_zone,
            "cycle_time": timing.get("cycle_time", 0),
        })

    # Summary
    print(f"\n{'='*60}")
    print("INTEGRATION SUMMARY:")
    print(f"  Total processed: {len(items)} items")
    print(f"  Total time: {total_cycle_time:.2f} sec")
    print(f"  Average cycle: {total_cycle_time/len(items):.2f} sec")
    print(f"  Throughput: {len(items)/total_cycle_time:.1f} items/sec")

    # Category breakdown
    cat_counts = {}
    for r in results:
        cat = r["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print(f"\nDistribution:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Robotic Sorting System"
    )
    parser.add_argument(
        "--mode",
        choices=["classify", "simulate", "robot", "full"],
        default="full",
        help="Mode of operation",
    )
    parser.add_argument(
        "--items",
        type=int,
        default=10,
        help="Number of items to process",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to 3D model file for classification",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save results JSON",
    )

    args = parser.parse_args()

    if args.mode == "classify":
        if args.input:
            # Classify single file
            classifier = ItemClassifier()
            result = classifier.classify_from_file(args.input)
            print(f"Category: {result.category.value}")
            print(f"Reason: {result.reason}")
        else:
            # Run demo
            run_classifier_demo()

    elif args.mode == "simulate":
        run_sorting_simulation(args.items)

    elif args.mode == "robot":
        run_robot_demo()

    elif args.mode == "full":
        results = run_full_system(args.items)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
