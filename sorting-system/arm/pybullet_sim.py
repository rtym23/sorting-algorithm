"""PyBullet-based physical simulation of the sorting cell.

This is the physics layer of the sorting system: it creates a ground plane, a
simplified conveyor, the four sorting zones and a robot, then processes items.
Unlike the analytic :mod:`arm.manipulator` model, the items are real rigid
bodies here — they are physically moved to their target zone, and the routing
is gated by the same robot feasibility checks (reach, payload, grip) so the
outcome is consistent with the rest of the pipeline.

Run headless (no window)::

    python arm/pybullet_sim.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

# Make sibling packages importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arm.manipulator import RobotManipulator
from classifier.sorter import Category, ItemClassifier
from config import get_config
from simulation.items import Item, ItemGenerator

logger = logging.getLogger(__name__)

MM_TO_M = 1e-3


@dataclass
class SimulationConfig:
    """PyBullet simulation options."""

    gui: bool = True
    time_step: float = 1.0 / 240.0
    conveyor_speed: float = 1.0  # m/s
    robot_reach: float = 1.2  # m
    items_to_process: int = 10


class PyBulletSimulation:
    """Physical (PyBullet) simulation of the sorting cell.

    Creates a scene with a ground plane, conveyor belt, robot manipulator and
    the four sorting zones (A, B, C, D), then classifies and "processes" items.
    """

    # Height of each zone pad above the ground (m); the item drop position is
    # computed on top of the pad so items visibly rest in their zone.
    ZONE_PAD_HEIGHT = {"A": 0.7, "B": 0.7, "C": 0.4, "D": 0.4}

    def __init__(self, config: SimulationConfig | None = None):
        self.config = config or SimulationConfig()
        self.classifier = ItemClassifier()
        self.generator = ItemGenerator()
        # Analytic robot model — provides the feasibility checks (reach,
        # payload, grip) that gate physical routing.
        self.robot = RobotManipulator()

        self.physics_client: int | None = None
        self.items_in_scene: dict[int, int] = {}
        self.zone_visuals: dict[str, int] = {}
        self.zone_pad_positions: dict[str, list] = {}

        # Zone positions in meters, read from the shared config (mm).
        zones = get_config().simulation.zones
        self.zone_positions = {
            key: zone.position.astype(float) * MM_TO_M
            for key, zone in zones.items()
        }

        # Statistics
        self.sorted_items = {"A": 0, "B": 0, "C": 0, "D": 0}
        self.failed_items = 0

    def initialize(self):
        """Initialize PyBullet simulation."""
        if self.config.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.config.time_step, physicsClientId=self.physics_client)

        # Load ground
        self.ground_id = p.loadURDF("plane.urdf", [0, 0, 0])

        # Create scene
        self._create_conveyor()
        self._create_zones()
        self._create_robot()

    def _create_conveyor(self):
        """Create visual representation of the conveyor."""
        # Conveyor belt (simplified as a box)
        conveyor_length = 6.0  # m
        conveyor_width = 0.5  # m
        conveyor_height = 0.1  # m
        conveyor_pos = [3.0, 5.0, 0.35]  # Center of conveyor

        conveyor_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[conveyor_length / 2, conveyor_width / 2, conveyor_height / 2],
        )
        conveyor_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[conveyor_length / 2, conveyor_width / 2, conveyor_height / 2],
            rgbaColor=[0.3, 0.3, 0.3, 1],
        )
        self.conveyor_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=conveyor_shape,
            baseVisualShapeIndex=conveyor_visual,
            basePosition=conveyor_pos,
        )

        # Side walls
        wall_height = 0.05
        wall_thickness = 0.02

        # Left wall
        left_wall_pos = [3.0, 5.0 - conveyor_width / 2, 0.4 + wall_height / 2]
        left_wall = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[conveyor_length / 2, wall_thickness / 2, wall_height / 2],
        )
        left_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[conveyor_length / 2, wall_thickness / 2, wall_height / 2],
            rgbaColor=[0.5, 0.5, 0.5, 1],
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=left_wall,
            baseVisualShapeIndex=left_visual,
            basePosition=left_wall_pos,
        )

        # Right wall
        right_wall_pos = [3.0, 5.0 + conveyor_width / 2, 0.4 + wall_height / 2]
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=left_wall,
            baseVisualShapeIndex=left_visual,
            basePosition=right_wall_pos,
        )

    def _create_zones(self):
        """Create visual representations of sorting zones."""
        zone_colors = {
            "A": [0, 1, 0, 0.3],  # Green - input
            "B": [0, 0, 1, 0.3],  # Blue - main sorter
            "C": [1, 0, 0, 0.3],  # Red - oversized
            "D": [1, 1, 0, 0.3],  # Yellow - repackaging
        }

        zone_sizes = {
            "A": [0.5, 0.7, 0.01],
            "B": [0.5, 0.7, 0.01],
            "C": [1.2, 0.8, 0.01],
            "D": [1.2, 0.8, 0.01],
        }

        for zone_name, pos in self.zone_positions.items():
            color = zone_colors[zone_name]
            size = zone_sizes[zone_name]
            # The pad is a thin box; the drop height is its top surface.
            z = pos[2] + self.ZONE_PAD_HEIGHT.get(zone_name, 0.0)
            pad_pos = [pos[0], pos[1], z - size[2] / 2]

            shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=size,
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=size,
                rgbaColor=color,
            )
            self.zone_visuals[zone_name] = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=shape,
                baseVisualShapeIndex=visual,
                basePosition=pad_pos,
            )
            self.zone_pad_positions[zone_name] = pad_pos

            # Add text label
            p.addUserDebugText(
                f"Zone {zone_name}",
                [pos[0], pos[1], z],
                textColorRGB=[1, 1, 1],
                textSize=1.5,
            )

    def _create_robot(self):
        """Create a simplified robot manipulator."""
        # Robot base
        robot_pos = [3.0, 5.0, 0.7]

        base_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.1,
            height=0.3,
        )
        base_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.1,
            length=0.3,
            rgbaColor=[0.2, 0.2, 0.8, 1],
        )
        self.robot_base = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_shape,
            baseVisualShapeIndex=base_visual,
            basePosition=[robot_pos[0], robot_pos[1], robot_pos[2] - 0.5],
        )

        # Robot arm (simplified as boxes)
        arm1_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.6, 0.05, 0.05],
        )
        arm1_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.6, 0.05, 0.05],
            rgbaColor=[0.3, 0.3, 0.9, 1],
        )
        self.arm1 = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=arm1_shape,
            baseVisualShapeIndex=arm1_visual,
            basePosition=[robot_pos[0] + 0.6, robot_pos[1], robot_pos[2] + 0.1],
        )

        # Gripper
        gripper_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.15, 0.05],
        )
        gripper_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.1, 0.15, 0.05],
            rgbaColor=[0.8, 0.2, 0.2, 1],
        )
        self.gripper = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=gripper_shape,
            baseVisualShapeIndex=gripper_visual,
            basePosition=[robot_pos[0] + 1.2, robot_pos[1], robot_pos[2] + 0.1],
        )

    def add_item_to_scene(self, item: Item, position: np.ndarray | None = None) -> int:
        """Add an item to the simulation scene."""
        if position is None:
            position = np.array([0.0, 5.0, 0.75])

        # Convert mm to meters
        dims_m = item.dimensions / 1000.0
        pos_m = position / 1000.0

        # Create box shape
        shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=dims_m / 2,
        )

        # Color based on category
        if item.is_round and item.roundness >= 0.8:
            color = [1, 0.5, 0, 1]  # Orange for round items
        elif np.any(item.dimensions > np.array([450, 320, 320])):
            color = [1, 0, 0, 1]  # Red for oversized
        else:
            color = [0, 1, 0, 1]  # Green for suitable

        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=dims_m / 2,
            rgbaColor=color,
        )

        item_id = p.createMultiBody(
            baseMass=0.5,
            baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,
            basePosition=pos_m.tolist(),
        )

        self.items_in_scene[item.id] = item_id
        return item_id

    def classify_and_route(self, item: Item) -> dict:
        """Classify an item and determine its route."""
        result = self.classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )

        # Determine target zone
        if result.category == Category.SUITABLE:
            target_zone = "B"
        elif result.category == Category.OVERSIZED:
            target_zone = "C"
        else:
            target_zone = "D"

        return {
            "category": result.category.value,
            "target_zone": target_zone,
            "reason": result.reason,
            "confidence": result.confidence,
        }

    def _drop_position(self, zone_name: str, item: Item) -> list:
        """Position (m) where an item rests on the given zone pad."""
        pad_pos = self.zone_pad_positions[zone_name]
        dims_m = item.dimensions / 1000.0
        return [pad_pos[0], pad_pos[1], pad_pos[2] + dims_m[2] / 2]

    def simulate_item_processing(self, item: Item) -> dict:
        """Classify an item, gate routing with robot feasibility checks and
        physically move the item to its target zone when possible."""
        # Add item to scene at conveyor start
        item_pos = np.array([0.0, 5.0, 0.75])
        body_id = self.add_item_to_scene(item, item_pos)

        # Classify
        classification = self.classify_and_route(item)
        target_zone = classification["target_zone"]

        # Simulate conveyor movement (simplified)
        p.stepSimulation()

        # Gate routing with the analytic robot model (reach, payload, grip).
        robot_result = self.robot.pick_and_place(
            item_position=item_pos * 1000.0,
            item_dimensions=item.dimensions,
            target_position=(
                np.asarray(self.zone_positions[target_zone], dtype=float) * 1000.0
            ),
        )
        success = robot_result["success"]

        classification.update({
            "success": success,
            "reason": robot_result.get("reason", ""),
            "cycle_time": robot_result.get("cycle_time", 0.0),
        })

        if success:
            # Physically move the item onto the target zone pad.
            p.resetBasePositionAndOrientation(
                body_id,
                self._drop_position(target_zone, item),
                [0, 0, 0, 1],
            )
            self.sorted_items[target_zone] += 1
            logger.info("Item %s -> Zone %s (%s)", item.name, target_zone,
                        classification["category"])
        else:
            self.failed_items += 1
            logger.warning("Item %s NOT routed to zone %s: %s", item.name,
                           target_zone, classification["reason"])

        return classification

    def run_simulation(self, num_items: int = 10) -> dict:
        """Run the full simulation."""
        self.initialize()

        results = []
        for i in range(num_items):
            item = self.generator.get_test_item(i % len(self.generator.TEST_ITEMS))
            result = self.simulate_item_processing(item)
            results.append(result)

            # Step simulation. Only throttle to real-time when the GUI is visible.
            steps = 100
            if self.config.gui:
                for _ in range(steps):
                    p.stepSimulation()
                    time.sleep(self.config.time_step)
            else:
                for _ in range(steps):
                    p.stepSimulation()

        # Generate report
        report = {
            "total_items": num_items,
            "successful": num_items - self.failed_items,
            "failed": self.failed_items,
            "sorted_items": self.sorted_items.copy(),
            "results": results,
        }

        return report

    def cleanup(self):
        """Clean up simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)


def run_headless_simulation(num_items: int = 20):
    """Run simulation without GUI for testing."""
    config = SimulationConfig(gui=False, items_to_process=num_items)
    sim = PyBulletSimulation(config)

    try:
        report = sim.run_simulation(num_items)
        print("=== PyBullet Simulation Results ===")
        print(f"Total items: {report['total_items']}")
        print(f"Successful: {report['successful']}")
        print(f"Failed: {report['failed']}")
        print(f"Sorted to zones: {report['sorted_items']}")

        for r in report["results"]:
            status = "OK" if r["success"] else "FAILED"
            print(f"\n[{status}] Item -> {r['category']}")
            print(f"  Target: Zone {r['target_zone']}")
            if r["success"]:
                print(f"  Cycle time: {r['cycle_time']:.2f} sec")
            else:
                print(f"  Reason: {r['reason']}")

    finally:
        sim.cleanup()


if __name__ == "__main__":
    run_headless_simulation()
