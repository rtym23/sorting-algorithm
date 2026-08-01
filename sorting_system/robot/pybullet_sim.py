import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Optional
from dataclasses import dataclass

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.sorter import ItemClassifier, Category
from simulation.items import Item, ConveyorBelt, ItemGenerator
from robot.manipulator import RobotManipulator, RobotConfig


@dataclass
class SimulationConfig:
    """PyBullet simulation configuration."""
    gui: bool = True
    time_step: float = 1.0 / 240.0
    conveyor_speed: float = 1.0  # m/s
    robot_reach: float = 1.2  # m
    items_to_process: int = 10


class PyBulletSimulation:
    """
    PyBullet-based simulation of the sorting cell.

    Creates a physical simulation with:
    - Ground plane
    - Conveyor belt
    - Robot manipulator (SCARA-style)
    - Sorting zones (A, B, C, D)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        if config is None:
            config = SimulationConfig()

        self.config = config
        self.classifier = ItemClassifier()
        self.generator = ItemGenerator()

        # Physics client
        self.physics_client = None
        self.items_in_scene = {}
        self.zone_visuals = {}

        # Zone positions (meters, converted from mm)
        self.zone_positions = {
            "A": np.array([0.0, 5.0, 0.7]),  # Input conveyor
            "B": np.array([2.0, 5.0, 0.7]),  # Main sorter
            "C": np.array([4.0, 2.0, 0.4]),  # Oversized
            "D": np.array([4.0, 8.0, 0.4]),  # Repackaging
        }

        # Statistics
        self.sorted_items = {"A": 0, "B": 0, "C": 0, "D": 0}

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
                basePosition=[pos[0], pos[1], pos[2] - 0.3],
            )

            # Add text label
            p.addUserDebugText(
                f"Zone {zone_name}",
                [pos[0], pos[1], pos[2]],
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

    def add_item_to_scene(self, item: Item, position: Optional[np.ndarray] = None) -> int:
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

    def simulate_item_processing(self, item: Item) -> dict:
        """Simulate processing of a single item."""
        # Add item to scene at conveyor start
        item_pos = np.array([0.0, 5.0, 0.75])
        self.add_item_to_scene(item, item_pos)

        # Classify
        classification = self.classify_and_route(item)

        # Simulate conveyor movement (simplified)
        conveyor_time = 6.0  # seconds to traverse conveyor
        p.stepSimulation()

        # Route to zone
        target_zone = classification["target_zone"]
        target_pos = self.zone_positions[target_zone]

        # Update statistics
        self.sorted_items[target_zone] += 1

        return classification

    def run_simulation(self, num_items: int = 10) -> dict:
        """Run the full simulation."""
        self.initialize()

        results = []
        for i in range(num_items):
            item = self.generator.get_test_item(i % len(self.generator.TEST_ITEMS))
            result = self.simulate_item_processing(item)
            results.append(result)

            # Step simulation
            for _ in range(100):
                p.stepSimulation()
                time.sleep(1.0 / 240.0)

        # Generate report
        report = {
            "total_items": num_items,
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
        print(f"Sorted to zones: {report['sorted_items']}")

        for r in report["results"]:
            print(f"\nItem -> {r['category']}")
            print(f"  Target: Zone {r['target_zone']}")
            print(f"  Reason: {r['reason']}")

    finally:
        sim.cleanup()


if __name__ == "__main__":
    run_headless_simulation()
