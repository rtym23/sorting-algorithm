"""Kinematic model of the robot manipulator for pick-and-place operations.

The model is deliberately simple: it computes move times from straight-line
distances and checks feasibility (reach, payload, grip) before acting. Physical
collision simulation lives in :mod:`arm.pybullet_sim`.
"""

from __future__ import annotations

import logging
from enum import Enum

import numpy as np

from config import RobotConfig

logger = logging.getLogger(__name__)

GRAVITY = 9.81  # m/s^2


class GripperState(Enum):
    """State of the gripper jaws."""

    OPEN = "open"
    CLOSED = "closed"


class Gripper:
    """Simple gripper model used for grip-feasibility checks."""

    def __init__(
        self,
        jaw_width: float = 500.0,  # mm
        max_force: float = 300.0,  # N
        grip_type: str = "parallel",
    ):
        self.jaw_width = jaw_width
        self.max_force = max_force
        self.grip_type = grip_type
        self.is_closed = False

    def can_grip(self, item_width: float) -> bool:
        """Whether the gripper can grasp an item of the given width."""
        if self.grip_type == "suction":
            return True
        if self.grip_type == "parallel":
            return item_width <= self.jaw_width
        # "adaptive" grippers handle a moderate range.
        return item_width <= self.jaw_width * 1.5

    def required_force(self, item_mass: float, safety_factor: float = 1.5) -> float:
        """Minimum clamping force needed to hold ``item_mass``."""
        return item_mass * GRAVITY * safety_factor

    def can_hold(self, item_mass: float) -> bool:
        """Whether the gripper can produce enough force for ``item_mass``."""
        return self.required_force(item_mass) <= self.max_force


class RobotManipulator:
    """Simulated robot performing pick-and-place operations.

    The robot moves through a waypoint at ``height`` above the pick/place
    positions, which keeps the carried item clear of the cell while travelling.
    """

    def __init__(self, config: RobotConfig | None = None):
        self.config = config if config is not None else self._default_config()
        self.gripper = Gripper(
            jaw_width=self.config.gripper_jaw_width,
            max_force=self.config.gripper_max_force,
            grip_type=self.config.gripper_type,
        )
        self.gripper_state = GripperState.OPEN
        self.current_item: dict | None = None
        self._position = self.config.base_position.copy()
        self._home_position = self.config.base_position.copy()

        # Performance tracking
        self.picks_completed = 0
        self.total_move_time = 0.0

    @staticmethod
    def _default_config() -> RobotConfig:
        """Default robot configuration, preferring values from config.yaml."""
        try:
            from config import get_config

            return get_config().robot
        except Exception as exc:
            logger.debug("Falling back to built-in robot defaults (%s)", exc)
            return RobotConfig(base_position=np.array([3000, 5000, 0]))

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    @property
    def position(self) -> np.ndarray:
        """Current end-effector position."""
        return self._position.copy()

    # ------------------------------------------------------------------ #
    # Motion
    # ------------------------------------------------------------------ #
    def move_to(self, target: np.ndarray, speed: float | None = None) -> float:
        """Move the end-effector to ``target``; returns the travel time (s).

        The end-effector is assumed to move in a straight line at constant
        speed. This is an approximation of the actual trajectory used in the
        physics simulation.
        """
        target = np.asarray(target, dtype=float)
        if target.shape != (3,):
            raise ValueError(f"target must be a 3-vector, got shape {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError(f"target contains non-finite values: {target}")

        speed = speed if speed and speed > 0 else self.config.move_speed
        distance = float(np.linalg.norm(target - self._position))
        duration = distance / speed

        self._position = target.copy()
        self.total_move_time += duration
        return duration

    def estimate_mass(self, item_dimensions: np.ndarray) -> float:
        """Estimate item mass (kg) from its dimensions and assumed density."""
        dims = np.asarray(item_dimensions, dtype=float)
        if np.any(dims <= 0) or not np.all(np.isfinite(dims)):
            return 0.0
        return float(np.prod(dims)) * self.config.item_density

    # ------------------------------------------------------------------ #
    # Feasibility
    # ------------------------------------------------------------------ #
    def can_reach(self, position) -> bool:
        """Whether the robot can reach ``position`` (within height and reach)."""
        if position is None or np.size(position) < 2:
            return False

        position = np.asarray(position, dtype=float)
        if not np.all(np.isfinite(position)):
            return False

        horizontal = float(
            np.linalg.norm(position[:2] - self.config.base_position[:2])
        )
        vertical = float(position[2]) if np.size(position) >= 3 else 0.0

        if vertical < 0.0 or vertical > self.config.height:
            return False
        return horizontal <= self.config.reach

    def _can_lift(self, item_dimensions: np.ndarray) -> bool:
        """Whether the item fits in the gripper and is not too heavy."""
        mass = self.estimate_mass(item_dimensions)
        if mass > self.config.max_payload:
            return False
        width = float(np.max(np.asarray(item_dimensions, dtype=float)[:2]))
        if not self.gripper.can_grip(width):
            return False
        return self.gripper.can_hold(mass)

    # ------------------------------------------------------------------ #
    # Pick / place
    # ------------------------------------------------------------------ #
    def pick(self, item_position: np.ndarray, item_dimensions: np.ndarray) -> bool:
        """Pick an item; returns True on success."""
        if self.gripper_state == GripperState.CLOSED:
            logger.warning("Pick refused: gripper is already closed")
            return False
        if not self.can_reach(item_position):
            logger.warning("Pick refused: item out of reach at %s", item_position)
            return False
        if not self._can_lift(item_dimensions):
            logger.warning("Pick refused: item cannot be lifted (dimensions %s)",
                           item_dimensions)
            return False

        self._move_with_clearance(item_position)
        self.gripper_state = GripperState.CLOSED
        self.gripper.is_closed = True
        self.current_item = {
            "position": np.asarray(item_position, dtype=float).copy(),
            "dimensions": np.asarray(item_dimensions, dtype=float).copy(),
        }
        self._lift(item_position)
        return True

    def place(self, target_position: np.ndarray) -> float:
        """Place the held item; returns the time taken (0.0 if no item held)."""
        if self.gripper_state == GripperState.OPEN:
            return 0.0

        self._move_with_clearance(target_position)
        self.gripper_state = GripperState.OPEN
        self.gripper.is_closed = False
        self.current_item = None
        self._lift(target_position)
        self.picks_completed += 1
        return 0.0  # Timing is accounted in pick_and_place via total_move_time.

    def return_to_home(self) -> float:
        """Move the end-effector back to the home position."""
        return self.move_to(self._home_position)

    def _move_with_clearance(self, target: np.ndarray) -> float:
        """Approach ``target`` from above via a waypoint at lift height."""
        waypoint = np.asarray(target, dtype=float).copy()
        waypoint[2] = self.config.height
        total = self.move_to(waypoint)
        total += self.move_to(target)
        return total

    def _lift(self, position: np.ndarray) -> float:
        """Lift from ``position`` back up to the clearance height."""
        lift_pos = np.asarray(position, dtype=float).copy()
        lift_pos[2] = self.config.height
        return self.move_to(lift_pos)

    # ------------------------------------------------------------------ #
    # Composite operation
    # ------------------------------------------------------------------ #
    def pick_and_place(
        self,
        item_position: np.ndarray,
        item_dimensions: np.ndarray,
        target_position: np.ndarray,
    ) -> dict:
        """Perform a full pick-and-place cycle and report timing.

        Returns a dict with a ``success`` flag and either a ``reason`` (on
        failure) or the phase timings (on success).
        """
        if item_position is None or target_position is None:
            return {"success": False, "reason": "Invalid position provided"}
        if item_dimensions is None:
            return {"success": False, "reason": "Invalid item dimensions"}

        try:
            item_position = np.asarray(item_position, dtype=float)
            target_position = np.asarray(target_position, dtype=float)
            item_dimensions = np.asarray(item_dimensions, dtype=float)
        except (TypeError, ValueError):
            return {"success": False, "reason": "Invalid position provided"}

        if not np.all(np.isfinite(item_position)) or not np.all(
            np.isfinite(target_position)
        ):
            return {"success": False, "reason": "Invalid position provided"}
        if item_dimensions.size != 3 or np.any(item_dimensions <= 0):
            return {"success": False, "reason": "Invalid item dimensions"}

        if not self.can_reach(item_position):
            return {
                "success": False,
                "reason": f"Item position {item_position.tolist()} out of robot "
                          f"reach ({self.config.reach} mm)",
            }
        if not self.can_reach(target_position):
            return {
                "success": False,
                "reason": f"Target position {target_position.tolist()} out of "
                          f"robot reach ({self.config.reach} mm)",
            }

        mass = self.estimate_mass(item_dimensions)
        if mass > self.config.max_payload:
            return {
                "success": False,
                "reason": f"Item too heavy for max payload "
                          f"({self.config.max_payload} kg)",
            }
        if not self.gripper.can_grip(float(np.max(item_dimensions[:2]))):
            return {
                "success": False,
                "reason": f"Item too wide for gripper "
                          f"({self.config.gripper_jaw_width} mm)",
            }

        if self.gripper_state == GripperState.CLOSED:
            return {"success": False, "reason": "Gripper already closed"}

        t0 = self.total_move_time
        if not self.pick(item_position, item_dimensions):
            return {"success": False, "reason": "Gripper already closed"}
        t1 = self.total_move_time

        self.place(target_position)
        t2 = self.total_move_time

        home_time = self.return_to_home()
        t3 = self.total_move_time

        return {
            "success": True,
            "pick_time": t1 - t0,
            "place_time": t2 - t1,
            "return_time": home_time,
            "cycle_time": t3 - t0,
        }

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def get_stats(self) -> dict:
        """Aggregate statistics about the robot's operation."""
        return {
            "picks_completed": self.picks_completed,
            "total_move_time": self.total_move_time,
            "avg_cycle_time": (
                self.total_move_time / self.picks_completed
                if self.picks_completed > 0
                else 0.0
            ),
            "gripper_state": self.gripper_state.value,
            "position": self._position.tolist(),
        }



