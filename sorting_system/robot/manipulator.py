import numpy as np
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class GripperState(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class RobotConfig:
    """Configuration for the robot manipulator."""
    base_position: np.ndarray
    reach: float = 1200.0  # mm
    height: float = 1500.0  # mm
    max_payload: float = 5.0  # kg
    cycle_time: float = 3.0  # seconds per pick-and-place


class RobotManipulator:
    """
    Simulated robot manipulator for pick-and-place operations.

    Supports SCARA or articulated robot configurations for conveyor sorting.
    """

    def __init__(self, config: Optional[RobotConfig] = None):
        if config is None:
            config = RobotConfig(
                base_position=np.array([3000, 5000, 0]),
                reach=1200,
                height=1500,
            )
        self.config = config
        self.gripper_state = GripperState.OPEN
        self.current_item = None
        self._position = config.base_position.copy()
        self._home_position = config.base_position.copy()

        # Performance tracking
        self.picks_completed = 0
        self.total_move_time = 0.0

    @property
    def position(self) -> np.ndarray:
        return self._position.copy()

    def move_to(self, target: np.ndarray, speed: float = 500.0) -> float:
        """
        Move robot to target position.
        Returns time taken for the move.
        """
        distance = np.linalg.norm(target - self._position)
        time = distance / speed  # seconds

        self._position = target.copy()
        self.total_move_time += time

        return time

    def pick(self, item_position: np.ndarray, item_dimensions: np.ndarray) -> bool:
        """
        Pick an item from the conveyor.
        Returns True if successful.
        """
        if self.gripper_state == GripperState.CLOSED:
            return False

        # Move to item
        approach_pos = item_position.copy()
        approach_pos[2] = self.config.height  # Approach from above
        self.move_to(approach_pos)

        # Descend to item
        self.move_to(item_position)

        # Close gripper
        self.gripper_state = GripperState.CLOSED
        self.current_item = {
            "position": item_position.copy(),
            "dimensions": item_dimensions.copy(),
        }

        # Lift
        lift_pos = item_position.copy()
        lift_pos[2] = self.config.height
        self.move_to(lift_pos)

        return True

    def place(self, target_zone_position: np.ndarray) -> float:
        """
        Place the current item at the target zone.
        Returns time taken for the operation.
        """
        if self.gripper_state == GripperState.OPEN:
            return 0.0

        total_time = 0.0

        # Move to above target zone
        approach_pos = target_zone_position.copy()
        approach_pos[2] = self.config.height
        total_time += self.move_to(approach_pos)

        # Descend
        total_time += self.move_to(target_zone_position)

        # Open gripper
        self.gripper_state = GripperState.OPEN
        self.current_item = None

        # Lift back up
        lift_pos = target_zone_position.copy()
        lift_pos[2] = self.config.height
        total_time += self.move_to(lift_pos)

        self.picks_completed += 1

        return total_time

    def return_to_home(self) -> float:
        """Return robot to home position."""
        return self.move_to(self._home_position)

    def pick_and_place(
        self,
        item_position: np.ndarray,
        item_dimensions: np.ndarray,
        target_position: np.ndarray,
    ) -> dict:
        """
        Perform a complete pick-and-place operation.
        Returns timing information.
        """
        timing = {}

        # Pick
        success = self.pick(item_position, item_dimensions)
        if not success:
            return {"success": False, "reason": "Gripper already closed"}

        timing["pick_time"] = self.total_move_time

        # Place
        place_time = self.place(target_position)
        timing["place_time"] = place_time
        timing["total_time"] = timing["pick_time"] + timing["place_time"]

        # Return home
        home_time = self.return_to_home()
        timing["return_time"] = home_time
        timing["cycle_time"] = timing["total_time"] + home_time

        timing["success"] = True

        return timing

    def can_reach(self, position: np.ndarray) -> bool:
        """Check if the robot can reach the given position."""
        horizontal_dist = np.linalg.norm(
            position[:2] - self.config.base_position[:2]
        )
        vertical_dist = abs(position[2] - self.config.height)

        total_dist = np.sqrt(horizontal_dist**2 + vertical_dist**2)
        return total_dist <= self.config.reach

    def get_stats(self) -> dict:
        """Get robot statistics."""
        return {
            "picks_completed": self.picks_completed,
            "total_move_time": self.total_move_time,
            "avg_cycle_time": (
                self.total_move_time / self.picks_completed
                if self.picks_completed > 0
                else 0
            ),
            "gripper_state": self.gripper_state.value,
            "position": self._position.tolist(),
        }


class Gripper:
    """Gripper configuration and simulation."""

    def __init__(
        self,
        jaw_width: float = 200.0,  # mm
        max_force: float = 50.0,  # N
        grip_type: str = "parallel",  # parallel, suction, adaptive
    ):
        self.jaw_width = jaw_width
        self.max_force = max_force
        self.grip_type = grip_type
        self.is_closed = False

    def can_grip(self, item_width: float) -> bool:
        """Check if gripper can grip an item of given width."""
        if self.grip_type == "parallel":
            return item_width <= self.jaw_width
        elif self.grip_type == "suction":
            return True  # Suction can grip any flat surface
        return False

    def grip_force(self, item_mass: float) -> float:
        """Calculate required grip force."""
        gravity = 9.81  # m/s^2
        safety_factor = 1.5
        return item_mass * gravity * safety_factor
