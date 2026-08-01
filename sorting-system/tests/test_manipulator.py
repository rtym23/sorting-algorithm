import numpy as np
import pytest

from arm.manipulator import (
    Gripper,
    GripperState,
    RobotConfig,
    RobotManipulator,
)


@pytest.fixture
def robot():
    return RobotManipulator()


class TestRobotManipulator:
    def test_move_to_timing(self):
        config = RobotConfig(base_position=np.array([0, 0, 0]))
        robot = RobotManipulator(config)
        t = robot.move_to(np.array([1000.0, 0, 0]))
        assert t == pytest.approx(1000.0 / config.move_speed)

    def test_move_to_uses_configured_speed(self):
        config = RobotConfig(
            base_position=np.array([0, 0, 0]),
            move_speed=1000.0,
        )
        robot = RobotManipulator(config)
        t = robot.move_to(np.array([500.0, 0, 0]))
        assert t == pytest.approx(0.5)

    def test_pick_and_place_success(self):
        robot = RobotManipulator()
        result = robot.pick_and_place(
            item_position=np.array([2500, 5000, 800]),
            item_dimensions=np.array([200, 150, 100]),
            target_position=np.array([2000, 5000, 800]),
        )
        assert result["success"] is True
        assert result["cycle_time"] > 0
        assert robot.picks_completed == 1
        assert robot.gripper_state == GripperState.OPEN

    def test_pick_and_place_invalid_dimensions(self, robot):
        result = robot.pick_and_place(
            item_position=np.array([2500, 5000, 800]),
            item_dimensions=np.array([0, 10, 10]),
            target_position=np.array([2000, 5000, 800]),
        )
        assert result["success"] is False
        assert "Invalid item dimensions" in result["reason"]

    def test_pick_and_place_none_positions(self, robot):
        result = robot.pick_and_place(None, np.array([10, 10, 10]), None)
        assert result["success"] is False
        assert "Invalid position" in result["reason"]

    def test_item_out_of_reach(self, robot):
        result = robot.pick_and_place(
            item_position=np.array([7000, 5000, 800]),
            item_dimensions=np.array([100, 100, 100]),
            target_position=np.array([2000, 5000, 800]),
        )
        assert result["success"] is False
        assert "out of robot reach" in result["reason"]

    def test_target_out_of_reach(self, robot):
        result = robot.pick_and_place(
            item_position=np.array([2500, 5000, 800]),
            item_dimensions=np.array([100, 100, 100]),
            target_position=np.array([8000, 9000, 800]),
        )
        assert result["success"] is False
        assert "out of robot reach" in result["reason"]

    def test_payload_too_heavy(self):
        robot = RobotManipulator()
        # 4000 x 4000 x 4000 mm at 500 kg/m^3 is far above a 20 kg payload.
        result = robot.pick_and_place(
            item_position=np.array([2500, 5000, 800]),
            item_dimensions=np.array([4000, 4000, 4000]),
            target_position=np.array([2000, 5000, 800]),
        )
        assert result["success"] is False
        assert "too heavy" in result["reason"]

    def test_can_reach_bounds(self, robot):
        assert robot.can_reach(np.array([3000, 5000, 0])) is True
        assert robot.can_reach(np.array([8000, 5000, 0])) is False
        assert robot.can_reach(np.array([3000, 5000, -1])) is False
        assert robot.can_reach(np.array([3000, 5000, 2000])) is False
        assert robot.can_reach(None) is False

    def test_double_pick_fails(self, robot):
        robot.pick(np.array([2500, 5000, 800]), np.array([100, 100, 100]))
        assert robot.gripper_state == GripperState.CLOSED
        # Second pick while gripper closed must fail.
        assert (
            robot.pick(np.array([2500, 5000, 800]), np.array([100, 100, 100]))
            is False
        )

    def test_place_without_item_returns_zero(self, robot):
        assert robot.place(np.array([2000, 5000, 800])) == 0.0

    def test_stats(self, robot):
        robot.pick_and_place(
            item_position=np.array([2500, 5000, 800]),
            item_dimensions=np.array([100, 100, 100]),
            target_position=np.array([2000, 5000, 800]),
        )
        stats = robot.get_stats()
        assert stats["picks_completed"] == 1
        assert stats["avg_cycle_time"] > 0


class TestGripper:
    def test_suction_grips_anything(self):
        gripper = Gripper(grip_type="suction")
        assert gripper.can_grip(10000.0) is True

    def test_parallel_grip_width_limit(self):
        gripper = Gripper(jaw_width=200.0)
        assert gripper.can_grip(150.0) is True
        assert gripper.can_grip(250.0) is False

    def test_can_hold_force_limit(self):
        # 6 kg at 1.5x safety needs ~88 N; 300 N gripper can hold it.
        gripper = Gripper(max_force=300.0)
        assert gripper.can_hold(6.0) is True
        assert gripper.can_hold(40.0) is False

    def test_required_force_scales_with_mass(self):
        gripper = Gripper()
        assert gripper.required_force(10.0) == pytest.approx(10.0 * 9.81 * 1.5)
