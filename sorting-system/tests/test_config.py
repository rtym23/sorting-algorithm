"""Tests for the configuration layer."""

import numpy as np
import pytest

from config import (
    AppConfig,
    ClassifierConfig,
    get_config,
    reset_config,
)
from config import (
    RobotConfig as AppRobotConfig,
)


class TestAppConfig:
    def test_loads_from_yaml(self):
        cfg = AppConfig.load()
        # Defaults documented in config.yaml
        assert cfg.robot.reach == 3500.0
        assert cfg.robot.height == 1500.0
        assert cfg.classifier.roundness_threshold == 0.8
        assert cfg.classifier.cross_section_angles == 4
        assert cfg.simulation.zones["B"].zone_type == "conveyor"
        assert cfg.simulation.zones["D"].zone_type == "rollcage"

    def test_singleton(self):
        reset_config()
        assert get_config() is get_config()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AppConfig.load(tmp_path / "nope.yaml")

    def test_robot_config_from_dict(self):
        robot = AppRobotConfig.from_dict(
            {
                "base_position": [1, 2, 3],
                "reach": 99.0,
                "gripper": {"jaw_width": 123.0, "max_force": 456.0},
            }
        )
        assert np.array_equal(robot.base_position, [1, 2, 3])
        assert robot.reach == 99.0
        assert robot.gripper_jaw_width == 123.0
        assert robot.gripper_max_force == 456.0

    def test_classifier_config_defaults(self):
        cfg = ClassifierConfig.from_dict({})
        assert cfg.min_size == 10.0
        assert cfg.roundness_threshold == 0.8
