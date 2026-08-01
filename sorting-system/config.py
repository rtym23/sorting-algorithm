"""Configuration management for the sorting system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class RobotConfig:
    """Robot manipulator configuration."""
    base_position: np.ndarray
    reach: float = 3500.0
    height: float = 1500.0
    max_payload: float = 20.0
    move_speed: float = 2000.0
    cycle_time: float = 3.0
    item_density: float = 5e-7
    gripper_jaw_width: float = 500.0
    gripper_max_force: float = 300.0
    gripper_type: str = "parallel"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotConfig:
        return cls(
            base_position=np.array(
                data.get("base_position", [3000, 5000, 0]), dtype=float
            ),
            reach=float(data.get("reach", 3500.0)),
            height=float(data.get("height", 1500.0)),
            max_payload=float(data.get("max_payload", 20.0)),
            move_speed=float(data.get("move_speed", 2000.0)),
            cycle_time=float(data.get("cycle_time", 3.0)),
            item_density=float(data.get("item_density", 5e-7)),
            gripper_jaw_width=float(data.get("gripper", {}).get("jaw_width", 500.0)),
            gripper_max_force=float(data.get("gripper", {}).get("max_force", 300.0)),
            gripper_type=str(data.get("gripper", {}).get("grip_type", "parallel")),
        )


@dataclass
class ClassifierConfig:
    """Feature extraction and classification configuration."""
    min_size: float = 10.0
    max_size_x: float = 450.0
    max_size_y: float = 320.0
    max_size_z: float = 320.0
    roundness_threshold: float = 0.8
    cross_section_angles: int = 36
    cache_enabled: bool = True
    monte_carlo_samples: int = 50
    welzl_randomized: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassifierConfig:
        return cls(
            min_size=float(data.get("min_size", 10.0)),
            max_size_x=float(data.get("max_size_x", 450.0)),
            max_size_y=float(data.get("max_size_y", 320.0)),
            max_size_z=float(data.get("max_size_z", 320.0)),
            roundness_threshold=float(data.get("roundness_threshold", 0.8)),
            cross_section_angles=int(data.get("cross_section_angles", 36)),
            cache_enabled=bool(data.get("cache_enabled", True)),
            monte_carlo_samples=int(data.get("monte_carlo_samples", 50)),
            welzl_randomized=bool(data.get("welzl_randomized", True)),
        )


@dataclass
class ConveyorConfig:
    """Conveyor belt configuration."""
    length: float = 6000.0
    width: float = 500.0
    height: float = 700.0
    speed: float = 1000.0


@dataclass
class ZoneConfig:
    """Sorting zone configuration."""
    name: str
    position: np.ndarray
    size: np.ndarray
    zone_type: str


@dataclass
class SimulationConfig:
    """Sorting cell simulation configuration."""
    conveyor: ConveyorConfig
    cell_working_area_x: float = 6000.0
    cell_working_area_y: float = 10000.0
    max_time: float = 30.0
    dt: float = 0.01
    zones: dict[str, ZoneConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SimulationConfig:
        conveyor_data = data.get("conveyor", {})
        zones_data = data.get("zones", {})

        zones = {}
        for key, zdata in zones_data.items():
            zones[key] = ZoneConfig(
                name=str(zdata.get("name", key)),
                position=np.array(zdata.get("position", [0, 0, 0]), dtype=float),
                size=np.array(zdata.get("size", [100, 100, 100]), dtype=float),
                zone_type=str(zdata.get("type", "conveyor")),
            )

        return cls(
            conveyor=ConveyorConfig(
                length=float(conveyor_data.get("length", 6000.0)),
                width=float(conveyor_data.get("width", 500.0)),
                height=float(conveyor_data.get("height", 700.0)),
                speed=float(conveyor_data.get("speed", 1000.0)),
            ),
            cell_working_area_x=float(
                data.get("cell", {}).get("working_area_x", 6000.0)
            ),
            cell_working_area_y=float(
                data.get("cell", {}).get("working_area_y", 10000.0)
            ),
            max_time=float(data.get("cell", {}).get("max_time", 30.0)),
            dt=float(data.get("cell", {}).get("dt", 0.01)),
            zones=zones,
        )


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoggingConfig:
        return cls(
            level=str(data.get("level", "INFO")),
            format=str(
                data.get(
                    "format",
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                )
            ),
            file=str(data.get("file", "")),
        )


@dataclass
class AppConfig:
    """Main application configuration."""
    robot: RobotConfig
    classifier: ClassifierConfig
    simulation: SimulationConfig
    logging: LoggingConfig

    @classmethod
    def load(cls, path: str | None = None) -> AppConfig:
        """Load configuration from YAML file."""
        if path is None:
            # Default to config.yaml in the same directory as this file
            base_dir = Path(__file__).parent
            path = base_dir / "config.yaml"

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls(
            robot=RobotConfig.from_dict(data.get("robot", {})),
            classifier=ClassifierConfig.from_dict(data.get("classifier", {})),
            simulation=SimulationConfig.from_dict(data.get("simulation", {})),
            logging=LoggingConfig.from_dict(data.get("logging", {})),
        )


# Global config instance (lazy loaded)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the global configuration instance (singleton pattern)."""
    global _config
    if _config is None:
        _config = AppConfig.load()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


