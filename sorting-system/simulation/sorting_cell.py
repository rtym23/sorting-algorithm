"""Simulation of the sorting cell: conveyor, zones and routing logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from arm.manipulator import RobotManipulator
from classifier.sorter import Category, ItemClassifier
from config import SimulationConfig, get_config

from .items import ConveyorBelt, Item, ItemGenerator

logger = logging.getLogger(__name__)


class SortingZone(Enum):
    """Named zones in the sorting cell."""

    INPUT = "A"
    MAIN_SORTER = "B"
    OVERSIZED = "C"
    REPACKAGING = "D"


@dataclass
class ZoneConfig:
    """Geometry of a sorting zone."""

    name: str
    position: np.ndarray
    size: np.ndarray  # [length, width, height]
    zone_type: str  # "conveyor" or "rollcage"


@dataclass
class SortingEvent:
    """Record of one item being routed to a zone."""

    item: Item
    category: Category
    target_zone: SortingZone
    timestamp: float
    success: bool
    reason: str = ""


# Category -> destination zone
CATEGORY_TO_ZONE: dict[Category, SortingZone] = {
    Category.SUITABLE: SortingZone.MAIN_SORTER,
    Category.OVERSIZED: SortingZone.OVERSIZED,
    Category.NEEDS_REPACKAGING: SortingZone.REPACKAGING,
}


class SortingCell:
    """Full sorting cell simulation.

    Layout (working area 6000 x 10000 mm):

    * A — conveyor input
    * B — main sorter conveyor
    * C — oversized items roll-cage
    * D — repackaging roll-cage

    Items arrive on the conveyor, are classified and routed to the matching
    zone. A robot feasibility check decides whether a route actually succeeds.
    """

    def __init__(
        self,
        robot: RobotManipulator | None = None,
        config: SimulationConfig | None = None,
    ):
        self.config = config or get_config().simulation

        self.robot = robot if robot is not None else RobotManipulator()
        self.conveyor = self._build_conveyor()
        self.zones: dict[SortingZone, ZoneConfig] = self._build_zones()

        self.classifier = ItemClassifier()
        self.generator = ItemGenerator()

        self.events: list[SortingEvent] = []
        self.zone_counts = {zone: 0 for zone in SortingZone}

        self.timeout_items = 0
        self.failed_routes = 0

        self._time = 0.0
        self._item_counter = 0

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_conveyor(self) -> ConveyorBelt:
        cfg = self.config.conveyor
        return ConveyorBelt(
            length=cfg.length,
            width=cfg.width,
            height=cfg.height,
            speed=cfg.speed,
        )

    def _build_zones(self) -> dict[SortingZone, ZoneConfig]:
        zones = {}
        for key, zone in self.config.zones.items():
            try:
                zone_key = SortingZone(key.upper())
            except ValueError:
                logger.warning("Unknown zone key %r in config; skipping", key)
                continue
            zones[zone_key] = ZoneConfig(
                name=zone.name,
                position=zone.position,
                size=zone.size,
                zone_type=zone.zone_type,
            )
        return zones

    # ------------------------------------------------------------------ #
    # Flow
    # ------------------------------------------------------------------ #
    def add_item(self, item: Item) -> None:
        """Place an item on the conveyor."""
        self.conveyor.add_item(item)
        self._item_counter += 1

    def classify_item(self, item: Item) -> Category:
        """Classify an item and store the category on it."""
        result = self.classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )
        item.category = result.category.value
        return result.category

    @staticmethod
    def get_target_zone(category: Category) -> SortingZone:
        """Destination zone for a category (C is the fallback)."""
        return CATEGORY_TO_ZONE.get(category, SortingZone.OVERSIZED)

    def route_item(self, item: Item, category: Category) -> SortingEvent:
        """Route an item to the zone matching its category."""
        target_zone = self.get_target_zone(category)
        zone_cfg = self.zones.get(target_zone)

        if zone_cfg is None:
            success = False
            reason = f"Zone {target_zone.value} is not configured"
            self.failed_routes += 1
        elif self.robot.can_reach(zone_cfg.position):
            success = True
            reason = f"Routed to zone {target_zone.value}"
        else:
            success = False
            reason = (
                f"Target zone {target_zone.value} unreachable by robot "
                f"(reach {self.robot.config.reach} mm)"
            )
            self.failed_routes += 1

        event = SortingEvent(
            item=item,
            category=category,
            target_zone=target_zone,
            timestamp=self._time,
            success=success,
            reason=reason,
        )
        self.events.append(event)
        self.zone_counts[target_zone] += 1

        if not success:
            logger.warning("Route failed for %s: %s", item.name, reason)
        return event

    def update(self, dt: float) -> list[SortingEvent]:
        """Advance the simulation by ``dt`` seconds; return new events."""
        self._time += dt
        events: list[SortingEvent] = []

        for item in self.conveyor.update(dt):
            category = self.classify_item(item)
            events.append(self.route_item(item, category))
            self.conveyor.remove_item(item)

        return events

    # ------------------------------------------------------------------ #
    # Batch execution
    # ------------------------------------------------------------------ #
    def run_batch(
        self,
        items: list[Item],
        dt: float | None = None,
        max_time: float | None = None,
    ) -> list[SortingEvent]:
        """Process a batch of items and return all routing events."""
        dt = dt if dt is not None else self.config.dt
        max_time = max_time if max_time is not None else self.config.max_time

        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if max_time <= 0:
            raise ValueError(f"max_time must be positive, got {max_time}")

        for item in items:
            self.add_item(item)
            logger.debug("Queued item: %s", item.name)

        all_events: list[SortingEvent] = []
        progress_interval = max(1, len(items) // 10)
        while self._time < max_time and len(self.events) < len(items):
            all_events.extend(self.update(dt))
            processed = len(self.events)
            if processed % progress_interval == 0 and processed:
                logger.info(
                    "Progress: %d/%d items processed", processed, len(items)
                )

        self.timeout_items = max(0, len(items) - len(self.events))
        if self.timeout_items:
            logger.warning("%d item(s) not processed within %s sec",
                           self.timeout_items, max_time)
        return all_events

    def run_test_scenario(self) -> dict:
        """Run the built-in test scenario and return its report."""
        self.run_batch(self.generator.get_all_test_items())
        return self.get_report()

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def get_report(self) -> dict:
        """Aggregate a report of the whole simulation run."""
        successful = sum(1 for e in self.events if e.success)
        total = len(self.events)
        return {
            "total_items": total,
            "successful": successful,
            "failed": total - successful,
            "timeout_items": self.timeout_items,
            "success_rate": successful / total if total else 0.0,
            "zone_distribution": {
                zone.value: self.zone_counts[zone] for zone in SortingZone
            },
            "category_distribution": self._category_counts(),
            "total_time": self._time,
            "items_per_second": total / self._time if self._time > 0 else 0.0,
            "events": [
                {
                    "item": e.item.name,
                    "category": e.category.value,
                    "zone": e.target_zone.value,
                    "success": e.success,
                    "reason": e.reason,
                }
                for e in self.events
            ],
        }

    def _category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for event in self.events:
            cat = event.category.value
            counts[cat] = counts.get(cat, 0) + 1
        return counts
