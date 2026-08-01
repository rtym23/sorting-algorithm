import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from .items import Item, ConveyorBelt, ItemGenerator
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.sorter import ItemClassifier, Category


class SortingZone(Enum):
    """Zones in the sorting cell."""
    INPUT = "A"  # Point A - conveyor input
    MAIN_SORTER = "B"  # Point B - main sorter
    OVERSIZED = "C"  # Point C - oversized items
    REPACKAGING = "D"  # Point D - repackaging


@dataclass
class ZoneConfig:
    """Configuration for a sorting zone."""
    name: str
    position: np.ndarray
    size: np.ndarray  # [length, width, height]
    zone_type: str  # "conveyor" or "rollcage"


@dataclass
class SortingEvent:
    """Event of an item being sorted."""
    item: Item
    category: Category
    target_zone: SortingZone
    timestamp: float
    success: bool
    reason: str = ""


class SortingCell:
    """
    Full sorting cell simulation.

    Layout (based on the provided scheme):
    - Point A: Conveyor input (fixed)
    - Point B: Main sorter zone (fixed relative to A)
    - Point C: Oversized items zone (free position)
    - Point D: Repackaging zone (free position)

    Working area: 6000 x 10000 mm
    """

    def __init__(self):
        # Zone configurations
        self.zones = {
            SortingZone.INPUT: ZoneConfig(
                name="Точка подачи товаров",
                position=np.array([0, 5000, 0]),
                size=np.array([500, 700, 2000]),
                zone_type="conveyor",
            ),
            SortingZone.MAIN_SORTER: ZoneConfig(
                name="Зона основного сортировщика",
                position=np.array([2000, 5000, 0]),
                size=np.array([500, 700, 2000]),
                zone_type="conveyor",
            ),
            SortingZone.OVERSIZED: ZoneConfig(
                name="Зона товаров неправильных габаритов",
                position=np.array([4000, 2000, 0]),
                size=np.array([1200, 800, 800]),
                zone_type="rollcage",
            ),
            SortingZone.REPACKAGING: ZoneConfig(
                name="Зона товаров неправильной формы",
                position=np.array([4000, 8000, 0]),
                size=np.array([1200, 800, 800]),
                zone_type="rollcage",
            ),
        }

        # Conveyor
        self.conveyor = ConveyorBelt(
            length=6000,
            width=500,
            height=700,
            speed=1000,
        )

        # Classifier
        self.classifier = ItemClassifier()

        # Item generator
        self.generator = ItemGenerator()

        # Sorting history
        self.events: list[SortingEvent] = []

        # Zone counters
        self.zone_counts = {zone: 0 for zone in SortingZone}

        # Time tracking
        self._time = 0.0
        self._item_counter = 0

    def add_item(self, item: Item):
        """Add an item to the sorting cell."""
        self.conveyor.add_item(item)
        self._item_counter += 1

    def classify_item(self, item: Item) -> Category:
        """Classify an item based on its features."""
        result = self.classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        )
        item.category = result.category.value
        return result.category

    def get_target_zone(self, category: Category) -> SortingZone:
        """Get target zone for a category."""
        if category == Category.SUITABLE:
            return SortingZone.MAIN_SORTER
        elif category == Category.OVERSIZED:
            return SortingZone.OVERSIZED
        elif category == Category.NEEDS_REPACKAGING:
            return SortingZone.REPACKAGING
        else:
            return SortingZone.OVERSIZED

    def route_item(self, item: Item, category: Category) -> SortingEvent:
        """Route an item to the appropriate zone."""
        target_zone = self.get_target_zone(category)

        # Calculate success (simplified - in real system this would check feasibility)
        success = True
        reason = f"Направлен в зону {target_zone.value}"

        # Create event
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

        return event

    def update(self, dt: float) -> list[SortingEvent]:
        """
        Update the sorting cell by dt seconds.
        Returns list of sorting events.
        """
        self._time += dt
        events = []

        # Update conveyor
        arrived_items = self.conveyor.update(dt)

        # Process arrived items
        for item in arrived_items:
            # Classify
            category = self.classify_item(item)

            # Route
            event = self.route_item(item, category)
            events.append(event)

            # Remove from conveyor
            self.conveyor.remove_item(item)

        return events

    def run_batch(self, items: list[Item], dt: float = 0.01) -> list[SortingEvent]:
        """Run a batch of items through the sorting cell."""
        all_events = []

        # Add all items
        for item in items:
            self.add_item(item)

        # Run simulation
        max_time = 30.0  # 30 seconds max
        while self._time < max_time:
            events = self.update(dt)
            all_events.extend(events)

            # Check if all items processed
            if len(self.events) >= len(items):
                break

        return all_events

    def run_test_scenario(self) -> dict:
        """Run a test scenario with predefined items."""
        items = self.generator.get_all_test_items()
        events = self.run_batch(items)

        return self.get_report()

    def get_report(self) -> dict:
        """Get sorting report."""
        stats = {zone: count for zone, count in self.zone_counts.items()}

        # Classification breakdown
        cat_counts = {}
        for event in self.events:
            cat = event.category.value
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "total_items": len(self.events),
            "zone_distribution": {zone.value: count for zone, count in stats.items()},
            "category_distribution": cat_counts,
            "total_time": self._time,
            "items_per_second": len(self.events) / self._time if self._time > 0 else 0,
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


if __name__ == "__main__":
    cell = SortingCell()
    report = cell.run_test_scenario()

    print("=== Результаты сортировки ===")
    print(f"Всего обработано: {report['total_items']}")
    print(f"Время: {report['total_time']:.2f} сек")
    print(f"Производительность: {report['items_per_second']:.1f} items/sec")
    print()
    print("Распределение по зонам:")
    for zone, count in report["zone_distribution"].items():
        print(f"  {zone}: {count}")
    print()
    print("Распределение по категориям:")
    for cat, count in report["category_distribution"].items():
        print(f"  {cat}: {count}")
