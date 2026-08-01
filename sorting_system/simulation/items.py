import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import math


@dataclass
class Item:
    """Represents an item on the conveyor."""
    id: int
    name: str
    dimensions: np.ndarray  # [x, y, z] in mm
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_round: bool = False
    roundness: float = 0.0
    category: Optional[str] = None
    on_conveyor: bool = True

    @property
    def volume(self) -> float:
        return float(np.prod(self.dimensions))

    @property
    def max_dimension(self) -> float:
        return float(np.max(self.dimensions))

    def get_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        half = self.dimensions / 2.0
        return (self.position - half, self.position + half)


class ItemGenerator:
    """Generates random items for testing."""

    # Standard test items
    TEST_ITEMS = [
        ("Коробка_малая", np.array([100, 80, 60]), False, 0.3),
        ("Коробка_средняя", np.array([250, 200, 150]), False, 0.4),
        ("Коробка_большая", np.array([400, 300, 250]), False, 0.5),
        ("Цилиндр_малый", np.array([50, 50, 100]), True, 0.95),
        ("Цилиндр_большой", np.array([200, 200, 300]), True, 0.95),
        ("Шар", np.array([150, 150, 150]), True, 1.0),
        ("Куб", np.array([200, 200, 200]), False, 0.7),
        ("Параллелепипед", np.array([300, 150, 100]), False, 0.5),
        ("Тетраэдр", np.array([180, 180, 180]), False, 0.6),
        ("Миниатюра", np.array([5, 5, 5]), False, 0.3),
        ("Гигант", np.array([600, 400, 400]), False, 0.4),
        ("Труба", np.array([100, 100, 400]), True, 0.9),
        ("Конус", np.array([150, 150, 200]), True, 0.85),
        ("Коробка_узкая", np.array([350, 50, 50]), False, 0.3),
        ("Плоская_пластина", np.array([300, 200, 5]), False, 0.2),
    ]

    def __init__(self, start_id: int = 0):
        self._next_id = start_id

    def generate_random(self, seed: Optional[int] = None) -> Item:
        """Generate a random item with dimensions within reasonable range."""
        if seed is not None:
            np.random.seed(seed)

        # Random dimensions between 5mm and 500mm
        dims = np.random.uniform(5, 500, 3)

        # Randomly decide if it's round (cylinder/sphere)
        is_round = np.random.random() < 0.3
        if is_round:
            # Make two dimensions similar for roundness
            dims[1] = dims[0] * np.random.uniform(0.9, 1.1)

        roundness = np.random.uniform(0.1, 1.0)
        if is_round:
            roundness = np.random.uniform(0.8, 1.0)

        name = f"item_{self._next_id}"
        self._next_id += 1

        return Item(
            id=self._next_id - 1,
            name=name,
            dimensions=dims,
            is_round=is_round,
            roundness=roundness,
        )

    def get_test_item(self, index: int) -> Item:
        """Get a predefined test item by index."""
        if index >= len(self.TEST_ITEMS):
            raise IndexError(f"Test item index {index} out of range")

        name, dims, is_round, roundness = self.TEST_ITEMS[index]

        item = Item(
            id=self._next_id,
            name=name,
            dimensions=dims.copy(),
            is_round=is_round,
            roundness=roundness,
        )
        self._next_id += 1
        return item

    def get_all_test_items(self) -> list[Item]:
        """Get all predefined test items."""
        return [self.get_test_item(i) for i in range(len(self.TEST_ITEMS))]


class ConveyorBelt:
    """Simulates a conveyor belt with items moving along it."""

    def __init__(
        self,
        length: float = 6000.0,  # mm
        width: float = 500.0,  # mm
        height: float = 700.0,  # mm (from ground)
        speed: float = 1000.0,  # mm/s (= 1 m/s)
    ):
        self.length = length
        self.width = width
        self.height = height
        self.speed = speed
        self.items: list[Item] = []
        self._time = 0.0

        # Conveyor geometry
        self.start_x = 0.0
        self.end_x = length

    def add_item(self, item: Item, position: Optional[np.ndarray] = None):
        """Add an item to the conveyor."""
        if position is None:
            position = np.array([self.start_x + 50, self.width / 2, self.height + item.dimensions[2] / 2])

        item.position = position.copy()
        item.velocity = np.array([self.speed, 0, 0])
        item.on_conveyor = True
        self.items.append(item)

    def update(self, dt: float) -> list[Item]:
        """
        Update conveyor state by dt seconds.
        Returns list of items that have reached the end.
        """
        self._time += dt
        arrived = []

        for item in self.items:
            if item.on_conveyor:
                # Move item along conveyor
                item.position += item.velocity * dt

                # Check if item reached the end
                if item.position[0] >= self.end_x:
                    arrived.append(item)
                    item.on_conveyor = False

        return arrived

    def get_item_at_position(self, x: float, tolerance: float = 100.0) -> Optional[Item]:
        """Find item closest to given x position."""
        best_item = None
        best_dist = float("inf")

        for item in self.items:
            if item.on_conveyor:
                dist = abs(item.position[0] - x)
                if dist < tolerance and dist < best_dist:
                    best_dist = dist
                    best_item = item

        return best_item

    def remove_item(self, item: Item):
        """Remove an item from the conveyor."""
        if item in self.items:
            self.items.remove(item)

    @property
    def time(self) -> float:
        return self._time

    def get_stats(self) -> dict:
        """Get conveyor statistics."""
        active = sum(1 for i in self.items if i.on_conveyor)
        return {
            "time": self._time,
            "total_items": len(self.items),
            "active_items": active,
            "speed_ms": self.speed / 1000,
        }
