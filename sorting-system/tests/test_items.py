import numpy as np
import pytest

from simulation.items import ConveyorBelt, Item, ItemGenerator


class TestItem:
    def test_volume(self):
        item = Item(id=0, name="test", dimensions=np.array([10, 20, 30]))
        assert item.volume == 6000.0

    def test_max_dimension(self):
        item = Item(id=0, name="test", dimensions=np.array([10, 500, 30]))
        assert item.max_dimension == 500.0

    def test_get_aabb(self):
        item = Item(id=0, name="test", dimensions=np.array([100, 80, 60]))
        item.position = np.array([1000, 500, 700])
        lo, hi = item.get_aabb()
        assert np.allclose(lo, [950, 460, 670])
        assert np.allclose(hi, [1050, 540, 730])


class TestItemGenerator:
    def test_test_item_out_of_range(self):
        gen = ItemGenerator()
        with pytest.raises(IndexError):
            gen.get_test_item(1000)

    def test_get_all_test_items(self):
        gen = ItemGenerator()
        items = gen.get_all_test_items()
        assert len(items) == len(ItemGenerator.TEST_ITEMS)

    def test_generate_random_deterministic(self):
        gen_a = ItemGenerator()
        gen_b = ItemGenerator()
        item_a = gen_a.generate_random(seed=42)
        item_b = gen_b.generate_random(seed=42)
        assert np.allclose(item_a.dimensions, item_b.dimensions)
        assert item_a.roundness == item_b.roundness


class TestConveyorBelt:
    def test_item_moves_and_arrives(self):
        belt = ConveyorBelt(length=500, width=100, height=700, speed=100)
        item = Item(id=0, name="test", dimensions=np.array([10, 10, 10]))
        belt.add_item(item, position=np.array([0, 50, 705]))

        assert item.on_conveyor is True
        assert item.position[0] == 0

        arrived = []
        for _ in range(10):
            arrived.extend(belt.update(dt=1.0))

        assert len(arrived) == 1
        assert arrived[0] is item
        assert item.on_conveyor is False

    def test_remove_item(self):
        belt = ConveyorBelt(length=500, width=100, height=700, speed=100)
        item = Item(id=0, name="test", dimensions=np.array([10, 10, 10]))
        belt.add_item(item, position=np.array([0, 50, 705]))
        assert len(belt.items) == 1
        belt.remove_item(item)
        assert len(belt.items) == 0

    def test_get_stats(self):
        belt = ConveyorBelt(length=500, width=100, height=700, speed=100)
        item = Item(id=0, name="test", dimensions=np.array([10, 10, 10]))
        belt.add_item(item, position=np.array([0, 50, 705]))
        stats = belt.get_stats()
        assert stats["total_items"] == 1
        assert stats["active_items"] == 1
        assert stats["speed_ms"] == 0.1
