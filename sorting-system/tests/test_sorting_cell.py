import numpy as np
import pytest

from arm.manipulator import RobotConfig, RobotManipulator
from classifier.sorter import Category
from simulation.items import ItemGenerator
from simulation.sorting_cell import SortingCell, SortingZone


@pytest.fixture
def generator():
    return ItemGenerator()


class TestSortingCell:
    def test_run_batch_processes_all(self, generator):
        cell = SortingCell()
        items = [generator.get_test_item(i) for i in range(5)]
        events = cell.run_batch(items)
        assert len(events) == 5

        report = cell.get_report()
        assert report["total_items"] == 5
        assert report["successful"] == 5
        assert report["failed"] == 0
        assert report["success_rate"] == 1.0

    def test_zone_distribution(self, generator):
        cell = SortingCell()
        items = [generator.get_test_item(i) for i in range(5)]
        cell.run_batch(items)
        report = cell.get_report()

        # Small/Medium/Large boxes -> B, cylinders -> D
        assert report["zone_distribution"][SortingZone.MAIN_SORTER.value] == 3
        assert report["zone_distribution"][SortingZone.REPACKAGING.value] == 2

    def test_get_target_zone(self):
        cell = SortingCell()
        assert cell.get_target_zone(Category.SUITABLE) == SortingZone.MAIN_SORTER
        assert cell.get_target_zone(Category.OVERSIZED) == SortingZone.OVERSIZED
        assert (
            cell.get_target_zone(Category.NEEDS_REPACKAGING)
            == SortingZone.REPACKAGING
        )

    def test_timeout_detects_unprocessed_items(self, generator):
        cell = SortingCell()
        items = [generator.get_test_item(0)]
        cell.run_batch(items, dt=0.01, max_time=0.01)
        # Item cannot reach conveyor end in 0.01 s -> flagged as timeout.
        assert cell.timeout_items >= 1

    def test_invalid_dt_raises(self, generator):
        cell = SortingCell()
        with pytest.raises(ValueError):
            cell.run_batch([], dt=0)
        with pytest.raises(ValueError):
            cell.run_batch([], dt=-1.0)
        with pytest.raises(ValueError):
            cell.run_batch([], dt=0.01, max_time=0)

    def test_unreachable_zone_marks_failure(self, generator):
        # A robot with tiny reach cannot reach any zone.
        robot = RobotManipulator(
            RobotConfig(base_position=np.array([0, 0, 0]), reach=10, height=100)
        )
        cell = SortingCell(robot=robot)
        items = [generator.get_test_item(0)]
        cell.run_batch(items)
        report = cell.get_report()

        assert report["successful"] == 0
        assert report["failed"] == 1
        assert report["success_rate"] == 0.0
        assert cell.failed_routes == 1
