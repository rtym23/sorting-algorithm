import numpy as np

import main
from arm.manipulator import RobotManipulator
from classifier.sorter import Category, ItemClassifier
from simulation.items import ItemGenerator
from simulation.sorting_cell import SortingCell


class TestClassifierDemo:
    def test_demo_runs(self, capsys):
        main.run_classifier_demo()
        captured = capsys.readouterr()
        assert "SUMMARY" in captured.out
        assert "SUITABLE" in captured.out or "Suitable" in captured.out


class TestSortingSimulation:
    def test_simulation_runs(self):
        report = main.run_sorting_simulation(5)
        assert report["total_items"] == 5
        assert report["successful"] == 5


class TestFullSystem:
    def test_full_system_success_for_test_items(self):
        results = main.run_full_system(5)
        assert len(results) == 5
        for r in results:
            assert r["success"] is True
            assert r["zone"] in ("B", "C", "D")

    def test_full_system_detects_payload_failure(self):
        # The "Giant" test item (600x400x400 mm) is far too heavy to lift.
        generator = ItemGenerator()
        giant = generator.get_test_item(10)  # index 10 = Giant
        assert giant.dimensions.tolist() == [600, 400, 400]

        robot = RobotManipulator()
        result = robot.pick_and_place(
            item_position=np.array([500, 5000, 800]),
            item_dimensions=giant.dimensions,
            target_position=np.array([4000, 2000, 400]),
        )
        assert result["success"] is False
        assert "too heavy" in result["reason"]


class TestEndToEndPipeline:
    def test_classify_route_and_pick(self):
        cell = SortingCell()
        classifier = ItemClassifier()
        generator = ItemGenerator()

        item = generator.get_test_item(0)  # Small_Box
        category = classifier.classify_from_dimensions(
            dimensions=item.dimensions,
            roundness=item.roundness,
            has_circle=item.is_round and item.roundness >= 0.8,
        ).category

        assert category == Category.SUITABLE
        target = cell.get_target_zone(category)

        cell.add_item(item)
        cell.conveyor.update(100.0)
        events = []
        for item_on_belt in list(cell.conveyor.items):
            cat = cell.classify_item(item_on_belt)
            events.append(cell.route_item(item_on_belt, cat))
            cell.conveyor.remove_item(item_on_belt)

        assert len(events) == 1
        assert events[0].target_zone == target
        assert events[0].success is True
