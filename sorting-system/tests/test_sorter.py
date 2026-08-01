import numpy as np
import pytest

from classifier.sorter import Category, ItemClassifier


@pytest.fixture
def classifier():
    return ItemClassifier()


class TestItemClassifier:
    def test_suitable_box(self, classifier):
        result = classifier.classify_from_dimensions(
            dimensions=np.array([100, 80, 60]),
            roundness=0.3,
            has_circle=False,
        )
        assert result.category == Category.SUITABLE
        assert result.confidence == 1.0

    def test_oversized_large(self, classifier):
        result = classifier.classify_from_dimensions(
            dimensions=np.array([600, 400, 400]),
            roundness=0.4,
            has_circle=False,
        )
        assert result.category == Category.OVERSIZED

    def test_oversized_too_small(self, classifier):
        result = classifier.classify_from_dimensions(
            dimensions=np.array([5, 5, 5]),
            roundness=0.3,
            has_circle=False,
        )
        assert result.category == Category.OVERSIZED

    def test_needs_repackaging_circle(self, classifier):
        result = classifier.classify_from_dimensions(
            dimensions=np.array([50, 50, 100]),
            roundness=0.95,
            has_circle=True,
        )
        assert result.category == Category.NEEDS_REPACKAGING

    def test_dimensions_priority_over_shape(self, classifier):
        # Circular AND oversized -> must be OVERSIZED (dimensions checked first).
        result = classifier.classify_from_dimensions(
            dimensions=np.array([600, 400, 400]),
            roundness=0.95,
            has_circle=True,
        )
        assert result.category == Category.OVERSIZED

    def test_boundary_dimensions_accepted(self, classifier):
        result = classifier.classify_from_dimensions(
            dimensions=np.array([450, 320, 320]),
            roundness=0.0,
            has_circle=False,
        )
        assert result.category == Category.SUITABLE

    def test_boundary_dimensions_rejected(self, classifier):
        result = classifier.classify_from_dimensions(
            dimensions=np.array([450.1, 320, 320]),
            roundness=0.0,
            has_circle=False,
        )
        assert result.category == Category.OVERSIZED

    def test_statistics(self, classifier):
        results = [
            classifier.classify_from_dimensions(np.array([100, 100, 100])),
            classifier.classify_from_dimensions(
                np.array([100, 100, 100]), roundness=0.9, has_circle=True
            ),
            classifier.classify_from_dimensions(np.array([600, 400, 400])),
        ]
        stats = classifier.get_statistics(results)
        assert stats["total"] == 3
        assert stats["by_category"][Category.SUITABLE.value] == 1
        assert stats["by_category"][Category.OVERSIZED.value] == 1
        assert stats["by_category"][Category.NEEDS_REPACKAGING.value] == 1

    def test_batch_classify_missing_file(self, classifier, capsys):
        results = classifier.batch_classify(["does_not_exist.stl"])
        assert results == []
        captured = capsys.readouterr()
        assert "Error classifying" in captured.out
