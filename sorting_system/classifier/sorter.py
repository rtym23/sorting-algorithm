from enum import Enum
from dataclasses import dataclass
import numpy as np
from .feature_extractor import FeatureExtractor, GeometricFeatures


class Category(Enum):
    SUITABLE = "Подходит для сортировки"
    OVERSIZED = "Не по габаритам"
    NEEDS_REPACKAGING = "Не по габаритам без доупаковки"


@dataclass
class ClassificationResult:
    category: Category
    confidence: float
    features: GeometricFeatures
    reason: str


class ItemClassifier:
    """
    Classifies items into one of three categories based on geometric features.

    Classification rules (in priority order):
    1. Check dimensions first (stricter constraint)
    2. Then check for circle in cross-section

    Priority: dimensions > shape
    """

    def __init__(self):
        self.extractor = FeatureExtractor()

    def classify_from_file(self, file_path: str) -> ClassificationResult:
        """Classify an item from its 3D model file."""
        features = self.extractor.extract_features(file_path)
        return self.classify(features)

    def classify(self, features: GeometricFeatures) -> ClassificationResult:
        """Classify an item based on its geometric features."""
        dims_ok = self.extractor.check_dimensions(features)

        # Priority 1: Check dimensions (stricter)
        if not dims_ok:
            dim_str = f"{features.dimensions[0]:.1f}x{features.dimensions[1]:.1f}x{features.dimensions[2]:.1f}"
            return ClassificationResult(
                category=Category.OVERSIZED,
                confidence=1.0,
                features=features,
                reason=(
                    f"Размеры {dim_str} мм выходят за допустимые "
                    f"габариты (10-450 x 10-320 x 10-320 мм)"
                ),
            )

        # Priority 2: Check for circle in cross-section
        if features.has_circle_in_section:
            return ClassificationResult(
                category=Category.NEEDS_REPACKAGING,
                confidence=features.roundness_coefficient,
                features=features,
                reason=(
                    f"Обнаружен круг в сечении "
                    f"(коэффициент округлости: {features.roundness_coefficient:.3f}, "
                    f"порог: 0.8). Товар требует доупаковки."
                ),
            )

        # Default: suitable for sorting
        return ClassificationResult(
            category=Category.SUITABLE,
            confidence=1.0,
            features=features,
            reason=(
                f"Размеры в допустимых пределах, круг в сечении не обнаружен "
                f"(коэффициент: {features.roundness_coefficient:.3f}). "
                f"Товар направляется в основной сортировщик."
            ),
        )

    def classify_from_dimensions(
        self,
        dimensions: np.ndarray,
        roundness: float = 0.0,
        has_circle: bool = False,
    ) -> ClassificationResult:
        """Classify based on pre-computed features (for simulation integration)."""
        features = GeometricFeatures(
            dimensions=dimensions,
            volume=0.0,
            bounding_box=(np.zeros(3), dimensions),
            inscribed_radius=0.0,
            circumscribed_radius=0.0 if roundness == 0 else 1.0,
            roundness_coefficient=roundness,
            has_circle_in_section=has_circle,
            cross_sections_analyzed=0,
        )
        return self.classify(features)

    def batch_classify(self, file_paths: list[str]) -> list[ClassificationResult]:
        """Classify multiple items."""
        results = []
        for path in file_paths:
            try:
                result = self.classify_from_file(path)
                results.append(result)
            except Exception as e:
                print(f"Error classifying {path}: {e}")
        return results

    def get_statistics(self, results: list[ClassificationResult]) -> dict:
        """Get classification statistics."""
        stats = {cat: 0 for cat in Category}
        for r in results:
            stats[r.category] += 1
        return {
            "total": len(results),
            "by_category": {cat.value: count for cat, count in stats.items()},
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sorter.py <path_to_3d_model>")
        sys.exit(1)

    classifier = ItemClassifier()
    result = classifier.classify_from_file(sys.argv[1])

    print(f"Категория: {result.category.value}")
    print(f"Уверенность: {result.confidence:.3f}")
    print(f"Размеры: {result.features.dimensions}")
    print(f"Причина: {result.reason}")
