"""Classification rules that map item geometry to a sorting category."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np

from config import ClassifierConfig, get_config

from .feature_extractor import FeatureExtractor, GeometricFeatures

logger = logging.getLogger(__name__)


class Category(Enum):
    """Possible sorting categories for an item."""

    SUITABLE = "Suitable for sorting"
    OVERSIZED = "Oversized/undersized dimensions"
    NEEDS_REPACKAGING = "Needs repackaging due to circular cross-section"


@dataclass
class ClassificationResult:
    """Outcome of classifying a single item."""

    category: Category
    confidence: float
    features: GeometricFeatures
    reason: str


class ItemClassifier:
    """Classifies items into categories based on their geometric features.

    The rules are applied in priority order:

    1. **Dimensions first** — items outside the configured size window are
       sent to the oversized category, no matter their shape.
    2. **Then shape** — items within the size window but with a round
       cross-section are sent for repackaging.
    3. Everything else is suitable for sorting.
    """

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or get_config().classifier
        self.extractor = FeatureExtractor(config=self.config)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def classify_from_file(self, file_path: str) -> ClassificationResult:
        """Classify an item from its 3D model file."""
        features = self.extractor.extract_features(file_path)
        return self.classify(features)

    def classify(self, features: GeometricFeatures) -> ClassificationResult:
        """Classify an item from pre-extracted geometric features."""
        if not self._dimensions_ok(features.dimensions):
            dims = features.dimensions
            bounds = (
                f"{self.config.min_size}-{self.config.max_size_x} x "
                f"{self.config.min_size}-{self.config.max_size_y} x "
                f"{self.config.min_size}-{self.config.max_size_z} mm"
            )
            return ClassificationResult(
                category=Category.OVERSIZED,
                confidence=1.0,
                features=features,
                reason=(
                    f"Dimensions {dims[0]:.1f}x{dims[1]:.1f}x{dims[2]:.1f} mm "
                    f"outside acceptable bounds ({bounds})"
                ),
            )

        if features.has_circle_in_section:
            threshold = self.config.roundness_threshold
            return ClassificationResult(
                category=Category.NEEDS_REPACKAGING,
                confidence=features.roundness_coefficient,
                features=features,
                reason=(
                    f"Circular cross-section detected (roundness coefficient: "
                    f"{features.roundness_coefficient:.3f}, threshold: "
                    f"{threshold}). Item requires repackaging."
                ),
            )

        return ClassificationResult(
            category=Category.SUITABLE,
            confidence=1.0,
            features=features,
            reason=(
                f"Dimensions within acceptable range, no circular cross-section "
                f"detected (coefficient: {features.roundness_coefficient:.3f}). "
                f"Item routed to main sorter."
            ),
        )

    def classify_from_dimensions(
        self,
        dimensions: np.ndarray,
        roundness: float = 0.0,
        has_circle: bool = False,
    ) -> ClassificationResult:
        """Classify from pre-computed dimensions (used by the simulation)."""
        features = GeometricFeatures(
            dimensions=np.asarray(dimensions, dtype=float),
            volume=0.0,
            bounding_box=(np.zeros(3), np.asarray(dimensions, dtype=float)),
            inscribed_radius=0.0,
            circumscribed_radius=0.0 if roundness == 0 else 1.0,
            roundness_coefficient=float(roundness),
            has_circle_in_section=bool(has_circle),
            cross_sections_analyzed=0,
        )
        return self.classify(features)

    def batch_classify(self, file_paths: list[str]) -> list[ClassificationResult]:
        """Classify several model files, skipping the ones that fail."""
        results: list[ClassificationResult] = []
        for path in file_paths:
            try:
                results.append(self.classify_from_file(path))
            except Exception as exc:
                logger.error("Error classifying %s: %s", path, exc)
                print(f"Error classifying {path}: {exc}")
        return results

    def get_statistics(self, results: list[ClassificationResult]) -> dict:
        """Count results by category."""
        stats = {cat: 0 for cat in Category}
        for result in results:
            stats[result.category] += 1
        return {
            "total": len(results),
            "by_category": {cat.value: count for cat, count in stats.items()},
        }

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _dimensions_ok(self, dimensions: np.ndarray) -> bool:
        """Whether the item dimensions fall within the configured window."""
        dims = np.asarray(dimensions, dtype=float)
        if dims.ndim != 1 or dims.size != 3 or not np.all(np.isfinite(dims)):
            logger.warning("Unexpected dimensions shape %s; treating as invalid",
                           getattr(dims, "shape", None))
            return False
        return bool(
            self.config.min_size <= dims[0] <= self.config.max_size_x
            and self.config.min_size <= dims[1] <= self.config.max_size_y
            and self.config.min_size <= dims[2] <= self.config.max_size_z
        )
