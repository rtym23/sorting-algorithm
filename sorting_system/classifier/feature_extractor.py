import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass


@dataclass
class GeometricFeatures:
    dimensions: np.ndarray  # [x, y, z] in mm
    volume: float  # mm^3
    bounding_box: tuple  # (min_point, max_point)
    inscribed_radius: float  # radius of largest inscribed circle in any cross-section
    circumscribed_radius: float  # radius of smallest circumscribed circle in any cross-section
    roundness_coefficient: float  # inscribed / circumscribed
    has_circle_in_section: bool  # True if roundness >= 0.8 in ANY cross-section
    cross_sections_analyzed: int


class FeatureExtractor:
    """Extracts geometric features from 3D models (STL/STEP) for classification."""

    MIN_SIZE = 10.0  # mm
    MAX_SIZE_X = 450.0  # mm
    MAX_SIZE_Y = 320.0  # mm
    MAX_SIZE_Z = 320.0  # mm
    ROUNDNESS_THRESHOLD = 0.8

    def __init__(self):
        pass

    def load_mesh(self, file_path: str) -> trimesh.Trimesh:
        """Load a 3D model from STL or STEP file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        mesh = trimesh.load(str(path), force="mesh")
        return mesh

    def get_bounding_box(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Get dimensions of the axis-aligned bounding box in mm."""
        bounds = mesh.bounding_box.extents
        return np.array(bounds)

    def compute_cross_section_roundness(
        self, mesh: trimesh.Trimesh, num_angles: int = 36
    ) -> tuple[float, float, int]:
        """
        Compute roundness coefficient for cross-sections at multiple angles.

        For each rotation angle, we take a cross-section perpendicular to the XY plane,
        fit inscribed and circumscribed circles, and compute the ratio.

        Returns: (max_roundness, inscribed_radius, circumscribed_radius, sections_analyzed)
        """
        best_roundness = 0.0
        best_inscribed = 0.0
        best_circumscribed = float("inf")
        sections_analyzed = 0

        angles = np.linspace(0, np.pi, num_angles, endpoint=False)

        for angle in angles:
            rotation = trimesh.transformations.rotation_matrix(angle, [0, 0, 1])

            rotated = mesh.copy()
            rotated.apply_transform(rotation)

            # Get the section at z=0 (middle of the object)
            z_center = (rotated.bounds[0][2] + rotated.bounds[1][2]) / 2

            try:
                section = rotated.section(
                    plane_origin=[0, 0, z_center],
                    plane_normal=[0, 0, 1],
                )

                if section is None:
                    continue

                # Convert to planar representation
                _, = section.to_planar()

                # Get the 2D outline
                if hasattr(section, "vertices") and len(section.vertices) > 2:
                    points = section.vertices[:, :2]

                    # Compute circumscribed circle (minimum enclosing circle)
                    circumscribed = self._minimum_enclosing_circle(points)

                    # Compute inscribed circle (maximum inscribed circle)
                    inscribed = self._maximum_inscribed_circle(points)

                    if circumscribed > 0:
                        roundness = inscribed / circumscribed
                        sections_analyzed += 1

                        if roundness > best_roundness:
                            best_roundness = roundness
                            best_inscribed = inscribed
                            best_circumscribed = circumscribed

            except Exception:
                continue

        return best_roundness, best_inscribed, best_circumscribed, sections_analyzed

    def _minimum_enclosing_circle(self, points: np.ndarray) -> float:
        """Compute radius of the minimum enclosing circle using Welzl's algorithm approximation."""
        if len(points) == 0:
            return 0.0

        # Simple approximation: use the farthest pair of points
        from scipy.spatial.distance import pdist, squareform

        if len(points) < 2:
            return 0.0

        dist_matrix = squareform(pdist(points))
        max_dist = np.max(dist_matrix)
        return max_dist / 2.0

    def _maximum_inscribed_circle(self, points: np.ndarray) -> float:
        """Compute radius of the maximum inscribed circle using Monte Carlo sampling."""
        if len(points) < 3:
            return 0.0

        # Compute centroid
        centroid = np.mean(points, axis=0)

        # Check if centroid is inside the polygon
        if not self._point_in_polygon(centroid, points):
            # Find a point inside using the centroid of triangles
            from scipy.spatial import Delaunay

            try:
                tri = Delaunay(points)
                # Use centroid of first triangle
                simplex = tri.simplices[0]
                centroid = np.mean(points[simplex], axis=0)
            except Exception:
                return 0.0

        # Monte Carlo: find maximum radius such that circle centered at centroid is inside
        max_radius = float("inf")
        for p in points:
            dist = np.linalg.norm(p - centroid)
            max_radius = min(max_radius, dist)

        # Refine with sampling
        best_radius = max_radius
        num_samples = 100
        for _ in range(num_samples):
            # Try random offsets from centroid
            offset = np.random.randn(2) * max_radius * 0.3
            test_center = centroid + offset

            if self._point_in_polygon(test_center, points):
                min_dist = min(np.linalg.norm(p - test_center) for p in points)
                if min_dist > best_radius:
                    best_radius = min_dist
                    centroid = test_center

        return best_radius

    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """Check if a point is inside a polygon using ray casting."""
        n = len(polygon)
        inside = False
        x, y = point
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def extract_features(self, file_path: str) -> GeometricFeatures:
        """Extract all geometric features from a 3D model."""
        mesh = self.load_mesh(file_path)

        # Get bounding box dimensions
        dimensions = self.get_bounding_box(mesh)

        # Get volume
        volume = mesh.volume if mesh.is_watertight else 0.0

        # Get bounding box bounds
        bounds = (mesh.bounds[0], mesh.bounds[1])

        # Compute cross-section roundness
        roundness, inscribed, circumscribed, sections = (
            self.compute_cross_section_roundness(mesh)
        )

        has_circle = roundness >= self.ROUNDNESS_THRESHOLD

        return GeometricFeatures(
            dimensions=dimensions,
            volume=volume,
            bounding_box=bounds,
            inscribed_radius=inscribed,
            circumscribed_radius=circumscribed,
            roundness_coefficient=roundness,
            has_circle_in_section=has_circle,
            cross_sections_analyzed=sections,
        )

    def check_dimensions(self, features: GeometricFeatures) -> bool:
        """Check if dimensions are within acceptable range."""
        dims = features.dimensions
        return (
            dims[0] >= self.MIN_SIZE
            and dims[1] >= self.MIN_SIZE
            and dims[2] >= self.MIN_SIZE
            and dims[0] <= self.MAX_SIZE_X
            and dims[1] <= self.MAX_SIZE_Y
            and dims[2] <= self.MAX_SIZE_Z
        )
