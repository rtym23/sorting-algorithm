"""Extraction of geometric features from 3D models (STL/STEP).

The features are used by the classifier to decide how to route an item.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from config import ClassifierConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class GeometricFeatures:
    """Features extracted from a single 3D model."""

    dimensions: np.ndarray  # [x, y, z] in mm
    volume: float  # mm^3
    bounding_box: tuple  # (min_point, max_point)
    inscribed_radius: float  # radius of the largest inscribed circle in a section
    circumscribed_radius: float  # radius of the smallest enclosing circle in a section
    roundness_coefficient: float  # inscribed / circumscribed
    has_circle_in_section: bool  # True if roundness >= threshold in ANY section
    cross_sections_analyzed: int


class FeatureExtractor:
    """Extracts geometric features from 3D models for classification.

    Two types of sections are inspected so that a round cross-section is
    found regardless of the orientation of the object:

    * three sections through the centroid, perpendicular to the X/Y/Z axes;
    * optionally, a rotational sweep of the section plane around each axis
      (controlled by ``ClassifierConfig.cross_section_angles``).

    Extraction results are cached by file path, size and modification time so
    that repeated calls (e.g. batch classification) do not re-parse the mesh.
    """

    ROUNDNESS_THRESHOLD = 0.8

    def __init__(self, config: ClassifierConfig | None = None):
        self.config = config or get_config().classifier
        self._cache: dict[tuple, GeometricFeatures] = {}

    # ------------------------------------------------------------------ #
    # Mesh loading
    # ------------------------------------------------------------------ #
    # STEP (ISO 10303) geometry is always expressed in metres: the trimesh
    # backend (cascadio) normalises the file's native units to SI before
    # tessellating. The whole system works in millimetres, so STEP models are
    # rescaled here and STL models (typically authored in mm) are left as-is.
    STEP_MM_PER_UNIT = 1000.0

    def load_mesh(self, file_path: str) -> trimesh.Trimesh:
        """Load and validate a 3D model, returning a single ``Trimesh``.

        STEP/STP models are rescaled from metres to millimetres so that all
        features are reported in the system's native units.

        Raises:
            FileNotFoundError: if the file does not exist.
            ValueError: if the file cannot be loaded as a mesh or is empty.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            loaded = trimesh.load(str(path), force="mesh")
        except ModuleNotFoundError as exc:
            raise ValueError(
                f"Cannot load '{file_path}': missing backend module "
                f"{exc.name!r}. Install it with 'pip install cascadio'."
            ) from exc
        mesh = self._to_single_mesh(loaded)

        if mesh is None or len(mesh.vertices) == 0:
            raise ValueError(
                f"No geometry found in '{file_path}' — the file may be "
                f"corrupted or unsupported."
            )

        if path.suffix.lower() in (".step", ".stp"):
            mesh.apply_scale(self.STEP_MM_PER_UNIT)
        return mesh

    @staticmethod
    def _to_single_mesh(loaded) -> trimesh.Trimesh | None:
        """Coalesce a ``trimesh.Scene`` into a single mesh when possible."""
        if isinstance(loaded, trimesh.Scene):
            submeshes = [g for g in loaded.geometry.values() if g is not None]
            if not submeshes:
                return None
            if len(submeshes) == 1:
                return submeshes[0]
            try:
                return trimesh.util.concatenate(submeshes)
            except Exception:
                return None
        return loaded if isinstance(loaded, trimesh.Trimesh) else None

    # ------------------------------------------------------------------ #
    # Feature extraction
    # ------------------------------------------------------------------ #
    def extract_features(self, file_path: str) -> GeometricFeatures:
        """Extract all geometric features for a model file."""
        key = self._cache_key(file_path)
        cached = self._cache.get(key)
        if cached is not None and self.config.cache_enabled:
            logger.debug("Feature cache hit for %s", file_path)
            return self._copy_features(cached)

        mesh = self.load_mesh(file_path)
        features = self._extract_from_mesh(mesh)

        if self.config.cache_enabled:
            self._cache[key] = self._copy_features(features)
        return features

    def _extract_from_mesh(self, mesh: trimesh.Trimesh) -> GeometricFeatures:
        dimensions = self.get_bounding_box(mesh)
        volume = self._safe_volume(mesh)
        bounds = (mesh.bounds[0], mesh.bounds[1])

        roundness, inscribed, circumscribed, sections = (
            self.compute_cross_section_roundness(mesh)
        )
        has_circle = bool(roundness >= self.config.roundness_threshold)

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

    @staticmethod
    def _copy_features(features: GeometricFeatures) -> GeometricFeatures:
        return GeometricFeatures(
            dimensions=features.dimensions.copy(),
            volume=features.volume,
            bounding_box=(features.bounding_box[0].copy(),
                          features.bounding_box[1].copy()),
            inscribed_radius=features.inscribed_radius,
            circumscribed_radius=features.circumscribed_radius,
            roundness_coefficient=features.roundness_coefficient,
            has_circle_in_section=features.has_circle_in_section,
            cross_sections_analyzed=features.cross_sections_analyzed,
        )

    def _cache_key(self, file_path: str) -> tuple:
        """Build a cache key from the file path and its stat info."""
        try:
            stat = os.stat(file_path)
            return (file_path, stat.st_size, stat.st_mtime_ns)
        except OSError:
            return (file_path,)

    @staticmethod
    def get_bounding_box(mesh: trimesh.Trimesh) -> np.ndarray:
        """Dimensions of the axis-aligned bounding box, in mm."""
        return np.asarray(mesh.bounding_box.extents, dtype=float)

    def _safe_volume(self, mesh: trimesh.Trimesh) -> float:
        """Return the mesh volume, tolerating imperfect (non-watertight) meshes."""
        try:
            if mesh.is_watertight:
                return float(mesh.volume)
            # Non-watertight meshes are common in real STL exports. Fall back to
            # the convex hull volume so the number stays meaningful.
            hull_volume = float(mesh.convex_hull.volume)
            logger.warning(
                "Mesh is not watertight (volume unreliable); using convex hull "
                "volume %.1f mm^3", hull_volume,
            )
            return hull_volume
        except Exception:
            logger.warning("Could not compute mesh volume; using 0.0")
            return 0.0

    # ------------------------------------------------------------------ #
    # Roundness from cross-sections
    # ------------------------------------------------------------------ #
    def compute_cross_section_roundness(
        self, mesh: trimesh.Trimesh, num_angles: int | None = None
    ) -> tuple[float, float, float, int]:
        """Compute the roundness of the roundest cross-section of a mesh.

        The object is sliced through its centroid; the section planes are the
        three principal-axis planes, plus an optional rotational sweep around
        each axis (``num_angles`` steps). The ratio of the largest inscribed
        circle to the smallest enclosing circle is computed for every section;
        the maximum ratio over all sections is returned.

        Returns:
            (max_roundness, inscribed_radius, circumscribed_radius,
             sections_analyzed)
        """
        threshold = self.config.roundness_threshold
        num_angles = (
            self.config.cross_section_angles
            if num_angles is None
            else max(0, int(num_angles))
        )

        centroid = np.asarray(mesh.bounds).mean(axis=0)
        axes = np.eye(3)

        best_roundness = 0.0
        best_inscribed = 0.0
        best_circumscribed = float("inf")
        sections_analyzed = 0

        for axis in axes:
            normals = [axis]
            if num_angles > 0:
                normals = self._swept_normals(axis, num_angles)

            for normal in normals:
                section = self._section_at(mesh, centroid, normal)
                if section is None:
                    continue

                points = self._section_points(section)
                if len(points) < 3:
                    continue

                circumscribed = self._minimum_enclosing_circle(points)
                inscribed = self._maximum_inscribed_circle(points)
                sections_analyzed += 1

                if circumscribed > 0:
                    roundness = inscribed / circumscribed
                    if roundness > best_roundness:
                        best_roundness = roundness
                        best_inscribed = inscribed
                        best_circumscribed = circumscribed

                    # We only need to know whether ANY section is round.
                    if roundness >= threshold:
                        return (
                            roundness,
                            inscribed,
                            circumscribed,
                            sections_analyzed,
                        )

        return best_roundness, best_inscribed, best_circumscribed, sections_analyzed

    @staticmethod
    def _swept_normals(axis: np.ndarray, num_angles: int) -> list[np.ndarray]:
        """Normals of ``num_angles`` section planes rotating around ``axis``.

        The planes all contain ``axis``, i.e. their normals lie in the plane
        perpendicular to ``axis``. Rotating the section plane around the axis
        therefore sweeps it through the whole range of orientations.
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)

        # Build an orthonormal basis {u, v} for the plane perpendicular to axis.
        reference = (
            np.array([1.0, 0.0, 0.0])
            if abs(axis[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        u = np.cross(reference, axis)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        v /= np.linalg.norm(v)

        normals = []
        for angle in np.linspace(0.0, np.pi, num_angles, endpoint=False):
            normals.append(u * np.cos(angle) + v * np.sin(angle))
        return normals

    @staticmethod
    def _section_at(
        mesh: trimesh.Trimesh, origin: np.ndarray, normal: np.ndarray
    ) -> trimesh.path.Path3D | None:
        """Slice the mesh with a plane, tolerating edge cases."""
        try:
            section = mesh.section(
                plane_origin=origin,
                plane_normal=normal,
            )
        except Exception as exc:
            logger.debug("Section failed: %s", exc)
            return None
        if section is None or len(section.entities) == 0:
            return None
        return section

    @staticmethod
    def _section_points(section) -> np.ndarray:
        """Extract 2D outline points from a section path."""
        try:
            path_2d, _ = section.to_2D()
            vertices = np.asarray(path_2d.vertices, dtype=float)
            if vertices.ndim == 2 and len(vertices) > 0:
                return vertices[:, :2]
        except Exception:
            pass
        return np.empty((0, 2))

    # ------------------------------------------------------------------ #
    # Smallest enclosing circle (Welzl's algorithm)
    # ------------------------------------------------------------------ #
    def _minimum_enclosing_circle(self, points: np.ndarray) -> float:
        """Radius of the minimum enclosing circle (Welzl, O(n) expected)."""
        if len(points) == 0:
            return 0.0
        if len(points) == 1:
            return 0.0

        unique = np.unique(points, axis=0)
        if len(unique) < 2:
            return 0.0

        center, radius = self._welzl(list(map(tuple, unique)))
        return float(radius)

    def _welzl(self, points: list[tuple]) -> tuple:
        """Iterative incremental Welzl algorithm returning (center, radius).

        The recursive formulation recurses once per point, which blows the
        stack on section outlines with thousands of vertices. This iterative
        variant is equivalent but has constant recursion depth.
        """
        pts = list(points)
        random.shuffle(pts)

        center = np.asarray(pts[0], dtype=float)
        radius = 0.0

        def outside(point) -> bool:
            return self._dist_sq(center, point) > radius * radius + 1e-9

        n = len(pts)
        for i in range(n):
            p = pts[i]
            if outside(p):
                center = np.asarray(p, dtype=float)
                radius = 0.0
                for j in range(i):
                    q = pts[j]
                    if outside(q):
                        # Smallest circle through p and q (diameter circle).
                        pa = np.asarray(p, dtype=float)
                        qa = np.asarray(q, dtype=float)
                        center = (pa + qa) / 2.0
                        radius = float(np.linalg.norm(pa - qa) / 2.0)
                        for k in range(j):
                            r = pts[k]
                            if outside(r):
                                center, radius = self._circle_from_points([p, q, r])
        return center, radius

    @staticmethod
    def _circle_from_points(boundary) -> tuple:
        """Circle determined by 0..3 boundary points (trivial cases)."""
        n = len(boundary)
        if n == 0:
            return (np.zeros(2), 0.0)
        if n == 1:
            return (np.asarray(boundary[0], dtype=float), 0.0)
        if n == 2:
            a = np.asarray(boundary[0], dtype=float)
            b = np.asarray(boundary[1], dtype=float)
            return ((a + b) / 2.0, float(np.linalg.norm(a - b) / 2.0))

        # Three points: center is the intersection of perpendicular bisectors.
        a = np.asarray(boundary[0], dtype=float)
        b = np.asarray(boundary[1], dtype=float)
        c = np.asarray(boundary[2], dtype=float)

        ax, ay = a
        bx, by = b
        cx, cy = c
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            # Collinear points — fall back to the two-point case.
            ab = (a + b) / 2.0
            return (ab, float(np.linalg.norm(a - b) / 2.0))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
              + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
              + (cx * cx + cy * cy) * (bx - ax)) / d
        center = np.array([ux, uy])
        radius = float(np.linalg.norm(a - center))
        return (center, radius)

    @staticmethod
    def _dist_sq(center, p) -> float:
        return float(np.dot(np.asarray(p) - center, np.asarray(p) - center))

    # ------------------------------------------------------------------ #
    # Largest inscribed circle
    # ------------------------------------------------------------------ #
    def _maximum_inscribed_circle(self, points: np.ndarray) -> float:
        """Radius of the largest circle that fits inside the section outline.

        The initial estimate is the distance from the polygon centroid to the
        nearest edge, which is exact for convex polygons. A short Monte Carlo
        refinement improves the estimate for concave outlines.
        """
        points = np.unique(points, axis=0)
        if len(points) < 3:
            return 0.0

        # Order vertices so that consecutive edges form the polygon outline.
        ordered = self._order_by_angle(points)
        if ordered is None:
            return 0.0

        centroid = self._polygon_centroid(ordered)
        if not self._point_in_polygon(centroid, ordered):
            return 0.0

        best_radius = self._distance_to_nearest_edge(centroid, ordered)
        best_center = centroid

        rng = np.random.default_rng(0)
        lo, hi = ordered.min(axis=0), ordered.max(axis=0)
        span = hi - lo
        if span[0] > 0 and span[1] > 0:
            for _ in range(self.config.monte_carlo_samples):
                candidate = best_center + (rng.random(2) - 0.5) * span * 0.4
                if not self._point_in_polygon(candidate, ordered):
                    continue
                radius = self._distance_to_nearest_edge(candidate, ordered)
                if radius > best_radius:
                    best_radius = radius
                    best_center = candidate

        return float(best_radius)

    @staticmethod
    def _order_by_angle(points: np.ndarray) -> np.ndarray | None:
        """Order points counter-clockwise by polar angle around their mean."""
        pts = np.asarray(points, dtype=float)
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        return pts[order]

    @staticmethod
    def _polygon_centroid(points: np.ndarray) -> np.ndarray:
        """Area centroid of a polygon via the shoelace formula."""
        pts = np.asarray(points, dtype=float)
        x, y = pts[:, 0], pts[:, 1]
        # Cyclic shift
        xr = np.roll(x, -1)
        yr = np.roll(y, -1)
        cross = x * yr - xr * y
        area = 0.5 * cross.sum()
        if abs(area) < 1e-12:
            return pts.mean(axis=0)
        cx = ((x + xr) * cross).sum() / (6.0 * area)
        cy = ((y + yr) * cross).sum() / (6.0 * area)
        return np.array([cx, cy])

    @staticmethod
    def _distance_to_nearest_edge(point: np.ndarray, polygon: np.ndarray) -> float:
        """Smallest distance from ``point`` to any polygon edge (segment).

        Vectorised over all edges so it stays fast even for section outlines
        with thousands of vertices.
        """
        p = np.asarray(point, dtype=float)
        poly = np.asarray(polygon, dtype=float)
        a = poly
        b = np.roll(poly, -1, axis=0)
        ab = b - a
        ab2 = np.einsum("ij,ij->i", ab, ab)
        t = np.einsum("ij,ij->i", p - a, ab) / np.where(ab2 > 0, ab2, 1.0)
        t = np.clip(t, 0.0, 1.0)
        closest = a + t[:, None] * ab
        dists = np.linalg.norm(closest - p, axis=1)
        return float(dists.min())

    @staticmethod
    def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
        """Ray-casting point-in-polygon test (vectorised)."""
        poly = np.asarray(polygon, dtype=float)
        x, y = float(point[0]), float(point[1])
        xi, yi = poly[:, 0], poly[:, 1]
        xj, yj = np.roll(xi, 1), np.roll(yi, 1)
        crosses = (yi > y) != (yj > y)
        denom = np.where(yj - yi != 0, yj - yi, 1.0)
        term = (xj - xi) * (y - yi) / denom + xi
        intersects = crosses & (x < term)
        return bool(np.count_nonzero(intersects) % 2 == 1)

    # ------------------------------------------------------------------ #
    # Dimension checks
    # ------------------------------------------------------------------ #
    def check_dimensions(self, features: GeometricFeatures) -> bool:
        """Whether the item dimensions are within the acceptable range."""
        dims = np.asarray(features.dimensions, dtype=float)
        if dims.ndim != 1 or dims.size != 3:
            return False
        return bool(
            self.config.min_size <= dims[0] <= self.config.max_size_x
            and self.config.min_size <= dims[1] <= self.config.max_size_y
            and self.config.min_size <= dims[2] <= self.config.max_size_z
        )
