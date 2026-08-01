import numpy as np
import pytest
import trimesh

from classifier.feature_extractor import FeatureExtractor


@pytest.fixture
def extractor():
    return FeatureExtractor()


class TestFeatureExtractor:
    def test_missing_file_raises(self, extractor):
        with pytest.raises(FileNotFoundError):
            extractor.load_mesh("does_not_exist.stl")

    def test_check_dimensions_valid(self, extractor):
        features = type(
            "F",
            (),
            {"dimensions": np.array([100, 80, 60])},
        )()
        assert extractor.check_dimensions(features) is True

    @pytest.mark.parametrize(
        "dims",
        [
            np.array([5, 100, 100]),  # too small on X
            np.array([100, 100, 500]),  # too big on Z
            np.array([900, 100, 100]),  # too big on X
        ],
    )
    def test_check_dimensions_invalid(self, extractor, dims):
        features = type("F", (), {"dimensions": dims})()
        assert extractor.check_dimensions(features) is False

    def test_extract_box_features(self, extractor, tmp_path):
        mesh = trimesh.creation.box(extents=[100, 80, 60])
        path = tmp_path / "box.stl"
        mesh.export(str(path))

        features = extractor.extract_features(str(path))
        assert np.allclose(features.dimensions, [100, 80, 60], atol=1.0)
        # A box has a rectangular cross-section -> low roundness.
        assert features.roundness_coefficient < 0.8
        assert features.has_circle_in_section is False

    def test_extract_cylinder_has_circle(self, extractor, tmp_path):
        mesh = trimesh.creation.cylinder(radius=25, height=100)
        path = tmp_path / "cylinder.stl"
        mesh.export(str(path))

        features = extractor.extract_features(str(path))
        assert features.roundness_coefficient >= 0.8
        assert features.has_circle_in_section is True

    def test_extract_tilted_cylinder_has_circle(self, tmp_path):
        # A cylinder tilted 45 degrees must still be recognised as round: a
        # rotational sweep of section planes catches the circular section.
        mesh = trimesh.creation.cylinder(radius=25, height=100, sections=48)
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                np.deg2rad(45), [1, 0, 0]
            )
        )
        path = tmp_path / "tilted_cylinder.stl"
        mesh.export(str(path))

        features = FeatureExtractor().extract_features(str(path))
        assert features.has_circle_in_section is True
        assert features.roundness_coefficient >= 0.8

    def test_extract_non_watertight_still_reports_volume(self, extractor, tmp_path):
        # A mesh with half its faces removed is not watertight; volume
        # extraction must not crash and should fall back to the convex hull.
        box = trimesh.creation.box(extents=[100, 80, 60])
        mesh = trimesh.Trimesh(
            vertices=box.vertices.copy(),
            faces=box.faces[:-6].copy(),
            process=False,
        )
        path = tmp_path / "leaky_box.stl"
        mesh.export(str(path))

        features = extractor.extract_features(str(path))
        assert features.volume > 0.0

    def test_extract_caches_results(self, extractor, tmp_path):
        mesh = trimesh.creation.box(extents=[100, 80, 60])
        path = tmp_path / "cached_box.stl"
        mesh.export(str(path))

        first = extractor.extract_features(str(path))
        second = extractor.extract_features(str(path))
        assert np.array_equal(first.dimensions, second.dimensions)

        # Extracting the same file again must hit the cache (fast).
        import time
        start = time.perf_counter()
        extractor.extract_features(str(path))
        assert time.perf_counter() - start < 0.05
