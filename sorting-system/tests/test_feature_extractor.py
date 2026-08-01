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

    def test_large_section_no_recursion(self, extractor):
        # A section outline with ~2000 vertices previously blew the recursion
        # limit of the recursive Welzl implementation.
        angles = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
        pts = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        radius = extractor._minimum_enclosing_circle(pts)
        assert radius == pytest.approx(1.0, abs=1e-6)

    def test_extract_high_res_cylinder_no_recursion(self, extractor, tmp_path):
        # A 1500-segment cylinder produces a 1500-gon cross-section, which
        # must not crash the extractor.
        mesh = trimesh.creation.cylinder(radius=25, height=100, sections=1500)
        path = tmp_path / "hi_res_cylinder.stl"
        mesh.export(str(path))

        features = extractor.extract_features(str(path))
        assert features.roundness_coefficient >= 0.8
        assert features.has_circle_in_section is True

    def test_step_mesh_scaled_to_mm(self, tmp_path, monkeypatch):
        # STEP backends (cascadio) return geometry in metres; the extractor
        # must rescale it to the system's millimetre units.
        def fake_load(*args, **kwargs):
            return trimesh.creation.box(extents=[1.0, 0.8, 0.6])

        monkeypatch.setattr("trimesh.load", fake_load)
        path = tmp_path / "model.stp"
        path.write_bytes(b"dummy step content")

        loaded = FeatureExtractor().load_mesh(str(path))
        assert np.allclose(
            loaded.bounding_box.extents, [1000, 800, 600], atol=1e-6
        )

    def test_stl_mesh_not_scaled(self, tmp_path, monkeypatch):
        # STL models are typically authored in mm and must be left untouched.
        def fake_load(*args, **kwargs):
            return trimesh.creation.box(extents=[100, 80, 60])

        monkeypatch.setattr("trimesh.load", fake_load)
        path = tmp_path / "model.stl"
        path.write_bytes(b"dummy stl content")

        loaded = FeatureExtractor().load_mesh(str(path))
        assert np.allclose(loaded.bounding_box.extents, [100, 80, 60])
