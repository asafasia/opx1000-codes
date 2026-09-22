import numpy as np
import pytest
from calibration_utils.iq_blobs.plotting import _nearest_center_boundary_segments


def test_triangle_has_three_branches_with_a_common_equidistant_junction():
    centers = np.array([[-1., 0.], [1., 0.], [0., 2.]])
    segments = _nearest_center_boundary_segments(centers, (-3, 3), (-2, 3))
    junction = np.array([0., 0.75])
    assert len(segments) == 3
    for segment in segments:
        assert np.min(np.linalg.norm(segment - junction, axis=1)) < 1e-10
    np.testing.assert_allclose(np.linalg.norm(centers - junction, axis=1), 1.25)


@pytest.mark.parametrize("centers, limits", [
    ([[-1, 0], [1, 0], [0, 2]], ((-3, 3), (-2, 3))),
    ([[-1, 0], [0, 0], [1, 0]], ((-2, 2), (-1, 1))),
    ([[-1, 0], [1, 0], [0, 0.01]], ((-2, 2), (-1, 1))),
])
def test_every_branch_is_a_real_nearest_center_decision_boundary(centers, limits):
    centers = np.asarray(centers, dtype=float)
    segments = _nearest_center_boundary_segments(centers, *limits)
    assert segments
    for segment in segments:
        assert np.all(segment[:, 0] >= limits[0][0] - 1e-10)
        assert np.all(segment[:, 0] <= limits[0][1] + 1e-10)
        assert np.all(segment[:, 1] >= limits[1][0] - 1e-10)
        assert np.all(segment[:, 1] <= limits[1][1] + 1e-10)
        for fraction in np.linspace(0.1, 0.9, 9):
            point = segment[0] * (1-fraction) + segment[1] * fraction
            distance = np.sort(np.sum((centers - point)**2, axis=1))
            assert distance[0] == pytest.approx(distance[1], abs=1e-9)
        midpoint = segment.mean(axis=0)
        direction = segment[1] - segment[0]
        normal = np.array([-direction[1], direction[0]])
        normal = normal / np.linalg.norm(normal) * 1e-6
        assert np.argmin(np.sum((centers - midpoint - normal)**2, axis=1)) != np.argmin(np.sum((centers - midpoint + normal)**2, axis=1))


def test_collinear_centers_have_only_two_parallel_boundaries():
    segments = _nearest_center_boundary_segments([[-1, 0], [0, 0], [1, 0]], (-2, 2), (-1, 1))
    assert len(segments) == 2
    assert sorted(segment[0, 0] for segment in segments) == [-0.5, 0.5]


def test_invalid_or_coincident_centers_do_not_create_spurious_lines():
    assert _nearest_center_boundary_segments([[0, 0], [0, 0], [0, 0]], (-1, 1), (-1, 1)) == []
    assert _nearest_center_boundary_segments([[0, 0], [1, float("nan")]], (-1, 1), (-1, 1)) == []


def test_colored_regions_tile_view_and_match_the_nearest_centers():
    from calibration_utils.iq_blobs.plotting import _nearest_center_regions
    centers = np.array([[-1., 0.], [1., 0.], [0., 2.]])
    regions = _nearest_center_regions(centers, (-3, 3), (-2, 3))
    area = 0.
    for index, polygon in enumerate(regions):
        area += abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))) / 2
        interior = polygon.mean(axis=0)
        assert np.argmin(np.sum((centers - interior)**2, axis=1)) == index
        for vertex in polygon:
            distances = np.sum((centers - vertex)**2, axis=1)
            assert distances[index] <= distances.min() + 1e-10
    assert area == pytest.approx(30.)
