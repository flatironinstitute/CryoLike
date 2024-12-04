import numpy as np

from cryolike.grids.cartesian_grid import (
    CartesianGrid2D,
    CartesianGrid3D,
    _compute_grid_dims,
    _setup_grid,
)

def test_setup_grid_returns_right_result_count():
    r_2d = np.array([3., 3.])
    n_xels_2d = np.array([6, 6])
    (axes, xels) = _setup_grid(r_2d, n_xels_2d, False)
    assert len(axes) == 2
    assert len(xels) == 2
    r_3d = np.array([3., 3., 3.])
    n_xels_3d = np.array([6, 6, 6])
    (axes, xels) = _setup_grid(r_3d, n_xels_3d, False)
    assert len(axes) == 3
    assert len(xels) == 3

def test_setup_grid_returns_good_axes():
    r_2d = np.array([4., 1.])
    n_xels_2d = np.array([4, 2])
    r_3d = np.array([6., 6., 4.])
    n_xels_3d = np.array([6, 8, 10])
    rs = [r_2d, r_3d]
    n_xels = [n_xels_2d, n_xels_3d]
    for i in range(len(rs)):
        (axes, _) = _setup_grid(rs[i], n_xels[i], False)
        for j in range(len(axes)):
            assert len(axes[j]) == n_xels[i][j]
            assert axes[j][0] == -1 * rs[i][j]
            assert axes[j][-1] == rs[i][j] - (2*rs[i][j]/(n_xels[i][j]))

def test_setup_grid_returns_good_mesh():
    r_2d = np.array([4., 1.])
    n_xels_2d = np.array([4, 2])
    r_3d = np.array([6., 6., 4.])
    n_xels_3d = np.array([6, 8, 10])
    rs = [r_2d, r_3d]
    n_xels = [n_xels_2d, n_xels_3d]
    for i in range(len(rs)):
        (axes, xels) = _setup_grid(rs[i], n_xels[i], False)
        for j in range(len(xels)):
            assert np.all(xels[j].shape == n_xels[i])
            # I haven't been able to sort out exactly how to express the relation in Numpy, alas
            # assert np.all(xels[j] == np.array(axes[j] * n_xels[i][j]).transpose())

def test_compute_grid_dims_computes_2d():
    n_xels_in = [2, 3]
    size = 3
    (n_xels, xel_size, box_size, radius) = _compute_grid_dims(2, n_xels_in, size)
    expected_size = np.ones((2,)) * size
    np.testing.assert_allclose(n_xels, n_xels_in)
    np.testing.assert_allclose(xel_size, expected_size)
    np.testing.assert_allclose(box_size, np.array(n_xels_in) * expected_size)
    np.testing.assert_allclose(radius, box_size * 0.5)

def test_compute_grid_dims_computes_3d():
    n_xels_in = [3, 4, 5]
    size = [5., 4., 3.]
    (n_xels, xel_size, box_size, radius) = _compute_grid_dims(3, n_xels_in, size)
    np.testing.assert_allclose(n_xels, n_xels_in)
    np.testing.assert_allclose(xel_size, np.array(size))
    np.testing.assert_allclose(box_size, np.array(n_xels_in) * np.array(size))
    np.testing.assert_allclose(radius, box_size * 0.5)

def test_cartesian_grid_2d():
    n_pixels = 4
    pixel_size = 5.
    endpoint = False
    cg2d = CartesianGrid2D(n_pixels, pixel_size, endpoint)
    assert cg2d.endpoint == endpoint
    np.testing.assert_allclose(cg2d.n_pixels, np.ones((2,)) * n_pixels)
    np.testing.assert_allclose(cg2d.pixel_size, np.ones((2,)) * pixel_size)
    np.testing.assert_allclose(cg2d.box_size, cg2d.n_pixels * cg2d.pixel_size)
    np.testing.assert_allclose(cg2d.radius, cg2d.box_size * 0.5)
    np.testing.assert_allclose(cg2d.x_axis, [-10., -5., 0., 5.])
    np.testing.assert_allclose(cg2d.y_axis, cg2d.x_axis)
    np.testing.assert_allclose(cg2d.x_pixels, cg2d.y_pixels.transpose())
    assert cg2d.n_pixels_total == 16

def test_cartesian_grid_3d():
    n_voxels = [10, 10, 20]
    voxel_size = 2.
    endpoint = False
    cg3d = CartesianGrid3D(n_voxels, voxel_size, endpoint)
    assert cg3d.endpoint == endpoint
    np.testing.assert_allclose(cg3d.n_voxels, n_voxels)
    np.testing.assert_allclose(cg3d.voxel_size, np.ones((3,)) * voxel_size)
    np.testing.assert_allclose(cg3d.box_size, cg3d.n_voxels * cg3d.voxel_size)
    np.testing.assert_allclose(cg3d.radius, cg3d.box_size * 0.5)
    np.testing.assert_allclose(cg3d.x_axis, cg3d.y_axis)
    assert(cg3d.z_axis.shape == (n_voxels[2],))
    # TODO: might be good to check the grids are correct
    assert cg3d.n_voxels_total == 10 * 10 * 20
    
    
if __name__ == "__main__":
    test_setup_grid_returns_right_result_count()
    test_setup_grid_returns_good_axes()
    test_setup_grid_returns_good_mesh()
    test_compute_grid_dims_computes_2d()
    test_compute_grid_dims_computes_3d()
    test_cartesian_grid_2d()
    test_cartesian_grid_3d()
    print("Cartesian grid tests passed")
