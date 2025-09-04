from unittest.mock import patch, Mock
import numpy as np
import numpy.testing as npt
import torch
from torch.testing import assert_close
from pytest import raises, mark
from math import ceil


from stacks_fixtures import (
    make_basic_Templates,
    make_image_tensor,
    make_mock_data_obj,
    make_mock_phys_grid,
    make_mock_polar_grid,
    make_mock_viewing_angles,
    mock_get_fourier_slices
)

from cryolike.util import AtomShape, AtomicModel, Precision, get_device
from cryolike.metadata import ViewingAngles

from cryolike.stacks.template import (
    Templates,
    _get_circles,
    _get_fourier_slices,
    _parse_atomic_model,
    _get_shared_kernel_params,
    _iterate_kernel_with_memory_constraints
)

PKG = "cryolike.stacks.template"


def test_init_throws_on_improper_viewing_angles():
    phys_data = make_mock_data_obj(
        make_image_tensor(10, 10, 10),
        make_mock_phys_grid(10, 10, 1.)
    )

    with raises(ValueError, match="Number of viewing angles"):
        viewing_angles = ViewingAngles.from_viewing_distance(3.)
        assert viewing_angles.n_angles != 10
        _ = Templates(phys_data=phys_data, viewing_angles=viewing_angles)


def test_get_circles():
    device = torch.device('cpu')
    n_angles = 10
    n_inplanes = 8
    torch_float_type = torch.float32
    viewing_angles = make_mock_viewing_angles(n_angles)
    polar_grid = make_mock_polar_grid(2, n_inplanes)
    circles = _get_circles(viewing_angles, polar_grid, torch_float_type, device)
    ## check that the circles are centered at the origin
    centers = torch.mean(circles, dim=1)
    assert_close(centers, torch.zeros((n_angles, 3), dtype=torch_float_type))
    ## check that the circles all have the same radius
    radii = torch.norm(circles, dim=2)
    assert_close(radii, torch.ones((n_angles, n_inplanes), dtype=torch_float_type))
    ## check that the circles are all in the plane normal to the viewing direction
    sin_polar = torch.sin(viewing_angles.polars)
    cos_polar = torch.cos(viewing_angles.polars)
    sin_azimu = torch.sin(viewing_angles.azimus)
    cos_azimu = torch.cos(viewing_angles.azimus)
    orientation_vector = torch.stack((sin_polar * cos_azimu, sin_polar * sin_azimu, cos_polar), dim=1)
    dot_products = torch.sum(circles * orientation_vector.unsqueeze(1), dim=2)
    assert_close(dot_products, torch.zeros((n_angles, 8), dtype=torch_float_type))
    ## check that the circles are angularly equidistant from each other
    points_dot_first_point = torch.sum(circles * circles[:,0,:].unsqueeze(1), dim=2)
    assert_close(points_dot_first_point, torch.cos(torch.arange(0, 2 * np.pi, 2 * np.pi / n_inplanes)).unsqueeze(0).repeat(n_angles, 1))


def test_parse_atomic_model():
    model = AtomicModel()
    polar_grid = make_mock_polar_grid(n_shells=10, n_inplanes=12)
    box_size = np.array([3., 1.])
    float_t = torch.float64
    for device in ['cpu', 'cuda']:
        _device = get_device(device)

        coordinate_scaling_factor = -4. * np.pi / 3.
        radius_scaling_factor = 2. / 3. # box-max is 3.
        pi_rad_factor = 2 * np.pi ** 2

        expected_rad = torch.tensor(model.atom_radii * radius_scaling_factor, dtype=float_t, device=_device)
        expected_shells = torch.tensor(polar_grid.radius_shells, dtype=float_t, device=_device)
        expected_coords = torch.tensor(model.atomic_coordinates, dtype=float_t, device=_device)

        res = _parse_atomic_model(model, polar_grid, box_size, float_t, _device)

        assert_close(res.atomic_radius_scaled, expected_rad)
        assert_close(res.radius_shells, expected_shells)
        assert_close(res.radius_shells_sq, res.radius_shells ** 2)
        assert_close(res.pi_atomic_radius_sq_times_two, pi_rad_factor * res.atomic_radius_scaled ** 2)
        assert_close(res.atomic_coordinates_scaled, expected_coords.T * coordinate_scaling_factor)


@mark.parametrize('precision,uniform,device,output_device', [
    (Precision.SINGLE,True,"cpu","cpu"),
    (Precision.SINGLE,True,"cuda","cpu"),
    (Precision.DOUBLE,False,"cuda","cpu"),
    (Precision.SINGLE,True,"cuda","cuda"),
    (Precision.DOUBLE,False,"cuda","cuda"),
])
@patch(f"{PKG}._parse_atomic_model")
def test_get_shared_kernel_params(parse_model: Mock, precision: Precision, uniform: bool, device: str, output_device: str):

    _device = get_device(device)
    _output_device = get_device(output_device)

    n_templates = 10
    n_shells = 12
    n_inplanes = 14
    n_points_per_img = n_shells * n_inplanes
    box_size = np.array([2, 2.])
    viewing_angles = make_mock_viewing_angles(n_templates)
    polar_grid = make_mock_polar_grid(n_shells, n_inplanes)
    polar_grid.uniform = uniform

    if precision == Precision.DOUBLE:
        t_float = torch.float64
        t_complex = torch.complex128
    else:
        t_float = torch.float32
        t_complex = torch.complex64
    if uniform:
        target_shape = (n_templates, n_shells, n_inplanes)
    else:
        target_shape = (n_templates, n_points_per_img)

    atomic_model = AtomicModel()
    with patch(f"{PKG}._get_circles") as gc:
        res = _get_shared_kernel_params(
            atomic_model,
            viewing_angles,
            polar_grid,
            box_size,
            device=device,
            output_device=output_device,
            precision=precision
        )
        assert res.xyz_template_points == gc.return_value

    print("_device", _device, "res.device", res.device)

    assert res.parsed_model == parse_model.return_value
    parse_model.assert_called_once_with(atomic_model, polar_grid, box_size, float_type=t_float, device=_device)
    assert_close(res.templates_fourier, torch.zeros(target_shape, dtype=t_complex, device=_output_device))
    assert res.polar_grid == polar_grid
    assert res.n_atoms == atomic_model.n_atoms
    assert res.n_templates == n_templates
    assert res.device == _device
    assert res.torch_float_type == t_float
    assert res.torch_complex_type == t_complex


## Do we still need these two? i.e. what's different from test_generate_from_positions_defaults?
@mark.xfail
def test_make_uniform_hard_sphere_kernel():
    raise NotImplementedError


@mark.xfail
def test_make_uniform_gaussian_kernel():
    raise NotImplementedError


###### LOOK AT THIS ######
# @mark.xfail
# @mark.parametrize('max_batch_size,verbose', [(10, False), (5, True), (0, False)])
@mark.parametrize('max_batch_size,device', [(10, 'cuda'), (5, 'cuda'), (0, 'cuda'),
                                             (10, 'cpu'), (5, 'cpu'), (0, 'cpu')])
def test_iterate_kernel(max_batch_size: int, device: str):
    def kernel_op(start: int, end: int):
        if end - start > max_batch_size:
            raise torch.cuda.OutOfMemoryError
    kernel = Mock(side_effect=kernel_op)
    _device = get_device(device)

    n_templates = 27
    verbose = True

    min_batch_size = 1
    batch_size = n_templates
    expected_to_fail_completely = min_batch_size > max_batch_size
    expected_failed_attempts = 0

    if expected_to_fail_completely:
        with raises(MemoryError, match="Insufficient memory"):
            _iterate_kernel_with_memory_constraints(n_templates, kernel, verbose)
        return

    while batch_size > max_batch_size:
        batch_size //= 2
        expected_failed_attempts += 1
    expected_realized_batch_size = batch_size
    expected_iterations = ceil(n_templates / expected_realized_batch_size) + expected_failed_attempts


    with patch('builtins.print') as _print:
        _iterate_kernel_with_memory_constraints(n_templates, kernel, verbose)

        if verbose:
            assert _print.call_count == 1 + expected_failed_attempts
        assert kernel.call_count == expected_iterations


@mark.parametrize('uniform',[True, False])
def test_get_fourier_slices(uniform: bool):
    device = torch.device('cpu')
    n_angles = 10
    n_shells = 4
    n_inplanes = 8
    torch_float_type = torch.float32
    viewing_angles = make_mock_viewing_angles(n_angles)
    polar_grid = make_mock_polar_grid(n_shells, n_inplanes, uniform)
    if not uniform:
        # NOTE: WE ARE NOT YET TESTING NON-UNIFORM POLAR GRIDS
        return
    fourier_slices = _get_fourier_slices(polar_grid, viewing_angles, torch_float_type, device)
    ## check that the circles on the slices have the correct radius
    radii = torch.norm(fourier_slices, dim=2)
    radius_shells = torch.tensor(polar_grid.radius_shells, dtype = torch_float_type, device = device)
    if uniform:
        expected_radii = radius_shells.unsqueeze(0).unsqueeze(2).repeat(n_angles, 1, n_inplanes).reshape(n_angles, n_shells * n_inplanes)
        assert_close(radii, expected_radii)
    else:
        # unreachable--we aren't testing non-uniform grids yet
        expected_radii = radius_shells.unsqueeze(0).unsqueeze(2).repeat(n_angles, 1, n_inplanes).reshape(n_angles, n_shells * n_inplanes)
        assert_close(radii, expected_radii)
    ## other checks are unnecessary because they are already checked in test_get_circles


###### LOOK AT THIS ######
# @mark.xfail
# @mark.parametrize('atom_shape', [AtomShape.DEFAULT, AtomShape.HARD_SPHERE, AtomShape.GAUSSIAN])
@mark.parametrize('atom_shape,device', [(AtomShape.DEFAULT, 'cpu'), (AtomShape.HARD_SPHERE, 'cpu'), (AtomShape.GAUSSIAN, 'cpu'),
                                        (AtomShape.DEFAULT, 'cuda'), (AtomShape.HARD_SPHERE, 'cuda'), (AtomShape.GAUSSIAN, 'cuda')])
@patch(f"{PKG}._get_shared_kernel_params")
# NOTE: This test is ONLY testing that the defaults passed are the ones we expect.
# It is NOT a test of the underlying position generation logic, which is tested in microscopy/test_nufft_3d.
# This test generates some coverage for _iterate_kernel_with_memory_constraints.
def test_generate_from_positions_defaults(mock_get_shared_params: Mock, atom_shape: AtomShape, device: str):

    n_imgs = 3
    model = Mock()
    angles = make_mock_viewing_angles(n_imgs)
    grid = make_mock_polar_grid()
    expected_box_size = np.array([2., 2.])

    mock_params = Mock()
    mock_params.n_templates = n_imgs
    mock_get_shared_params.return_value = mock_params

    mock_templates_tensor = make_image_tensor(n_imgs, 2, 2, target_fourier=True)
    def kernel_effect(start: int, end: int):
        if end >= mock_params.n_templates - 1:
            mock_params.templates_fourier = mock_templates_tensor

    if atom_shape == AtomShape.GAUSSIAN:
        kernel_fn = f"{PKG}._make_uniform_gaussian_kernel"
    else: # use hard shell for explicit or default
        kernel_fn = f"{PKG}._make_uniform_hard_sphere_kernel"
    with patch(kernel_fn) as mock_get_kernel:
        mock_kernel = Mock(side_effect=kernel_effect)
        mock_get_kernel.return_value = mock_kernel

        res = Templates.generate_from_positions(
            atomic_model=model,
            viewing_angles=angles,
            polar_grid=grid,
            box_size=2.,
            atom_shape=atom_shape,
            device=device,
            output_device="cpu"
        )
        mock_get_kernel.assert_called_once()
        assert mock_kernel.call_count == 1
    
    assert_close(res.images_fourier, mock_templates_tensor)
    mock_get_shared_params.assert_called_once()

    calls = mock_get_shared_params.call_args[1]
    assert calls['atomic_model'] == model
    assert calls['viewing_angles'] == angles
    assert calls['polar_grid'] == grid
    npt.assert_allclose(calls['box_size'], expected_box_size)
    assert calls['precision'] == Precision.DEFAULT
    assert calls['device'] == get_device(device)
    assert calls['output_device'] == get_device('cpu')


@patch(f"{PKG}._get_shared_kernel_params")
def test_generate_from_positions_throws_on_non_uniform_grid(mock_get_params: Mock):
    grid = make_mock_polar_grid(uniform=False)
    with raises(NotImplementedError, match="not implemented yet"):
        _ = Templates.generate_from_positions(
            atomic_model=Mock(),
            viewing_angles=Mock(),
            polar_grid=grid,
            box_size=1.0,
        )


@patch(f"{PKG}._get_shared_kernel_params")
def test_generate_from_positions_throws_on_bad_atom_shape(mock_get_params: Mock):
    grid = make_mock_polar_grid()
    badval = Mock()
    badval.value = "NOT SUPPORTED"
    with raises(ValueError, match="Unknown atom shape"):
        _ = Templates.generate_from_positions(
            atomic_model=Mock(),
            viewing_angles=Mock(),
            polar_grid=grid,
            box_size=2.,
            atom_shape=badval # type: ignore
        )


# volume_phys_to_fourier_points --> should return n_img x n_shells x n_inplanes scalar complex floats
# offsets should be scalar real floats, same shape as n_img x n_shells x n_inplanes
# (volume phys is in physical space, it's n_voxel x n_voxel x n_voxel)
@mark.parametrize('uniform', [True, False])
@patch(f"{PKG}.volume_phys_to_fourier_points")
@patch(f"{PKG}._get_fourier_slices")
def test_generate_from_physical_volume(mock_slices: Mock, mock_phys_to_fourier: Mock, uniform: bool):
    n_imgs = 5
    n_shells = 6
    n_inplanes = 3
    n_pts_per_img = n_shells * n_inplanes
    device = get_device("cpu")

    # We don't need to mock the return value--we aren't actually using it
    # if you feel better about it, can set mock_volume.density_physical = any torch tensor
    mock_volume = Mock()
    mock_pgrid = make_mock_polar_grid(n_shells=n_shells, n_inplanes=n_inplanes)
    # Note: we are NOT actually testing non-uniform grid functionality;
    # merely setting the flag to see if the generate_from_physical_volume
    # looks for it.
    # The only visible difference for it is that uniform grids are reshaped
    # back to n_templates x n_shells x n_inplanes, while non-uniforms
    # stay as a list of points per image.
    mock_pgrid.uniform = uniform
    x_pts = torch.tensor(mock_pgrid.x_points)
    y_pts = torch.tensor(mock_pgrid.y_points)
    mock_offset = torch.sinc(2. * x_pts) * torch.sinc(2. * y_pts)
    mock_offset = mock_offset.to(device)

    views = make_mock_viewing_angles(n_imgs)
    # We never actually use the returned value except to call another fn we're mocking, thus
    # we don't actually care what it returns. (By default, calling a Mock returns a new Mock
    # instance, so we don't need to do anything to set the return value.)
    # mock_slices.side_effect = mock_get_fourier_slices(mock_pgrid, views, torch.float32, torch.device('cpu'))

    f_return_1 = torch.arange(n_imgs * n_pts_per_img, device=device).reshape((n_imgs, n_pts_per_img))
    f_return_1 = f_return_1.to(torch.complex64)
    # templates - offset should make the templates mean-0.
    # We've just picked mock values that enforce that.
    f_return_2 = torch.mean(f_return_1, dim=1, keepdim=True).repeat((1, n_pts_per_img)) / mock_offset

    mock_phys_to_fourier.side_effect = [f_return_1, f_return_2]

    res = Templates.generate_from_physical_volume(
        mock_volume,
        mock_pgrid,
        views
    )

    assert res.polar_grid == mock_pgrid
    assert res.images_fourier.dtype == torch.complex64
    if uniform:
        assert_close(res.images_fourier.shape, (n_imgs, n_shells, n_inplanes))
        flat_imgs = torch.reshape(res.images_fourier, (n_imgs, n_shells * n_inplanes))
    else:
        assert_close(res.images_fourier.shape, (n_imgs, n_pts_per_img))
        flat_imgs = res.images_fourier
    assert_close(torch.mean(flat_imgs, dim=1), torch.zeros(n_imgs, device=flat_imgs.device, dtype=flat_imgs.dtype))


def test_generate_from_physical_volume_raises_if_no_phys_density():
    volume = Mock()
    volume.density_physical = None
    with raises(ValueError, match="No physical volume"):
        Templates.generate_from_physical_volume(
            volume,
            polar_grid=Mock(),
            viewing_angles=Mock()
        )


@mark.parametrize('uniform', [True, False])
@patch(f"{PKG}._get_fourier_slices")
def test_generate_from_function(mock_slices: Mock, uniform: bool):
    n_imgs = 10
    n_shells = 3
    n_inplanes = 6
    mock_slices.side_effect = mock_get_fourier_slices
    fn = lambda x: torch.sum(x, -1) + 0j  # sum the x, y, z vals for each point
    views = make_mock_viewing_angles(n_imgs)
    polar_grid = make_mock_polar_grid(n_shells, n_inplanes)
    polar_grid.uniform = uniform

    res = Templates.generate_from_function(fn, views, polar_grid, device="cpu", precision=Precision.SINGLE)

    assert res.polar_grid == polar_grid
    assert res.n_images == n_imgs

    mock_slices.assert_called_once()
    args = mock_slices.call_args
    assert args[0][0] == polar_grid
    assert args[0][1] == views
    assert args[1]['float_type'] == torch.float32
    assert args[1]['device'] == torch.device('cpu')

    expected_imgs = torch.sum(mock_get_fourier_slices(polar_grid, views, torch.complex64, torch.device('cpu')), dim=-1)
    assert_close(res.images_fourier, expected_imgs)
    # Empirically, it's doing the same thing for both branches
    # if uniform:
    #     assert_close(res.images_fourier, expected_imgs)
    # else:
    #     assert_close(res.images_fourier, expected_imgs)


non_callable = torch.arange(10)
bad_shape_fn = lambda x: x
fn_returns_single = lambda x: x[:,:,0].reshape(x.shape[:-1]).to(dtype=torch.complex64)
fn_returns_double = lambda x: x[:,:,0].reshape(x.shape[:-1]).to(dtype=torch.complex128)
@mark.parametrize('fn,msg,precision', [
    (non_callable, "must be callable", Precision.DOUBLE),
    (bad_shape_fn, "takes a tensor of shape", Precision.DOUBLE),
    (fn_returns_single, "appropriate type", Precision.DOUBLE),
    (fn_returns_double, "appropriate type", Precision.SINGLE)
])
def test_generate_from_function_throws_on_bad_fn(fn, msg, precision):
    polar_grid = make_mock_polar_grid()
    views = make_mock_viewing_angles(10)

    with raises(ValueError, match=msg):
        _ = Templates.generate_from_function(fn, views, polar_grid, precision=precision)


def test_to_images():
    t_cimg_and_fgrid = make_basic_Templates(10, with_cimgs=True, with_polargrid=True)
    t_fimg_and_cgrid = make_basic_Templates(11, with_fimgs=True, with_physgrid=True)
    t_cimg_no_fgrid = make_basic_Templates(12, with_cimgs=True)
    t_fimg_no_cgrid = make_basic_Templates(13, with_fimgs=True)

    src = t_cimg_and_fgrid
    res = src.to_images()
    assert_close(res.images_phys, src.images_phys)
    assert res.images_fourier.shape[0] == 0
    npt.assert_allclose(res.phys_grid.pixel_size, src.phys_grid.pixel_size)
    assert res.polar_grid == src.polar_grid

    src = t_fimg_and_cgrid
    res = src.to_images()
    assert_close(res.images_fourier, src.images_fourier)
    assert not res.has_physical_images()
    npt.assert_allclose(res.phys_grid.pixel_size, src.phys_grid.pixel_size)
    assert res.polar_grid == src.polar_grid

    src = t_fimg_no_cgrid
    res = src.to_images()
    assert_close(res.images_fourier, src.images_fourier)
    assert not res.has_physical_images()
    assert res.polar_grid == src.polar_grid
    assert getattr(res, "phys_grid", None) is None

    src = t_cimg_no_fgrid
    res = src.to_images()
    assert_close(res.images_phys, src.images_phys)
    assert res.images_fourier.shape[0] == 0
    npt.assert_allclose(res.phys_grid.pixel_size, src.phys_grid.pixel_size)
    assert getattr(res, "polar_grid", None) is None
