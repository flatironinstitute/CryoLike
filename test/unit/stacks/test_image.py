from unittest.mock import patch, Mock
import numpy as np
import numpy.testing as npt
import torch
from torch.testing import assert_close
from pytest import raises, mark
import re

from cryolike.grids import PolarGrid, CartesianGrid2D
from cryolike.util import Precision, NormType, get_device

from cryolike.stacks.image import (
    Images
)

PKG = "cryolike.stacks.image"


from stacks_fixtures import (
    make_basic_Templates,
    make_image_tensor,
    make_mock_data_obj,
    make_mock_phys_grid,
    make_mock_polar_grid,
    make_mock_viewing_angles
)

def test_ctor_sets_physical_data():
    n_im_in_stack = 3
    img_x_dim = 4
    img_y_dim = 5

    mock_polar_grid = make_mock_polar_grid()
    mock_phys_grid = make_mock_phys_grid(img_x_dim, img_y_dim, 3.)
    mock_phys_im = make_image_tensor(n_im_in_stack, img_x_dim, img_y_dim)
    mock_phys_data = make_mock_data_obj(mock_phys_im, mock_phys_grid)

    viewings = Mock()
    ctf = Mock()

    res = Images(mock_phys_data, fourier_data=mock_polar_grid, viewing_angles=viewings, ctf=ctf)

    assert res.n_images == n_im_in_stack
    assert_close(res.images_phys, mock_phys_im)
    assert res.images_fourier.shape[0] == 0
    npt.assert_allclose(mock_phys_grid.box_size, res.box_size)
    assert res.phys_grid == mock_phys_grid
    assert res.polar_grid == mock_polar_grid
    assert res.filename is None


def test_ctor_sets_fourier_data():
    n_im_in_stack = 13
    img_s_dim = 4
    img_w_dim = 5

    four_im = make_image_tensor(n_im_in_stack, img_s_dim, img_w_dim, target_fourier=True)
    mock_polar_grid = make_mock_polar_grid()
    mock_fourier_data = make_mock_data_obj(four_im, mock_polar_grid)

    res = Images(fourier_data=mock_fourier_data)

    assert res.n_images == n_im_in_stack
    assert_close(res.images_fourier, four_im)
    assert res.images_phys.shape[0] == 0
    assert res.polar_grid == mock_polar_grid
    assert res.ctf is None
    assert res.viewing_angles is None
    assert res.filename is None


def test_ctor_box_size_logic():
    mock_polar_grid = make_mock_polar_grid()
    mock_fourier_im = make_image_tensor(1, 4, 4, target_fourier=True)
    mock_f_data = make_mock_data_obj(mock_fourier_im, mock_polar_grid)

    # we've already checked that a phys grid box size will be used.
    # check that explicit box size will trump a phys grid
    mock_phys_grid = make_mock_phys_grid(1, 1, 4.)
    explicit_box_size = np.array([10., 10.])

    res = Images(phys_data=mock_phys_grid, fourier_data=mock_f_data, box_size=explicit_box_size)
    assert res.phys_grid == mock_phys_grid
    npt.assert_allclose(explicit_box_size, res.box_size)

    # Check that default is used when no box size or phys grid is passed
    res = Images(fourier_data=mock_f_data)
    npt.assert_allclose(res.box_size, np.array([2., 2.]))
    
    # check expansion of floats
    res = Images(fourier_data=mock_f_data, box_size=5.)
    npt.assert_allclose(res.box_size, np.array([5., 5.]))

    # check truncation of over-long box size arrays
    res = Images(fourier_data=mock_f_data, box_size=np.array([6., 6., 6., 6.]))
    npt.assert_allclose(res.box_size, np.array([6., 6.]))


def test_ctor_throws_if_no_images_provided():
    mock_polar_grid = Mock()
    mock_polar_grid.__class__ = PolarGrid           # type: ignore
    mock_phys_grid = Mock()
    mock_phys_grid.__class__ = CartesianGrid2D      # type: ignore

    with raises(ValueError, match="No images provided"):
        _ = Images()

    with raises(ValueError, match="No images provided"):
        _ = Images(phys_data=mock_phys_grid, fourier_data=mock_polar_grid)


def test_check_image_array_promotes_single_image_tensors():
    img_x_dim = 4
    img_y_dim = 4
    img = torch.arange(img_x_dim * img_y_dim, dtype=torch.float64).reshape([img_x_dim, img_y_dim])
    assert img.shape[0] == img_x_dim
    assert len(img.shape) == 2
    mock_phys_grid = make_mock_phys_grid(img_x_dim, img_y_dim, 1.)
    phys_data = make_mock_data_obj(img, mock_phys_grid)

    res = Images(phys_data=phys_data)

    assert res.n_images == 1
    assert res.images_phys.shape[0] == 1
    assert_close(res.images_phys[0], img)


def test_check_image_array_throws_on_too_many_phys_img_dimensions():
    imgs = make_image_tensor(4, 5, 5)
    imgs = imgs.reshape([2, 2, 5, 5])
    mock_phys_grid = make_mock_phys_grid(5, 5, 1.)
    mock_phys_data = make_mock_data_obj(imgs, mock_phys_grid)

    with raises(ValueError, match="Invalid shape for images"):
        _ = Images(phys_data=mock_phys_data)


def test_check_image_array_checks_image_counts():
    n_im_in_stack = 13
    four_im = make_image_tensor(n_im_in_stack, 5, 5, target_fourier=True)
    mock_fdata = make_mock_data_obj(four_im, make_mock_polar_grid())

    sut = Images(fourier_data=mock_fdata)
    sut.n_images = 8

    with raises(ValueError, match="the fourier images array has 13"):
        sut._check_image_array()

    phys_im = make_image_tensor(n_im_in_stack, 5, 5)
    mock_cgrid = make_mock_phys_grid(5, 5, .2)
    mock_pdata = make_mock_data_obj(phys_im, mock_cgrid)

    sut = Images(phys_data=mock_pdata)
    sut.n_images = 8

    with raises(ValueError, match="the physical images array has 13"):
        sut._check_image_array()


def test_check_image_array_checks_required_grids():
    n_im_in_stack = 5
    p_imgs = make_image_tensor(n_im_in_stack, 12, 12)
    mock_cgrid = make_mock_phys_grid(12, 12, 2.)

    sut = Images(phys_data=make_mock_data_obj(p_imgs, mock_cgrid))
    sut.phys_grid = None # type: ignore
    with raises(ValueError, match="Physical grid is not defined"):
        sut._check_image_array()

    f_imgs = make_image_tensor(n_im_in_stack, 12, 12, target_fourier=True)
    mock_pgrid = make_mock_polar_grid()

    sut = Images(fourier_data=make_mock_data_obj(f_imgs, mock_pgrid))
    sut.polar_grid = None # type: ignore
    with raises(ValueError, match="Polar grid is not defined"):
        sut._check_image_array()


def test_check_phys_imgs_checks_grid_pixels_against_shape():
    n_im = 3
    p_imgs = make_image_tensor(n_im, 6, 6)
    mock_cgrid_bad_x = make_mock_phys_grid(5, 6, .1)
    mock_cgrid_bad_y = make_mock_phys_grid(6, 5, .1)

    with raises(ValueError, match="Dimension mismatch"):
        _ = Images(phys_data=make_mock_data_obj(p_imgs, mock_cgrid_bad_x))
    with raises(ValueError, match="Dimension mismatch"):
        _ = Images(phys_data=make_mock_data_obj(p_imgs, mock_cgrid_bad_y))


def test_has_images():
    n_im = 3
    p_imgs = make_image_tensor(n_im, 6, 6)
    mock_cgrid = make_mock_phys_grid(6, 6, 1.)
    sut = Images(phys_data=make_mock_data_obj(p_imgs, mock_cgrid))
    assert sut.has_physical_images()
    assert not sut.has_fourier_images()

    f_imgs = make_image_tensor(n_im, 4, 4, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    sut = Images(fourier_data=make_mock_data_obj(f_imgs, mock_fgrid))
    assert sut.has_fourier_images()
    assert not sut.has_physical_images()


def test_get_item_size():
    n_im = 3
    p_imgs = make_image_tensor(n_im, 6, 6)
    mock_cgrid = make_mock_phys_grid(6, 6, 1.)
    expected_size = (p_imgs.element_size() * p_imgs.nelement()) / n_im
    sut = Images(phys_data=make_mock_data_obj(p_imgs, mock_cgrid))
    assert sut.get_item_size('fourier') == 0
    assert sut.get_item_size('physical') == expected_size

    n_f_im = 12
    f_imgs = make_image_tensor(n_f_im, 12, 16, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    expected_size = (f_imgs.element_size() * f_imgs.nelement()) / n_f_im
    sut = Images(fourier_data=make_mock_data_obj(f_imgs, mock_fgrid))
    assert sut.get_item_size('physical') == 0
    assert sut.get_item_size('fourier') == expected_size

    with raises(NotImplementedError):
        sut.get_item_size('unallowed') # type: ignore



@patch("builtins.print")
def test_check_phys_imgs_warns_on_box_size_inconsistency(mock_print: Mock):
    p_imgs = make_image_tensor(3, 5, 5)
    mock_cgrid = make_mock_phys_grid(5, 5, .1)
    explicit_box = 5.

    _ = Images(phys_data=make_mock_data_obj(p_imgs, mock_cgrid), box_size=explicit_box)

    warning = mock_print.call_args_list[0].args[0]
    mock_print.assert_called()
    assert re.match(".*is outside tolerance.*", warning)



def test_ensure_images():
    fourier_im = Images(fourier_data=make_mock_data_obj(
        make_image_tensor(2, 3, 3, target_fourier=True),
        make_mock_polar_grid(),
    ))
    phys_im = Images(phys_data=make_mock_data_obj(
        make_image_tensor(2, 3, 3),
        make_mock_phys_grid(3, 3, .1)
    ))

    with raises(ValueError, match="Physical images not found"):
        fourier_im._ensure_phys_images()
    fourier_im._ensure_fourier_images()     # should succeed
    with raises(ValueError, match="Fourier images not found"):
        phys_im._ensure_fourier_images()
    phys_im._ensure_phys_images()           # should succeed


def test_from_mrc_throws_on_bad_file_extension():
    with raises(ValueError, match="Invalid file format"):
        _ = Images.from_mrc("somefile.txt", None)


def _make_mock_mrc_file(*, single_img: bool = False, neg_pixel_size: bool = False):
    mock_res = Mock()
    mock_res.n_imgs = 1 if single_img else 5
    mock_res.x_npix = 25
    mock_res.y_npix = 25
    mock_res.voxel_size = Mock()
    mock_res.voxel_size.x = 12.
    mock_res.voxel_size.y = 12.
    mock_res.expected_box = np.array([
        mock_res.x_npix * mock_res.voxel_size.x,
        mock_res.y_npix * mock_res.voxel_size.y
    ])
    mock_pimg = make_image_tensor(
        mock_res.n_imgs,
        mock_res.x_npix,
        mock_res.y_npix
    ).numpy()
    if single_img:
        mock_pimg = mock_pimg[0]
    mock_res.data = mock_pimg
    if neg_pixel_size:
        mock_res.voxel_size.x *= -1.
        mock_res.voxel_size.y *= -1.
    mock_res.set_data = Mock()
    return mock_res


def _make_mock_mrcfile_ctxt(ret: Mock):
    mock_ctxt = Mock()
    mock_ctxt.__enter__ = Mock(return_value=ret)
    # return False to avoid swallowing exceptions
    mock_ctxt.__exit__ = Mock(return_value=False)
    return mock_ctxt


@patch(f"{PKG}.mrcfile")
def test_from_mrc_loads_correctly(mock_mrcfile: Mock):
    fn = "myfile.mrc"
    mock_mrc = _make_mock_mrc_file()
    mock_ctxt = _make_mock_mrcfile_ctxt(mock_mrc)
    mock_mrcfile.open = Mock(return_value=mock_ctxt)
    expected_box = mock_mrc.expected_box

    res = Images.from_mrc(filename=fn, pixel_size=None)

    mock_mrcfile.open.assert_called_once_with(fn)
    assert res.filename == fn
    assert_close(res.images_phys, torch.from_numpy(mock_mrc.data))
    npt.assert_allclose(res.box_size, expected_box)


@patch(f"{PKG}.mrcfile")
def test_from_mrc_with_single_image_mrc_file(mock_mrcfile: Mock):
    fn = "myfile.mrcs"
    # try with single-image mrc file
    mock_mrc = _make_mock_mrc_file(single_img=True)
    mock_ctxt = _make_mock_mrcfile_ctxt(mock_mrc)
    mock_mrcfile.open = Mock(return_value=mock_ctxt)

    res = Images.from_mrc(filename=fn, pixel_size=None)
    assert res.n_images == 1
    assert res.images_phys.shape == torch.Size([1, mock_mrc.x_npix, mock_mrc.y_npix])


@patch(f"{PKG}.mrcfile")
def test_from_mrc_honors_explicit_pixel_size(mock_mrcfile: Mock):
    fn = "myfile.mrc"
    mock_mrc = _make_mock_mrc_file()
    mock_ctxt = _make_mock_mrcfile_ctxt(mock_mrc)
    mock_mrcfile.open = Mock(return_value=mock_ctxt)
    pixel_size = [2., 2.]
    expected_box = np.array([
        mock_mrc.x_npix * pixel_size[0],
        mock_mrc.y_npix * pixel_size[1]
        ])

    res = Images.from_mrc(filename=fn, pixel_size=pixel_size)
    npt.assert_allclose(res.box_size, expected_box)


@patch(f"{PKG}.mrcfile")
def test_from_mrc_throws_on_negative_pixel_size(mock_mrcfile: Mock):
    fn = "myfile.mrc"
    mock_mrc = _make_mock_mrc_file(neg_pixel_size=True)
    mock_ctxt = _make_mock_mrcfile_ctxt(mock_mrc)
    mock_mrcfile.open = Mock(return_value=mock_ctxt)

    with raises(ValueError, match="contains non-positive pixel sizes"):
        _ = Images.from_mrc(filename=fn, pixel_size=None)


@patch(f"{PKG}.mrcfile")
def test_save_to_mrc(mock_mrcfile: Mock):
    fn = "outfile.mrc"
    mock_mrc = _make_mock_mrc_file()
    mock_ctxt = _make_mock_mrcfile_ctxt(mock_mrc)
    mock_mrcfile.new = Mock(return_value=mock_ctxt)
    assert isinstance(mock_mrcfile.new, Mock)

    n_im = 3
    img_x = 25
    img_y = 15

    mock_pgrid = make_mock_phys_grid(img_x, img_y, 1.)
    mock_pimg = make_image_tensor(n_im, img_x, img_y)
    img = Images(phys_data=make_mock_data_obj(mock_pimg, mock_pgrid))

    img.save_to_mrc(fn)
    mock_mrcfile.new.assert_called_once_with(fn, overwrite=True)
    assert isinstance(mock_mrc.set_data, Mock)
    mock_mrc.set_data.assert_called_once_with(img.images_phys)
    assert mock_mrc.voxel_size[0] == mock_pgrid.pixel_size[0]
    assert mock_mrc.voxel_size[1] == mock_pgrid.pixel_size[1]
    assert mock_mrc.voxel_size[2] == 1.


def test_modify_pixel_size():
    x_npix = 25
    y_npix = 25
    n_imgs = 10
    mock_pgrid = make_mock_phys_grid(x_npix, y_npix, 1.)
    mock_pimg = make_image_tensor(n_imgs, x_npix, y_npix)
    
    res = Images(phys_data=make_mock_data_obj(mock_pimg, mock_pgrid))
    new_size = 13.
    expected_box = np.array([
        mock_pgrid.n_pixels[0] * new_size,
        mock_pgrid.n_pixels[1] * new_size
    ])
    res.modify_pixel_size(new_size)
    
    assert res.phys_grid != mock_pgrid
    npt.assert_allclose(res.box_size, expected_box)


def test_pad_or_trim_images():
    x_npix = 50
    y_npix = 50
    n_imgs = 10
    mock_pimg = make_image_tensor(n_imgs, x_npix, y_npix)
    mock_cgrid = make_mock_phys_grid(x_npix, y_npix, 1.)
    new_y_dim = y_npix + 10
    mock_new_cgrid = make_mock_phys_grid(x_npix, new_y_dim, 1.)

    sut = Images(phys_data=make_mock_data_obj(mock_pimg, mock_cgrid))
    sut.phys_grid = mock_new_cgrid
    sut._pad_or_trim_images_if_needed()
    # x-dim not changed
    assert sut.images_phys.shape[0] == n_imgs
    assert sut.images_phys.shape[1] == x_npix
    
    # assert y-dim now matches new grid's
    assert sut.images_phys.shape[2] == new_y_dim
    # based on hard-coded values, should've inserted 5 0s
    # at start and end of y-index. Assert this.
    left_pad = sut.images_phys[:,:,0:5]
    right_pad = sut.images_phys[:,:,-5:]
    assert np.allclose(left_pad, np.zeros_like(left_pad))
    assert np.allclose(right_pad, np.zeros_like(right_pad))
    # assert the non-padded parts are untouched
    orig = sut.images_phys[:,:,5:-5]
    assert np.allclose(orig, mock_pimg)


def test_pad_or_trim_images_trims_oversize_imgs():
    x_npix = 50
    y_npix = 50
    n_imgs = 10
    mock_pimg = make_image_tensor(n_imgs, x_npix, y_npix)
    mock_cgrid = make_mock_phys_grid(x_npix, y_npix, 1.)
    new_x_dim = x_npix - 10
    new_y_dim = y_npix - 20
    mock_new_cgrid = make_mock_phys_grid(new_x_dim, new_y_dim, 1.)

    sut = Images(phys_data=make_mock_data_obj(mock_pimg, mock_cgrid))
    sut.phys_grid = mock_new_cgrid
    sut._pad_or_trim_images_if_needed()
    assert sut.images_phys.shape[0] == n_imgs
    assert sut.images_phys.shape[1] == new_x_dim
    assert sut.images_phys.shape[2] == new_y_dim
    # assert we took the middle n values from the source img dims
    sliced = mock_pimg[:,5:45,10:40]
    assert np.allclose(sliced, sut.images_phys)


@patch(f"{PKG}.torch.from_numpy")
def test_pad_images_skips_empty_phys_imgs(mock_from_npy: Mock):
    four_im = make_image_tensor(2, 12, 12, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    
    sut = Images(fourier_data=make_mock_data_obj(four_im, mock_fgrid))
    
    sut._pad_or_trim_images_if_needed()
    mock_from_npy.assert_not_called()


def test_change_images_phys_size():
    x_npix = 50
    y_npix = 50
    n_imgs = 10
    mock_pimg = make_image_tensor(n_imgs, x_npix, y_npix)
    mock_cgrid = make_mock_phys_grid(x_npix, y_npix, 1.)

    sut = Images(phys_data=make_mock_data_obj(mock_pimg, mock_cgrid))
    sut.change_images_phys_size(box_size=15.)
    assert np.allclose(sut.box_size, np.array([15., 15.]))
    # TODO: Should probably assert that the new grid has an n_pixels computed to fit new box size

    sut = Images(phys_data=make_mock_data_obj(mock_pimg, mock_cgrid))
    sut.change_images_phys_size(n_pixels=75)
    assert np.allclose(sut.phys_grid.n_pixels, np.array([75, 75]))
    assert sut.images_phys.shape[1] == 75
    assert sut.images_phys.shape[2] == 75


def test_change_images_phys_size_throws_on_bad_params():
    mock_fimg = make_image_tensor(5, 10, 10, target_fourier=True)
    mock_pgrid = make_mock_polar_grid()
    sut = Images(fourier_data=make_mock_data_obj(mock_fimg, mock_pgrid))

    with raises(ValueError, match="Only one of n_pixels or box_size"):
        sut.change_images_phys_size(5, 10.)
    with raises(ValueError, match="Either n_pixels or box_size"):
        sut.change_images_phys_size(None, None)
    with raises(ValueError, match="no physical images"):
        sut.change_images_phys_size(n_pixels=5, box_size=None)



@patch(f"{PKG}.cartesian_phys_to_fourier_polar")
def test_transform_to_fourier(mock_conv: Mock):
    n_imgs = 10
    mock_cimg = make_image_tensor(n_imgs, 20, 20)
    mock_cgrid = make_mock_phys_grid(20, 20, 1.)
    mock_fgrid = make_mock_polar_grid(n_shells=10, n_inplanes=12)
    mock_fimg_ret = make_image_tensor(n_imgs, 6, 20, target_fourier=True)
    mock_conv.return_value = mock_fimg_ret

    sut = Images(
        phys_data=make_mock_data_obj(mock_cimg, mock_cgrid),
        fourier_data=mock_fgrid
    )
    sut.transform_to_fourier()

    mock_conv.assert_called_once_with(
        grid_cartesian_phys=mock_cgrid,
        grid_fourier_polar = mock_fgrid,
        images_phys = mock_cimg,
        eps = 1e-12,
        precision = Precision.DOUBLE,
        device = get_device('cuda')
    )
    assert sut.images_fourier.shape[0] == n_imgs
    assert sut.images_fourier.shape[1] == 10
    assert sut.images_fourier.shape[2] == 12
    assert_close(sut.images_fourier.flatten(), mock_fimg_ret.flatten())
    assert sut.images_fourier.device == sut.images_phys.device


@patch(f"{PKG}.cartesian_phys_to_fourier_polar")
def test_transform_to_fourier_with_new_polar_grid(mock_conv: Mock):
    n_imgs = 10
    mock_cimg = make_image_tensor(n_imgs, 20, 20)
    mock_cgrid = make_mock_phys_grid(20, 20, 1.)
    mock_fgrid = make_mock_polar_grid(n_shells=10, n_inplanes=12)
    mock_fgrid_nonu = make_mock_polar_grid(uniform=False)
    mock_fimg_ret = make_image_tensor(n_imgs, 6, 20, target_fourier=True)
    mock_conv.return_value = mock_fimg_ret

    sut = Images(
        phys_data=make_mock_data_obj(mock_cimg, mock_cgrid),
        fourier_data=mock_fgrid
    )
    sut.transform_to_fourier(polar_grid=mock_fgrid_nonu, precision=Precision.SINGLE)

    mock_conv.assert_called_once_with(
        grid_cartesian_phys=mock_cgrid,
        grid_fourier_polar = mock_fgrid_nonu,
        images_phys = mock_cimg,
        eps = 1e-12,
        precision = Precision.SINGLE,
        device = get_device('cuda')
    )
    assert sut.polar_grid == mock_fgrid_nonu
    assert sut.images_fourier.shape[1] == mock_fimg_ret.shape[1]
    assert sut.images_fourier.shape[2] == mock_fimg_ret.shape[2]
    assert_close(sut.images_fourier.flatten(), mock_fimg_ret.flatten())
    assert sut.images_fourier.device == sut.images_phys.device


def test_transform_to_fourier_throws_with_no_polar_grid():
    n_imgs = 10
    mock_cimg = make_image_tensor(n_imgs, 20, 20)
    mock_cgrid = make_mock_phys_grid(20, 20, 1.)

    sut = Images(phys_data=make_mock_data_obj(mock_cimg, mock_cgrid))
    with raises(ValueError, match="No polar grid found"):
        sut.transform_to_fourier()


@patch(f"{PKG}.fourier_polar_to_cartesian_phys")
def test_transform_to_spatial(mock_conv: Mock):
    n_imgs = 10
    mock_fimg = make_image_tensor(n_imgs, 5, 12, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    mock_cgrid = make_mock_phys_grid(25, 25, 1.)
    mock_cimg_ret = make_image_tensor(n_imgs, 25, 25)
    mock_conv.return_value = mock_cimg_ret

    sut = Images(phys_data=mock_cgrid, fourier_data=make_mock_data_obj(mock_fimg, mock_fgrid))
    sut.transform_to_spatial()
    assert_close(mock_cimg_ret, sut.images_phys)
    mock_conv.assert_called_once()
    args = mock_conv.call_args[1]
    assert args["grid_fourier_polar"] == mock_fgrid
    assert args["grid_cartesian_phys"] == mock_cgrid
    assert_close(args["image_polar"].flatten(), mock_fimg.flatten())
    assert args["eps"] == 1e-12
    assert args["precision"] == Precision.DOUBLE
    assert args["device"] == get_device(sut.images_fourier.device)
    assert sut.images_phys.device == sut.images_fourier.device


@patch(f"{PKG}.fourier_polar_to_cartesian_phys")
@patch("builtins.print")
def test_transform_to_spatial_with_limit(mock_print: Mock, mock_conv: Mock):
    n_imgs = 10
    transform_limit = 5
    mock_fimg = make_image_tensor(n_imgs, 5, 12, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    mock_cgrid = make_mock_phys_grid(25, 25, 1.)
    mock_cimg_ret = make_image_tensor(transform_limit, 25, 25)
    mock_conv.return_value = mock_cimg_ret

    sut = Images(phys_data=mock_cgrid, fourier_data=make_mock_data_obj(mock_fimg, mock_fgrid))
    xformed = sut.transform_to_spatial(max_to_transform=transform_limit)
    assert sut.images_phys.shape[0] == 0
    assert_close(xformed, mock_cimg_ret)
    mock_conv.assert_called_once()
    args = mock_conv.call_args[1]
    assert args["grid_fourier_polar"] == mock_fgrid
    assert args["grid_cartesian_phys"] == mock_cgrid
    assert_close(args["image_polar"].flatten(), mock_fimg[:transform_limit].flatten())
    assert args["eps"] == 1e-12
    assert args["precision"] == Precision.DOUBLE
    assert args["device"] == get_device(sut.images_fourier.device)

    mock_print.assert_called_once_with(f"Transforming only the first {transform_limit} images, probably for testing or plotting. Transformed images will be returned but not persisted.")

    # Assert we do not persist transformed images if max_to_transform is not the default -1
    xformed = sut.transform_to_spatial(max_to_transform=0)
    assert sut.images_phys.shape[0] == 0
    args = mock_conv.call_args[1]
    assert_close(args["image_polar"].flatten(), mock_fimg.flatten())

    xformed = sut.transform_to_spatial(max_to_transform=n_imgs * 10)
    assert sut.images_phys.shape[0] == 0
    args = mock_conv.call_args[1]
    assert_close(args["image_polar"].flatten(), mock_fimg.flatten())


@patch("builtins.print")
@patch(f"{PKG}.fourier_polar_to_cartesian_phys")
def test_transform_to_spatial_with_new_phys_grid(mock_conv: Mock, mock_print: Mock):
    n_imgs = 10
    mock_fimg = make_image_tensor(n_imgs, 5, 12, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    mock_cgrid = make_mock_phys_grid(25, 25, 1.)
    mock_cimg_ret = make_image_tensor(n_imgs, 25, 25)
    mock_conv.return_value = mock_cimg_ret

    sut = Images(fourier_data=make_mock_data_obj(mock_fimg, mock_fgrid))
    sut.transform_to_spatial(grid=mock_cgrid, precision=Precision.SINGLE)
    assert sut.phys_grid == mock_cgrid
    assert_close(sut.images_phys, mock_cimg_ret)
    mock_conv.assert_called_once()
    args = mock_conv.call_args[1]
    assert args["grid_fourier_polar"] == mock_fgrid
    assert args["grid_cartesian_phys"] == mock_cgrid
    assert_close(args["image_polar"].flatten(), mock_fimg.flatten())
    assert args["eps"] == 1e-12
    assert args["precision"] == Precision.SINGLE
    assert args["device"] == get_device(sut.images_fourier.device)
    assert sut.images_phys.device == sut.images_fourier.device
    mock_print.assert_called_once_with("Precision single provided, overriding the existing precision.")


@patch("builtins.print")
@patch(f"{PKG}.fourier_polar_to_cartesian_phys")
def test_transform_to_spatial_changing_precision(mock_conv: Mock, mock_print: Mock):
    n_imgs = 10
    mock_fimg = make_image_tensor(n_imgs, 5, 12, target_fourier=True)
    mock_fimg = mock_fimg.to(dtype=torch.complex64)
    mock_fgrid = make_mock_polar_grid()
    mock_cgrid = make_mock_phys_grid(25, 25, 1.)
    mock_cimg_ret = make_image_tensor(n_imgs, 25, 25)
    mock_conv.return_value = mock_cimg_ret

    sut = Images(fourier_data=make_mock_data_obj(mock_fimg, mock_fgrid))
    sut.transform_to_spatial(grid=mock_cgrid, precision=Precision.DOUBLE)
    mock_conv.assert_called_once()
    args = mock_conv.call_args[1]
    assert args["precision"] == Precision.DOUBLE
    mock_print.assert_called_once_with("Precision double provided, overriding the existing precision.")

    # Make sure we don't get any new printing when explicitly telling it the correct precision
    sut.transform_to_spatial(grid=mock_cgrid, precision=Precision.SINGLE)
    mock_print.assert_called_once()


def test_transform_to_spatial_throws_with_no_phys_grid():
    n_imgs = 10
    mock_fimg = make_image_tensor(n_imgs, 5, 12, target_fourier=True)
    mock_fgrid = make_mock_polar_grid()
    sut = Images(fourier_data=make_mock_data_obj(mock_fimg, mock_fgrid))

    with raises(ValueError, match="No physical grid found"):
        sut.transform_to_spatial()


def test_center_physical_image_signal():
    img1 = make_image_tensor(1, 5, 5) + 10.
    img2 = make_image_tensor(1, 5, 5) + 20.

    cimg = torch.concat([img1, img2])
    cgrid = make_mock_phys_grid(5, 5, 1.)
    
    sut = Images(phys_data=make_mock_data_obj(cimg, cgrid))
    sut.center_physical_image_signal(norm_type=NormType.MAX)
    expected_maxnorm = (torch.arange(25, dtype=torch.float64) - 12.) / 12.
    assert_close(sut.images_phys[0].flatten(), expected_maxnorm)
    assert_close(sut.images_phys[1].flatten(), expected_maxnorm)

    sut = Images(phys_data=make_mock_data_obj(cimg, cgrid))
    sut.center_physical_image_signal(norm_type=NormType.STD)
    expected_std = (torch.arange(25, dtype=torch.float64) - 12.) / torch.std(torch.arange(25, dtype=torch.float64) - 12.)
    assert_close(sut.images_phys[0].flatten(), expected_std)
    assert_close(sut.images_phys[1].flatten(), expected_std)

    with raises(ValueError, match="Unreachable"):
        sut.center_physical_image_signal(norm_type=-1) # type: ignore


def test_apply_ctf():
    fgrid = make_mock_polar_grid()
    fgrid_nonuniform = make_mock_polar_grid(uniform=False)
    fimgs = make_image_tensor(5, 12, 12, target_fourier=True)
    mock_ctf = Mock()
    mock_ctf.apply = Mock(side_effect=lambda x: x)
    assert isinstance(mock_ctf.apply, Mock)
    sut = Images(fourier_data=make_mock_data_obj(fimgs, fgrid))
    assert sut.ctf is None

    res = sut.apply_ctf(mock_ctf)
    assert sut.ctf == mock_ctf
    mock_ctf.apply.assert_called_once()
    assert_close(mock_ctf.apply.call_args[0][0], fimgs)
    assert_close(sut.images_fourier, res)
    
    sut.polar_grid = fgrid_nonuniform
    with raises(NotImplementedError, match="Non-uniform Fourier images not implemented yet"):
        sut.apply_ctf(mock_ctf)


@patch(f"{PKG}.torch.randn_like")
def test_add_noise_phys(mock_randn: Mock):
    snr = torch.tensor([3.])

    cimg = make_image_tensor(2, 10, 10)
    mock_randn.return_value = torch.ones_like(cimg)
    cgrid = make_mock_phys_grid(10, 10, 1.)
    imgs = Images(phys_data=make_mock_data_obj(cimg, cgrid))

    with raises(ValueError, match="positive"):
        imgs.add_noise_phys(-4.)

    # we expect sigma_noise to be [33.0832, 87.9081]
    (updated, noise) = imgs.add_noise_phys(snr)
    _expected_sigma_noise = torch.tensor([33.083228, 87.908096])
    _expanded_noise = _expected_sigma_noise.unsqueeze(1).unsqueeze(2) * torch.ones_like(cimg)
    assert_close(imgs.images_phys, updated)
    assert_close(imgs.images_phys.flatten(), (cimg + _expanded_noise).flatten())
    npt.assert_allclose(noise, _expected_sigma_noise.numpy())

    # And make sure it also works for numpy-typed SNR values
    imgs = Images(phys_data=make_mock_data_obj(cimg, cgrid))
    (updated, noise) = imgs.add_noise_phys(snr.numpy())
    assert_close(imgs.images_phys, updated)
    assert_close(imgs.images_phys.flatten(), (cimg + _expanded_noise).flatten())
    npt.assert_allclose(noise, _expected_sigma_noise.numpy())


def test_add_noise_phys_throws_on_no_images():
    fimg = make_image_tensor(2, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    fourier_imgs = Images(fourier_data=make_mock_data_obj(fimg, fgrid))

    with raises(ValueError, match="physical images are not set"):
        fourier_imgs.add_noise_phys(5.)


@patch(f"{PKG}.torch.randn_like")
def test_add_noise_fourier(mock_randn: Mock):
    snr = torch.tensor([3.])

    fimg = make_image_tensor(2, 10, 10, target_fourier=True)
    mock_randn.return_value = torch.ones_like(fimg)
    fgrid = make_mock_polar_grid()
    def mock_integrate(x: torch.Tensor):
        # mocked version applies weights of 1
        return torch.sum(x, dim = -1) * (2 * np.pi) ** 2
    fgrid.integrate = Mock(side_effect=lambda x: mock_integrate(x))
    img = Images(fourier_data=make_mock_data_obj(fimg, fgrid))

    with raises(ValueError, match="positive"):
        img.add_noise_fourier(-5.)

    _expected_sigma_noise = torch.sqrt(mock_integrate(fimg.abs().pow(2)) / snr)
    _expanded_noise = _expected_sigma_noise.unsqueeze(1).unsqueeze(2) * torch.ones_like(fimg, dtype=torch.complex128)
    (updated, noise) = img.add_noise_fourier(snr)
    assert_close(img.images_fourier, updated)
    assert_close(img.images_fourier.flatten(), (fimg + _expanded_noise).flatten())
    npt.assert_allclose(noise, _expected_sigma_noise.flatten().numpy())

    # repeat with numpy-formatted
    snr = snr.numpy()
    img = Images(fourier_data=make_mock_data_obj(fimg, fgrid))
    (updated, noise) = img.add_noise_fourier(snr)
    assert_close(img.images_fourier, updated)
    assert_close(img.images_fourier.flatten(), (fimg + _expanded_noise).flatten())
    npt.assert_allclose(noise, _expected_sigma_noise.flatten().numpy())


def test_add_noise_fourier_throws_on_no_imgs():
    with raises(ValueError, match="fourier images are not set"):
        cimg = make_image_tensor(2, 10, 10)
        cgrid = make_mock_phys_grid(10, 10, 1.)
        phys_imgs = Images(phys_data=make_mock_data_obj(cimg, cgrid))
        phys_imgs.add_noise_fourier()


# Checks:
# - basic happy path
# - handles adding (0, 0) for even-numbered entries
# - handles one or both displacement counts being < 1
@mark.parametrize("x_disp,y_disp", [(5, 7), (5, -2), (0, 3), (-3, -4), (4, 4)])
def test_set_displacement_grid(x_disp, y_disp):
    fimg = make_image_tensor(2, 10, 10, target_fourier=True)
    pimg = make_image_tensor(2, 10, 10, target_fourier=False)
    fgrid = make_mock_polar_grid()
    pgrid = make_mock_phys_grid(10, 10, -1) # we aren't actually using the box size
    pixel_size = pgrid.pixel_size[0]
    img = Images(
        phys_data=make_mock_data_obj(pimg, pgrid),
        fourier_data=make_mock_data_obj(fimg, fgrid)
    )
    assert img.box_size is not None
    assert img.displacement_grid_angstrom.size() == torch.Size([2, 1])
    assert_close(img.displacement_grid_angstrom, torch.tensor([[0.], [0.]]))

    max_d_pixel = 4
    x_disp = max(1, x_disp)
    y_disp = max(1, y_disp)
    disp_t = x_disp * y_disp
    max_d_angstrom = max_d_pixel * pixel_size
    if disp_t == 1:
        max_d_angstrom = 0.
    if disp_t % 2 == 0:
        disp_t += 1
    expected_x = max_d_angstrom if x_disp > 1 else 0.
    expected_y = max_d_angstrom if y_disp > 1 else 0.

    img.set_displacement_grid(max_d_pixel, x_disp, y_disp)
    assert img.n_displacements == disp_t
    assert img.displacement_grid_angstrom.size() == torch.Size([2, disp_t])
    assert torch.min(img.displacement_grid_angstrom[0]).item() == -1 * expected_x
    assert torch.min(img.displacement_grid_angstrom[1]).item() == -1 * expected_y
    assert torch.max(img.displacement_grid_angstrom[0]).item() == expected_x
    assert torch.max(img.displacement_grid_angstrom[1]).item() == expected_y

    assert img._translation_matrix is not None
    for i in range(img.n_displacements):
        if (img.displacement_grid_angstrom[0,i] < 1e-6 and
            img.displacement_grid_angstrom[1,i] < 1e-6):
            return
    # If the displacement grid does not contain an entry that's
    # ~(0.,0.), then fail the test
    assert False


def test_set_displacement_grid_throws_on_indeterminate_size():
    fimg = make_image_tensor(2, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    img = Images(
        fourier_data=make_mock_data_obj(fimg, fgrid)
    )
    assert img.box_size is not None
    assert img.displacement_grid_angstrom.size() == torch.Size([2, 1])
    assert_close(img.displacement_grid_angstrom, torch.tensor([[0.], [0.]]))

    with raises(ValueError, match="If pixel size is not provided"):
        img.set_displacement_grid(2, 3, 5)


def test_set_displacement_grid_warns_on_rectangular_box():
    fimg = make_image_tensor(2, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    img = Images(
        fourier_data=make_mock_data_obj(fimg, fgrid)
    )
    assert img.box_size is not None
    assert img.displacement_grid_angstrom.size() == torch.Size([2, 1])
    assert_close(img.displacement_grid_angstrom, torch.tensor([[0.], [0.]]))
    img.box_size = np.array([2., 2.5])
    with patch('builtins.print') as _print:
        img.set_displacement_grid(3, 3, 3, 1.)
        _print.assert_called_once()


def test_set_displacement_grid_doesnt_set_translation_when_no_polar_grid():
    pimg = make_image_tensor(2, 10, 10, target_fourier=False)
    pgrid = make_mock_phys_grid(10, 10, -1) # we aren't actually using the box size
    img = Images(
        phys_data=make_mock_data_obj(pimg, pgrid),
    )
    assert img.box_size is not None
    assert img.displacement_grid_angstrom.size() == torch.Size([2, 1])
    assert_close(img.displacement_grid_angstrom, torch.tensor([[0.], [0.]]))
    assert getattr(img, '_translation_matrix', None) is None


# Right now this is just checking that the returned tensor
# is the expected shape; actually double-checking the projection
# would require much more careful mocking
def test_project_images_over_displacements():
    x_disp = 5
    y_disp = 3
    fimg = make_image_tensor(20, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    fgrid.get_fourier_translation_kernel = Mock(
        side_effect=lambda _x, _y, bs1, bs2, p, d:\
            torch.ones((len(_x), 10, 10))
    )
    pgrid = make_mock_phys_grid(10, 10, -1) # we aren't actually using the box size
    img = Images(
        phys_data=pgrid,
        fourier_data=make_mock_data_obj(fimg, fgrid)
    )
    assert img.box_size is not None
    assert img.displacement_grid_angstrom.size() == torch.Size([2, 1])
    assert_close(img.displacement_grid_angstrom, torch.tensor([[0.], [0.]]))

    max_d_pixel = 4
    disp_t = x_disp * y_disp

    img.set_displacement_grid(max_d_pixel, x_disp, y_disp)
    assert img.n_displacements == disp_t
    assert img.images_fourier is not None
    assert isinstance(img.images_fourier, torch.Tensor)

    foo = img.project_images_over_displacements(0, 4)
    assert foo.device == img.images_fourier.device
    assert foo.shape == torch.Size([4, disp_t, 10, 10])


def test_project_images_over_displacements_throws_on_no_matrix():
    pimg = make_image_tensor(2, 10, 10, target_fourier=False)
    pgrid = make_mock_phys_grid(10, 10, -1) # we aren't actually using the box size
    img = Images(
        phys_data=make_mock_data_obj(pimg, pgrid),
    )
    assert img.box_size is not None
    assert img.displacement_grid_angstrom.size() == torch.Size([2, 1])
    assert_close(img.displacement_grid_angstrom, torch.tensor([[0.], [0.]]))
    assert getattr(img, '_translation_matrix', None) is None

    with raises(ValueError, match="Translation kernel was never set"):
        _ = img.project_images_over_displacements(0, 4)


def test_displace_fourier_images_scalar_displacement():
    def inv_kernel(t: torch.Tensor):
        s = t.shape[-1]
        i = torch.eye(s, dtype=t.dtype) * -1.
        return i

    n_imgs = 5
    fimgs = make_image_tensor(n_imgs, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    my_inv = inv_kernel(fimgs)
    mock_tk = Mock(return_value = my_inv)
    fgrid.get_fourier_translation_kernel = mock_tk
    sut = Images(fourier_data=make_mock_data_obj(fimgs, fgrid))

    x_disp = 4.
    y_disp = 6.
    sut.displace_fourier_images(x_disp, y_disp)
    assert_close(sut.images_fourier, fimgs * my_inv)
    calls = mock_tk.call_args[0]
    assert_close(calls[0], torch.tensor([x_disp], dtype=torch.float64))
    assert_close(calls[1], torch.tensor([y_disp], dtype=torch.float64))
    assert calls[2] == 2.0
    assert calls[3] == 2.0
    assert calls[4] == Precision.DOUBLE
    assert calls[5] == sut.images_fourier.device


def test_displace_fourier_images_summed_displacements():
    def inv_kernel(t: torch.Tensor):
        s = t.shape[-1]
        i = torch.eye(s, dtype=t.dtype) * -1.
        return i

    n_imgs = 5
    fimgs = make_image_tensor(n_imgs, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    my_inv = inv_kernel(fimgs)
    mock_tk = Mock(return_value = my_inv)
    fgrid.get_fourier_translation_kernel = mock_tk
    sut = Images(fourier_data=make_mock_data_obj(fimgs, fgrid))

    sut = Images(fourier_data=make_mock_data_obj(fimgs, fgrid))
    x_disp = torch.tensor([4., 6.], dtype=torch.float64)
    y_disp = torch.tensor([6., 3.], dtype=torch.float64)
    sut.displace_fourier_images(x_disp, y_disp)
    assert_close(sut.images_fourier, fimgs * my_inv)
    calls = mock_tk.call_args[0]
    assert_close(calls[0], torch.sum(x_disp).unsqueeze(0))
    assert_close(calls[1], torch.sum(y_disp).unsqueeze(0))


def test_displace_fourier_images_displacement_per_image():
    def inv_kernel(t: torch.Tensor):
        s = t.shape[-1]
        i = torch.eye(s, dtype=t.dtype) * -1.
        return i

    n_imgs = 5
    fimgs = make_image_tensor(n_imgs, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    my_inv = inv_kernel(fimgs)
    mock_tk = Mock(return_value = my_inv)
    fgrid.get_fourier_translation_kernel = mock_tk
    sut = Images(fourier_data=make_mock_data_obj(fimgs.to(dtype=torch.complex64), fgrid),
                 box_size=np.array([4., 4.])
                )
    x_disp3 = np.array([2., 4., 6., 8., 10.])
    y_disp3 = np.array([10., 8., 6., 4., 2.])
    sut.displace_fourier_images(x_disp3, y_disp3, displacement_per_image=True)
    assert_close(sut.images_fourier, fimgs * my_inv)
    calls = mock_tk.call_args[0]
    assert_close(calls[0], torch.tensor([2., 4., 6., 8., 10.], dtype=torch.float32))
    assert_close(calls[1], torch.tensor([10., 8., 6., 4., 2.], dtype=torch.float32))
    assert calls[4] == Precision.SINGLE


def test_displace_fourier_images_throws_on_bad_displacement_lengths():
    n_imgs = 5
    fimgs = make_image_tensor(n_imgs, 6, 6, target_fourier=True)
    fgrid = make_mock_polar_grid()
    img = Images(fourier_data=make_mock_data_obj(fimgs, fgrid))

    with raises(ValueError, match="must provide one displacement per image"):
        img.displace_fourier_images(x_displacements=np.array([1., 2.]), y_displacements=1., displacement_per_image=True)


def test_normalize_images_phys():
    n_imgs = 5
    cimg = make_image_tensor(n_imgs, 25, 25)
    cgrid = make_mock_phys_grid(25, 25, 1.)

    sut = Images(phys_data=make_mock_data_obj(cimg.clone(), cgrid))
    norm = sut.normalize_images_phys(1, use_max=True)
    assert torch.all(sut.images_phys >= 0.)
    assert torch.all(sut.images_phys <= 1.)
    assert_close(norm,
                 torch.tensor([624., 1249., 1874., 2499., 3124.],
                              dtype=torch.float64
                            ).unsqueeze(1).unsqueeze(2))
    assert_close(sut.images_phys * norm, cimg)

    sut = Images(phys_data=make_mock_data_obj(cimg.clone(), cgrid))
    lp_norm = sut.normalize_images_phys(2)
    assert_close(lp_norm, torch.norm(cimg, dim=(1,2), p=2))
    assert_close(sut.images_phys * lp_norm[:,None,None], cimg)


def test_normalize_images_fourier_maxval():
    n_imgs = 5
    fimg = make_image_tensor(n_imgs, 10, 10).to(dtype=torch.complex128)
    fgrid = make_mock_polar_grid(10, 10)
    fgrid.integrate = Mock(side_effect=lambda x: torch.sum(x, dim = -1) * (2 * np.pi) ** 2)

    sut = Images(fourier_data=make_mock_data_obj(fimg.clone(), fgrid))
    norm = sut.normalize_images_fourier(1, use_max=True)
    assert_close(norm, torch.tensor([99., 199., 299., 399., 499.], dtype=torch.float64).reshape([5, 1, 1]))
    assert torch.all(sut.images_fourier.real >= 0.)
    assert torch.all(sut.images_fourier.real <= 1.)
    assert torch.all(sut.images_fourier.imag == 0.)
    assert_close(sut.images_fourier * norm, fimg)


def test_normalize_images_fourier_lpnorm():
    n_imgs = 5
    ord = 2
    fimg = make_image_tensor(n_imgs, 10, 10).to(dtype=torch.complex128)
    fgrid = make_mock_polar_grid(10, 10)
    # fgrid.integrate = Mock(side_effect=lambda x: torch.sum(x, dim = -1) * (2 * np.pi) ** 2)

    sut = Images(fourier_data=make_mock_data_obj(fimg.clone(), fgrid))
    norm = sut.normalize_images_fourier(ord, use_max=False)
    norm_new = fgrid.integrate(sut.images_fourier.abs().pow(ord)).pow(1.0 / ord)
    assert_close(norm_new, torch.ones_like(norm_new))


def test_make_rotation_tensor():
    n_imgs = 5
    fimg = make_image_tensor(n_imgs, 10, 10, target_fourier=True)
    fgrid = make_mock_polar_grid()
    expected = torch.ones(n_imgs, dtype=torch.float64) * 3.
    sut = Images(fourier_data=make_mock_data_obj(fimg, fgrid))
    
    # test with scalar
    res = sut._make_rotation_tensor(3.)
    assert_close(res, expected)

    # test with numpy
    res = sut._make_rotation_tensor(np.array([3.] * n_imgs))
    assert_close(res, expected)

    # test with tensor
    res = sut._make_rotation_tensor(torch.ones(n_imgs) * 3.)
    assert_close(res, expected)

    # test with one-element array
    res = sut._make_rotation_tensor(torch.tensor([3.]))
    assert_close(res, expected)

    # not 1-d
    with raises(ValueError, match="must be a 1D array"):
        sut._make_rotation_tensor(torch.ones(10).reshape([5, 2]))

    # wrong number of rotations
    with raises(ValueError, match="Number of rotations must be equal"):
        sut._make_rotation_tensor(torch.ones(n_imgs + 1))


def test_rotate_images_fourier_discrete():
    n_imgs = 5
    n_inplanes = 10
    fimg = make_image_tensor(n_imgs, 10, n_inplanes).to(dtype=torch.complex128)
    fgrid = make_mock_polar_grid(10, n_inplanes)
    rot = torch.arange(n_imgs, dtype=torch.float64) * (-2.0 * np.pi / n_inplanes)
    
    sut = Images(fourier_data=make_mock_data_obj(fimg.clone(), fgrid))
    sut.rotate_images_fourier_discrete(rot)
    for i in range(n_imgs):
        assert_close(sut.images_fourier[i], torch.roll(fimg[i], i, dims=-1))


def test_filter_padded_images():
    n_imgs = 5
    cgrid = make_mock_phys_grid(5, 5, 1.)
    cimg = make_image_tensor(n_imgs, 5, 5) # uses arange: each element s.b. distinct
    # The function-under-test detects padding as any cases where the same image
    # has the same pixel values in the first 2 or last 2 rows or columns.
    # We'll test each of those cases for images 0, 1, 3, and 4, and then
    # expect the filter operation to return an array with a single image (the
    # original image 2).

    cimg[0, 1, :] = cimg[0, 0, :]
    cimg[1, :, 1] = cimg[1, :, 0]
    cimg[3,-1, :] = cimg[3,-2, :]
    cimg[4, :,-2] = cimg[4, :,-1]
    sut = Images(phys_data=make_mock_data_obj(cimg, cgrid))

    not_padded = sut.filter_padded_images(rtol=1e-3)
    assert isinstance(not_padded, np.ndarray)
    npt.assert_allclose(not_padded, np.array([False, False, True, False, False]))
    assert sut.n_images == 1
    assert_close(sut.images_phys[0], cimg[2])


@patch(f"{PKG}.np.allclose")
def test_filter_padded_images_is_no_op_if_no_phys_imgs(mock_allclose: Mock):
    sut = Images(fourier_data=make_mock_data_obj(
        img=make_image_tensor(5, 10, 10, target_fourier=True),
        grid=make_mock_polar_grid()
    ))
    sut.filter_padded_images()
    mock_allclose.assert_not_called()


def test_get_power_spectrum():
    n_imgs = 5
    n_shells = 10
    fimg = make_image_tensor(n_imgs, n_shells, 10).to(dtype=torch.complex128)
    fgrid = make_mock_polar_grid(n_shells=n_shells)
    fgrid.radius_shells = np.arange(n_shells) + 1.
    expected_resolutions = 1. / fgrid.radius_shells
    expected_powers = np.array([61828.5, 66018.5, 70408.5,  74998.5,  79788.5,
                                84778.5, 89968.5, 95358.5, 100948.5, 106738.5])
    sut = Images(fourier_data=make_mock_data_obj(fimg, fgrid))
    (spectrum, res) = sut.get_power_spectrum()
    npt.assert_allclose(spectrum, expected_powers)
    npt.assert_allclose(res, expected_resolutions)


def test_get_power_spectrum_throws_on_no_fourier_imgs():
    phys_img = Images(make_mock_data_obj(
        make_image_tensor(5, 5, 5),
        make_mock_phys_grid(5, 5, 1.)
    ))
    with raises(ValueError, match="Fourier images not found"):
        phys_img.get_power_spectrum()


def test_select_images():
    n_imgs = 5
    cimg = make_image_tensor(n_imgs, 10, 10)
    fimg = cimg.to(dtype=torch.complex128)
    cgrid = make_mock_phys_grid(10, 10, 1.)
    fgrid = make_mock_polar_grid()
    sut = Images(phys_data=make_mock_data_obj(cimg, cgrid),
                 fourier_data=make_mock_data_obj(fimg, fgrid))
    indices = [0, 2, 4]
    sut.select_images(indices)
    
    assert sut.n_images == 3
    assert_close(sut.images_phys[0], cimg[0])
    assert_close(sut.images_phys[1], cimg[2])
    assert_close(sut.images_phys[2], cimg[4])
    assert_close(sut.images_fourier[0], fimg[0])
    assert_close(sut.images_fourier[1], fimg[2])
    assert_close(sut.images_fourier[2], fimg[4])


def test_downsample_images_phys():
    n_imgs = 3
    cimg = make_image_tensor(n_imgs, 10, 10)
    cgrid = make_mock_phys_grid(10, 10, 1.)

    # mean-pool
    sut = Images(phys_data=make_mock_data_obj(cimg.clone(), cgrid))
    # pixel count went from [10, 10] to [5, 5] but pixel_size went from
    # [10, 10] to [20, 20]
    expected_box = np.array([100., 100.])
    res = sut.downsample_images_phys(2, 'mean')
    assert_close(res, sut.images_phys)
    assert sut.images_phys.shape[0] == cimg.shape[0]
    assert sut.images_phys.shape[1] == cimg.shape[1] / 2
    assert sut.images_phys.shape[2] == cimg.shape[2] / 2
    npt.assert_allclose(sut.phys_grid.pixel_size, cgrid.pixel_size * 2.)
    npt.assert_allclose(sut.phys_grid.n_pixels, cgrid.n_pixels / 2)
    assert_close(sut.images_phys, torch.nn.functional.avg_pool2d(cimg, 2, 2))
    npt.assert_allclose(sut.box_size, expected_box)

    # max-pool
    sut = Images(phys_data=make_mock_data_obj(cimg.clone(), cgrid))
    # new pixel count is [2, 2] and new size is [40., 40.]
    expected_box = np.array([80., 80.])
    res = sut.downsample_images_phys(4, 'max')
    assert_close(res, sut.images_phys)
    assert sut.images_phys.shape[0] == cimg.shape[0]
    assert sut.images_phys.shape[1] == cimg.shape[1] // 4
    assert sut.images_phys.shape[2] == cimg.shape[2] // 4
    npt.assert_allclose(sut.phys_grid.pixel_size, cgrid.pixel_size * 4.)
    npt.assert_allclose(sut.phys_grid.n_pixels, cgrid.n_pixels // 4)
    assert_close(sut.images_phys, torch.nn.functional.max_pool2d(cimg, 4, 4))
    npt.assert_allclose(sut.box_size, expected_box)


@patch(f"{PKG}.torch.nn.functional")
def test_downsample_images_phys_is_no_op_with_factor_1(mock_torch: Mock):
    sut = Images(phys_data=make_mock_data_obj(make_image_tensor(2, 2, 2), make_mock_phys_grid(2, 2, 1.)))
    mock_torch.avg_pool2d = Mock()
    mock_torch.max_pool2d = Mock()
    res = sut.downsample_images_phys(1)
    assert res is None
    mock_torch.avg_pool2d.assert_not_called()
    mock_torch.max_pool2d.assert_not_called()


def test_downsample_images_phys_throws_on_bad_factor():
    sut = Images(phys_data=make_mock_data_obj(make_image_tensor(2, 2, 2), make_mock_phys_grid(2, 2, 1.)))
    with raises(ValueError, match="multiple of 2"):
        sut.downsample_images_phys(3)
    with raises(ValueError, match="must be positive"):
        sut.downsample_images_phys(-2)


def test_downsample_images_phys_throws_on_no_phys_imgs():
    sut = Images(fourier_data=make_mock_data_obj(
        make_image_tensor(3, 3, 3, target_fourier=True),
        make_mock_polar_grid()
    ))
    with raises(ValueError, match="Physical images not found"):
        sut.downsample_images_phys(2)
