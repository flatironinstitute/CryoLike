from unittest.mock import patch, Mock
from pytest import raises
import numpy as np
import numpy.testing as npt

from cryolike.util import Precision, AtomShape
from cryolike.grids import CartesianGrid2D, PolarGrid
from cryolike.metadata.image_descriptor import ImageDescriptor

PKG = "cryolike.metadata.image_descriptor"


@patch(f"{PKG}.ViewingAngles")
def test_ctor(va: Mock):
    prec = Precision.SINGLE
    cgrid = Mock()
    pgrid = Mock()
    angles = Mock()
    radii = None
    selection = "my selection"
    shape = AtomShape.HARD_SPHERE

    res = ImageDescriptor(
        precision=prec,
        cartesian_grid=cgrid,
        polar_grid=pgrid,
        viewing_angles=angles,
        viewing_distance=None,
        atom_radii=radii,
        atom_selection=selection,
        use_protein_residue_model=True,
        atom_shape=shape
    )
    assert res.precision == prec
    assert res.cartesian_grid == cgrid
    assert res.polar_grid == pgrid
    assert res.viewing_angles == angles
    assert res.viewing_distance is None
    assert res.atom_radii is None
    assert res.atom_selection == selection
    assert res.use_protein_residue_model
    assert res.atom_shape == shape

    distance = 10.
    res2 = ImageDescriptor(
        precision=prec,
        cartesian_grid=cgrid,
        polar_grid=pgrid,
        viewing_angles=None,
        viewing_distance=distance,
        atom_radii=radii,
        atom_selection=selection,
        use_protein_residue_model=True,
        atom_shape=shape
    )
    va.from_viewing_distance.assert_called_once_with(distance)
    assert res2.viewing_angles == va.from_viewing_distance.return_value
    assert res2.viewing_distance == distance
    assert res2.viewing_angles != angles


# def test_ctor_throws_on_no_viewing_data():
#     prec = Precision.SINGLE
#     cgrid = Mock()
#     pgrid = Mock()

#     with raises(ValueError, match="viewing distance must be set"):
#         _ = ImageDescriptor(
#             precision=prec,
#             cartesian_grid=cgrid,
#             polar_grid=pgrid,
#             viewing_angles=None,
#             viewing_distance=None
#         )


def test_get_3d_box_size():
    cgrid = Mock()
    pgrid = Mock()
    cgrid.box_size = [1., 1.]
    sut = ImageDescriptor(
        precision=Precision.SINGLE,
        cartesian_grid=cgrid,
        polar_grid=pgrid,
        viewing_angles=None,
        viewing_distance=10.
    )

    res = sut.get_3d_box_size()
    npt.assert_allclose(res, np.array([1., 1., 1.]))

    cgrid.box_size = [1., 3.]
    with raises(NotImplementedError, match="Need 3rd dimension"):
        _ = sut.get_3d_box_size()


def test_is_compatible_with_imagestack():
    prec = Precision.SINGLE
    pgrid_1 = PolarGrid(radius_max=2., dist_radii=5., n_inplanes=10)
    pgrid_2 = PolarGrid(radius_max=3., dist_radii=6., n_inplanes=12)

    cgrid_1 = CartesianGrid2D(15, 15.)
    cgrid_2 = CartesianGrid2D(25, 25.)

    sut = ImageDescriptor(prec, cgrid_1, pgrid_1, None, 1.)

    matches = Mock()
    matches.polar_grid = pgrid_1
    matches.phys_grid = cgrid_1

    wrong_pgrid = Mock()
    wrong_pgrid.phys_grid = cgrid_1
    wrong_pgrid.polar_grid = pgrid_2

    wrong_cgrid = Mock()
    wrong_cgrid.polar_grid = pgrid_1
    wrong_cgrid.phys_grid = cgrid_2
    
    assert sut.is_compatible_with_imagestack(matches)
    
    assert not sut.is_compatible_with_imagestack(wrong_pgrid)
    assert not sut.is_compatible_with_imagestack(wrong_cgrid)

    wrong_cgrid.phys_grid = None
    wrong_pgrid.polar_grid = None
    assert sut.is_compatible_with_imagestack(wrong_pgrid)
    assert sut.is_compatible_with_imagestack(wrong_cgrid)


def test_is_compatible_with():
    prec = Precision.SINGLE
    pgrid_1 = PolarGrid(radius_max=2., dist_radii=5., n_inplanes=10)
    pgrid_2 = PolarGrid(radius_max=3., dist_radii=6., n_inplanes=12)

    cgrid_1 = CartesianGrid2D(15, 15.)
    cgrid_2 = CartesianGrid2D(25, 25.)

    sut = ImageDescriptor(prec, cgrid_1, pgrid_1, None, 1.)

    matches = ImageDescriptor(prec, cgrid_1, pgrid_1, None, 1.)
    wrong_pgrid = ImageDescriptor(prec, cgrid_1, pgrid_2, None, 1.)
    wrong_cgrid = ImageDescriptor(prec, cgrid_2, pgrid_1, None, 1.)

    assert sut.is_compatible_with(matches)
    assert not sut.is_compatible_with(wrong_pgrid)
    assert not sut.is_compatible_with(wrong_cgrid)
    wrong_pgrid.polar_grid = None # type: ignore
    assert sut.is_compatible_with(wrong_pgrid)
    wrong_cgrid.cartesian_grid = None # type: ignore
    assert sut.is_compatible_with(wrong_cgrid)


def test_is_compatible_with_pdb():
    prec = Precision.SINGLE
    cgrid = Mock()
    pgrid = Mock()
    distance = 10.

    sut = ImageDescriptor(
            precision=prec,
            cartesian_grid=cgrid,
            polar_grid=pgrid,
            viewing_angles=None,
            viewing_distance=distance,
            use_protein_residue_model=False
        )
    assert not sut.is_compatible_with_pdb()

    sut.use_protein_residue_model = True
    assert sut.is_compatible_with_pdb()

    sut.use_protein_residue_model = False
    sut.atom_radii = -0.5
    assert not sut.is_compatible_with_pdb()

    sut.atom_radii = 0.5
    assert sut.is_compatible_with_pdb()


# def test_serialization_throws_on_no_viewing_distance():
#     sut = ImageDescriptor(
#         Precision.SINGLE, Mock(), Mock(), viewing_angles=Mock(), viewing_distance=None
#     )
#     with raises(NotImplementedError, match="no explicit viewing distance"):
#         _ = sut.serialize()


def test_serialization_roundtrip():
    prec = Precision.SINGLE
    cgrid = CartesianGrid2D(n_pixels=5, pixel_size=1.)
    pgrid = PolarGrid(radius_max=5., dist_radii=1., n_inplanes=15)
    distance = 10.
    radii = None
    selection = "my selection"
    shape = AtomShape.HARD_SPHERE

    sut = ImageDescriptor(
        precision=prec,
        cartesian_grid=cgrid,
        polar_grid=pgrid,
        viewing_angles=None,
        viewing_distance=distance,
        atom_radii=radii,
        atom_selection=selection,
        use_protein_residue_model=True,
        atom_shape=shape
    )
    roundtrip = ImageDescriptor.from_dict(sut.to_dict())

    assert roundtrip.precision == sut.precision
    assert roundtrip.viewing_distance == sut.viewing_distance
    assert roundtrip.atom_radii == sut.atom_radii
    assert roundtrip.atom_selection == sut.atom_selection
    assert roundtrip.use_protein_residue_model == sut.use_protein_residue_model
    assert roundtrip.atom_shape == sut.atom_shape

    assert roundtrip.cartesian_grid != sut.cartesian_grid
    npt.assert_allclose(roundtrip.cartesian_grid.n_pixels, sut.cartesian_grid.n_pixels)
    npt.assert_allclose(roundtrip.cartesian_grid.pixel_size, sut.cartesian_grid.pixel_size)

    assert roundtrip.polar_grid != sut.polar_grid
    assert roundtrip.polar_grid.radius_max == sut.polar_grid.radius_max
    assert roundtrip.polar_grid.dist_radii == sut.polar_grid.dist_radii
    assert roundtrip.polar_grid.n_inplanes == sut.polar_grid.n_inplanes


@patch(f"{PKG}.save_descriptors")
def test_save(save: Mock):
    distance = 10.
    sut = ImageDescriptor(
        precision=Precision.DOUBLE,
        cartesian_grid=CartesianGrid2D(n_pixels=5, pixel_size=1.),
        polar_grid=PolarGrid(radius_max=5., dist_radii=1., n_inplanes=15),
        viewing_angles=None,
        viewing_distance=distance,
    )

    as_dict = sut.to_dict()
    filename = "my/file"

    sut.save(filename)
    save.assert_called_once()
    args = save.call_args[0]
    assert args[0] == filename
    res_dict = args[1]
    for k in res_dict.keys():
        if isinstance(as_dict[k], np.ndarray):
            npt.assert_allclose(as_dict[k], res_dict[k])
        else:
            assert as_dict[k] == res_dict[k]
    assert len(res_dict.keys()) == len(as_dict.keys())


@patch("builtins.print")
def test_print(_print: Mock):
    cgrid = CartesianGrid2D(n_pixels=5, pixel_size=1.)
    pgrid = PolarGrid(radius_max=5., dist_radii=1., n_inplanes=15)
    distance = 10.
    sut = ImageDescriptor(Precision.DOUBLE, cgrid, pgrid, None, distance)
    sut.print()
    assert _print.call_count == 13


def test_from_individual_values():
    n_pixels = 5
    pixel_size = 3.
    resolution = 2.
    radii = .4

    res = ImageDescriptor.from_individual_values(
        n_pixels=n_pixels,
        pixel_size=pixel_size,
        resolution_factor=resolution,
        atom_radii=radii,
        precision='single',
        atom_shape='gaussian'
    )

    assert res.precision == Precision.SINGLE
    assert res.atom_shape == AtomShape.GAUSSIAN
    assert res.atom_radii == radii

    assert res.viewing_distance == 1.0 / (4.0 * np.pi) / resolution
    npt.assert_allclose(res.cartesian_grid.n_pixels, np.array([n_pixels, n_pixels]))
    npt.assert_allclose(res.cartesian_grid.pixel_size, np.array([pixel_size, pixel_size]))

    assert res.polar_grid.radius_max == resolution * n_pixels * .25
    assert res.polar_grid.dist_radii == np.pi / 2.0 / (2.0 * np.pi)
    assert res.polar_grid.n_inplanes == n_pixels * 2


# Currently unreachable, but save the test for when we are persisting these
# as strings instead of enum values
# @patch("builtins.print")
# def test_from_dict_exceptions(_print: Mock):
#     ref = ImageDescriptor(
#         Precision.SINGLE,
#         CartesianGrid2D(n_pixels=15, pixel_size=4.),
#         PolarGrid(radius_max=10, dist_radii=15., n_inplanes=10),
#         viewing_angles=None,
#         viewing_distance=10.
#     )
#     as_dict = ref.to_dict()
#     as_dict["atom_shape"] = "invalid"
#     as_dict["precision"] = "invalid"
#     res = ImageDescriptor.from_dict(as_dict)
#     assert _print.call_count == 2
#     assert res.precision == Precision.DEFAULT
#     assert res.atom_shape == AtomShape.DEFAULT
