from pytest import mark

import numpy as np
import numpy.testing as npt

from cryolike.metadata import ImageDescriptor, LensDescriptor
from cryolike.metadata.file_mgmt import (
    save_combined_params,
    load_combined_params
)


lens_defaults = {
    'defocusU': np.array([1., 2., 3.]),
    'defocusV': np.array([3., 4., 5.]),
    'defocusAngle': np.array([90., 180., 135.]),
    'phaseShift': np.array([45., 135., 60.]),
    'sphericalAberration': 3.2,
    'voltage': 320.,
    'amplitudeContrast': 0.5,
    'angleRotation': np.array([0., .5, .6]),
    'angleTilt': np.array([0., .1, .2]),
    'anglePsi': np.array([0., .3, .6]),
    'ref_pixel_size': np.array([2.5, 2.5]),
    'files': np.array(['file 1', 'file 2']),
    'idxs': np.array([[1, 4, 5], [2, 3, 7]]),
    'ctfBfactor': 12.,
    'ctfScalefactor': np.array([.5])
}

img_complete = {
    'precision': 'single',
    'n_pixels': 4,
    'pixel_size': 2.,
    'n_inplanes': 12,
    'atom_radii': 1.1,
    'atom_selection': "my selection",
    'use_protein_residue_model': False,
    'atom_shape': "hard-sphere"
}

img_with_nones = {
    'precision': 'single',
    'n_pixels': 4,
    'pixel_size': 2.,
    'n_inplanes': 12,
}


def _assert_img_desc_equality(a: ImageDescriptor, b: ImageDescriptor):
    assert a.precision == b.precision
    npt.assert_array_equal(a.cartesian_grid.n_pixels, b.cartesian_grid.n_pixels)
    npt.assert_array_equal(a.cartesian_grid.pixel_size, b.cartesian_grid.pixel_size)
    assert a.polar_grid.dist_radii == b.polar_grid.dist_radii
    assert a.polar_grid.radius_max == b.polar_grid.radius_max
    assert a.polar_grid.n_inplanes == b.polar_grid.n_inplanes
    assert a.viewing_distance == b.viewing_distance
    if a.atom_radii is None:
        assert b.atom_radii is None
    else:
        assert b.atom_radii == a.atom_radii
    if a.atom_selection is None:
        assert b.atom_selection is None
    else:
        assert b.atom_selection == a.atom_selection
    assert a.use_protein_residue_model == b.use_protein_residue_model
    assert a.atom_shape == b.atom_shape


def _assert_lens_desc_equality(a: LensDescriptor, b: LensDescriptor):
    for x in lens_defaults.keys():
        orig = getattr(a, x, None)
        res = getattr(b, x, None)
        if orig is None:
            assert res is None
            continue
        assert res is not None
        if isinstance(orig, np.ndarray):
            npt.assert_array_equal(res, orig)
            continue
        assert res == orig


@mark.parametrize("use_nones",[True, False])
def test_load_combined_params_rt(tmp_path, use_nones):
    fn = tmp_path / "my_file.npz"
    if use_nones:
        lens_desc = LensDescriptor()
        img_desc = ImageDescriptor.from_individual_values(**img_with_nones)
    else:
        lens_desc = LensDescriptor(**lens_defaults, defocusAngle_degree=False, phaseShift_degree=False)
        img_desc = ImageDescriptor.from_individual_values(**img_complete)
    
    n_imgs_this_stack = 100

    save_combined_params(fn, img_desc, lens_desc, n_imgs_this_stack)
    (rt_img, rt_lens) = load_combined_params(fn)

    # assert
    _assert_img_desc_equality(img_desc, rt_img)
    _assert_lens_desc_equality(lens_desc, rt_lens)        
