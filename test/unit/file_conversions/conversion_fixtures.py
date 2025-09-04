from unittest.mock import Mock
from typing import Literal
import torch
import numpy as np
from pathlib import Path

from cryolike.metadata import ImageDescriptor
from cryolike.util import Precision
from cryolike.metadata.lens_descriptor import LensDescriptor
from cryolike.file_mgmt import make_dir
from cryolike.file_conversions.particle_stacks_converter import (
    ParticleStackConverter,
    DataSource,
    StarfileInput,
    Indexed,
    SequentialCryosparc
)

FIX_IMG_DESC = ImageDescriptor.from_individual_values(
    n_pixels = 3,
    pixel_size = 1.,
    precision = Precision.SINGLE,
    resolution_factor=1.0,
    viewing_distance=2.0,
    n_inplanes = 5,
)

FIX_OUTPUT_DIR = "YOU_SHOULD_NOT_SEE_THIS/output/"


def get_base_converter(tmp_path: Path, seed_files: list[str] = [], **kwargs) -> ParticleStackConverter:
    """Wrapper for the ParticleStackConverter constructor that prevents
    actually creating the output directories.

    This function uses the fixture image descriptor FIX_IMG_DESC and
    fixture output directory FIX_OUTPUT_DIR as its image descriptor
    and output directory. While these values are hard-coded, the function
    will pass any provided keyword arguments through to the constructor,
    for all the rest of the constructor's arguments.

    Returns:
        ParticleStackConverter: A ParticleStackConverter object ready for
            testing.
    """
    f_out = tmp_path / FIX_OUTPUT_DIR
    make_dir(f_out, '')
    for x in seed_files:
        p = f_out / x
        p.write_text("")

    return ParticleStackConverter(FIX_IMG_DESC, str(f_out), **kwargs)


def make_mock_imagestack(length: int = 10) -> Mock:
    """Makes a mock object representing an Images object. The mock
    will have physical and fourier image tensors of the requested
    length, and defines the has_physical_images and has_fourier_images
    functions to return True. It also implements

    Args:
        length (int, optional): Length of the image stacks. Defaults to 10.

    Returns:
        Mock: A mock object capable of standing in for an Images object.
    """
    imgs = Mock()
    imgs.has_physical_images = Mock(return_value=length > 0)
    imgs.has_fourier_images = Mock(return_value=length > 0)
    imgs.images_phys = torch.arange(length) + 1.
    imgs.images_fourier = torch.arange(length) + 1. + float(length)
    imgs.n_images = length
    def select_images(imgs: Mock, selections: np.ndarray):
        cnt = len(selections)
        imgs.images_phys = imgs.images_phys[selections]
        imgs.images_fourier = imgs.images_fourier[selections]
        imgs.n_images = cnt

    imgs.select_images = Mock(side_effect=lambda x: select_images(imgs, x))
    return imgs


def make_lens_descriptor(length: int = 10) -> LensDescriptor:
    """Returns a basic LensDescriptor object for testing. This is not a
    mock, but an actual instance, although with nonsense data.

    Args:
        length (int, optional): Number of elements in the variable length
            descriptor arrays (defocus U, V, Angle, and phase shift). Defaults to 10.

    Returns:
        LensDescriptor: A LensDescriptor instance suitable for testing.
    """
    _array = np.arange(length) * 1.
    desc = LensDescriptor(defocusU=_array, defocusV=_array, defocusAngle=_array, phaseShift=_array)
    return desc


def make_datasource(
    type: Literal['starfile'] | Literal['indexed_cryosparc'] | Literal['indexed_starfile'] | Literal['sequential_cryosparc'],
    base_data: str,
    selection_count: int = 3
) -> DataSource:
    """Creates a DataSource instance with stub data. Convenience function for testing
    particle stack conversion.

    Args:
        type: Type of data source record
        base_data (str): Base name for files referred to in the record
        selection_count (int, optional): How many elements to include as selected
            indices for indexed cryosparc records. Unused for other types.
            Defaults to 3.

    Returns:
        DataSource: A DataSource record that can be consumed by the ParticleStackConverter.
    """
    if type == 'starfile':
        rec = StarfileInput(
            particle_file=f"{base_data}-particle.mrc",
            star_file=f"{base_data}-starfile.star",
            defocus_is_degree=True,
            phase_shift_is_degree=True
        )
        return (type, rec)
    elif type == 'indexed_cryosparc':
        rec = Indexed(
            mrc_file=f"{base_data}.mrc",
            selected_img_indices=np.arange(selection_count),
            selected_lensdesc_indices=np.arange(selection_count)
        )
        return ('indexed', rec)
    elif type == 'sequential_cryosparc':
        rec = SequentialCryosparc(mrc_file=f"{base_data}.mrc")
        return (type, rec)
    elif type == 'indexed_starfile':
        rec = Indexed(
            mrc_file=f"{base_data}.mrc",
            selected_img_indices=np.arange(selection_count) + 16,
            selected_lensdesc_indices=np.arange(selection_count)
        )
        return ('indexed', rec)
