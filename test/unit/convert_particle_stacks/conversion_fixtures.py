from unittest.mock import Mock, patch
from typing import Literal
import torch
import numpy as np
from os import path

from cryolike.metadata import ImageDescriptor
from cryolike.util import Precision
from cryolike.metadata.lens_descriptor import LensDescriptor
from cryolike.convert_particle_stacks.particle_stacks_pathing import OutputFilenames
from cryolike.convert_particle_stacks.particle_stacks_converter import (
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

FIX_OUTPUT_DIR = "my/output/"


def get_base_converter(**kwargs) -> ParticleStackConverter:
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
    # This is a somewhat hacky bit of mocking: we need to patch the OutputDirs ctor
    # in creating the ParticleStackConverter, in order to avoid actually creating
    # a bunch of directories.
    # One option would be to use the built-in temp directory fixtures, but then
    # that would need to be added to the parameter list and passed in to this function
    # from every test that uses this function. Bleh.
    # So instead I've opted to create a mock of the OutputFolders object and copy
    # over the definitions from the real one, and make sure that the folder stubs
    # are set up the way they are in the actual implementation.
    # There is probably a more elegant way to do this, and we can expect it to start
    # failing once we get around to centralizing our directory structure stuff.
    # But for now, the test pass without touching the filesystem, so I'm calling it.
    def get_output_fns(mock, i_stack):
        phys_stack = path.join(mock.folder_output_particles_phys, f"particles_phys_stack_{i_stack:06}.pt")
        fourier_base = path.join(mock.folder_output_particles_fft, f"particles_fourier_stack_{i_stack:06}")
        fourier_stack = fourier_base + ".pt"
        params_fn = fourier_base + ".npz"
        return OutputFilenames(phys_stack, fourier_stack, params_fn)
    
    with patch("cryolike.convert_particle_stacks.particle_stacks_converter.OutputFolders") as mock:
        mock.folder_output_plots = path.join(FIX_OUTPUT_DIR, 'plots')
        mock.folder_output_particles_fft = path.join(FIX_OUTPUT_DIR, "fft")
        mock.folder_output_particles_phys = path.join(FIX_OUTPUT_DIR, "phys")
        mock.get_output_filenames = lambda x, y: get_output_fns(x, y)

        return ParticleStackConverter(FIX_IMG_DESC, FIX_OUTPUT_DIR, **kwargs)


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
        return (type, rec)
    elif type == 'sequential_cryosparc':
        rec = SequentialCryosparc(mrc_file=f"{base_data}.mrc")
        return (type, rec)
    elif type == 'indexed_starfile':
        rec = Indexed(
            mrc_file=f"{base_data}.mrc",
            selected_img_indices=np.arange(selection_count) + 16,
            selected_lensdesc_indices=np.arange(selection_count)
        )
        return (type, rec)


def configure_mock_OutputFolders(OutputFolders: Mock) -> Mock:
    """Given a mock object used to patch the OutputFolders object, this
    configures that object to return the expected filenames. The purpose
    is to avoid modifying the actual filesystem, although that has been
    largely superseded by the get_base_converter function.

    Args:
        OutputFolders (Mock): Stand-in for the OutputFolders object,
            for stubbing out the actual object, which modifies the
            filesystem during its construction.

    Returns:
        Mock: The input OutputFolders mock, configured for use in testing.
    """
    output_filenames = Mock()
    output_filenames.phys_stack = "phys_stack"
    output_filenames.fourier_stack = "fourier_stack"
    output_filenames.params_filename = "params_filename"

    OutputFolders.folder_output_plots = "output_plots"
    OutputFolders.folder_output_particles_fft = "output_fft"
    OutputFolders.folder_output_particles_phys = "output_phys"
    OutputFolders.get_output_filenames = Mock(side_effect=lambda x: output_filenames)

    return OutputFolders
