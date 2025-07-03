Convert Particles
=================

This script demonstrates the usage of the
:py:func:`convert_particle_stacks_from_star_files()
<cryolike.convert_particle_stacks.particle_stacks_conversion.convert_particle_stacks_from_star_files>`
function from
the :py:mod:`cryolike.convert_particle_stacks.particle_stacks_conversion`
module. Please see the linked documentation for a brief description
of the possible parameters.

The :py:func:`convert_particle_stacks_from_star_files()
<cryolike.convert_particle_stacks.particle_stacks_conversion.convert_particle_stacks_from_star_files>`
function converts particle
stacks from a list of input files, such as MRC files and STAR files. It
sets various parameters for particle stack conversion, including the
pixel size, dataset name, particle file list, STAR file list, defocus
angle, phase shift, skip exist, and flag plots.

.. Parameters:

.. - `params_input` (str): The input parameters file path.
.. - `folder_output` (str): The output folder for saving
..   the converted particle stacks.
.. - `particle_file_list` (list): A list of particle file paths.
.. - `star_file_list` (list): A list of STAR file paths.
.. - `pixel_size` (float): The pixel size for particle stack conversion.
.. - `defocus_angle_is_degree` (bool): Whether the defocus angle
..   is in degrees or not.
.. - `phase_shift_is_degree` (bool): Whether the phase shift is in degrees or not.
.. - `skip_exist` (bool): Whether to skip existing files or not.
.. - `flag_plots` (bool): Whether to plot the converted particle stacks or not.

Example usage:

.. literalinclude:: ../../../example/convert_particle_stacks_example.py
    :language: python
