Convert Particles
=================

.. code-block:: python

   """
   This script demonstrates the usage of the `convert_particle_stacks_from_star_files` function from the `cryolike.convert_particle_stacks.particle_stacks_conversion` module.
   
    The `convert_particle_stacks_from_star_files` function converts particle stacks from a list of input files, such as MRC files and STAR files. It sets various parameters for particle stack conversion, including the pixel size, dataset name, particle file list, STAR file list, defocus angle, phase shift, skip exist, and flag plots.

    Parameters:
       - `params_input` (str): The input parameters file path.
       - `folder_output` (str): The output folder for saving the converted particle stacks.
       - `particle_file_list` (list): A list of particle file paths.
       - `star_file_list` (list): A list of STAR file paths.
       - `pixel_size` (float): The pixel size for particle stack conversion.
       - `defocus_angle_is_degree` (bool): Whether the defocus angle is in degrees or not.
       - `phase_shift_is_degree` (bool): Whether the phase shift is in degrees or not.
       - `skip_exist` (bool): Whether to skip existing files or not.
       - `flag_plots` (bool): Whether to plot the converted particle stacks or not.

    Returns:
        None

    Example usage:

    .. code-block:: python

       import numpy as np
       from cryolike.convert_particle_stacks.particle_stacks_conversion import convert_particle_stacks_from_star_files
       
       pixel_size = 1.346
       dataset_name = "apoferritin"
       particle_file_list = ["./data/particles/particles.mrcs"]
       star_file = ["./data/particles/particle_data.star"]
       
       convert_particle_stacks_from_star_files(
           params_input = "./output/templates/parameters.npz",
           folder_output = "./output/particles/",
           particle_file_list = particle_file_list,
           star_file_list = star_file,
           pixel_size = pixel_size,
           defocus_angle_is_degree = True,
           phase_shift_is_degree = True,
           skip_exist = False,
           flag_plots = True
       )
    """
