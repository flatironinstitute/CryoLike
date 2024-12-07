Run likelihood
=================

.. code-block:: python

   """
   This script demonstrates the usage of the `run_likelihood` function from the `cryolike.run_likelihood` module.

    The `run_likelihood` function computes the likelihood of cryo-EM images given a cryo-EM map and atomic model. It sets various parameters for likelihood computation, including the input parameters, template folder, particle folder, output folder, template index, number of stacks, skip exist, number of templates per batch, number of images per batch, search batch size, maximum displacement in pixels, number of displacements in x, number of displacements in y, return likelihood integrated pose Fourier, return likelihood optimal pose physical, return likelihood optimal pose Fourier, and verbose.

    Parameters:
       - `params_input` (str): The input parameters file path.
       - `folder_templates` (str): The folder containing the templates.
       - `folder_particles` (str): The folder containing the particles.
       - `folder_output` (str): The output folder for saving the likelihood results.
       - `i_template` (int): The index of the template.
       - `n_stacks` (int): The number of stacks.
       - `skip_exist` (bool): Whether to skip existing files or not.
       - `n_templates_per_batch` (int): The number of templates per batch.
       - `n_images_per_batch` (int): The number of images per batch.
       - `search_batch_size` (bool): Whether to search for the batch size or not.
       - `max_displacement_pixels` (float): The maximum displacement in pixels.
       - `n_displacements_x` (int): The number of displacements in x.
       - `n_displacements_y` (int): The number of displacements in y.
       - `return_likelihood_integrated_pose_fourier` (bool): Whether to return the likelihood integrated pose Fourier or not.
       - `return_likelihood_optimal_pose_physical` (bool): Whether to return the likelihood optimal pose physical or not.
       - `return_likelihood_optimal_pose_fourier` (bool): Whether to return the likelihood optimal pose Fourier or not.
       - `verbose` (bool): Whether to print verbose output or not.

    Returns:
        None

    Example usage:

    .. code-block:: python

       from cryolike.run_likelihood import run_likelihood
       
       for i_template in range(2):
           run_likelihood(
               params_input = "./output/templates/parameters.npz",
               folder_templates = "./output/templates/",
               folder_particles = "./output/particles/",
               folder_output = "./output/likelihood/",
               i_template = i_template,
               n_stacks = 1,
               skip_exist = False,
               n_templates_per_batch = 16,
               n_images_per_batch = 128,
               search_batch_size = True,
               max_displacement_pixels = 8.0,
               n_displacements_x = 16,
               n_displacements_y = 16,
               return_likelihood_integrated_pose_fourier = True,
               return_likelihood_optimal_pose_physical = True,
               return_likelihood_optimal_pose_fourier = True,
               verbose = True
           )

    """