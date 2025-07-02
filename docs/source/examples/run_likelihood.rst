Run likelihood
=================

TODO: LINK TO MORE EXTENSIVE DOCUMENTATION?
(We should already have a link to the function itself.)

This script demonstrates the usage of the
:py:func:`run_likelihood() <cryolike.run_likelihood.run_likelihood>`
function from the :py:mod:`cryolike.run_likelihood` module.

The :py:func:`run_likelihood() <cryolike.run_likelihood.run_likelihood>`
function computes the likelihood of observing given cryo-EM images,
assuming a given underlying 3D structure. The 3D structure is
represented as a set of 2D template images, observed from different
viewing angles.

The following parameters are required:

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
- `return_likelihood_integrated_pose_fourier` (bool): Whether to return the
  likelihood integrated pose Fourier or not.
- `return_likelihood_optimal_pose_physical` (bool): Whether to return the
  likelihood optimal pose physical or not.
- `return_likelihood_optimal_pose_fourier` (bool): Whether to return the
  likelihood optimal pose Fourier or not.
- `verbose` (bool): Whether to print verbose output or not.

Eaxmple usage:

.. literalinclude:: ../../../example/run_likelihood_example.py
    :language: python
