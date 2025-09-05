Image-to-Structure Likelihood Computation
##########################################

The main output of CryoLike is the likelihood of between each input ``Images``
and ``Templates`` created from 3D structures.

Overview
==========

At its heart, CryoLike offers a way to compute the likelihood of a given
observed 2D image to a particular 3D structure.
As described in the
:doc:`Mathematical Framework</about>`,
likelihood comparisons are based on comparing a stack of
images with a templates set using the cross-correlation.

:doc:`Templates sets</concepts/templateCreation>`
are projections of a single 3D structure into image
space from multiple viewing angles.
Images will be compared against these templates
at a number of different rotations and displacements, and the results
can be returned with several different means of aggregation.

.. admonition:: Note:

    The Templates and Images stacks are unlikely to fit fully in GPU
    memory all at once, so CryoLike batches the comparison over several sets.
    To reduce memory transfer overhead, we preference Templates as the outer
    set of objects to loop over. We may provide more customization options
    for this in the future.

Main outputs
************

The primary outputs of CryoLike are the *best cross-correlation*
for each image across every template set (each corresponding to a
3D structure), and the *integrated likelihood* for each image with
respect to each 3D structure.

Interface
==============

The ``run_likelihood`` module provides two wrapper functions that
serve as a convenient interface to
the underlying *iterator* and *aggregator* functions found in
:py:mod:`cryolike.likelihoods`. One wrapper returns the optimal pose
for each image
(:py:func:`cryolike.run_likelihood.run_likelihood_optimal_pose`),
and the other returns the full unaggregated cross correlation likelihood,
indexed by image, template, displacement, and inplane rotation
(:py:func:`cryolike.run_likelihood.run_likelihood_full_cross_correlation`).

For a worked example of this wrapper function in action, see the
:doc:`run likelihood example</examples/run_likelihood>`.

Both wrapper functions take the following parameters:

 - A configured file manager that handles fetching input files and writing
   output files to standard locations on the file system
 - A set of :doc:`image descriptor parameters</concepts/imageSettings>`, in
   on-disk or in-memory form (``params_input``)
 - A callback function that applies the appropriate displacement-search grid
   to every batch of templates
 - The index of the template file to process (``i_template``)
 - The number of image stacks to process (``n_stacks``)
 - Whether to skip processing when the output files appear to exist
   already (``skip_exist``)
 - Number of templates and images to use per batch, and whether to
   attempt to determine
   those values automatically (``n_templates_per_batch``,
   ``n_images_per_batch``, ``estimate_batch_size``)

The file manager is provided by the
:py:func:`cryolike.run_likelihood.configure_likelihood_files` function, and
the displacer is provided by the
:py:func:`cryolike.run_likelihood.configure_displacement` function. See the
:doc:`run likelihood example</examples/run_likelihood>` for example uses, and
the :doc:`/concepts/file_structure` documentation for more details about
expected file locations.


Input system
***************

We compute likelihood by matching images against templates.
We expect the templates to be located under the directory
specified by ``folder_templates`` and the images to be located
under the directory specified by ``folder_particles`` as passed to the
``configure_likelihood_files()`` function. Specifically:

 - There must be a "template file list"
   ``folder_templates/template_file_list.npy`` in the
   ``folder_templates`` directory which lists the available template stacks

   - The ``i_template`` parameter determines which of the template files
     in the template file list will be used

 - Templates themselves can be placed anywhere, provided the template
   file list has paths to them
 - Image stacks should be in
   ``folder_particles/fft/particles_fourier_stack_NUMBER.pt``

   - ``NUMBER`` here is a six-digit 0-padded increment starting from 0
   - Every image file should have a correspondingly-named metadata file
     with an ``.npz`` extension

It is anticipated that users may wish to run these comparisons in parallel,
especially when a cluster environment is available; hence the need for
the ``i_template`` parameter.


Displacement handling
***********************

The user specifies the displacement values to check using the
``n_displacements_x``, ``n_displacements_y``, and
``max_displacement_pixels`` parameters to the
:py:func:`cryolike.run_likelihood.configure_displacement` function,
which provides a callback that should be passed to the ``run_likelihood``
wrapper.

To compute the available displacements, the
``max_displacement_pixels`` is first
converted to Angstrom using the pixel size associated with
the image/template grids. The
resulting ``max_displacement`` is treated as a potential
displacement in either direction,
creating a total displacement length of ``2 * max_displacement`` in
both dimensions.
This distance is then
divided linearly into ``n_displacements_x`` and ``n_displacements_y``
steps, resulting in
a grid of displacement positions to test during cross-correlation
computation.

The set of displacements tested will be preserved in
``folder_output/displacements_set.pt``.

Possible outputs
=========================

CryoLike can return the computed values at the following levels of
aggregation. Note that the ``run_likelihood`` wrappers currently
only support computing optimal pose or providing the fully
unaggregated data, but other aggregation types are available in the
``cryolike.likelihoods.interface`` module (just swap out the
``compute_optimal_pose`` call for one of the other functions).


Output paths
**************

The wrapper functions write computed likelihoods to disk for
later review. The exact files written depend on which wrapper function
is called.

The root output directory is specified by the ``folder_output`` parameter.
Within that directory, the following paths will be used. Note that the
directories will be created if they do not exist.

In the case of a name collision between an output file and an existing
file, the existing file will be *overwritten* unless the ``skip_exist``
parameter is set *and* the complete set of output files are present.

For the following examples, assume ``folder_output`` is set to
``OUT``. ``N`` is the template number (the
value of ``i_template``), NOT zero-padded.
``STACK`` is the 6-digit 0-padded number, starting from 0, of the stack being
processed.

 - In all cases:

    - The actual set of displacement values used will be written to
      ``OUT/displacements_set.pt``

 - ``run_likelihood_optimal_pose()``: Will write the 5 Tensors
   :ref:`discussed above<optimal_pose>` to individual files:

     - ``OUT/templateN/cross_correlation/cross_correlation_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_template_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_displacement_x_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_displacement_y_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_inplane_rotation_stack_STACK.pt``

 - ``run_likelihood_full_cross_correlation()`` will, by contrast,
   write only a single file per image stack, to
   ``OUT/templateN/cross_correlation/cross_correlation_pose_msdw_stack_STACK.pt``


.. _integrated_likelihood:

Integrated Log-Likelihood
******************************

**TODO: this seems inadequate, & also doesn't distinguish between**
**ILL and cross-correlation likelihood**
The integrated likelihood is calculated by comparing
each image to each template in the Fourier-Bessel
representation using the cross-correlation
as described in the :doc:`Mathematical Framework</about>`.


Cross-correlation
******************************

.. _optimal_pose:

Optimal pose outputs
------------------------

This will return 5 1-dimensional Tensors, indexed by the image sequence index:

 - Best cross-correlation value for each image
   (``cross_correlation_M``).
   As described in the :doc:`Mathematical Framework</about>`,
   CryoLike calculates the cross-correlation between each image
   and each template. This tensor reports the numeric value of the
   best match achieved.
 - The template (by sequence number) of the best match
   (``optimal_template_M``), i.e. the template that produced
   the number in the corresponding index of ``cross_correlation_M``
 - The optimal x-displacement matching this image with the best-fitting
   template (``optimal_displacement_x_M``)
 - The optimal y-displacement matching this image with the best-fitting
   template (``optimal_displacement_y_M``)
 - The optimal inplane rotation matching this image with the best-fitting
   template (``optimal_inplane_rotation_M``)

.. admonition:: Example:

    So consider the values at index ``i``, which correspond to the image at index ``i`` in the
    input Images stack. Then:

    - ``cross_correlation_M[i]`` is the best alignment likelihood
    - ``optimal_template_M[i]`` is the index of the template that got the score above
    - ``optimal_displacement_x_M[i]`` and ``..._y_M[i]`` are the displacements resulting in that alignment score
    - ``optimal_inplane_rotation_M[i]`` is the rotation resulting in that alignment score


.. .. _optimal_displacement_rotations:

.. Optimized Displacement and Rotations
.. ----------------------------------------

.. This will return 4 2-dimensional Tensors. The outer (first) index
.. is the image sequence index,
.. and the inner (second) index is the template sequence index:

..  - Cross-correlation value for each image (``cross_correlation_SM``)
..  - The optimal x-displacement (``optimal_displacement_x_SM``)
..  - The optimal y-displacement (``optimal_displacement_y_SM``)
..  - The optimal inplane rotation (``optimal_inplane_rotation_SM``)

.. As these Tensors are two-dimensional, they are communicating the values
.. resulting in best alignment of each image and template.

.. .. admonition:: Example:

..     Consider indexing into these Tensors at outer index ``i`` and
..     inner index ``j`` . This will
..     correspond to the best-alignment values between the ``i`` th
..     image and ``j`` th tensor of the stack.
..     Then:

..       - ``cross_correlation_SM[i][j]`` is the best-alignment likelihood
..         score between image ``i`` and template ``j``
..       - ``optimal_displacement_x_SM[i][j]`` is the x-displacement resulting
..         in best alignment for this pair
..       - ``optimal_inplane_rotation_SM[i][j]`` is the rotational value
..         resulting in best alignment for this pair


.. .. _optimized_displacement:

.. Optimized Displacement
.. ------------------------

.. This data states the optimal displacements, de-aggregated over image,
.. template, and rotation.

.. This will return 3 3-dimensional Tensors. The outer (first) index is
.. the image sequence index, the
.. middle (second) index is the template sequence index, and the inner
.. (third) index is the index of the
.. corresponding rotational value (from the list of discrete rotations
.. used for comparison).

..  - Cross-correlation value for each image and template pair at each
..    possible rotational alignment (``cross_correlation_SMw``)
..  - Best X- and Y-displacements for each image-template pair at each
..    rotational alignment (``optimal_displacement_x_SMw`` and ``..._y_...``)

.. .. admonition:: Example:

..   Consider indexing into these Tensors at outer index ``i``, middle
..   index ``j``, and inner index ``k``. This
..   corresponds to looking at the alignment between the ``i`` th image
..   and ``j`` th template, at the ``k`` th rotation
..   value. Then:

..       - ``cross_correlation_SMw[i][j]`` is a 1-D slice with the
..         likelihood score of the best displacement value for each rotation
..       - ``optimal_displacement_x_SMw[i][j][k]`` is the displacement that
..         best aligns image ``i`` with template ``j`` when the image
..         has been rotated by the ``k`` th rotation value

.. .. _optimized_rotation:

.. Optimized Rotation
.. -----------------------

.. This data states the optimal rotations, de-aggregated over image, template,
.. and displacement index. It is very similar to the optimized displacement
.. return type above, except that it returns the best rotation for each
.. displacement, rather than the best displacement for each rotation.
.. It returns 2 3-D Tensors:

..  - The likelihood of alignment between the pair, at each displacement
..    value, given the most-likely angle of
..    rotation (``cross_correlation_SMd``)
..  - The rotation value generating that (best/likeliest)
..    alignment (``optimal_inplane_rotation_SMd``)

.. Note that the displacement grid is linearized, so we use only a single index
.. to indicate the displacement. This index refers to the displacements as
.. converted to the Fourier-space polar grid.

.. .. _complete_disagg:

.. Complete Disaggregated
.. --------------------------

.. This data provides a completely disaggregated view into the
.. cross-correlation
.. likelihood results. It returns a single 4-D Tensor, indexed
.. by image sequence
.. index, then template sequence index, then displacement index,
.. then rotation index.
.. The Tensor is ``cross_correlation_SMdw``.

.. Note that the displacement grid is linearized, so we use only a single index
.. to indicate the displacement. This index refers to the displacements as
.. converted to the Fourier-space polar grid.



Base Comparator
================

The underlying code that computes likelihood is found in the
``compute_cross_correlation`` function. For further information, see
:py:mod:`cryolike.likelihoods.kernels`.
