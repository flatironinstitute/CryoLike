Image-to-Structure Likelihood Computation
##########################################

The main output of CryoLike is the likelihood of between each input ``Images``
and ``Templates`` created from 3D structures.

Overview
==========

At its heart, CryoLike offers a way to compute the likelihood of a given
observed 2D image to a particular 3D structure.
As described in the
:doc:`Mathematical Framework</about.rst>`,
likelihood comparisons are based on comparing a stack of
images with a templates set using the cross-correlation.

:doc:`Templates sets</usage/templateCreation>`
are projections of a single 3D structure into image
space from multiple viewing angles.
Images will be compared against these templates
at a number of different rotations and displacements, and the results
can be returned with several different means of aggregation.

.. admonition:: Note:

    The Templates and Images stacks are unlikely to fit fully in GPU
    memory all at once, so CryoLike batches the comparison over several sets.
    To reduce memory transfer overhead, we preference Templates as the outer
    set of images to loop over.

Main Outputs
************

The primary outputs of CryoLike are the *best cross-correlation*
for each image across every template set (each corresponding to a
3D structure), and the *integrated likelihood* for each image with
respect to each 3D structure.

Interface
==============

The ``run_likelihood`` wrapper function exposes an interface to the underlying
``CrossCorrelationLikelihood`` object that incorporates convenience features
for file management and optionally attempts to find the best batch sizes for
available GPU hardware (if any).

For a worked example of this wrapper function, see the
:doc:`run likelihood example</examples/run_likelihood>`.


The ``run_likelihood`` function takes the following parameters:

 - A set of :doc:`image descriptor parameters</usage/imageSettings>`, in
   on-disk or in-memory form (``params_input``)
 - The path to the directory where templates are stored (``folder_templates``)
 - The path to the directory where image stacks are stored
   (``folder_particles``)
 - The root of the output directory (``folder_output``)
 - The index of the template file to process (``i_template``)
 - The number of image stacks to process (``n_stacks``)
 - Whether to skip processing when the output files appear to exist
   already (``skip_exist``)
 - Number of templates and images to use per batch, and whether to
   attempt to determine
   those values automatically (``n_templates_per_batch``,
   ``n_images_per_batch``, ``search_batch_size``)
 - The largest-size displacement to consider, and the number of
   displacements to consider in both directions
   (``max_displacement_pixels``, ``n_displacements_x``,
   ``n_displacements_y``)
 - Flags to configure which output files are written
   **[PC: Question is there a flag for the CC?]**
 
   - ``return_likelihood_integrated_pose_fourier``
   - ``return_optimal_pose``
   - ``optimized_inplane_rotation``
   - ``optimized_displacement``
   - ``optimized_viewing_angle``


Input System
***************

We compute likelihood by matching images against templates.
We expect the templates to be located under the directory
specified by ``folder_templates`` and the images to be located
under the directory specified by ``folder_particles``. Specifically:

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
``max_displacement_pixels`` parameters.

To compute the available displacements to try, the
``max_displacement_pixels`` is first
converted to Angstrom using the pixel size associated with
the image/template grids. The
resulting ``max_displacement`` is treated as a potential
displacement in either direction,
creating a total displacement length of ``2 * max_displacement``.
This distance is then
divided linearly into ``n_displacements_x`` and ``n_displacements_y``
steps, resulting in
a grid of displacement positions to test during cross-correlation
computation.

The set of displacements tested will be preserved in
``folder_output/displacements_set.pt``.

Possible Outputs
=========================

CryoLike can return the following aggregation levels of the computations.

Note that these correspond to the ``NamedTuple`` return-type classes defined in
``cross_correlation_likelihood.py``. For more detail, see
:py:mod:`cryolike.cross_correlation_likelihood`.

Output Type Selection
*************************

The ``run_likelihood()`` function exposes the following flags to control
which of the above return types will be returned, as well as which
additional likelihood reports will be written.

   - ``return_likelihood_integrated_pose_fourier``

If ``True``, we will additionally write a Tensor with the integrated
log likelihood of the Fourier-space cross-correlation (see the
:doc:`Mathematical Framework </about.rst>` and
:ref:`the Integrated likelihood section <_integrated_likelihood>`).

   - ``return_optimal_pose``

If true, we will output the Tensors described under
:ref:`the Optimal Pose section<optimal_pose>` below.
If this is set to true, the remaining  options will be ignored.

The remaining three options can be set individually, but the output will
depend on the chosen combination.


Output Paths
**************

The wrapper function writes computed likelihoods to disk for
later review. The exact files written depend on the requested outputs.

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

 - ``return_optimal_pose``: Will write the 5 Tensors
   :ref:`discussed above<optimal_pose>` to individual files:

     - ``OUT/templateN/cross_correlation/cross_correlation_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_template_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_displacement_x_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_displacement_y_stack_STACK.pt``
     - ``OUT/templateN/optimal_pose/optimal_inplane_rotation_stack_STACK.pt``

 - ``return_likelihood_integrated_pose_fourier``: will write the
   **TODO: WHATEVER THIS IS, I think the actual likelihoods?** to:

     - ``OUT/templateN/log_likelihood/log_likelihood_integrated_fourier_stack_STACK.pt``

 - ``return_likelihood_optimal_pose_fourier``: will write the
   **TODO: WHATEVER THIS IS** to:

     - ``OUT/templateN/log_likelihood/log_likelihood_optimal_fourier_stack_STACK.pt``

 - ``return_likelihood_optimal_pose_physical``: will write the
   **TODO: WHATEVER THIS IS** to:

     - ``OUT/templateN/log_likelihood/log_likelihood_optimal_physical_stack_STACK.pt``


.. _integrated_likelihood:

Integrated Log-Likelihood
******************************

The integrated likelihood is the main output of CryoLike.
It is calculated by comparing
each image to each template in the Fourier-Bessel
representation using the Cross-correlation
as described in the :doc:`Mathematical Framework</about.rst>`.

.. _optimal_pose:

Optimal Pose Outputs
****************************

This will return 5 1-dimensional Tensors, indexed by the image sequence index:

Cross-correlation
******************************

 - Best cross-correlation value for each image
   (``cross_correlation_S``) **[TO DO: CHECK _S]** The highest
   cross-correlation per image is a cryoLike output. As
   described in the :doc:`Mathematical Framework</about.rst>`,
   CryoLike calculates the cross-correlation between each image
   and each template.
 - The template (by sequence number) of the best match (``optimal_template_S``)
 - The optimal x-displacement matching this image with the best-fitting
   template (``optimal_displacement_x_S``)
 - The optimal y-displacement matching this image with the best-fitting
   template (``optimal_displacement_y_S``)
 - The optimal inplane rotation matching this image with the best-fitting
   template (``optimal_inplane_rotation_S``)

.. admonition:: Example:

    So consider the values at index ``i``, which correspond to the image at index ``i`` in the
    input Images stack. Then:

    - ``cross_correlation_S[i]`` is the best alignment likelihood
    - ``optimal_template_S[i]`` is the index of the template that got the score above
    - ``optimal_displacement_x_S[i]`` and ``..._y_S[i]`` are the displacements resulting in that alignment score
    - ``optimal_inplane_rotation_S[i]`` is the rotation resulting in that alignment score



**[TO DO::: I DONT KNOW IF THIS IS AN OUTPUT ANYMORE, and its
not referenced in the output section above]**

.. _optimal_displacement_rotations:

Optimized Displacement and Rotations
*********************************************

This will return 4 2-dimensional Tensors. The outer (first) index
is the image sequence index,
and the inner (second) index is the template sequence index:

 - Cross-correlation value for each image (``cross_correlation_SM``)
 - The optimal x-displacement (``optimal_displacement_x_SM``)
 - The optimal y-displacement (``optimal_displacement_y_SM``)
 - The optimal inplane rotation (``optimal_inplane_rotation_SM``)

As these Tensors are two-dimensional, they are communicating the values
resulting in best alignment of each image and template.

.. admonition:: Example:

    Consider indexing into these Tensors at outer index ``i`` and inner index ``j`` . This will
    correspond to the best-alignment values between the ``i`` th image and ``j`` th tensor of the stack.
    Then:

      - ``cross_correlation_SM[i][j]`` is the best-alignment likelihood score between image ``i`` and template ``j``
      - ``optimal_displacement_x_SM[i][j]`` is the x-displacement resulting in best alignment for this pair
      - ``optimal_inplane_rotation_SM[i][j]`` is the rotational value resulting in best alignment for this pair


.. _optimized_displacement:

Optimized Displacement
******************************

This data states the optimal displacements, de-aggregated over image,
template, and rotation.

This will return 3 3-dimensional Tensors. The outer (first) index is
the image sequence index, the
middle (second) index is the template sequence index, and the inner
(third) index is the index of the
corresponding rotational value (from the list of discrete rotations
used for comparison).

 - Cross-correlation value for each image and template pair at each
   possible rotational alignment (``cross_correlation_SMw``)
 - Best X- and Y-displacements for each image-template pair at each
   rotational alignment (``optimal_displacement_x_SMw`` and ``..._y_...``)

.. admonition:: Example:

  Consider indexing into these Tensors at outer index ``i``, middle
  index ``j``, and inner index ``k``. This
  corresponds to looking at the alignment between the ``i`` th image
  and ``j`` th template, at the ``k`` th rotation
  value. Then:

      - ``cross_correlation_SMw[i][j]`` is a 1-D slice with the
        likelihood score of the best displacement value for each rotation
      - ``optimal_displacement_x_SMw[i][j][k]`` is the displacement that
        best aligns image ``i`` with template ``j`` when the image
        has been rotated by the ``k`` th rotation value

.. _optimized_rotation:

Optimized Rotation
******************************

This data states the optimal rotations, de-aggregated over image, template,
and displacement index. It is very similar to the optimized displacement
return type above, except that it returns the best rotation for each
displacement, rather than the best displacement for each rotation.
It returns 2 3-D Tensors:

 - The likelihood of alignment between the pair, at each displacement
   value, given the most-likely angle of rotation (``cross_correlation_SMd``)
 - The rotation value generating that (best/likeliest)
   alignment (``optimal_inplane_rotation_SMd``)

**TODO: SAY SOMETHING ABOUT THE FACT WE ONLY USE
SINGLE INDEX FOR DISPLACEMENT.**

.. _complete_disagg:

Complete Disaggregated
******************************

This data provides a completely disaggregated view into the cross-correlation
likelihood results. It returns a single 4-D Tensor, indexed by image sequence
index, then template sequence index, then displacement index,
then rotation index.
The Tensor is ``cross_correlation_SMdw``.

**TODO: SAY SOMETHING ABOUT THE FACT WE USE
ONLY A SINGLE INDEX FOR DISPLACEMENT**




Base Comparator
================

The underlying code that computes likelihood is found in the
``CrossCorrelationLikelihood`` object. It contains many methods
for computing probability arrays, including ones which are not
yet supported by the wrapper, but are currently available.

For further information, see
:py:mod:`cryolike.cross_correlation_likelihood`.
