File and Directory Structure
################################

CryoLike's wrapper functions expect certain input and
output files to be located on the filesystem in
predictable locations. Most wrappers provide an optional
prefix to determine the root directory, but within that
root, the following expectations will hold.


Template Creation Patterns
===========================

See also :doc:`/examples/make_templates`


Inputs
------

No particular patterns. Users are expected to pass a full
valid path to any input files as part of the input list.


Outputs
-------

Prefix set by the ``folder_output`` parameter.

Assume:

 - ``folder_output`` is ``FOLDER_OUTPUT``
 - ``NAME`` is:

   - for files: the base name of a file, e.g. "file" for "files/file.pdb"
   - for in-memory arrays: ``templates_fourier_tensor_N`` where ``N`` is the number of
     that element in the input list

Generated template files will be written as:

 - ``FOLDER_OUTPUT/templates/templates_fourier_NAME.pt``
 - ``FOLDER_OUTPUT/templates/templates_fourier_NAME_1.pt`` if that file exists
   (counter will keep incrementing)
 - ``FOLDER_OUTPUT/templates/template_file_list.npy`` for the list of files written

If plots are requested, they will be placed in:

 - ``FOLDER_OUTPUT/plots/templates_fourier_NAME.png``
 - ``FOLDER_OUTPUT/plots/templates_phys_NAME.png``
 - ``FOLDER_OUTPUT/plots/power_spectrum_NAME.png``

Names for plots will *not* increment; in the event of repeated names, later plots
will overwrite earlier ones.


Image Conversion Patterns
===========================

See also :doc:`/examples/convert_particle_stacks`


Inputs
------

Depends on the wrapper function being used. Refer to the documentation
for the individual wrapper functions.

In general, "indexed" wraqppers expect the metadata file to contain either
a valid relative path to the MRC files, or expect the MRC file path basenames
to govern (with the files themselves located in the directory pointed to
by the ``folder_mrc`` parameter).


Outputs
-------

Prefix set by the ``folder_output`` parameter.

Assume ``folder_output`` = ``OUT`` and ``COUNTER`` is a 6-digit 0-padded count
of the total number of stacks converted so far. Then:

 - ``OUT/phys/particles_phys_stack_COUNTER.pt`` for physical image stacks
 - ``OUT/fft/particles_fourier_stack_COUNTER.pt`` for Fourier-space image stacks
 - ``OUT/fft/particles_fourier_stack_COUNTER.npz`` for image metadata
   (stored alongside the Fourier-space image stack)
 - ``OUT/plots/PLOT_NAME`` for plots, if requested, where ``PLOT_NAME`` matches
   the stack name or is ``power_spectrum_stack_COUNTER.png`` for power spectrum plot



Likelihood Output Patterns
==========================

See also :doc:`/usage/likelihoodComputation`


Inputs
------

For Templates: the actual paths will be listed in the ``template_file_list.npy``
file under the directory pointed to by the ``folder_templates`` parameter.

For Images: We expect to look at the Fourier-space image stacks stored following
the output conventions of the image conversion process. Specifically,
``FP/fft/particles_fourier_stack_NUMBER.pt``, where ``FP`` is the value of
the ``folder_particles`` parameter and ``NUMBER`` is a 6-digit 0-padded increment.

If the physical image stack is required (for reporting optimal pose likelihood against
physical poses), ``FP/phys/particles_phys_stack_NUMBER.pt``.


Outputs
-------

The exact files written will depend on the output type requested.

In general, if the ``folder_output`` parameter is set to ``OUT``,
all outputs will be under ``OUT/templateN``, where ``N`` is the
index number of the template in the templates list (i.e. the ``i_template``
parameter).

Then:

 - Cross-correlation outputs will go under ``OUT/templateN/cross_correlation/``
 - Optimal-pose outputs will go under ``OUT/templateN/optimal_pose/``
 - Integrated and log-likelihood outpus will go under ``OUT/templateN/log_likelihood/``
  