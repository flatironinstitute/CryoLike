Image Conversion
##############################

In CryoLike, "Images" (or an ``Images`` object) are stacks of experimentally-captured cryo-EM
images. In practice, we operate extensively on a Fourier-space representation of the images.

Internally, these images are represented as paired multi-dimensional arrays--PyTorch
Tensors--along with metadata values that describe how to interpret the arrays, and
associated data such as any associated contrast transfer function (CTF).

Because this format differs from the ones used by experimental software to store captured
images, CryoLike needs to ingest those image files and convert them to our internal
representation before comparisons can be made. As part of this process, we also offer
ways to restack the images into regularly-sized stacks, which can help ensure more efficient
use of computational resources.


.. contents::  Table of Contents


Overview
==========

To properly import an image, CryoLike needs both the MRC file (where the image itself
is actually stored) and a file or files, in Starfile or CryoSparc format, describing the
image capture equipment.

We support several formats:

 - Lists of Star files (with one MRC file per Star file)
 - Internally indexed Star files (one Star file describes images from many MRC files, as with ReLion files)
 - Internally indexed CryoSparc files (one CryoSparc file describes images from multiple listed MRC files)
 - CryoSparc with 'jobs' folder (one CryoSparc file describes the MRC images located in a specified folder)

Full details can be found at the API documentation. [TODO: XREF]


Images Metadata
------------------

During the conversion process, image stacks require two sets of metadata. One of these
is partially shared with the image templates: this is the "image descriptor," which
includes the discretization and scaling grids used to interpret the tensor representations
of the images. For more detail on this, see the
:doc:`image settings documentation</usage/imageSettings>`
and the TODO: XREF-API documentation, 

In addition, image conversion needs data describing the image capture apparatus--most notably
defocus and phase shift information. This metadata is expected to come from the Star file or
CryoSparc file. A full description of the expected file formats can be found in the
:doc:`file formats documentation</usage/formats>`


Interfaces
============

File conversion is managed by the ``ParticleStackConverter`` class. While it is possible
to interface with this class directly (see XREF TO API DOCUMENTATION), we also offer
wrapper functions for the four use cases (i.e. converting from individual Star files,
converting from indexed Star or CryoSparc files, and converting CryoSparc job directories).

These wrapper functions are discussed below.

For a basic example of converting ``Images`` from a set of Star files and particle files,
see the :doc:`image conversion example </examples/convert_particle_stacks>`.

OTHER EXAMPLES TK

For complete documentation of the API, see the API DOCUMENTATION
(IF WE HAVEN'T LINKED TO THAT ENOUGH ALREADY)

Wrapper functions
-----------------

We will discuss the four wrapper functions available for the four cases listed above.
These wrapper functions can be accessed at ``CryoLike.convert_particle_stacks``.

Common Parameters
****************************

 - The path to a file containing image descriptor information (``params_input``)
 - The root of the directory in which to output image stacks (``folder_output``)
 - The maximum (or target) number of images in a stack (``batch_size``)
 - A scalar downsampling factor (1, the default, means no downsampling; 2 would downsample
   by a factor of 2, etc) and downsample method ('mean' or 'max') (``downsample_factor``, ``downsample_type``)
 - A manually input pixel size (errors will be generated if this conflicts with the pixel
   size recorded in the source file). In Angstroms. (``pixel_size``)
 - Whether to output plots (``flag_plots``). Default (``True``) causes plots to be output.


Output
****************

All functions described here produce three (or four) kinds of output:

 #. Stacks of physical images, as Torch tensor (``.pt`` file)
 #. Stacks of Fourier-space images, as Torch tensor (``.pt`` file)
 #. A metadata file that records both the grids for the images and the capture apparatus metadata
 #. A set of plots, if requested

The user controls the root of the output direcvtory with the ``folder_output`` parameter
(by default, the wrapper functions use the current directory). Within the output directory,
files will use the following naming conventions, where ``OUT`` is the user-specified root
of the output directory and ``COUNTER`` is a 6-digit 0-padded count of the number of stacks
exported so far:

 #. Physical images: ``OUT/phys/particles_phys_stack_COUNTER.pt``
 #. Fourier images: ``OUT/fft/particles_fourier_stack_COUNTER.pt``
 #. Metadata file: ``OUT/fft/particles_fourier_stack_COUNTER.npz``
 #. Plots (if requested): ``OUT/plots/PlOT_NAME``, where ``PLOT_NAME`` matches the stack
    name for the physical or Fourier image files or is ``power_spectrum_stack_COUNTER.png``
    for the power spectrum plot

Most of the wrapper functions are *restacking* by default: they will read input image data
in one or multiple files, and output regular-sized stacks of ``batch_size`` images each,
except for the last stack (which has the remaining images). This will potentially combine
images from several input MRC files into a single stack.

The exception is the ``convert_particle_stacks_from_star_files()`` wrapper. This function
is intended to process pairs of Star files and MRC files, so it is assumed that the Star files
might have different (incompatible) settings. This function will output one or more stacks per
input MRC/Starfile pair: if a single input contains more than ``batch_size`` images, it will
split those images into multiple output stacks, but it will not combine images from multiple
inputs into a single stack.

The underlying converter can apply either logic to either type of input; please see the
TODO: API XREF HERE for more information.

.. admonition:: Example

  Suppose we have ``ONE.mrc``, ``TWO.mrc``, and ``THREE.mrc``, which have 7, 2, and 6 images,
  respectively. We call the wrapper with ``batch_size`` set to 10 and ``folder_output`` set to
  ``output``, with no plots.

  Most functions would produce the following files:
    
  - ``output/phys/particles_phys_stack_000000.pt`` (containing all 7 images from ``ONE.mrc``,
    both images from ``TWO.mrc``, and one image from ``THREE.mrc``)
  - ``output/phys/particles_phys_stack_000001.pt`` (containing the remaining 5 images from ``THREE.mrc``)
  - ``output/fft/particles_fourier_stack_000000.pt`` and ``..._000001.pt`` (containing Fourier-space
    representations of the same image stacks as above)
  - ``output/fft/particles_fourier_stack_000000.npz`` and ``..._000001.npz`` (containing metadata
    for the above stacks)
    
  The ``convert_particle_stacks_from_star_files()`` wrapper function would produce:

  - ``output/phys/particles_phys_stack_000000.pt`` (with only the 7 images from ``ONE.mrc``)
  - ``..._000001.pt`` (with only the 2 images from ``TWO.mrc``)
  - ``..._000003.pt`` (with only the 6 images from ``THREE.mrc``)
  - and so on for the Fourier-space and metadata file outputs.
  - If the ``batch_size`` were set to 5 instead,
    this function would emit 5 physical and 5 Fourier stacks, since ``ONE.mrc`` and ``THREE.mrc`` would be split
    so as not to exceed the batch size.

.. admonition:: Common Pitfalls

    TODO Something about make sure you have enough CTFs/defocus angles etc


``convert_particle_stacks_from_star_files()``
**********************************************

This function is designed to convert images stored in a series of MRC files, described
by a corresponding series of Star files. The two file lists should be of the same length.

API XREF LINK

In addition to the common parameters above, this function exposes the following parameters:

 - A list of Star files (``star-file_list``) and MRC files (``particle_file_list``). These
   lists should be the same length, with each Star file describing all the particles in the
   MRC file at the corresponding index. Paths may be absolute or relative to the directory
   where you are running the script.
 - Whether the defocus and phase shift angle measurements in the Star file are in degrees
   or radians (``defocus_angle_is_degree``, ``phase_shift_is_degree``). These fields are
   optional; if not provided, we assume angles are in degrees.

As described above, this wrapper function follows a different batching logic than the
other two: it never makes output stacks that combine images from multiple MRC files.


``convert_particle_stacks_from_indexed_star_files()``
****************************************************************

This function is designed to convert images stored in a series of MRC files, described
by a single Star file that refers to the images individually.

For more information about the expected file format, see :doc:`the formats page</usage/formats>`.

API XREF LINK

In addition to the common parameters above, this function exposes the following parameters:

 - A Star file referring to images in individual MRC files (``star_file``)
 - The location of the MRC files referred to (``folder_mrc``)
   
If the ``folder_mrc`` value is set, any path information in the Star file will be ignored; the MRC
files will be assumed to reside directly in this directory. If this value is NOT set,
then the system will use the paths in the Star file. Those paths will be assumed to
be relative to the current directory.


``convert_particle_stacks_from_cryosparc()``
****************************************************************

This function is designed to convert images stored in a series of MRC files, described
by a single CryoSparc file that refers to the images individually.

API XREF LINK

In addition to the common parameters above, this function exposes the following parameters:

 - The location of a CryoSparc file that refers to the MRC files (``file_cs``)
 - The root location of the MRC files (``folder_cryosparc``)
 - A maximum number of stacks to output before terminating (``n_stacks_max``); by default
   all files will be processed

As with the ``indexed_star_file()`` converter function, if the ``folder_cryosparc`` is
not set, we will assume that any path information in the CryoSparc file provides correct
relative paths to the MRC files. If the ``folder_cryosparc`` value is set, we will take
only the filename (without path information) from the CryoSparc index, and look for
those filenames within the ``folder_cryosparc`` directory.


``convert_particle_stacks_from_cryosparc_restack()``
****************************************************************

This function is designed to convert images stored in a CryoSparc job folder, described
by a single unified CryoSparc file. It expects to load all the images from all the MRC
files in the job directory, in order, until the sequence of MRC files is broken.

API XREF LINK

Instead of looking at explicitly specified MRC files, as in the "``indexed``" wrappers
above, this function attempts to process all MRC files that follow a certain naming
convention that reside within the same job directory. They are assumed to be all described
by the same CryoSparc file, which is expected to reside within the job directory. (The
CryoSparc file's location is not explicitly passed to this function.)

In addition to the common parameters above, this function exposes the following parameters:
 - The root location of the job folders (``folder_cryosparc``) 
 - The number identifying which sub-folder to process (``job_number``)
 - A maximum number of stacks to output before terminating (``n_stacks_max``); by default
   all files will be processed

All files are expected to reside in a "job folder" under the directory specified by the
``folder_cryosparc`` parameter. The details are best expressed by example:

.. admonition:: Example:

  Assume ``folder_cryosparc`` is set to ``cryofolder`` and ``job_number`` is set to ``2``.

  We expect the job directory to be ``cryofolder/J2`` and expect the following to exist:

   - ``cryofolder/J2/J2_passthrough_particles.cs``, a CryoSparc file with the metadata for
     all the images to be converted
   - One of the following sub-directories:

     - ``cryofolder/J2/restack`` containing files matching ``batch_NUMBER_restacked.mrc``, OR
     - ``cryofolder/J2/downsample`` containing files matching ``batch_NUMBER_downsample.mrc``
    
  where ``NUMBER`` is a sequential index starting with 0.
  
  If both the ``restack`` and ``downsample`` subdirectories exist, ``restack`` will be used.
  
  Note that ``downsample`` refers to any downsampling that has been done PRIOR TO use of the
  CryoSparc library. Within image processing, any downsampling is controlled
  by the ``downsample_factor`` and ``downsample_type`` parameters, as normal.

  The converter will then process every file in the chosen directory, starting with 0, until
  it cannot find a file matching the expected naming pattern. (Note that this means that a
  discontinuous numbering--going from ``batch_4_restacked.mrc`` to ``batch_6_restacked.mrc``--
  will cause processing to terminate.)

  The CryoSparc file is expected to have metadata for each of the MRC files' images, in order.



Using ``ParticleStackConverter`` directly
------------------------------------------------------

While the above wrappers are likely to meet most users' needs, it is also possible
to interact with the ``ParticleStackConverter`` class directly. This could be
useful for, for instance, interactively converting several different sources of
images.

In this event, the implementations of the wrapper functions [TODO: INSERT XREF LINK]
are instructive, as they all follow the same pattern:

 #. Instantiate the converter with basic information (parameters, output, stack settings)
 #. Load the converter with the input files to process
 #. Call the ``convert_stacks`` function to write out the processed batches

For further information, see the TODO: XREF API DOCUMENTATION or the code itself.
