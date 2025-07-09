Template Creation
##############################

In CryoLike, "Templates" (or a ``Templates`` object) are stacks of
images, with each image representing the 2D projection of a
base 3D structure or 3D map with a particular orientation ("viewing angle") relative
to the capture device.

Overview
==========

CryoLike can convert models in several formats:
 - MRC
 - PDB
The ``Templates`` object itself is virtually identical to the ``Images`` object: it consists
of a stack of images. The main functional difference is that the ``Templates`` requires that
viewing angles be recorded (while ``Images`` objects may leave this field empty).


Templates Metadata
------------------

As described in more detail in the :doc:`image settings documentation</usage/imageSettings>`,
there are some image descriptor metadata fields that apply only
to ``Templates`` during the conversion process. Other than viewing distance, 
these are only required or used for converting PDB files. 


Interfaces
============

Users should generate ``Templates`` using the ``make_templates_from_inputs()``
wrapper function, which generates all templates using previously specified parameters
for many PDBs of maps.  


For a basic example of generating ``Templates`` from PDB files, see the
:doc:`template generation example </examples/make_templates>`.

Wrapper function
----------------

The ``make_templates_from_inputs()`` wrapper function is exported from
``CryoLike.stacks``. It accepts:

 - a list of inputs ``list_of_inputs``: each input can be a path to a file (PDB, MRC),
   or a Numpy array or Torch tensor containing density data already loaded into memory.
 - The path to an image descriptor file (see the
   :doc:`image settings documentation</usage/imageSettings>` for more details)
 - Whether or not to output plotted samples of the converted Templates (``output_plots``)
 - The root output folder ``folder_output``, which defaults to a directory called ``templates``
   in the working directory
 - Whether to use verbose output

The wrapper function will emit one stack of Templates per input in the list. Each stack will
have a number of images equal to the number of viewing angles defined in the image descriptor
file. Template stacks will be written to the directory specified by the ``folder_output``
parameter (default ``./templates``). This directory will be created if it does not already exist.

Each file will be named following the pattern:
``templates_fourier_{name}.pt``, where ``{name}`` is

 - for file-based inputs, the name of the source PDB or MRC file (without extension)
 - for in-nemory inputs, ``tensor_{i}`` where ``{i}`` is the order of the item in the input list.

If a file by this name already exists--for instance, if the user processed two files named
``data.mrc``--then a counter will be appended to the output name.

Additionally, a file named ``template_file_list.npy`` will be written in the output folder
documenting the name of every file output by the run (as a Numpy array of strings).

.. admonition:: Example

    Given the following list of inputs:

    - ``folder1/model.pdb``
    - ``folder2/model.pdb``
    - [a volume represented as a Torch Tensor]

    with ``folder_output`` set to ``my_project/templates``, the function would write the files:

    - ``my_project/templates/templates_fourier_model.pt``
    - ``my_project/templates/templates_fourier_model_1.pt``
    - ``my_project/templates/templates_fourier_tensor_2.pt``
    - ``my_project/templates/template_file_list.npy``

If the user has requested that plots be created, they will be placed in ``<folder_output>/plots/``.
For every output Templates file, the plotter function will write three plots:

 - ``templates_fourier_{NAME}.png``
 - ``templates_phys_{NAME}.png``
 - ``power_spectrum_{NAME}.png``

where ``{NAME}`` follows the same pattern as for output files (except that repeat file names
will not generate a counter--in the event of name collisions, the later plots will overwrite
the earlier ones.)

These plots contain images of the
first 16 template images in physical and Fourier space, as well as the power spectrum of the
Fourier-space representation.


Additional methods
------------------

Most users are expected to use the wrapper function described above. However, the
``Templates`` class also exposes some functions that can generate a stack of templates
from a function, as well as the underlying calls to create templates from physical volumes
or from atom positions. 

