Concepts
==========================

These pages provide a more detailed explanation of the features offered
by the CryoLike package and the use cases it can handle.

For example usages in different scenarios, see the
:doc:`tutorials section </tutorials/tutorials>`.

For exhaustive description of the public-facing API, see
:doc:`the python API section </pythonapi/modules>`.


Overview
------------------------

The purpose of the CryoLike package is to allow users to compare 2D images
(captured through cryo-EM) with 3D structures from reference sources, such
as molecular dynamics or modelling tools,
in order to determine the likelihood of each image under a given set of
"templates" (possible rotations and translations of projections of
the 3D structure).

There are four main steps of a CryoLike run:

#. Setting the imaging and CryoLike parameters
   **[TODO: explain and reference]**
#. Generating the templates by projecting
   the 3D structures into 2D space
#. Converting the particles into CryoLike's
   internal representation
#. Running the comparison


Generated files
#################

The first three steps of this process emit several types of
files. Template generation and particle conversion create
two types of data files:

- *Templates*, which are sets of 2D projections of a 3D structure or map at
  known viewing angles (poses), and
- *Images*, or stacks of experimentally-captured cryoEM particle images

These are stored as `.pt` (Pytorch Tensor) files. We commonly
refer to them as "stacks," since they usually contain many images
per file.

In addition, the first step creates collection(s) of parameters
that describe how to interpret the images. These parameter
files are collectively called *metadata* files.
Metadata includes:

- Data that is relevant to both file types (such as the
  Cartesian-space and Fourier-space discretization grids used to
  interpret the 2D representations),
- Data that is specific to Templates (such as how the 3D atomic
  structures were interpreted), and
- Data that is specific to Images (such as the microscope properties,
  contrast transfer function, etc.)

A meaningful comparison between Images and Templates depends on
compatible metadata. Thus, CryoLike's likelihood computation
checks metadata compatibility for every Image and Template stack.


Creating parameter settings
-------------------------------------------------

CryoLike uses representations of images in both two-dimensional
Cartesian space, and in Fourier space. Interpreting these
representations requires two discretization grids: a
conventional grid for the Cartesian representation and a
polar grid for the Fourier representation. These grids, along with a
numeric precision (single or double), make up the required fields
of the ``ImageDescriptor`` object. The object also has
optional fields which control how CryoLike makes Templates
from reference structures. Note that the optional fields are
not used when creating or interpreting
(experimentally-captured) Images.

The first step to a CryoLike run is to create an image
descriptor file. Once this file is present,
you can proceed with Templates creation and Images conversion.
These may be done in parallel, if you
have appropriate compute resources.

For an example of creating an image descriptor, see the
:doc:`image settings creation example </examples/set_image_parameters>`.

For more details about the available options, see the
:doc:`detailed description of image parameters </concepts/imageSettings>`,
which also links
to the relevant parts of the CryoLike API for further information.


Creating templates
------------------------

CryoLike currently supports creating Templates from
``.mrc`` files (``.mrc``, ``.mrcs``, ``.map``) and from ``pdb`` files.
Additionally, Templates may be created from physical-density data in
other formats, if the data has already been loaded into memory as a
Numpy Array or PyTorch Tensor.

Template conversion places the generated files in a standard directory
structure and will additionally generate a list of the created files.

For a basic example of generating Templates from PDB files, see the
:doc:`template generation example </examples/make_templates>`.

For more details about the available options for Template creation, see the
:doc:`detailed description of Templates creation </concepts/templateCreation>`,
which includes details about the available parameters, supported formats,
and links to the relevant parts of the CryoLike API.


Converting images
------------------------

CryoLike currently supports reading CryoSparc and Starfile descriptor
files, and images stored in ``.mrc`` format. The conversion
process also handles "restacking," or converting Image inputs into
files with an equal number of images each.

Converted Images stacks are stored in a standard directory
structure. Each stack has its own parameters file; this file
records
provenance and capture data, as well as description of grids.
A description of the file layout can be found at the
:doc:`file structure documentation </concepts/file_structure>`.

For a basic example of converting Images from star files, see the
:doc:`image conversion example</examples/convert_particle_stacks>`.

For more details about the available options for Image conversion, see the
:doc:`detailed description of Images conversion </concepts/imageConversion>`,
which includes details about the available parameters, supported formats,
and links to relevant API calls.


Running the Likelihood computation
---------------------------------------------

CryoLike outputs a variety of optimal pose and likelihood data,
with several options for which degrees of freedom to consider
and how to aggregate the likelihoods.

For a basic example of likelihood computation, see the
:doc:`likelihood computation example</examples/run_likelihood>`.

For more details about the types of output CryoLike can generate,
see the
:doc:`detailed description of cross-correlational likelihood
computation</concepts/likelihoodComputation>`,
which includes details about the available parameters and output
specifications, as well
as links to relevant API calls and a description of the wrapper functions.


File Structure
------------------------

In the current implementation, CryoLike makes certain strong assumptions about
the location of its input files and where its output will be generated. The
main functions allow the user to change the top-level directory, but the
internal directory structure
is more rigid. This may be changed in future versions.

For more details about CryoLike's expected file locations, see the
:doc:`detailed description of
file/directory structure </concepts/file_structure>`.

To read more about the CryoLike concepts and functions see:

.. toctree::
    :maxdepth: 1

    imageSettings
    templateCreation
    imageConversion
    formats
    likelihoodComputation
    file_structure
