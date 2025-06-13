Concepts
==========================
.. Also considered: "Functionality in detail"

These pages provide a more detailed explanation of the features offered
by the CryoLike package and the use cases it can handle.

For example usages in different scenarios, see the NEED CROSS-REFERENCE
tutorials section.

For exhaustive description of the public-facing API, see the NEED CROSS-REFERENCE
python API section.


.. Check out the :doc:`usage` section for further information, including
  how to :doc:`install <installation>` the project.


.. toctree:: 
    :maxdepth: 2

    imageSettings
    templateCreation
    imageConversion
    likelihoodComputation
    file_structure
    formats


Overview
------------------------

The purpose of the CryoLike package is to allow users to compare 2D images
(captured through Cryo-EM) with potential 3D structures from reference sources,
in order to determine the likelihood of each image under a given set of
possible rotations and translations of projections of the 3D structure (templates).

Data files
##########

The two data types--known structures and experimental images--give rise to two
types of data files:

- *Templates*, which are sets of 2D projections of a 3D structure or map at known viewing directions (poses) and
- *Images*, or stacks of experimentally-captured cryoEM particle stacks

The user should begin by setting the imaging and cryoLike parameters [TODO: explain and reference]

Then, the user must *generate the Templates* by
projecting the 3D structres into 2D images space,
and *convert the particles* into the representation that CryoLike can handle.

The output of these processes are the parameter files, and the Image and Template stacks used to running the likelihood
comparison. These are stored as `.pt` (Pytorch Tensor) files.

In addition to the Images and Templates files, CryoLike also stores *metadata* files that describe
how to interpret them. This includes:

- Data that is relevant to both file types (such as the
  Cartesian-space and Fourier-space discretization grids used to interpret the 2D representations),
- Data that is specific to Templates (such as how the atomic structures were interpreted), and
- Data that is specific to Images (such as the microscope properties,
  contrast transfer function, etc.)

A meaningful comparison between Images and Templates depends on compatible metadata. Thus,
CryoLike's likelihood computation checks metadata compatibility for every Image and Template stack.


Creating parameter settings
------------------------

CryoLike's representations of images depend upon two grids: one in two-dimensional Cartesian space,
and one in Fourier space with polar coordinates. These grids, along with a precision (single or double),
make up the required fields of the ``ImageDescriptor`` object. The object also has optional fields which
control how CryoLike makes Templates from reference structures; the optional fields are not used when
creating or interpreting Images.

[TODO: I THINK THIS CHANGED]

The first step to a CryoLike run is to create an image descriptor file. Once this file is present,
you can proceed with Templates creation and Images conversion. These may be done in parallel, if you
have appropriate compute resources.

For an example of creating an image descriptor, see the
:doc:`image settings creation example </examples/set_image_parameters>`.

For more details about the available options, see the
:doc:`detailed description of image parameters </usage/imageSettings>`, which also links
to the relevant API calls for further information.


Creating Templates
------------------------

CryoLike currently supports creating Templates from ``.mrc`` files (``.mrc``, ``.mrcs``, ``.map``)
and from ``pdb`` files. Additionally, Templates may be created from physical-density data in
other formats, if the data has already been loaded into memory as a Numpy Array or PyTorch Tensor.

Template conversion will place the generated files in a standard directory structure and will
additionally generate a list of the created files.

For a basic example of generating Templates from PDB files, see the
:doc:`template generation example </examples/make_templates>`.

For more details about the available options for Template creation, see the
:doc:`detailed description of Templates creation </usage/templateCreation>`, which includes
details about the available parameters, supported formats, and links to the relevant API calls.


Converting images
------------------------

CryoLike currently supports reading CryoSparc and Starfile image sources and images
stored in ``.mrc`` format. The conversion
process also handles restacking Images into standard sizes.

Converted Images stacks will be stored in a standard directory structure. Each stack will have
its own parameters file recording provenance and capture data, as well as description of grids.

For a basic example of converting Images from star files, see the
:doc:`image conversion example</examples/convert_particle_stacks>`.

For more details about the available options for Image conversion, see the
:doc:`detailed description of Images conversion </usage/imageConversion>`, which includes
details about the available parameters, supported formats, and links to relevant API calls.


Running the Likelihood computation
------------------------

CryoLike outputs a variety of optimal pose and likelihood data, with several options for which
degrees of freedom to consider and how to aggregate the likelihoods.

For a basic example of running likelihood, see the
:doc:`likelihood computation example</examples/run_likelihood>`.

For more details about the types of output CryoLike can generate, see the
:doc:`detailed description of cross-correlational likelihood computation</usage/likelihoodComputation>`,
which includes details about the available parameters and output specifications, as well
as links to relevant API calls and a description of the wrapper functions.


File Structure
------------------------

In the current implementation, CryoLike makes certain strong assumptions about
the location of its input files and where its output will be generated. The main functions
allow the user to change the top-level directory, but the internal directory structure
is more rigid. This may be changed in future versions.

For more details about CryoLike's expected file locations, see the
:doc:`detailed description of file/directory structure </usage/file_structure>`.

