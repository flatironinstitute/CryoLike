Parameter setup: Image Descriptor 
##############################

The Image Descriptor object contains the parameters required
to interpret a set of 2D images (Images stack or Templates). An
``ImageDescriptor`` is a required input to the functions for making
Templates or converting experimentally-captured Images.

Available fields
============================

The following fields are present on the ``ImageDescriptor`` object.
In general, users will set them through a simpler interface
(see :ref:`the Interface section <imagesettings_interface>` below and the :doc:`/examples/set_image_parameters`).

Shared fields
----------------------------

The ``ImageDescriptor`` object has three fields that must
be present, whether it is being used for Images or for Templates.

These are:
 - A ``CartesianGrid2D`` instance.
 - A ``PolarGrid`` instance.
 - A ``Precision`` (single or double)


The ``CartesianGrid2D`` describes a two-dimensional grid.
While the class supports rectangular grids and pixel shapes,
by default this will be a square grid defined by the number
of pixels per side (``n_pixels``) and the pixel size (``pixel_size`` in Angstroms). 

The ``PolarGrid`` describes a polar-coordinate grid used to
discretize the Fourier-space representation of the image.
The underlying class supports non-uniform grids, but
CryoLike as a whole does not yet support them. Uniform grids
can be completely described by a number of shells (radii),
the radial distances of the shells, and the number of in-plane
points per shell.

The ``Precision`` can be set to single or double precision.


Template-specific fields
----------------------------

The following ``ImageDescriptor`` fields are only used in
the process of converting 3D volume data to sets of Templates.
Their values have no effect on captured Images, and they are
not used for Templates after the initial conversion process.
They are saved as part of the metadata file for future reference.
These are the specific fields potentially used for Templates creation:

 - A ``viewing distance`` (in Angstroms) that deterimes the number of *ViewingAngles*
 - The ``resolution_factor`` in [0,1]
 - The ``radii`` (in Angstroms) of the atoms or beads in the model 
 - A string identifying the ``selected atoms`` to take from the structure if ``PDB`` format is used.
 - Whether to use the ``default protein residue model`` atomic sizes
 - The ``shape`` of the atoms (i.e. whether to interpret them as hard spheres or Gaussian probability clouds with a given width defined by `radii`)


The *ViewingAngles* are the different orientations of the
device lens which will be considered in projecting the 3D model
onto a 2D plane. These could be set manually, but in current
usage they are computed automatically from the
``viewing distance``. A smaller ``viewing distance`` implies a finer angular search grid, 
and therefore creates more templates and increses the computational costs.
Once the viewing distance is defined, the grid for the :math:`\alpha` and :math:`\beta`
Euler angles is computed, and the rotation as exempliefed in the figure below is applied to 
generate each template.

 [TODO: INCLUDE ADI's figure]

The ``resolution_factor`` determines the maximum frequency in the Fourier polar grid up to 
which the cross-correlation and likelihood computations will be performed. ``resolution_factor=1`` 
performs calculations up to Nyqst (2 ``pixel_size``)
The maximum frequency radius is  the *resolution_factor/Nyqst*, and therefore 
a lower ``resolution_factor`` implies a lower frequency up to which to perform the calculation.  

Note that, since each Template is the 2D projection from a
specific angle, the number of viewing angles should match the number
of Templates in the Templates file.

The remaining fields are required only for interpreting PDB models (not maps)
and will be ignored otherwise. Note that to create a map from a PDB file there are two options.
The user must either specify a value for the ``atom radii``  that determines the radius of the ``shape`` obeject 
(Gaussian or sphere) to be centered on the position of the ``atom selection`` or set the
``default protein residue model`` flag for proteins, which models a 
single bead per amino acid with a specific Vaan der Waals radii centered on the :math:`C_\alpha` as in `Cossio and Hummer, 2013 <https://www.sciencedirect.com/science/article/abs/pii/S1047847713002712?via%3Dihub>`_


Compatibility
============================

Image Descriptors can describe the images in Images stacks or in
Templates. Two Image Descriptors are considered compatible if they
use the same 2D Cartesian grid and Fourier-space polar grid.

All other fields are ignored for the purposes of compatibility.
So there is no issue with running a comparison between the
Images converted using one ImageDescriptor and Templates constructed
with another ImageDescriptor--as long as the descriptors describe
the same grid, they are interoperable.


.. _imagesettings_interface:

Interface
============================

The main way for a user to create an ``ImageDescriptor`` instance
is by the ``ImageDescriptor.from_individual_values()`` function.

The following parameters are accepted:

 - Precision (as a string ``single`` or ``double``, or CryoLike enum representation)
 - For setting the Cartesian grid:

   - Number of pixels per side of the grid (``n_pixels``) and
   - size, in Angstroms, of each pixel (``pixel_size``).
   - Grids and pixels are assumed to be square.
   - These are the only required fields--the rest will be set to defaults if not provided.

 - For setting the polar grid:

   - number of points per shell (``n_inplanes``)
   - ``resolution_factor`` in [0,1] for deteriming the maximum number of frequency radii in the polar grid. 
   If not specified, a ``resolution_factor`` of 1 will be used, which takes the maximum frequency radii up to Nyqst.

 - For Template generation:
  
    - A ``viewing distance``, to compute the viewing angles to use for 3D-to-2D projection
    - ``atomic radii`` (a scalar value, in Angstrom)
    - ``atom selection`` (string)
    - ``atom shape`` (hard-shell or Gaussian)
    - whether to use the default ``protein residue model``

 - For the outputs:
  [TO DO:]
    - output folder
    - output name
