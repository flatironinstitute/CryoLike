Parameter-setting example
==========================

CryoLike requires setting various parameters for image conversion,
template creation, and likelihood computation. This example
demontrates how to set these parameters using a script, which
will write the collection of parameters to the file system for
reuse in other functions. The parameters are collected and held
in an **TODO:ADD METADATA PY LINK** ``ImageDescriptor`` object.

The imaging parameters include the number of voxels and voxel size
in the template model, the precision for computations, a
description of the angles from which to view the template
model and the number of inplane rotations to consider for
each projection. For PDB models, we also accept the atomic
radii and selection of atoms to include. These are optional;
if not specified, amino-acid-specific defaults will be used,
and all atoms in the model will be included.

Currently, the user is advised to make sure the number of
voxels and voxel size agrees with the cryo-EM data to avoid
inconsistent results.

.. literalinclude:: ../../../example/set_image_parameters_example.py
    :language: python
