Parameter-setting example
==========================

CryoLike requires setting various parameters for image conversion,
template creation, and likelihood computation. This example
demonstrates how to set these parameters using a script.

The imaging parameters include the number of voxels, voxel size,
precision, viewing distance, number of inplanes, atom radii,
atom selection, and whether to use the default protein residue
model for the atomic radii for each amino acid type.

Currently, the user is advised to make sure the number of
voxels and voxel size agrees with the cryo-EM data to avoid
inconsistent results.

The atomic radii and atom selection are optional. If the user
does not specify the atomic radii, the function will use the
default atomic radii for each amino acid type.

If the user does not specify the atom selection, the function
will use all atoms in the atomic model.

.. literalinclude:: ../../../example/set_image_parameters_example.py
    :language: python
