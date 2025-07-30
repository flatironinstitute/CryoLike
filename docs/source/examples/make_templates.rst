Make templates from cryo-EM maps or atomic models
========================================================

This script demonstrates making templates from
cryo-EM density maps or atomic models with CryoLike.

The wrapper function for making templates is
:py:func:`make_templates_from_inputs
<cryolike.stacks.make_templates_from_inputs_api.make_templates_from_inputs>`
in the :py:mod:`cryolike.stacks
<cryolike.stacks.make_templates_from_inputs_api>` module.
The function takes a list of inputs for the cryo-EM
density map or atomic model, the imaging parameters perviously set (see :doc:`/examples/set_image_parameters`), and
the output folder. For a brief overview of the possible parameters,
please see the linked API documentation.

.. - The input for a 3D map can be the filename of the cryo-EM density map in
..   MRC format, or a 3-D numpy array or a 3-D torch tensor.
.. - The input for an atomic model can be a filename of the atomic model in
..   PDB format.
.. - The imaging parameters include:

..   -  the number of voxels
..   -  voxel size
..   -  precision,
..   -  viewing distance
..   -  number of inplane rotations
..   -  atom radii
..   -  atom selection
..   -  whether to use the default protein residue model
..      for the atomic radii for each amino acid type

Currently, the user is advised to make sure the number of voxels and
voxel size agree with the cryo-EM data to avoid inconsistent results.

The atomic radii and atom selection are only relevant for PDB inputs,
and are optional. If atomic radii is unspecified, the function will
use a standard set of atomic radii for each amino acid type.
If no atom selection is specified, the function will use
all atoms in the atomic model.

.. The output folder is where the templates will be saved.
.. The function will generate templates for each input and save them in
.. the specified output folder.

Example usage:

.. literalinclude:: ../../../example/make_templates_from_inputs_example.py
    :language: python

.. The parameters are as follows:

.. - `list_of_inputs`: list of inputs for the cryo-EM density
..   maps or atomic models
.. - `n_voxels`: number of voxels in each dimension of the template
.. - `voxel_size`: size of each voxel in Angstroms
.. - `precision`: precision of the template, either 'single'
..   or 'double'
.. - `viewing_distance`: angular distance between two viewing
..   angles of the template in radians
.. - `n_inplanes`: number of inplanes in the template
.. - `atom_radii`: radius of each atom in Angstroms
.. - `atom_selection`: selection of atoms to include in
..   the template
.. - `use_protein_residue_model`: boolean indicating whether to use
..   the default protein residue model for the atomic radii for each
..   amino acid type
.. - `atom_shape`: shape of the atoms in the template, either 'hard-sphere'
..   or 'gaussian'
.. - `folder_output`: folder where the templates will be saved
.. - `verbose`: boolean indicating whether to print progress messages

.. note: explain every possible ways of generating the templates
