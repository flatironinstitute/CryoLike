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
density map or atomic model, the imaging parameters perviously set
(see :doc:`/examples/set_image_parameters`), and
the output folder. For a brief overview of the possible parameters,
please see the linked API documentation.

Currently, the user is advised to make sure the number of voxels and
voxel size agree with the cryo-EM data to avoid inconsistent results.

The atomic radii and atom selection are only relevant for PDB inputs,
and are optional. If atomic radii is unspecified, the function will
use a standard set of atomic radii for each amino acid type.
If no atom selection is specified, the function will use
all atoms in the atomic model.

Example usage:

.. literalinclude:: ../../../example/make_templates_from_inputs_example.py
    :language: python
