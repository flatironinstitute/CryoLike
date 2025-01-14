Make templates from cryo-EM maps or atomic models
========================================================

In this tutorial, we will show how to make templates from cryo-EM density maps or atomic models with CryoLike.

The wrapper function for making templates is ``make_templates_from_inputs`` in the ``cryolike.stacks`` module. It takes a list of inputs for the cryo-EM density map or atomic model, the imaging parametersm, and the output folder location as arguments.
The input for a density map can be filename of the cryo-EM density map in MRC format, or a 3-D numpy array or a 3-D torch tensor.
The input for an atomic model can be a filename of the atomic model in PDB format.
The imaging parameters include the number of voxels, voxel size, precision, viewing distance, number of inplanes, atom radii, atom selection, and whether to use the default protein residue model for the atomic radii for each amino acid type.
Currently, the user is advised to make sure the number of voxels and voxel size agrees with the cryo-EM data to avoid inconsistent results.
The atomic radii and atom selection are optional. If the user does not specify the atomic radii, the function will use the default atomic radii for each amino acid type.
If the user does not specify the atom selection, the function will use all atoms in the atomic model.
The output folder is where the templates will be saved.
The function will generate templates for each input and save them in the specified output folder.

If we look at the example code:

.. code-block:: python

    import numpy as np
    from cryolike.stacks import make_templates_from_inputs

    list_of_inputs = [
        "data/pdb/apoferritin_ca_apo_sym.pdb",
        "data/pdb/apoferritin_ca_apo_def.pdb",
        "data/map/emd_2788.map", ## just to show that we can use a map as input
    ]
    make_templates_from_inputs(
        list_of_inputs = list_of_inputs,
        n_voxels = 132,
        voxel_size = 1.346,
        resolution_factor = 1.0,
        precision = 'single', # 'single' or 'double'
        viewing_distance = 8.0 / (4.0 * np.pi),
        n_inplanes =  256,
        # atom_radii = 3.0,
        # atom_selection = "name CA",
        use_protein_residue_model = True,
        atom_shape = 'hard-sphere',#'gaussian',#
        folder_output = "./output/templates/",
        verbose = True
    )

The parameters are as follows:
    - `list_of_inputs`: list of inputs for the cryo-EM density map or atomic model
    - `n_voxels`: number of voxels in each dimension of the template
    - `voxel_size`: size of each voxel in Angstroms
    - `precision`: precision of the template, either 'single' or 'double'
    - `viewing_distance`: angular distance between two viewing angles of the template in radians
    - `n_inplanes`: number of inplanes in the template
    - `atom_radii`: radius of each atom in Angstroms
    - `atom_selection`: selection of atoms to include in the template
    - `use_protein_residue_model`: boolean indicating whether to use the default protein residue model for the atomic radii for each amino acid type
    - `atom_shape`: shape of the atoms in the template, either 'hard-sphere' or 'gaussian'
    - `folder_output`: folder where the templates will be saved
    - `verbose`: boolean indicating whether to print progress messages

.. note: explain every possible ways of generating the templates