Make templates from cryo-EM maps or atomic models
========================================================

In this tutorial, we will show how to make templates from cryo-EM density maps or atomic models with CryoLike.

The wrapper function for making templates is ``make_templates_from_inputs`` in the ``cryolike.stacks`` module. 
It takes a list of inputs for the cryo-EM density map or atomic model, the imaging parameters, and the output folder location as arguments.
The input for a 3D map can be the filename of the cryo-EM density map in MRC format, or a 3-D numpy array or a 3-D torch tensor.
The input for an atomic model can be a filename of the atomic model in PDB format.
The imaging parameters include the number of voxels, voxel size, precision, viewing distance, number of inplanes, 
atom radii, atom selection, and whether to use the default protein residue model for the atomic radii for each amino acid type.
Currently, the user is advised to make sure the number of voxels and voxel size agrees with the cryo-EM data to avoid inconsistent results.

The atomic radii and atom selection are optional. If the user does not specify the atomic radii, the function will use the default atomic radii for each amino acid type.
If the user does not specify the atom selection, the function will use all atoms in the atomic model.
The output folder is where the templates will be saved.
The function will generate templates for each input and save them in the specified output folder.

If we look at the example code:

[TO DO: This changed]

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
    - `list_of_inputs`: list of inputs for the cryo-EM density maps or atomic models
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

.. code-block:: python

   """
   This script demonstrates the usage of the `make_templates_from_inputs` function from the `cryolike.stacks.make_templates_from_inputs_api` module.
   
   The `make_templates_from_inputs` function generates templates from a list of input files, such as PDB files and EM maps. It sets various parameters for template generation, including the number of voxels, voxel size, precision, viewing distance, number of inplanes, atom radii, atom selection, and more.
   
   Parameters:
      - `list_of_inputs` (list): A list of input file paths.
      - `n_voxels` (int): The number of voxels for template generation.
      - `voxel_size` (float): The size of each voxel.
      - `resolution_factor` (float): The resolution factor for template generation.
      - `precision` (str): The precision for template generation, either 'single' or 'double'.
      - `viewing_distance` (float): The viewing distance for template generation.
      - `n_inplanes` (int): The number of inplanes for template generation.
      - `atom_radii` (float): The atom radii for template generation.
      - `atom_selection` (str): The atom selection for template generation.
      - `use_protein_residue_model` (bool): Whether to use the protein residue model or not.
      - `atom_shape` (str): The atom shape for template generation, either 'hard-sphere' or 'gaussian'.
      - `folder_output` (str): The output folder for saving the generated templates.
      - `verbose` (bool): Whether to print verbose output or not.
   
   Returns:
      None
   
   Example usage:


  python3 plot_example.py [LINK TO EXAMPLES]
   
   .. code-block:: python
   
      import numpy as np
      from cryolike.stacks.make_templates_from_inputs_api import make_templates_from_inputs
      
      list_of_inputs = [
         "data/pdb/apoferritin_ca_apo_sym.pdb",
         "data/pdb/apoferritin_ca_apo_def.pdb",
         "data/map/emd_2788.map",
      ]
      
      print("Setting parameters...")
      n_voxels = 132
      voxel_size = 1.346
      precision = 'single'
      viewing_distance = 8.0 / (4.0 * np.pi)
      n_inplanes = 256
      atom_radii = 3.0
      atom_selection = "name CA"
      
      make_templates_from_inputs(
         list_of_inputs=list_of_inputs,
         n_voxels=n_voxels,
         voxel_size=voxel_size,
         resolution_factor=1.0,
         precision=precision,
         viewing_distance=viewing_distance,
         n_inplanes=n_inplanes,
         atom_radii=atom_radii,
         atom_selection=atom_selection,
         use_protein_residue_model=True,
         atom_shape='hard-sphere',
         folder_output="./output/templates/",
         verbose=True
      )
   """
