Make Templates from Inputs
==========================

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
