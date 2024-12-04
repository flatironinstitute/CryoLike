import numpy as np
from cryolike.stacks.make_templates_from_inputs import make_templates_from_inputs

list_of_inputs = [
    "data/pdb/apoferritin_ca_apo_sym.pdb",
    "data/pdb/apoferritin_ca_apo_def.pdb",
    "data/map/emd_2788.map", ## just to show that we can use a map as input
]

print("Setting parameters...")
n_voxels = 132
voxel_size = 1.346
precision = 'single' # 'single' or 'double'
viewing_distance = 8.0 / (4.0 * np.pi)
n_inplanes = 256
atom_radii = 3.0
atom_selection = "name CA"
make_templates_from_inputs(
    list_of_inputs = list_of_inputs,
    n_voxels = n_voxels,
    voxel_size = voxel_size,
    resolution_factor = 1.0,
    precision = precision,
    viewing_distance = viewing_distance,
    n_inplanes =  n_inplanes,
    atom_radii = atom_radii,
    atom_selection = atom_selection,
    use_protein_residue_model = True, # False,
    atom_shape = 'hard-sphere', # 'gaussian',
    folder_output = "./output/templates/",
    verbose = True
)