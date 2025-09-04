import os
from numpy import pi

from cryolike import ImageDescriptor

verbose = True

if verbose:
    print("Setting parameters...")
n_voxels = 132
voxel_size = 1.346
precision = 'single' # 'single' or 'double'
viewing_distance = 8.0 / (4.0 * pi)
n_inplanes = 256
# atom_radii = 3.0
atom_selection = "name CA"
folder_output = './output/templates/'
os.makedirs(folder_output, exist_ok=True)

image_parameters_filename = os.path.join(folder_output, "parameters.npz")
image_parameters = ImageDescriptor.from_individual_values(
    n_pixels = n_voxels,
    pixel_size = voxel_size,
    resolution_factor = 1.0,
    precision = precision,
    viewing_distance = viewing_distance,
    n_inplanes = n_inplanes,
    use_protein_residue_model = True,
    atom_shape = 'hard-sphere' # or 'gaussian'
)
if verbose:
    image_parameters.print()
image_parameters.save(image_parameters_filename)
