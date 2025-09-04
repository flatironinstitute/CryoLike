import os

from cryolike import make_templates_from_inputs

# Assume you have already run set_image_parameters_example with the same
# value of folder_output.
folder_output = './output/templates/'
image_parameters_filename = os.path.join(folder_output, "parameters.npz")
list_of_inputs = [
    # density,
    # density_torch,
    # "data/map/emd_2788.map",
    "data/pdb/apoferritin_ca_apo_sym.pdb",
    "data/pdb/apoferritin_ca_apo_def.pdb"
]
verbose = True

make_templates_from_inputs(
    list_of_inputs = list_of_inputs,
    image_parameters_file=image_parameters_filename,
    folder_output = folder_output,
    verbose = verbose
)
