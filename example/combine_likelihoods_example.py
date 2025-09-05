import torch
import numpy as np
from pathlib import Path

# NOTE: A similar effect can be achieved using the
# cryolike.util.stitch_log_likelihood_matrices() function.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_templates = 2

# this matches the folder_output value used in the run_likelihood example
base_output_folder = Path("./output/likelihood/")

# Initialize a list to store the concatenated results for each template folder
all_templates_np = []

for x in range(0, num_templates):
    template_dir = base_output_folder / f"template{x}" / "log_likelihood"
    likelihood_file = template_dir / "log_likelihood_integrated_fourier_stack_000000.pt"
    likelihood_tensor = torch.load(likelihood_file)
    # convert likelihood tensor to numpy and append to final list
    all_templates_np.append(likelihood_tensor.cpu().numpy())
    
# Convert the final list of arrays into a single array (concatenating along rows)
final_result_numpy = np.vstack(all_templates_np)

# Save the collected log likelihood to a text file
np.savetxt("./output/LogLikeMat.txt", final_result_numpy.T)
