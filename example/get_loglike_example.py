import os, sys
import torch
import numpy as np
from pathlib import Path


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# number of templates
numtemp = int(sys.argv[1])

# Define the base directory (adjust this to your actual base directory)
base_dir = Path("./output/likelihood/")

# Initialize a list to store the concatenated results for each template folder
all_templates_concat = []

for x in range(0, numtemp):
    template_dir = base_dir / f"template{x}" / "log_likelihood"

    files_concat = []

    for y in range(1):
        file_path = os.path.join(template_dir, f"log_likelihood_integrated_fourier_stack_00000{y}.pt")

        # Load the .pt file (assuming each file contains a tensor)
        tensor = torch.load(file_path)

        # Append the tensor to the list
        files_concat.append(tensor)


    concatenated = torch.cat(files_concat, dim=0)

    # Convert to NumPy and append the result to the final list
    concatenated_numpy = concatenated.cpu().numpy()
    all_templates_concat.append(concatenated_numpy)

# Convert the final list of arrays into a single array (concatenating along rows)
final_result_numpy = np.vstack(all_templates_concat)

# Save the logLike to a text file
np.savetxt("LogLikeMat.txt", final_result_numpy.T)

