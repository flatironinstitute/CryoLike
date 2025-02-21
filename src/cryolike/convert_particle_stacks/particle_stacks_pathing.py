from os import path, makedirs
from typing import Literal, NamedTuple


class OutputFilenames(NamedTuple):
    phys_stack: str
    fourier_stack: str
    params_filename: str


class OutputFolders():
    folder_output_plots: str
    folder_output_particles_fft: str
    folder_output_particles_phys: str

    def __init__(self, folder_output: str):
        self.folder_output_plots = path.join(folder_output, 'plots')
        makedirs(self.folder_output_plots, exist_ok=True)
        self.folder_output_particles_fft = path.join(folder_output, "fft")
        makedirs(self.folder_output_particles_fft, exist_ok=True)
        self.folder_output_particles_phys = path.join(folder_output, "phys")
        makedirs(self.folder_output_particles_phys, exist_ok=True)


    def get_output_filenames(self, i_stack: int) -> OutputFilenames:
        phys_stack = path.join(self.folder_output_particles_phys, f"particles_phys_stack_{i_stack:06}.pt")
        fourier_base = path.join(self.folder_output_particles_fft, f"particles_fourier_stack_{i_stack:06}")
        fourier_stack = fourier_base + ".pt"
        params_fn = fourier_base + ".npz"
        return OutputFilenames(phys_stack, fourier_stack, params_fn)


class JobPaths():
    file_cs: str
    folder_type: Literal["restacks"] | Literal["downsample"]
    restacks_folder: str
    downsample_folder: str


    def __init__(self, folder_cryosparc: str, job_number: int):
        folder_job = path.join(folder_cryosparc, "J%d" % job_number)
        if not path.exists(folder_job):
            raise ValueError("Error: folder not found: ", folder_job)
        self.restacks_folder = path.join(folder_job, "restack")
        self.downsample_folder = path.join(folder_job, "downsample")
        if not path.exists(self.restacks_folder) and not path.exists(self.downsample_folder):
            raise ValueError("Error: folder not found: ", self.restacks_folder, " and ", self.downsample_folder)
        self.folder_type = "restacks" if path.exists(self.restacks_folder) else "downsample"
        self.file_cs = path.join(folder_job, "J%d_passthrough_particles.cs" % job_number)
        

    def get_mrc_filename(self, i_file: int):
        # TODO: Might want to look for them in both padded and unpadded versions
        if self.folder_type == "restacks":
            mrc_path = path.join(self.restacks_folder, f"batch_{i_file}_restacked.mrc")
        elif self.folder_type == "downsample":
            mrc_path = path.join(self.downsample_folder, f"batch_{i_file:06}_downsample.mrc")
        else:
            raise NotImplementedError("Impossible value for paths folder type.")
        if not path.exists(mrc_path):
            print(f"Looked for file {mrc_path} but it does not exist, breaking the loop and continuing to the next step...")
            return None
        return mrc_path

