import glob
from numpy import load
from pathlib import Path
from typing import NamedTuple

from .run_likelihood_file_mgmt import LikelihoodFileManager
from .file_base import make_dir, check_files_exist
from cryolike.util import OutputConfiguration


class PostProcessOutputTree(NamedTuple):
    FourierMatrix: Path
    PhysMatrix: Path
    IntegratedMatrix: Path
    CrossCorrelationMatrix: Path


class PostProcessSources(NamedTuple):
    FourierStacks: list[Path]
    PhysStacks: list[Path]
    IntegratedStacks: list[Path]
    CrossCorrelationStacks: list[Path]


class PostProcessFileManager():
    _output_directory: Path
    _batch_directory: Path
    _template_root: Path
    _particles_root: Path
    _output_matrix_root: Path


    def __init__(self,
        output_directory: str,
        batch_directory: str = '',
        template_directory: str = '',
        particles_directory: str = ''
    ):
        self._output_directory = Path(output_directory)
        self._output_matrix_root = self._output_directory / 'likelihood_matrix'
        make_dir(self._output_matrix_root, '')

        if len(batch_directory) > 0:
            self._batch_directory = Path(batch_directory)
        else:
            self._batch_directory = self._output_directory / 'likelihood'
        if len(template_directory) > 0:
            self._template_root = Path(template_directory)
        else:
            self._template_root = self._output_directory / 'templates'
        if len(particles_directory) > 0:
            self._particles_root = Path(particles_directory)
        else:
            self._particles_root = self._output_directory / 'particles'


    def confirm_counts(self,
        n_templates: int = 0,
        n_image_stacks: int = 0
    ):
        if n_templates > 0 and n_image_stacks > 0:
            return n_templates, n_image_stacks

        if n_templates <= 0:
            n_templates = len(load(self._template_root / 'template_file_list.npy'))
        if n_image_stacks <= 0:
            n_image_stacks = len(glob.glob(str(self._particles_root / 'phys/*')))

        return n_templates, n_image_stacks


    def get_source_lists(self,
        n_templates: int = 0,
        n_image_stacks: int = 0,
        phys: bool = False,
        opt: bool = False,
        integrated: bool = False,
        cc: bool = False
    ):
        if n_templates < 1 or n_image_stacks < 1:
            raise ValueError("Number of templates/stacks must be set to positive values.")

        # I think this is how these match up
        # NOTE: It might've been better to just return everything...
        config = OutputConfiguration(
            return_likelihood_integrated_pose_fourier=integrated,
            return_likelihood_optimal_pose_physical=phys,
            return_likelihood_optimal_pose_fourier=opt,
            return_optimal_pose=cc,
        )

        opt_fourier_list = []
        phys_list = []
        int_fourier_list = []
        cc_list = []
        for i_t in range(n_templates):
            mgr = LikelihoodFileManager(
                folder_output=str(self._batch_directory),
                folder_templates=str(self._template_root),
                folder_particles=str(self._particles_root),
                i_template=i_t,
                dry_run=True
            )
            for i_s in range(n_image_stacks):
                filenames = mgr._get_output_filenames(i_s, config)
                if filenames.optimal_fourier_pose_likelihood_file is not None:
                    opt_fourier_list.append(filenames.optimal_fourier_pose_likelihood_file)
                if filenames.optimal_phys_pose_likelihood_file is not None:
                    phys_list.append(filenames.optimal_phys_pose_likelihood_file)
                if filenames.integrated_pose_file is not None:
                    int_fourier_list.append(filenames.integrated_pose_file)
                if filenames.cross_corr_file is not None:
                    cc_list.append(filenames.cross_corr_file)
        
        for x, name in [(opt_fourier_list, 'opt_fourier'),
                        (phys_list, 'phys'),
                        (int_fourier_list, 'int'),
                        (cc_list, 'cc')]:
            all_exist, missings = check_files_exist(x)
            if all_exist:
                continue
            missings_f = "\n\t".join(missings)
            print(f"Files missing from the {name} list:\n{missings_f}")

        return PostProcessSources(
            FourierStacks=opt_fourier_list,
            PhysStacks=phys_list,
            IntegratedStacks=int_fourier_list,
            CrossCorrelationStacks=cc_list
        )


    def get_output_targets(self) -> PostProcessOutputTree:
        r = self._output_matrix_root
        four  = r / 'optimal_fourier_log_likelihood_matrix.pt'
        phys  = r / 'optimal_physical_log_likelihood_matrix.pt'
        integ = r / 'integrated_fourier_log_likelihood_matrix.pt'
        cc    = r / 'cross_correlation_matrix.pt'

        return PostProcessOutputTree(
            FourierMatrix= four,
            PhysMatrix = phys,
            IntegratedMatrix=integ,
            CrossCorrelationMatrix=cc
        )
