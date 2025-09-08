from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING
from torch import load, save, stack, Tensor
from numpy import load as np_load

if TYPE_CHECKING: # pragma: no cover
    from cryolike.likelihoods import OptimalPoseReturn, CrossCorrelationReturn

from .file_base import make_dir, get_input_filename, FILE_TYPES, check_files_exist
from cryolike.stacks.template import Templates
from cryolike.stacks.image import Images
from cryolike.microscopy import CTF
from cryolike.grids import FourierImages, PhysicalImages
from cryolike.util import OutputConfiguration, Precision
from cryolike.metadata import ImageDescriptor, load_combined_params

class LikelihoodOutputFiles(NamedTuple):
    # cross correlation pose
    cross_corr_pose_file: Path | None
    # integrated pose (fourier)
    integrated_pose_file: Path | None
    # optimal pose (f)
    optimal_fourier_pose_likelihood_file: Path | None
    # optimal pose (p)
    optimal_phys_pose_likelihood_file: Path | None
    # optimal pose
    cross_corr_file: Path | None
    template_indices_file: Path | None
    x_displacement_file: Path | None
    y_displacement_file: Path | None
    inplane_rotation_file: Path | None


@dataclass
class LikelihoodOutputDataSources():
    full_pose: CrossCorrelationReturn | None = None
    optimal_pose: OptimalPoseReturn | None = None
    ll_fourier_integrated: Tensor | None = None
    ll_optimal_fourier_pose: Tensor | None = None
    ll_optimal_phys_pose: Tensor | None = None


class LikelihoodFileManager():
    _template_src: Path
    _particles_fft_src: Path
    _particles_phys_src: Path
    _output_base: Path
    _output_log_likelihood: Path
    _output_cross_correlation: Path
    _output_optimal_pose: Path
    _displacements_saved: bool

    def __init__(self,
        folder_output: str,
        folder_templates: str,
        folder_particles: str,
        n_stacks_to_process: int = 1,
        i_template: int = 0,
        phys_needed: bool = False,
        dry_run: bool = False
    ):
        self._template_src = Path(folder_templates)
        _particles = Path(folder_particles)
        self._particles_fft_src = _particles / 'fft'
        self._particles_phys_src = _particles / 'phys'
        if not dry_run:
            self._check_input_files_exist(n_stacks_to_process, phys_needed)
        self._make_tree(folder_output, i_template, dry_run)
        self._displacements_saved = False


    def _make_tree(self, folder_output: str, i_template: int, dry_run: bool = False):
        self._output_base = Path(folder_output)
        _output_template = self._output_base / f"template{i_template}"
        self._output_log_likelihood = _output_template / "log_likelihood"
        self._output_cross_correlation = _output_template / "cross_correlation"
        self._output_optimal_pose = _output_template / "optimal_pose"

        if not dry_run:
            for x in [
                self._output_log_likelihood,
                self._output_cross_correlation,
                self._output_optimal_pose
            ]:
                make_dir(x, '')


    def _get_input_filename(self, i_stack: int, type: FILE_TYPES) -> Path:
        if type == 'phys':
            return get_input_filename(self._particles_phys_src, i_stack, type)
        return get_input_filename(self._particles_fft_src, i_stack, type)


    def _get_template_filename(self) -> Path:
        return self._template_src / 'template_file_list.npy'


    def _check_input_files_exist(self, n_stacks: int, phys_needed: bool):
        file_paths = []
        for i_stack in range(n_stacks):
            file_paths.append(self._get_input_filename(i_stack, 'fourier'))
            file_paths.append(self._get_input_filename(i_stack, 'params'))
            if phys_needed:
                phys_file = self._get_input_filename(i_stack, 'phys')
                file_paths.append(phys_file)
        (all_exist, misses) = check_files_exist(file_paths)
        if not all_exist:
            errs = "\n\t".join(misses)
            raise ValueError(f'Files not found:\n{errs}')


    def _get_output_filenames(self, i_stack: int, types: OutputConfiguration) -> LikelihoodOutputFiles:
        i_ext = f"stack_{i_stack:06}.pt"
        
        integrated = self._output_log_likelihood / f"log_likelihood_integrated_fourier_{i_ext}"
        opt_f_pose_ll = self._output_log_likelihood / f"log_likelihood_optimal_fourier_{i_ext}"
        opt_p_pose_ll = self._output_log_likelihood / f"log_likelihood_optimal_physical_{i_ext}"
        xcorr = self._output_cross_correlation / f"cross_correlation_{i_ext}"
        xcorr_pose = self._output_cross_correlation / f"cross_correlation_pose_msdw_{i_ext}"
        template_indices = self._output_optimal_pose / f"optimal_template_{i_ext}"
        disp_x = self._output_optimal_pose / f"optimal_displacement_x_{i_ext}"
        disp_y = self._output_optimal_pose / f"optimal_displacement_y_{i_ext}"
        inplane_rotation = self._output_optimal_pose / f"optimal_inplane_rotation_{i_ext}"

        if types.cross_correlation_pose:
            # cross correlation pose (the full msdw tensor) is mutually exclusive with all other
            # return types
            return LikelihoodOutputFiles(xcorr_pose, None, None, None, None, None, None, None, None)

        return LikelihoodOutputFiles(
            cross_corr_pose_file=None,
            integrated_pose_file=integrated if types.integrated_likelihood_fourier else None,
            optimal_fourier_pose_likelihood_file=opt_f_pose_ll if types.optimal_fourier_pose_likelihood else None,
            optimal_phys_pose_likelihood_file=opt_p_pose_ll if types.optimal_phys_pose_likelihood else None,
            cross_corr_file=xcorr if types.optimal_pose else None,
            template_indices_file=template_indices if types.optimal_pose else None,
            x_displacement_file=disp_x if types.optimal_pose else None,
            y_displacement_file=disp_y if types.optimal_pose else None,
            inplane_rotation_file=inplane_rotation if types.optimal_pose else None,
        )


    def load_template(self, params_input: str | ImageDescriptor, i_template: int):
        image_desc = ImageDescriptor.ensure(params_input)
        (torch_float_type, _, _) = image_desc.precision.get_dtypes(default=Precision.SINGLE)

        template_file_list = np_load(self._get_template_filename(), allow_pickle = True)
        template_file = template_file_list[i_template]
        print("template_file: ", template_file)

        templates_fourier = load(template_file, weights_only=True)
        fourier_templates_data = FourierImages(templates_fourier, image_desc.polar_grid)
        tp = Templates(
            fourier_data = fourier_templates_data,
            phys_data = image_desc.cartesian_grid,
            viewing_angles = image_desc.viewing_angles
        )
        return (tp, image_desc, torch_float_type)


    def load_img_stack(self, i_stack: int, image_desc: ImageDescriptor):
        print("stack number: ", i_stack)
        image_fourier_file = self._get_input_filename(i_stack, 'fourier')
        image_param_file = self._get_input_filename(i_stack, 'params')
        images_fourier = load(image_fourier_file, weights_only=True)
        (stack_img_desc, stack_lens_desc) = load_combined_params(image_param_file)
        if not image_desc.is_compatible_with(stack_img_desc):
            raise ValueError("Incompatible image parameters")

        fourier_images = FourierImages(images_fourier, stack_img_desc.polar_grid)
        im = Images(fourier_data=fourier_images, phys_data=stack_img_desc.cartesian_grid)
        ctf = CTF(
            ctf_descriptor=stack_lens_desc,
            polar_grid = stack_img_desc.polar_grid,
            box_size = stack_img_desc.cartesian_grid.box_size[0],
            anisotropy = True
        )

        return (im, ctf)

    
    def load_phys_stack(self, i_stack: int, image_desc: ImageDescriptor):
        images_phys = load(self._get_input_filename(i_stack, 'phys'), weights_only=True)
        phys_image_data = PhysicalImages(images_phys, pixel_size=image_desc.cartesian_grid.pixel_size)
        im_phys = Images(phys_data=phys_image_data, fourier_data=None)
        return im_phys


    def save_displacements(self, displacement_grid: Tensor):
        """Persists the realized set of displacements used in cross-correlation to a
        predictable location. Ensures this is only done once per call to run_likelihood.

        Args:
            displacements (Tensor): Realized grid of displacements, in Angstrom.
                This can be imported directly from the Images/Templates object,
                and should be indexed as [[x or y], [list-of-displacements]]
                i.e. its shape should be (2, n_displacements).
        """
        if self._displacements_saved:
            return
        
        displacements_filename = self._output_base / 'displacements_set.pt'
        displacements = displacement_grid.cpu().numpy()
        save(displacements, displacements_filename)
        self._displacements_saved = True


    def outputs_exist(self,
        i_stack: int,
        files: OutputConfiguration
    ) -> bool:
        fns = self._get_output_filenames(i_stack, files)
        expected_files = [x for x in fns if x is not None]
        (all_exist, _) = check_files_exist(expected_files)

        return all_exist


    def write_outputs(self,
        i_stack: int,
        outs: OutputConfiguration,
        out_data: LikelihoodOutputDataSources
    ):
        out_fns = self._get_output_filenames(i_stack, outs)
        if (outs.cross_correlation_pose):
            assert out_fns.cross_corr_pose_file is not None
            assert out_data.full_pose is not None
            save(out_data.full_pose, out_fns.cross_corr_pose_file)
            # cross-correlation-pose is defined as mutex with other outputs
            return
        if (outs.integrated_likelihood_fourier):
            assert out_fns.integrated_pose_file is not None
            assert out_data.ll_fourier_integrated is not None
            save(out_data.ll_fourier_integrated, out_fns.integrated_pose_file)
        if (outs.optimal_fourier_pose_likelihood):
            assert out_fns.optimal_fourier_pose_likelihood_file is not None
            assert out_data.ll_optimal_fourier_pose is not None
            save(out_data.ll_optimal_fourier_pose, out_fns.optimal_fourier_pose_likelihood_file)
        if (outs.optimal_phys_pose_likelihood):
            assert out_fns.optimal_phys_pose_likelihood_file is not None
            assert out_data.ll_optimal_phys_pose is not None
            save(out_data.ll_optimal_phys_pose, out_fns.optimal_phys_pose_likelihood_file)
        if (outs.optimal_pose):
            assert out_data.optimal_pose is not None
            assert out_fns.cross_corr_file is not None
            assert out_fns.template_indices_file is not None
            assert out_fns.x_displacement_file is not None
            assert out_fns.y_displacement_file is not None
            assert out_fns.inplane_rotation_file is not None
            save(out_data.optimal_pose.cross_correlation_M, out_fns.cross_corr_file)
            save(out_data.optimal_pose.optimal_template_M, out_fns.template_indices_file)
            save(out_data.optimal_pose.optimal_displacement_x_M, out_fns.x_displacement_file)
            save(out_data.optimal_pose.optimal_displacement_y_M, out_fns.y_displacement_file)
            save(out_data.optimal_pose.optimal_inplane_rotation_M, out_fns.inplane_rotation_file)
