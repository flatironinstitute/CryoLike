from torch import cuda, device, Tensor, cuda
from typing import Callable, Literal

from cryolike.file_mgmt import LikelihoodFileManager, LikelihoodOutputDataSources
from cryolike.stacks import Templates, Images
from cryolike.microscopy import CTF
from cryolike.likelihoods.likelihood import calc_likelihood_optimal_pose
from cryolike.likelihoods import (
    OptimalPoseReturn,
    compute_optimal_pose,
    compute_cross_correlation_complete,
    template_first_comparator
)
from cryolike.metadata import ImageDescriptor
from cryolike.util import  OutputConfiguration, Precision


def configure_likelihood_files(
    folder_templates: str,
    folder_particles: str,
    folder_output: str = '',
    n_stacks: int = 1,
    i_template: int = 0,
    return_likelihood_optimal_pose_physical : bool = False,
) -> LikelihoodFileManager:
    if return_likelihood_optimal_pose_physical:
        raise NotImplementedError("Physical likelihood is still under development and not yet available. Please use Fourier likelihood instead.")

    return LikelihoodFileManager(
        folder_output,
        folder_templates,
        folder_particles,
        n_stacks,
        i_template,
        return_likelihood_optimal_pose_physical
    )


displacement_configurator_T = Callable[[Templates], None]
def configure_displacement(
    max_displacement_pixels: float = 8.0,
    n_displacements_x: int = -1,
    n_displacements_y: int = -1,
) -> displacement_configurator_T:
    def template_grid_setter(tp: Templates):
        tp.set_displacement_grid(
            max_displacement_pixels,
            n_displacements_x,
            n_displacements_y
        )
    return template_grid_setter


def run_likelihood_optimal_pose(
    file_config: LikelihoodFileManager,
    params_input: str | ImageDescriptor,
    displacer: displacement_configurator_T,
    template_index: int = 0,
    n_stacks: int = 1,
    skip_exist: bool = False,
    n_templates_per_batch: int = 1,
    n_images_per_batch: int = 128,
    estimate_batch_size: bool = False,
    return_likelihood_optimal_pose_fourier: bool = False,
    return_likelihood_integrated_pose_fourier: bool = False
):
    """Function to run cross-correlation likelihood of a template stack against
    potentially several image stacks meeting the same name convention. It is
    mostly a convenience wrapper around the user-facing functions in
    cryolike.likelihoods.interface, as it automates calling the likelihood
    computation against multiple input files. Note that the API currently
    provided by this function is quite likely to change, as it is only a
    convenience wrapper.

    This version will write files containing the optimal x- and y-displacement
    of each template-image pair, as well as the optimal rotation for each pair.
    Additionally, if return_likelihood_optimal_pose_fourier is set, a file will
    be written containing the per-image log likelihood of the optimal pose; and
    if return_likelihood_integrated_pose_fourier is set, a file will be written
    containing the per-image integrated log likelihood over all poses.

    Args:
        file_config (LikelihoodFileManager): A file manager obtained by
            calling configure_likelihood_files, which handles consistent
            naming on the file system
        params_input (str | ImageDescriptor): An ImageDescriptor or a
            path to where one has been saved on the file system
        displacer (displacement_configurator_T): The result of calling
            configure_displacement (technically a callback)
        template_index (int, optional): The index of the template file,
            among similarly-named template files, to use for comparison.
            Defaults to 0. Note that this refers to a particular *file*
            among similarly-named files; it does not refer to an individual
            template within a template stack (stored in a single file).
        n_stacks (int, optional): How many image stacks to process.
            Defaults to 1.
        skip_exist (bool, optional): Whether to skip processing of any
            image stack which appears (from existing file names) to have
            been processed already. Defaults to False.
        n_templates_per_batch (int, optional): The number of templates to
            try to compare at once in memory. Defaults to 1. Higher numbers
            should result in more efficient computation, particularly on
            GPUs, but may need to be reduced if out-of-memory errors occur.
        n_images_per_batch (int, optional): The number of images to try to
            compare at once in memory. Defaults to 128. Higher numbers should
            result in more efficient computation, particularly on GPUs, but
            may need to be reduced if out-of-memory errors occur.
        estimate_batch_size (bool, optional): Whether to attempt to compute
            an optimal number of templates and images per batch. Defaults to
            False. This functionality is not yet very refined.
        return_likelihood_optimal_pose_fourier (bool, optional): Whether to
            return the optimal Fourier pose (as opposed to the optimal rotation
            and displacement per-template). Defaults to False.
        return_likelihood_integrated_pose_fourier (bool, optional): Whether to
            output the integrated log likelihood (in Fourier space) for each
            image. Defaults to False.
    """
    outputs = OutputConfiguration(
        return_cross_correlation_pose=False,
        return_likelihood_integrated_pose_fourier=return_likelihood_integrated_pose_fourier,
        return_likelihood_optimal_pose_fourier=return_likelihood_optimal_pose_fourier,
        return_likelihood_optimal_pose_physical=False,  # Not yet supported
        return_optimal_pose=True,
        optimized_inplane_rotation=True,
        optimized_displacement=True,
        optimized_viewing_angle=True
    )

    (tp, image_desc, _) = file_config.load_template(params_input, template_index)
    precision = image_desc.precision
    displacer(tp)
    optimal_pose_ll_partial = _get_optimal_pose_log_likelihood_partial(tp, precision)
    file_config.save_displacements(tp.displacement_grid_angstrom)

    for i_stack in range(n_stacks):
        if skip_exist and file_config.outputs_exist(i_stack, outputs):
            # NOTE: this is not a foolproof way to check if the files exist, as the files could be corrupted
            print(f"Skipping stack number: {i_stack} as all output files already exist")
            continue
        (im, ctf) = file_config.load_img_stack(i_stack, image_desc)

        if estimate_batch_size:
            n_templates_per_batch, n_images_per_batch = _compute_batch_sizes(tp, im, precision)

        iterator = template_first_comparator(
            device=device('cuda'),
            images=im,
            templates=tp,
            ctf=ctf,
            n_images_per_batch=n_images_per_batch,
            n_templates_per_batch=n_templates_per_batch,
            return_integrated_likelihood=False,
            precision=Precision.DEFAULT
        )

        out_data = LikelihoodOutputDataSources()
        optimal_pose, log_likelihood_fourier_integrated = compute_optimal_pose(
            iterator,
            tp,
            im,
            precision,
            include_integrated_log_likelihood=True
        )
        out_data.optimal_pose = optimal_pose
        out_data.ll_fourier_integrated = log_likelihood_fourier_integrated
        if (outputs.optimal_fourier_pose_likelihood):
            out_data.ll_optimal_fourier_pose = optimal_pose_ll_partial(
                im,
                ctf,
                optimal_pose,
                "fourier"
            )
        if (outputs.optimal_phys_pose_likelihood):
            raise NotImplementedError
            im_phys = file_config.load_phys_stack(i_stack, image_desc)
            out_data.ll_optimal_phys_pose = optimal_pose_ll_partial(im_phys, ctf, optimal_pose, 'phys')
            del im_phys
        
        file_config.write_outputs(i_stack, outputs, out_data)
        cuda.empty_cache()  # Clear GPU memory after each stack to avoid memory issues


def run_likelihood_full_cross_correlation(
    file_config: LikelihoodFileManager,
    params_input: str | ImageDescriptor,
    displacer: displacement_configurator_T,
    template_index: int = 0,
    n_stacks: int = 1,
    skip_exist: bool = False,
    n_templates_per_batch: int = 1,
    n_images_per_batch: int = 128,
    estimate_batch_size: bool = False,
):
    """Function to run cross-correlation likelihood of a template stack against
    potentially several image stacks meeting the same name convention. It is
    mostly a convenience wrapper around the user-facing functions in
    cryolike.likelihoods.interface, as it automates calling the likelihood
    computation against multiple input files. Note that the API currently
    provided by this function is quite likely to change, as it is only a
    convenience wrapper.
    
    This version will write a single file per image stack,
    containing a 4-tensor showing the cross-correlation likelihood between
    each image and template pair at each displacement and rotation.

    Args:
        file_config (LikelihoodFileManager): A file manager obtained by
            calling configure_likelihood_files, which handles consistent
            naming on the file system
        params_input (str | ImageDescriptor): An ImageDescriptor or a
            path to where one has been saved on the file system
        displacer (displacement_configurator_T): The result of calling
            configure_displacement (technically a callback)
        template_index (int, optional): The index of the template file,
            among similarly-named template files, to use for comparison.
            Defaults to 0. Note that this refers to a particular *file*
            among similarly-named files; it does not refer to an individual
            template within a template stack (stored in a single file).
        n_stacks (int, optional): How many image stacks to process.
            Defaults to 1.
        skip_exist (bool, optional): Whether to skip processing of any
            image stack which appears (from existing file names) to have
            been processed already. Defaults to False.
        n_templates_per_batch (int, optional): The number of templates to
            try to compare at once in memory. Defaults to 1. Higher numbers
            should result in more efficient computation, particularly on
            GPUs, but may need to be reduced if out-of-memory errors occur.
        n_images_per_batch (int, optional): The number of images to try to
            compare at once in memory. Defaults to 128. Higher numbers should
            result in more efficient computation, particularly on GPUs, but
            may need to be reduced if out-of-memory errors occur.
        estimate_batch_size (bool, optional): Whether to attempt to compute
            an optimal number of templates and images per batch. Defaults to
            False. This functionality is not yet very refined.
    """

    outputs = OutputConfiguration(
        return_optimal_pose=False,
        optimized_inplane_rotation=False,
        optimized_displacement=False,
        optimized_viewing_angle=False,
        return_cross_correlation_pose=True
    )

    (tp, image_desc, _) = file_config.load_template(params_input, template_index)
    precision = image_desc.precision
    displacer(tp)
    file_config.save_displacements(tp.displacement_grid_angstrom)

    for i_stack in range(n_stacks):
        if skip_exist and file_config.outputs_exist(i_stack, outputs):
            # NOTE: this is not a foolproof way to check if the files exist, as the files could be corrupted
            print(f"Skipping stack number: {i_stack} as all output files already exist")
            continue
        (im, ctf) = file_config.load_img_stack(i_stack, image_desc)

        if estimate_batch_size:
            n_templates_per_batch, n_images_per_batch = _compute_batch_sizes(tp, im, precision)

        iterator = template_first_comparator(
            device=device('cuda'),
            images=im,
            templates=tp,
            ctf=ctf,
            n_images_per_batch=n_images_per_batch,
            n_templates_per_batch=n_templates_per_batch,
            return_integrated_likelihood=False,
            precision=Precision.DEFAULT
        )
        out_data = LikelihoodOutputDataSources()
        full_cross_correlation_pose = compute_cross_correlation_complete(
            iterator,
            tp,
            im,
            precision,
            False
        )
        out_data.full_pose = full_cross_correlation_pose
        file_config.write_outputs(i_stack, outputs, out_data)
        cuda.empty_cache()  # Clear GPU memory after each stack to avoid memory issues


def _compute_batch_sizes(
    templates: Templates,
    images: Images,
    precision: Precision,
):
    if not cuda.is_available():
        raise ValueError("Requested to estimate batch sizes but no GPU.")

    (torch_float_type, torch_complex_type, _) = Precision.get_dtypes(precision, default=Precision.DOUBLE)
    n_templates = templates.n_images
    n_imgs = images.n_images
    n_shells = templates.polar_grid.n_shells
    n_inplanes = templates.polar_grid.n_inplanes
    n_disp = templates.n_displacements
    
    size_float = torch_float_type.itemsize ## in bytes
    size_complex = torch_complex_type.itemsize ## in bytes

    def memory_usage_batchsize(bs_temp: int, bs_img: int) -> int:
        ## This is a rough estimate of the memory usage for the given batch sizes
        ## Only consider the significant memory usage
        
        _usage = 0
        # _usage += n_shells * n_inplanes * 8192 * size_complex ## aten::clone
        # _usage += n_shells * n_inplanes * 8192 * size_complex ## aten::empty_like
        _usage += bs_temp * n_shells * n_inplanes * size_complex ## sqrtweighted_fourier_templates_mnw
        _usage += bs_img * n_shells * n_inplanes * size_complex ## images_fourier_snw
        _usage += bs_temp * n_disp * n_shells * n_inplanes * size_complex ## sqrtweighted_fourier_templates_bessel_mdnq
        _usage += bs_img * n_shells * n_inplanes * size_complex ## sqrtweighted_image_fourier_bessel_conj_snq
        _usage += bs_temp * bs_img * n_shells * n_inplanes * size_complex ## CTF_sqrtweighted_fourier_templates_smnw
        _usage += bs_temp * bs_img * n_disp * n_inplanes * size_complex ## cross_correlation_smdq
        _usage += bs_temp * bs_img * n_disp * n_inplanes * size_complex ## cross_correlation_smdw
        _usage += bs_temp * bs_img * n_disp * (n_inplanes // 2 + 1) * size_complex ## aten::_fft_c2r
        _usage += bs_temp * bs_img * n_disp * n_inplanes * size_complex ## aten::fft_irfft

        return _usage ## in bytes

    free_bytes = cuda.mem_get_info()[0]
    batch_size_templates = n_templates
    batch_size_images = n_imgs
    while(True):
        _usage = memory_usage_batchsize(batch_size_templates, batch_size_images)
        if _usage > free_bytes:
            if batch_size_templates > 1 and batch_size_images > 1:
                if batch_size_templates > batch_size_images:
                    batch_size_templates //= 2
                else:
                    batch_size_images //= 2
            elif batch_size_templates > 1:
                batch_size_templates //= 2
            elif batch_size_images > 1:
                batch_size_images //= 2
            else:
                raise ValueError("Cannot estimate batch sizes, as both batch sizes are already 1.")
        else:
            break

    print(f"Estimated batch sizes: {batch_size_templates} templates, {batch_size_images} images")

    return (batch_size_templates, batch_size_images)



T_OptPosePartial = Callable[[Images, CTF, OptimalPoseReturn, Literal['phys'] | Literal['fourier']], Tensor]
def _get_optimal_pose_log_likelihood_partial(tp: Templates, precision: Precision):
    def _inner(
        im: Images,
        ctf: CTF,
        optimal_pose: OptimalPoseReturn,
        mode: Literal['phys'] | Literal['fourier'] = 'fourier'
    ) -> Tensor:
        res =  calc_likelihood_optimal_pose(
            template = tp,
            image = im,
            ctf = ctf,
            mode = mode,
            template_indices = optimal_pose.optimal_template_M,
            displacements_x = optimal_pose.optimal_displacement_x_M,
            displacements_y = optimal_pose.optimal_displacement_y_M,
            inplane_rotations = optimal_pose.optimal_inplane_rotation_M,
            return_distance = False,
            return_likelihood = True,
            precision = precision,
            use_cuda = True
        )
        assert isinstance(res, Tensor)
        return res
    return _inner
