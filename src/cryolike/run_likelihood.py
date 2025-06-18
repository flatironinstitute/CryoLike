from torch import cuda, device, dtype, Tensor, cuda
from math import ceil
from typing import Callable, Literal

from cryolike.file_mgmt import LikelihoodFileManager, LikelihoodOutputDataSources
from cryolike.stacks import Templates, Images
from cryolike.microscopy import CTF
from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood, OptimalPoseReturn
from cryolike.likelihood import LikelihoodFourierModel
from cryolike.metadata import ImageDescriptor
from cryolike.util import CrossCorrelationReturnType, OutputConfiguration, Precision

## TODO: implement functionality: optimized_inplane_rotation, optimized_displacement, optimized_viewing_angle

def run_likelihood(
    params_input: str | ImageDescriptor, # parameters
    folder_templates: str, # folder with templates
    folder_particles: str, # folder with particles
    i_template: int = 0, # index of the template
    n_stacks: int = 1, # number of stacks to process
    skip_exist: bool = False, # skip if the output files exist
    n_templates_per_batch: int = 1, # number of templates per batch
    n_images_per_batch: int = 128, # number of images per batch
    estimate_batch_size: bool = False, # search for the batch size that fits in the GPU memory
    max_displacement_pixels: float = 8.0, # maximum displacement in pixels
    n_displacements_x: int = -1, # number of displacements in x
    n_displacements_y: int = -1, # number of displacements in y
    return_cross_correlation_pose: bool = False,
    return_likelihood_integrated_pose_fourier : bool = False, # return integrated likelihood in fourier space
    return_likelihood_optimal_pose_physical : bool = False, # return likelihood of optimal pose in physical space
    return_likelihood_optimal_pose_fourier : bool = False, # return likelihood of optimal pose in fourier space
    return_optimal_pose : bool = True, # return optimal pose
    optimized_inplane_rotation : bool = True, # optimize inplane rotation
    optimized_displacement : bool = True, # optimize displacement
    optimized_viewing_angle : bool = True, # optimize viewing angle
    folder_output: str = '', # output folder
    verbose : bool = False # verbose mode
):
    """Run likelihood calculation with template files and particle files

    Attributes:
        params_input (str or ImageDescriptor): ImageDescriptor object (describing the grids used for
            the image values), or the file path to a serialized ImageDescriptor
        folder_templates (str): folder with templates
        folder_particles (str): folder with particles
        i_template (int): index of the template
        n_stacks (int): number of stacks to process
        skip_exist (bool): skip if the output files exist
        n_templates_per_batch (int): number of templates per batch
        n_images_per_batch (int): number of images per batch
        estimate_batch_size (bool): compute the batch size that fits in the GPU memory. If True,
            n_templates_per_batch and n_images_per_batch will be ignored.
        max_displacement_pixels (float): maximum displacement in pixels
        n_displacements_x (int): number of displacements in x
        n_displacements_y (int): number of displacements in y
        return_cross_correlation_pose (bool): return the cross correlation of each image and each pose.
            If True, no other return types will be computed.
        return_likelihood_integrated_pose_fourier (bool): return integrated likelihood in fourier space
        return_likelihood_optimal_pose_physical (bool): return likelihood of optimal pose in physical space
        return_likelihood_optimal_pose_fourier (bool): return likelihood of optimal pose in fourier space
        return_optimal_pose (bool): return optimal pose
        optimized_inplane_rotation (bool): optimize inplane rotation
        optimized_displacement (bool): optimize displacement
        optimized_viewing_angle (bool): optimize viewing angle
        folder_output (str): output folder
        verbose (bool): verbose mode
    """

    # TODO: Push this to a separate config creation
    filemgr = LikelihoodFileManager(
        folder_output,
        folder_templates,
        folder_particles,
        n_stacks,
        i_template,
        return_likelihood_optimal_pose_physical
    )

    # TODO: Push this to a separate config creation
    outputs = OutputConfiguration(
        return_cross_correlation_pose=return_cross_correlation_pose,
        return_likelihood_integrated_pose_fourier=return_likelihood_integrated_pose_fourier,
        return_likelihood_optimal_pose_physical=return_likelihood_optimal_pose_physical,
        return_likelihood_optimal_pose_fourier=return_likelihood_optimal_pose_fourier,
        return_optimal_pose=return_optimal_pose,
        optimized_inplane_rotation=optimized_inplane_rotation,
        optimized_displacement=optimized_displacement,
        optimized_viewing_angle=optimized_viewing_angle
    )

    if outputs.optimal_phys_pose_likelihood:
        raise NotImplementedError("Physical likelihood is still under development and not yet available. Please use Fourier likelihood instead.")

    (tp, image_desc, torch_float_type, max_displacement) = filemgr.load_template(params_input, max_displacement_pixels, i_template)
    optimal_pose_ll_partial = _get_optimal_pose_log_likelihood_partial(tp, image_desc)

    cclik = CrossCorrelationLikelihood(
        templates = tp,
        max_displacement = max_displacement,
        n_displacements_x = n_displacements_x,
        n_displacements_y = n_displacements_y,
        precision = image_desc.precision,
        device = 'cuda',
        verbose = verbose
    )
    filemgr.save_displacements(cclik.x_displacements_expt_scale, cclik.y_displacements_expt_scale)
    per_loop_computation = _get_batch_likelihood_computation_partial(
        outputs,
        cclik,
        optimal_pose_ll_partial,
        torch_float_type,
        filemgr
    )

    for i_stack in range(n_stacks):
        if skip_exist and filemgr.outputs_exist(i_stack, outputs):
            # NOTE: this is not a foolproof way to check if the files exist, as the files could be corrupted
            print(f"Skipping stack number: {i_stack} as all output files already exist")
            continue
        (im, ctf) = filemgr.load_img_stack(i_stack, image_desc)

        if estimate_batch_size:
            n_templates_per_batch, n_images_per_batch = _compute_batch_sizes(tp, im, cclik)

        out_data = per_loop_computation(im, ctf, n_images_per_batch, n_templates_per_batch, i_stack, image_desc)
        filemgr.write_outputs(i_stack, outputs, out_data)
        cuda.empty_cache()


def _compute_batch_sizes(
    templates: Templates,
    images: Images,
    cclik: CrossCorrelationLikelihood
):
    if not cuda.is_available():
        raise ValueError("Requested to estimate batch sizes but no GPU.")

    n_templates = templates.n_images
    n_imgs = images.n_images
    n_shells = templates.polar_grid.n_shells
    n_inplanes = templates.polar_grid.n_inplanes
    n_disp = cclik.n_displacements
    size_float = cclik.torch_float_type.itemsize ## in bytes
    size_complex = cclik.torch_complex_type.itemsize ## in bytes

    def memory_usage_batchsize(bs_temp: int, bs_img: int) -> int:
        ## This is a rough estimate of the memory usage for the given batch sizes
        ## Only consider the significant memory usage
        
        _usage = 0
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
                raise ValueError("Cannot estimate batch sizes, as both batch sizes are already 1. Possibly the GPU memory is too small for the cross correlation calculation. Try reducing the resolution.")
        else:
            break

    print(f"Estimated batch sizes: {batch_size_templates} templates, {batch_size_images} images")

    return (batch_size_templates, batch_size_images)



T_OptPosePartial = Callable[[Images, CTF, OptimalPoseReturn, Literal['phys'] | Literal['fourier']], Tensor]
def _get_optimal_pose_log_likelihood_partial(tp: Templates, image_desc: ImageDescriptor):
    def _inner(
        im: Images,
        ctf: CTF,
        optimal_pose: OptimalPoseReturn,
        mode: Literal['phys'] | Literal['fourier'] = 'fourier'
    ) -> Tensor:
        # TODO: This can be revised once calc_likelihood_etc is cleaned up
        likelihood_fourier_model =  LikelihoodFourierModel(
            model = tp,
            polar_grid = tp.polar_grid,
            box_size = tp.box_size,
            n_pixels = im.phys_grid.n_pixels[0] ** 2,
            viewing_angles = None,
            atom_shape = None,
            precision = Precision.SINGLE,
            device = device('cuda'),
            verbose = False
        )
        res = likelihood_fourier_model(
            images = im,
            template_indices = optimal_pose.optimal_template_S,
            x_displacements = optimal_pose.optimal_displacement_x_S,
            y_displacements = optimal_pose.optimal_displacement_y_S,
            gammas = optimal_pose.optimal_inplane_rotation_S,
            ctf = ctf,
            verbose = False
        )
        assert isinstance(res, Tensor)
        return res
    return _inner


def _get_batch_likelihood_computation_partial(
    outputs: OutputConfiguration,
    cclik: CrossCorrelationLikelihood,
    optimal_pose_ll_partial: T_OptPosePartial,
    torch_float_type: dtype,
    filemgr: LikelihoodFileManager
):
    def full_fn(im: Images, ctf: CTF, n_im_per_batch: int, n_tem_per_batch: int, i_stack: int, image_desc: ImageDescriptor):
        out_data = LikelihoodOutputDataSources()
        full_cross_correlation_pose = cclik.compute_cross_correlation_complete(
            device=device("cuda"),
            images_fourier = im.images_fourier,
            ctf= ctf.ctf.to(torch_float_type),
            n_pixels_phys = im.phys_grid.n_pixels[0].item() * im.phys_grid.n_pixels[1].item(),
            n_images_per_batch=n_im_per_batch,
            n_templates_per_batch=n_tem_per_batch,
            return_integrated_likelihood=False
        )
        out_data.full_pose = full_cross_correlation_pose
        return out_data

    def fourier_partial(optimal_pose: OptimalPoseReturn, i_stack: int, image_desc: ImageDescriptor, im: Images, ctf: CTF, out_data: LikelihoodOutputDataSources):
        ll_opt_pose_fourier = optimal_pose_ll_partial(im, ctf, optimal_pose, 'fourier')
        out_data.ll_optimal_fourier_pose = ll_opt_pose_fourier
    
    def phys_partial(optimal_pose: OptimalPoseReturn, i_stack: int, image_desc: ImageDescriptor, im: Images, ctf: CTF, out_data: LikelihoodOutputDataSources):
        im_phys = filemgr.load_phys_stack(i_stack, image_desc)
        log_likelihood_optimal_pose_physical_images_ = \
            optimal_pose_ll_partial(im_phys, ctf, optimal_pose, 'phys')
        out_data.ll_optimal_phys_pose = log_likelihood_optimal_pose_physical_images_
        del im_phys

    optimal_pose_partials = []
    if outputs.optimal_fourier_pose_likelihood:
        optimal_pose_partials.append(fourier_partial)
    if outputs.optimal_phys_pose_likelihood:
        optimal_pose_partials.append(phys_partial)

    def optimal_pose_partial(im: Images, ctf: CTF, n_im_per_batch: int, n_tem_per_batch: int, i_stack: int, image_desc: ImageDescriptor):
        out_data = LikelihoodOutputDataSources()
        optimal_pose, log_likelihood_fourier_integrated = cclik._compute_cross_correlation_likelihood(
            device=device("cuda"),
            images_fourier = im.images_fourier,
            ctf = ctf.ctf.to(torch_float_type),
            n_pixels_phys = im.phys_grid.n_pixels[0].item() * im.phys_grid.n_pixels[1].item(),
            n_images_per_batch=n_im_per_batch,
            n_templates_per_batch=n_tem_per_batch,
            return_type=CrossCorrelationReturnType.OPTIMAL_POSE,
            return_integrated_likelihood=True
        )
        out_data.optimal_pose = optimal_pose
        out_data.ll_fourier_integrated = log_likelihood_fourier_integrated
        _ = [x(optimal_pose, i_stack, image_desc, im, ctf, out_data) for x in optimal_pose_partials]
        return out_data

    if outputs.cross_correlation_pose:
        return full_fn
    else:
        return optimal_pose_partial
