from torch import cuda, device, Tensor, cuda
from typing import Callable, Literal
from math import ceil
import torch

from cryolike.file_mgmt import LikelihoodFileManager, LikelihoodOutputDataSources
from cryolike.stacks import Templates, Images
from cryolike.microscopy import CTF
from cryolike.likelihoods.likelihood import calc_likelihood_optimal_pose
from cryolike.likelihoods import (
    GeneratorType,
    compute_optimal_pose,
    compute_cross_correlation_complete,
    template_first_comparator
)
from cryolike.metadata import ImageDescriptor
from cryolike.util import  OutputConfiguration, Precision

T_PoseKernel = Callable[[GeneratorType, Templates, Images, CTF, Precision, OutputConfiguration], LikelihoodOutputDataSources]


def configure_likelihood_files(
    folder_templates: str,
    folder_particles: str,
    folder_output: str = '',
    n_stacks: int = 1,
    i_template: int = 0,
    return_likelihood_optimal_pose_physical : bool = False,
) -> LikelihoodFileManager:
    """Sets up an object that is responsible for ensuring name consistency
    for files input to or output by the run_likelihood wrappers.

    Args:
        folder_templates (str): Root of the folder where template files can be found
        folder_particles (str): Root of the folder where image stack files can be found
        folder_output (str, optional): Root of the folder to use for output. Defaults to ''.
        n_stacks (int, optional): Maximum number of image stacks to process. Defaults to 1.
        i_template (int, optional): Index of the template file within the templates directory. Defaults to 0.
        return_likelihood_optimal_pose_physical (bool, optional): Whether to additionally
            compute the likelihood of the optimal pose in physical-space representation.
            Not currently implemented. Defaults to False.

    Returns:
        LikelihoodFileManager:
            An object that provides consistent file name handling. 
    """
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
    """Sets up an object (callback) that sets the grid of displacements to search
    for each template file before processing.

    Args:
        max_displacement_pixels (float, optional): Maximum number of pixels to
            displace in either x or y dimension for search. Defaults to 8.0.
            Note that the full range will be from -(this value) to +(this value).
        n_displacements_x (int, optional): Number of displacements to search in the
            x-direction. Defaults to -1.
        n_displacements_y (int, optional): Number of displacements to search in the
            y-direction. Defaults to -1.

    Returns:
        displacement_configurator_T: A callback that ensures the displacement search
            grid is correctly set.
    """
    def template_grid_setter(tp: Templates):
        tp.set_displacement_grid(
            max_displacement_pixels,
            n_displacements_x,
            n_displacements_y
        )
    return template_grid_setter


def _get_smallest_batch(working_val: int, max_items: int):
    """Returns the smallest batch size that results in the same
    number of batches as the known-working size.

    The rationale is there's no point using larger batch sizes
    that result in the same number of batches; if you have a
    remainder, you have a remainder, and the larger batch size
    will only risk needless OOM situations and down-estimation.

    Args:
        working_val (int): A batch size known or believed to work
        max_items (int): The total number of items being batched

    Returns:
        int: The smallest batch size resulting in the same batch count.
    """
    if working_val < 1 or max_items < 1:
        raise ValueError("Pools and batch sizes must be positive.")
    batches_current = ceil(max_items / working_val)
    return ceil(max_items / batches_current)


class _BatchConfig():
    """Class that handles batch size estimation.

    Attributes:
        active (bool): Whether we are doing batch size adaptation; if False,
            we won't compute anything.
        last_worked (bool): Whether the last iteration completed successfully.
            Allows the upward size adjustment to work when getting a new
            stack, which depends on the current stack image count but
            shouldn't be run before a successful baseline is set.
        t_batch (int): Current batch size for templates
        i_batch (int): Current batch size for images
        t_total (int): Total templates in the stack (which is constant across
            all image stacks)
        i_hard_ceiling (int): Lowest image batch size known to cause failure
        i_hard_floor (int): Highest image batch size known to work
        t_hard_ceiling (int): Lowest template batch size known to cause OOM failure
        t_hard_floor (int): Highest template batch size known to work
    """
    active: bool
    last_worked: bool
    t_batch: int
    i_batch: int
    t_total: int
    i_hard_ceiling: int
    i_hard_floor: int
    t_hard_ceiling: int
    t_hard_floor: int

    def __init__(self, do_discovery: bool, t_batch: int | None, i_batch: int | None, t_total: int):
        if not do_discovery and (t_batch is None or i_batch is None):
            raise ValueError("Must provide concrete batch sizes if not doing discovery")
        if ((t_batch is not None and t_batch < 1) or (i_batch is not None and i_batch < 1)):
            raise ValueError("User-proposed batch sizes must be positive, if set.")
        self.last_worked = False
        self.active = do_discovery
        self.t_total = t_total
        self.t_batch = -1 if t_batch is None else t_batch
        self.i_batch = -1 if i_batch is None else i_batch
        self.i_hard_ceiling = -1
        self.i_hard_floor = 0
        self.t_hard_ceiling = -1
        self.t_hard_floor = -1


    def guess_initial_batches(self, i_total: int):
        # Don't overwrite user's explicit guesses for t_batch or i_batch
        if self.t_batch == -1:
            self.t_batch = self.t_total
        if self.i_batch == -1:
            self.i_batch = max(i_total // 8, 1) # intentionally start small here
        return


    def adjust_batches_up(self, i_total: int):
        if not self.active:
            return
        if not self.last_worked:
            self.last_worked = True
            return
        if self.i_batch <= 0 or self.t_batch <= 0:
            raise ValueError("Broken assumption: must have had a working batch")

        # we know the current template size worked, so we should never go lower
        self.t_hard_floor = _get_smallest_batch(self.t_batch, self.t_total)
        # and if we're already doing full batch, we can't increase it, so just
        # set the current value as the ceiling.
        # This depends on us using the same template stack per batch config
        # (a reasonable assumption); if we could allow other templates/if the
        # template total were to change, our information would not be as strong
        # regarding whether this is a true ceiling for even larger stacks.
        if (self.t_batch >= self.t_total):
            self.t_hard_ceiling = self.t_batch

        # If we don't have a hard template batch ceiling, we're still discovering
        # how much we can push it. So adjust that only and return
        if (self.t_hard_ceiling <= 0):
            self._find_higher_template_size()
            return

        self.i_hard_floor = max(self.i_batch, self.i_hard_floor)
        if self.i_hard_ceiling <= 0:
            # No information about best possible batch sizes, so
            # go for broke: see if we can do a single image batch.
            self.i_batch = i_total
            return
 
        # By this point we have hard ceilings for everything.
        # So we aren't adjusting template batch size any more,
        # and should do binary search on possible image batch sizes.
        batches_at_ceiling = ceil(i_total / self.i_hard_ceiling)
        batches_at_current = ceil(i_total / self.i_batch)

        # Only adjust if we might reduce the batch count by doing so
        if batches_at_ceiling < batches_at_current:
            self.i_batch = (self.i_hard_ceiling + self.i_batch) // 2


    def _find_higher_template_size(self):
        if not self.active:
            return
        curr_batch_count = ceil(self.t_total / self.t_batch)
        if curr_batch_count < 2:
            # This can no longer happen, because we're checking the batch
            # size against the total template size before we enter this fn
            raise ValueError("Impossible: full batch worked but we didn't set a ceiling.")
        ballpark_batch_size = ceil(self.t_total / (curr_batch_count - 1))
        self.t_batch = _get_smallest_batch(ballpark_batch_size, self.t_total)


    def adjust_batches_down(self):
        if not self.active:
            return
        self.last_worked = False
        if self.t_batch == 1 and self.i_batch == 1:
            raise ValueError("Unfixable: OOM error but both batch sizes are already 1.")
        if self.t_batch == 1 and self.t_hard_floor <= 0:
            self.t_hard_ceiling = 1
            self.t_hard_floor = 1

        # We know that we just ran a batch and it didn't work for OOM reasons.
        # case 1: We don't know the floor for the template size.
        # Response: try increasing the batch count by 1.
        # (We'll stop adjusting template batch size once we find one that works)
        if (self.t_hard_floor <= 0):
            self.t_hard_ceiling = self.t_batch
            self.t_batch = _get_smallest_batch(self.t_batch - 1, self.t_total)
            return
        else:
            # We found a working template size, so we'll only change the image size.
            # Take half the distance between our current (non-working) value and
            # the (known-working) floor.
            self.i_hard_ceiling = self.i_batch
            self.i_batch = (self.i_batch + self.i_hard_floor) // 2
            return


def run_likelihood_optimal_pose(
    file_config: LikelihoodFileManager,
    params_input: str | ImageDescriptor,
    displacer: displacement_configurator_T,
    template_index: int = 0,
    n_stacks: int = 1,
    skip_exist: bool = False,
    n_templates_per_batch: int | None = None,
    n_images_per_batch: int | None = None,
    discover_batch_size: bool = False,
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
            try to compare at once in memory. Higher numbers
            should result in more efficient computation, particularly on
            GPUs, but may need to be reduced if out-of-memory errors occur.
            If set to None (the default), requires discover_batch_size to
            be set, and will be initially set to the full template count.
        n_images_per_batch (int, optional): The number of images to try to
            compare at once in memory. Higher numbers should
            result in more efficient computation, particularly on GPUs, but
            may need to be reduced if out-of-memory errors occur.
            If set to None (the default), requires discover_batch_size to
            be set, and will be initially guessed as 1/8 of the total images
            for the first iteration.
        discover_batch_size (bool, optional): Whether to attempt to compute
            an optimal number of templates and images per batch. Defaults to
            False. If False, explicit template and image batch sizes will be
            used (and any resulting out-of-memory errors will cause processing
            failure). If set True, image and template batch sizes will be adjusted
            for subsequent image stacks with the same template, starting with
            the values of the n_templates_per_batch and n_images_per_batch
            parameters (which are taken as hints rather than hard requirements).
            OOM errors will be caught, and errors will only be generated if
            the system still runs out of memory with a single image and template
            per batch. Otherwise, the logic will first adjust the template batch
            size to the value that gives the smallest total number of template
            batches without running out of memory. Once this value is found,
            the image batch size will be adjusted by halves between a hard
            ceiling (a batch size that resulted in OOM) and a hard floor (the
            largest known-working batch size), with adjustment stopping once
            these parameters would no longer decrease the total number of image
            batches.
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

    (tp, img_desc, precision) = _prepare_loop(
        file_config,
        params_input,
        displacer,
        template_index
    )

    sizes = _BatchConfig(discover_batch_size, n_templates_per_batch, n_images_per_batch, tp.n_images)

    _run_stack_loop(
        n_stacks,
        skip_exist,
        file_config,
        outputs,
        img_desc,
        sizes,
        tp,
        precision,
        return_integrated_pose_likelihood_fourier=return_likelihood_integrated_pose_fourier,
        kernel=_optimal_pose_kernel
    )


def run_likelihood_full_cross_correlation(
    file_config: LikelihoodFileManager,
    params_input: str | ImageDescriptor,
    displacer: displacement_configurator_T,
    template_index: int = 0,
    n_stacks: int = 1,
    skip_exist: bool = False,
    n_templates_per_batch: int | None = None,
    n_images_per_batch: int | None = None,
    discover_batch_size: bool = False,
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
            try to compare at once in memory. Higher numbers
            should result in more efficient computation, particularly on
            GPUs, but may need to be reduced if out-of-memory errors occur.
            If set to None (the default), requires discover_batch_size to
            be set, and will be initially set to the full template count.
        n_images_per_batch (int, optional): The number of images to try to
            compare at once in memory. Higher numbers should
            result in more efficient computation, particularly on GPUs, but
            may need to be reduced if out-of-memory errors occur.
            If set to None (the default), requires discover_batch_size to
            be set, and will be initially guessed as 1/8 of the total images
            for the first iteration.
        discover_batch_size (bool, optional): Whether to attempt to compute
            an optimal number of templates and images per batch. Defaults to
            False. If False, explicit template and image batch sizes will be
            used (and any resulting out-of-memory errors will cause processing
            failure). If set True, image and template batch sizes will be adjusted
            for subsequent image stacks with the same template, starting with
            the values of the n_templates_per_batch and n_images_per_batch
            parameters (which are taken as hints rather than hard requirements).
            OOM errors will be caught, and errors will only be generated if
            the system still runs out of memory with a single image and template
            per batch. Otherwise, the logic will first adjust the template batch
            size to the value that gives the smallest total number of template
            batches without running out of memory. Once this value is found,
            the image batch size will be adjusted by halves between a hard
            ceiling (a batch size that resulted in OOM) and a hard floor (the
            largest known-working batch size), with adjustment stopping once
            these parameters would no longer decrease the total number of image
            batches.

    """

    outputs = OutputConfiguration(
        return_optimal_pose=False,
        optimized_inplane_rotation=False,
        optimized_displacement=False,
        optimized_viewing_angle=False,
        return_cross_correlation_pose=True
    )

    (tp, img_desc, precision) = _prepare_loop(
        file_config,
        params_input,
        displacer,
        template_index)

    sizes = _BatchConfig(discover_batch_size, n_templates_per_batch, n_images_per_batch, tp.n_images)

    _run_stack_loop(
        n_stacks,
        skip_exist,
        file_config,
        outputs,
        img_desc,
        sizes,
        tp,
        precision,
        return_integrated_pose_likelihood_fourier=False,
        kernel=_full_cross_correlation_kernel
    )


def _prepare_loop(
    file_config: LikelihoodFileManager,
    params_input: str | ImageDescriptor,
    displacer: displacement_configurator_T,
    template_index: int = 0
):
    (tp, image_desc, _) = file_config.load_template(params_input, template_index)
    precision = image_desc.precision
    displacer(tp)
    file_config.save_displacements(tp.displacement_grid_angstrom)
    return (tp, image_desc, precision)


def _run_stack_loop(
    n_stacks: int,
    skip_exist: bool,
    file_config: LikelihoodFileManager,
    outputs: OutputConfiguration,
    image_desc: ImageDescriptor,
    sizes: _BatchConfig,
    tp: Templates,
    precision: Precision,
    return_integrated_pose_likelihood_fourier: bool,
    kernel: T_PoseKernel,
):
    for i_stack in range(n_stacks):
        if skip_exist and file_config.outputs_exist(i_stack, outputs):
            # NOTE: this is not a foolproof way to check if the files exist, as the files could be corrupted
            print(f"Skipping stack number: {i_stack} as all output files already exist")
            continue
        (im, ctf) = file_config.load_img_stack(i_stack, image_desc)
        total_images = im.n_images
        sizes.guess_initial_batches(total_images)
        retry_oom = True

        while(retry_oom):
            sizes.adjust_batches_up(total_images)
            print(f"Using batch sizes: {sizes.t_batch} t, {sizes.i_batch} i")
            iterator = template_first_comparator(
                device=device('cuda'),
                images=im,
                templates=tp,
                ctf=ctf,
                n_images_per_batch=sizes.i_batch,
                n_templates_per_batch=sizes.t_batch,
                return_integrated_likelihood=return_integrated_pose_likelihood_fourier,
                precision=precision
            )
            try:
                out_data = kernel(iterator, tp, im, ctf, precision, outputs)
                file_config.write_outputs(i_stack, outputs, out_data)
                cuda.empty_cache()
                retry_oom = False
            except torch.cuda.OutOfMemoryError:
                if sizes.active:
                    sizes.adjust_batches_down()
                else:
                    raise


def _get_optimal_pose_log_likelihood(
    tp: Templates,
    im: Images,
    ctf: CTF,
    precision: Precision,
    out_data: LikelihoodOutputDataSources,
    mode: Literal['phys'] | Literal['fourier'] = 'fourier'
) -> None:
    if mode == 'phys':
        raise NotImplementedError
        ## NOTE: dependency on file config; this needs to be reworked if we
        ## ever wind up supporting this use case
        im_phys = file_config.load_phys_stack(i_stack, image_desc)
        out_data.ll_optimal_phys_pose = partial(im_phys, ctf, optimal_pose, 'phys')
        del im_phys
    optimal_pose = out_data.optimal_pose
    assert optimal_pose is not None
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
    if mode == 'fourier':
        out_data.ll_optimal_fourier_pose = res
    if mode == 'phys':
        out_data.ll_optimal_phys_pose = res
    return


# Note: outputs, ctf are unused but maintain a consistent interface
# for the other kernel(s).
def _full_cross_correlation_kernel(
    iterator: GeneratorType,
    tp: Templates,
    im: Images,
    ctf: CTF,
    precision: Precision,
    outputs: OutputConfiguration,
):
    out_data = LikelihoodOutputDataSources()
    full_cross_correlation_pose = compute_cross_correlation_complete(
        iterator,
        tp,
        im,
        precision,
        include_integrated_log_likelihood=False
    )
    out_data.full_pose = full_cross_correlation_pose
    return out_data


def _optimal_pose_kernel(
    iterator: GeneratorType,
    tp: Templates,
    im: Images,
    ctf: CTF,
    precision: Precision,
    outputs: OutputConfiguration,
):
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
    if outputs.optimal_fourier_pose_likelihood:
        _get_optimal_pose_log_likelihood(
            tp,
            im,
            ctf,
            precision,
            out_data,
            'fourier'
        )
    return out_data
