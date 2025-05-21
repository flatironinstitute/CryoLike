import torch
import numpy as np
import sys, os
from time import time, sleep


from cryolike.grids import FourierImages, PhysicalImages
from cryolike.stacks import Templates, Images
from cryolike.microscopy import CTF
from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood
from cryolike.likelihood import calc_likelihood_optimal_pose
from cryolike.metadata import ViewingAngles, ImageDescriptor, load_combined_params
from cryolike.util import Precision, CrossCorrelationReturnType
## TODO: implement functionality : skip_exist, optimized_inplane_rotation, optimized_displacement, optimized_viewing_angle
## TODO: REFACTOR

def run_likelihood_batch(
    params_input: str | ImageDescriptor, # parameters
    folder_templates: str, # folder with templates
    folder_particles: str, # folder with particles
    i_template: int = 0, # index of the template
    i_stack: int = 0, # index of the template
    skip_exist: bool = False, # skip if the output files exist
    n_templates_per_batch: int = 1, # number of templates per batch
    n_images_per_batch: int = 128, # number of images per batch
    search_batch_size: bool = False, # search for the batch size that fits in the GPU memory
    max_displacement_pixels: float = 8.0, # maximum displacement in pixels
    n_displacements_x: int = -1, # number of displacements in x
    n_displacements_y: int = -1, # number of displacements in y
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
        search_batch_size (bool): search for the batch size that fits in the GPU memory
        max_displacement_pixels (float): maximum displacement in pixels
        n_displacements_x (int): number of displacements in x
        n_displacements_y (int): number of displacements in y
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
    if search_batch_size:
        list_n_images_per_batch = []
        list_n_templates_per_batch = []
        while n_images_per_batch > 0:
            list_n_images_per_batch.append(n_images_per_batch)
            n_images_per_batch = n_images_per_batch // 2
        while n_templates_per_batch > 0:
            list_n_templates_per_batch.append(n_templates_per_batch)
            n_templates_per_batch = n_templates_per_batch // 2
        n_n_images_per_batch = len(list_n_images_per_batch)
        n_n_templates_per_batch = len(list_n_templates_per_batch)
    else:
        n_n_images_per_batch = 0
        n_n_templates_per_batch = 0

    ## load parameters
    image_desc = ImageDescriptor.ensure(params_input)
    
    (torch_float_type, torch_complex_type, _) = image_desc.precision.get_dtypes(Precision.SINGLE)

    max_displacement = max_displacement_pixels * image_desc.cartesian_grid.pixel_size[0]
    flag_returned_displacements = False
    
    template_file_list = np.load(os.path.join(folder_templates, 'template_file_list.npy'), allow_pickle = True)
    folder_particles_fft = os.path.join(folder_particles, 'fft')
    print(folder_particles_fft)
    template_file = template_file_list[i_template]
    print("template_file: ", template_file)

    templates_fourier = torch.load(template_file, weights_only=True)
    fourier_templates_data = FourierImages(templates_fourier, image_desc.polar_grid)
    tp = Templates(
        fourier_data = fourier_templates_data,
        phys_data = image_desc.cartesian_grid,
        viewing_angles = image_desc.viewing_angles
    )

    folder_output_template = os.path.join(folder_output, 'template%d' % i_template)
    folder_output_log_likelihood = os.path.join(folder_output_template, 'log_likelihood')
    os.makedirs(folder_output_log_likelihood, exist_ok = True)
    folder_output_cross_correlation = os.path.join(folder_output_template, 'cross_correlation')
    os.makedirs(folder_output_cross_correlation, exist_ok = True)
    folder_output_optimal_pose = os.path.join(folder_output_template, 'optimal_pose')
    os.makedirs(folder_output_optimal_pose, exist_ok = True)

    if skip_exist:
        file_integrity = True
        if return_likelihood_integrated_pose_fourier:
            filename_output_log_likelihood = os.path.join(folder_output_log_likelihood, f'log_likelihood_S_stack_{i_stack:06}.pt')
            file_integrity = os.path.exists(filename_output_log_likelihood) and file_integrity
        if return_likelihood_optimal_pose_fourier:
            filename_log_likelihood_optimal_pose_fourier = os.path.join(folder_output_log_likelihood, f'log_likelihood_fourier_S_stack_{i_stack:06}.pt')
            file_integrity = os.path.exists(filename_log_likelihood_optimal_pose_fourier) and file_integrity
        if return_likelihood_optimal_pose_physical:
            filename_log_likelihood_optimal_pose_physical = os.path.join(folder_output_log_likelihood, f'log_likelihood_phys_S_stack_{i_stack:06}.pt')
            file_integrity = os.path.exists(filename_log_likelihood_optimal_pose_physical) and file_integrity
        if return_optimal_pose:
            filename_output_cross_correlation = os.path.join(folder_output_cross_correlation, f'cross_correlation_S_stack_{i_stack:06}.pt')
            filename_output_optimal_template_indices = os.path.join(folder_output_optimal_pose, f'optimal_template_S_stack_{i_stack:06}.pt')
            filename_output_optimal_displacement_x =  os.path.join(folder_output_optimal_pose, f'optimal_displacement_x_S_stack_{i_stack:06}.pt')
            filename_output_optimal_displacement_y = os.path.join(folder_output_optimal_pose, f'optimal_displacement_y_S_stack_{i_stack:06}.pt')
            filename_output_optimal_inplane_rotation = os.path.join(folder_output_optimal_pose, f'optimal_inplane_rotation_S_stack_{i_stack:06}.pt')
            file_integrity = os.path.exists(filename_output_cross_correlation) and os.path.exists(filename_output_optimal_template_indices) and os.path.exists(filename_output_optimal_displacement_x) and os.path.exists(filename_output_optimal_displacement_y) and os.path.exists(filename_output_optimal_inplane_rotation) and file_integrity
        if file_integrity:
            print("Skipping stack number: ", i_stack)
            return
    
    print("stack number: ", i_stack)
    image_fourier_file = os.path.join(folder_particles_fft, f'particles_fourier_stack_{i_stack:06}.pt')
    image_param_file = os.path.join(folder_particles_fft, f'particles_fourier_stack_{i_stack:06}.npz')
    if not os.path.exists(image_fourier_file):
        raise ValueError("File not found: %s" % image_fourier_file)
    if not os.path.exists(image_param_file):
        raise ValueError("File not found: %s" % image_param_file)
    images_fourier = torch.load(image_fourier_file, weights_only=True)
    (stack_img_desc, stack_lens_desc) = load_combined_params(image_param_file)
    if not image_desc.is_compatible_with(stack_img_desc):
        raise ValueError("Incompatible image parameters")
    

    fourier_images = FourierImages(images_fourier, stack_img_desc.polar_grid)
    im = Images(fourier_data=fourier_images, phys_data=stack_img_desc.cartesian_grid)
    ctf = CTF(
        ctf_descriptor=stack_lens_desc,
        polar_grid = stack_img_desc.polar_grid,
        box_size = stack_img_desc.cartesian_grid.box_size[0], ## TODO: check this hard-coded index
        anisotropy = True
    )

    ctf_tensor = torch.tensor(ctf.ctf, dtype = torch_float_type)
    if search_batch_size:
        success = False
        for i_diag in range(n_n_images_per_batch + n_n_templates_per_batch):
            i_x = i_diag
            i_y = 0
            while i_x >= 0:
                cc = None
                if i_x < n_n_templates_per_batch and i_y < n_n_images_per_batch:
                    n_templates_per_batch = list_n_templates_per_batch[i_x]
                    n_images_per_batch = list_n_images_per_batch[i_y]
                    try:
                        cclik = CrossCorrelationLikelihood(
                            templates = tp,
                            max_displacement = max_displacement,
                            # n_displacements = n_displacements,
                            n_displacements_x = n_displacements_x,
                            n_displacements_y = n_displacements_y,
                            precision = image_desc.precision,
                            device = 'cuda',
                            verbose = verbose
                        )
                        if not flag_returned_displacements:
                            displacements_set = torch.stack([cclik.x_displacements_expt_scale, cclik.y_displacements_expt_scale], dim = 1).T.cpu().numpy()
                            torch.save(displacements_set, os.path.join(folder_output, 'displacements_set.pt'))
                            flag_returned_displacements = True
                        optimal_pose, log_likelihood_fourier_integrated = cclik._compute_cross_correlation_likelihood(
                            device=torch.device("cuda"),
                            images_fourier = im.images_fourier,
                            ctf = ctf_tensor,
                            n_pixels_phys = im.phys_grid.n_pixels[0] * im.phys_grid.n_pixels[1],
                            n_images_per_batch=n_images_per_batch,
                            n_templates_per_batch=n_templates_per_batch,
                            return_type=CrossCorrelationReturnType.OPTIMAL_POSE,
                            return_integrated_likelihood=True
                        )
                        success = True
                    except torch.cuda.OutOfMemoryError:
                        del cc
                        torch.cuda.empty_cache()
                        print('n_templates_per_batch', n_templates_per_batch, 'n_images_per_batch', n_images_per_batch, 'out of memory')
                torch.cuda.empty_cache()
                i_x -= 1
                i_y += 1
            if success:
                break
        if not success:
            raise ValueError("Out of memory")
    else:
        cclik = CrossCorrelationLikelihood(
            templates = tp,
            max_displacement = max_displacement,
            # n_displacements = n_displacements,
            n_displacements_x = n_displacements_x,
            n_displacements_y = n_displacements_y,
            precision = image_desc.precision,
            device = 'cuda',
            verbose = verbose
        )
        if not flag_returned_displacements:
            # displacements_set = torch.stack([cclik.x_displacements, cclik.y_displacements], dim = 1).T.cpu().numpy()
            # TODO: CHECK: NOTE THIS CHANGE in variable name. Did this get broken in a previous PR?
            displacements_set = torch.stack([cclik.x_displacements_expt_scale, cclik.y_displacements_expt_scale], dim = 1).T.cpu().numpy()
            torch.save(displacements_set, os.path.join(folder_output, 'displacements_set.pt'))
            flag_returned_displacements = True
        optimal_pose, log_likelihood_fourier_integrated = cclik._compute_cross_correlation_likelihood(
            device=torch.device("cuda"),
            images_fourier = im.images_fourier,
            ctf = ctf_tensor,
            n_pixels_phys = im.phys_grid.n_pixels[0] * im.phys_grid.n_pixels[1],
            n_images_per_batch=n_templates_per_batch,
            n_templates_per_batch=n_images_per_batch,
            return_type=CrossCorrelationReturnType.OPTIMAL_POSE,
            return_integrated_likelihood=True
        )
    if return_optimal_pose:
        filename_output_cross_correlation = os.path.join(folder_output_cross_correlation, f'cross_correlation_S_stack_{i_stack:06}.pt')
        filename_output_optimal_template_indices = os.path.join(folder_output_optimal_pose, f'optimal_template_S_stack_{i_stack:06}.pt')
        filename_output_optimal_displacement_x =  os.path.join(folder_output_optimal_pose, f'optimal_displacement_x_S_stack_{i_stack:06}.pt')
        filename_output_optimal_displacement_y = os.path.join(folder_output_optimal_pose, f'optimal_displacement_y_S_stack_{i_stack:06}.pt')
        filename_output_optimal_inplane_rotation = os.path.join(folder_output_optimal_pose, f'optimal_inplane_rotation_S_stack_{i_stack:06}.pt')
        torch.save(optimal_pose.cross_correlation_S, filename_output_cross_correlation)
        torch.save(optimal_pose.optimal_template_S, filename_output_optimal_template_indices)
        torch.save(optimal_pose.optimal_displacement_x_S, filename_output_optimal_displacement_x)
        torch.save(optimal_pose.optimal_displacement_y_S, filename_output_optimal_displacement_y)
        torch.save(optimal_pose.optimal_inplane_rotation_S, filename_output_optimal_inplane_rotation)
    if return_likelihood_integrated_pose_fourier:
        filename_output_log_likelihood = os.path.join(folder_output_log_likelihood, f'log_likelihood_S_stack_{i_stack:06}.pt')
        torch.save(log_likelihood_fourier_integrated, filename_output_log_likelihood)
    if return_likelihood_optimal_pose_fourier:
        log_likelihood_optimal_pose_fourier_images_ = calc_likelihood_optimal_pose(
            template = tp,
            image = im,
            ctf = ctf,
            mode = "fourier",
            template_indices = optimal_pose.optimal_template_S,
            displacements_x = optimal_pose.optimal_displacement_x_S,
            displacements_y = optimal_pose.optimal_displacement_y_S,
            inplane_rotations = optimal_pose.optimal_inplane_rotation_S,
            return_distance = False,
            return_likelihood = True,
            precision = image_desc.precision,
            use_cuda = True
        )
        filename_log_likelihood_optimal_pose_fourier = os.path.join(folder_output_log_likelihood, f'log_likelihood_fourier_S_stack_{i_stack:06}.pt')
        torch.save(log_likelihood_optimal_pose_fourier_images_, filename_log_likelihood_optimal_pose_fourier)
    
    if return_likelihood_optimal_pose_physical:
        folder_particles_phys = os.path.join(folder_particles, 'phys')
        images_phys = torch.load(os.path.join(folder_particles_phys, f'particles_phys_stack_{i_stack:06}.pt'), weights_only=True)
        phys_image_data = PhysicalImages(images_phys, pixel_size=image_desc.cartesian_grid.pixel_size)
        im_phys = Images(phys_data=phys_image_data, fourier_data=None)
        log_likelihood_optimal_pose_physical_images_ = calc_likelihood_optimal_pose(
            template = tp,
            image = im_phys,
            ctf = ctf,
            mode = "phys",
            template_indices = optimal_pose.optimal_template_S,
            displacements_x = optimal_pose.optimal_displacement_x_S,
            displacements_y = optimal_pose.optimal_displacement_y_S,
            inplane_rotations = optimal_pose.optimal_inplane_rotation_S,
            return_distance = False,
            return_likelihood = True,
            precision = image_desc.precision,
            use_cuda = True
        )
        filename_log_likelihood_optimal_pose_physical = os.path.join(folder_output_log_likelihood, f'log_likelihood_phys_S_stack_{i_stack:06}.pt') 
        torch.save(log_likelihood_optimal_pose_physical_images_, filename_log_likelihood_optimal_pose_physical)
        del images_phys

    del im
    del ctf
    del images_fourier

    try:
        del cc
    except:
        pass
    del tp
    del templates_fourier
