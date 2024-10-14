from cryolike.viewing_angles import ViewingAngles
import torch
import numpy as np
import sys, os
from time import time, sleep
from cryolike.util.enums import Precision, CrossCorrelationReturnType

from cryolike.polar_grid import PolarGrid
from cryolike.cartesian_grid import CartesianGrid2D
from cryolike.template import Templates
from cryolike.image import FourierImages, Images, PhysicalImages
from cryolike.ctf import CTF, LensDescriptor
from cryolike.cross_correlation_likelihood import CrossCorrelationLikelihood
from cryolike.likelihood import calc_distance_optimal_templates_vs_physical_images
from cryolike.parameters import load_parameters
from cryolike.util.typechecks import set_precision

## TODO: implement functionality : skip_exist, optimized_inplane_rotation, optimized_displacement, optimized_viewing_angle
def run_likelihood(
    params_input: str | dict = None, # parameters
    folder_templates: str = None, # folder with templates
    folder_particles: str = None, # folder with particles
    i_template: int = 0, # index of the template
    n_stacks: int = 1, # number of stacks to process
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
    folder_output: str = None, # output folder
    verbose : bool = False # verbose mode
):
    """Run likelihood calculation with template files and particle files

    Attributes:
        params_input (str or dict): parameters file or dictionary
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

    ## load parameters
    params = load_parameters(params_input)
    
    (torch_float_type, torch_complex_type, _) = set_precision(params.precision, Precision.DOUBLE)

    polar_grid = PolarGrid(
        radius_max = params.radius_max,
        dist_radii = params.dist_radii,
        n_inplanes = params.n_inplanes,
        uniform = True
    )
    n_shells = polar_grid.n_shells
    n_points = polar_grid.n_points
    max_displacement = max_displacement_pixels * params.voxel_size[0]
        
    phys_grid = CartesianGrid2D(
        n_pixels = params.n_voxels,
        pixel_size = params.voxel_size,
    )
    flag_returned_displacements = False
    
    template_file_list = np.load(os.path.join(folder_templates, 'template_file_list.npy'), allow_pickle = True)
    folder_particles_fft = os.path.join(folder_particles, 'fft')
    template_file = template_file_list[i_template]
    print("template_file: ", template_file)

    templates_fourier = torch.load(template_file)
    fourier_templates_data = FourierImages(templates_fourier, polar_grid)
    viewing_angles = ViewingAngles.from_viewing_distance(params.viewing_distance)
    tp = Templates(
        fourier_templates_data = fourier_templates_data,
        box_size = params.box_size,
        viewing_angles = viewing_angles
    )
    n_shells = polar_grid.n_shells
    thetas = polar_grid.theta_shell
    radii = polar_grid.radius_shells

    folder_output_template = os.path.join(folder_output, 'template%d' % i_template)
    folder_output_log_likelihood = os.path.join(folder_output_template, 'log_likelihood')
    os.makedirs(folder_output_log_likelihood, exist_ok = True)
    folder_output_cross_correlation = os.path.join(folder_output_template, 'cross_correlation')
    os.makedirs(folder_output_cross_correlation, exist_ok = True)
    folder_output_optimal_pose = os.path.join(folder_output_template, 'optimal_pose')
    os.makedirs(folder_output_optimal_pose, exist_ok = True)

    for i_stack in range(n_stacks):
        
        print("stack number: ", i_stack)
        image_fourier_file = os.path.join(folder_particles_fft, f'particles_fourier_stack_{i_stack:06}.pt')
        image_param_file = os.path.join(folder_particles_fft, f'particles_fourier_stack_{i_stack:06}.npz')
        if not os.path.exists(image_fourier_file):
            raise ValueError("File not found: %s" % image_fourier_file)
        if not os.path.exists(image_param_file):
            raise ValueError("File not found: %s" % image_param_file)
        images_fourier = torch.load(image_fourier_file)
        image_param_compressed = np.load(image_param_file)

        box_size = image_param_compressed['box_size']
        n_pixels = image_param_compressed['n_pixels']
        pixel_size = image_param_compressed['pixel_size']

        defocusU_stack = image_param_compressed['defocusU']
        defocusV_stack = image_param_compressed['defocusV']
        defocusAng_stack = image_param_compressed['defocusAng']
        sphericalAberration = image_param_compressed['sphericalAberration']
        voltage = image_param_compressed['voltage']
        amplitudeContrast = image_param_compressed['amplitudeContrast']
        phaseShift_stack = image_param_compressed['phaseShift']
        
        defocus_angle_is_degree = image_param_compressed['defocus_angle_is_degree']
        phase_shift_is_degree = image_param_compressed['phase_shift_is_degree']

        fourier_images = FourierImages(images_fourier, polar_grid)
        im = Images(fourier_images_data=fourier_images, box_size=box_size, phys_grid=phys_grid)
        device = LensDescriptor(
            defocusU = defocusU_stack,
            defocusV = defocusV_stack,
            defocusAng = defocusAng_stack,
            defocusAng_degree = defocus_angle_is_degree,
            sphericalAberration = sphericalAberration,
            voltage = voltage,
            amplitudeContrast = amplitudeContrast,
            phaseShift = phaseShift_stack,
            phaseShift_degree = phase_shift_is_degree,)
        ctf = CTF(
            ctf_descriptor=device,
            polar_grid = polar_grid,
            box_size = box_size[0], ## TODO: check this hard-coded index
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
                                precision = params.precision,
                                device = 'cuda',
                                verbose = verbose
                            )
                            if not flag_returned_displacements:
                                displacements_set = torch.stack([cclik.x_displacements_expt_scale, cclik.y_displacements_expt_scale], dim = 1).T.cpu().numpy()
                                torch.save(displacements_set, os.path.join(folder_output, 'displacements_set.pt'))
                                flag_returned_displacements = True
                            assert im.images_fourier is not None
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
                precision = params.precision,
                device = 'cuda',
                verbose = verbose
            )
            if not flag_returned_displacements:
                displacements_set = torch.stack([cclik.x_displacements, cclik.y_displacements], dim = 1).T.cpu().numpy()
                torch.save(displacements_set, os.path.join(folder_output, 'displacements_set.pt'))
                flag_returned_displacements = True
            assert im.images_fourier is not None
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
            log_likelihood_optimal_pose_fourier_images_ = calc_distance_optimal_templates_vs_physical_images(
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
                precision = params.precision,
                use_cuda = True
            )
            filename_log_likelihood_optimal_pose_fourier = os.path.join(folder_output_log_likelihood, f'log_likelihood_fourier_S_stack_{i_stack:06}.pt')
            torch.save(log_likelihood_optimal_pose_fourier_images_, filename_log_likelihood_optimal_pose_fourier)
        
        if return_likelihood_optimal_pose_physical:
            folder_particles_phys = os.path.join(folder_particles, 'phys')
            images_phys = torch.load(os.path.join(folder_particles_phys, f'particles_phys_stack_{i_stack:06}.pt'))
            phys_image_data = PhysicalImages(images_phys, pixel_size=pixel_size)
            im_phys = Images(phys_images_data=phys_image_data, fourier_images_data=None, box_size=box_size)
            log_likelihood_optimal_pose_physical_images_ = calc_distance_optimal_templates_vs_physical_images(
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
                precision = params.precision,
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