import numpy as np
import torch
from finufft import nufft2d1 as nufft2d1_cpu
from finufft import nufft2d2 as nufft2d2_cpu
from finufft import nufft3d2 as nufft3d2_cpu

from cryolike.grids import CartesianGrid2D, PolarGrid, Volume
from cryolike.util import Precision

# TODO: Use the one in device_handling
def _check_cuda_available() -> bool:
    try:
        if torch.cuda.is_available():
            return True
    except:
        print("Warning: CUDA is not available, using CPU!")
    return False

def fourier_polar_to_cartesian_phys(
    grid_fourier_polar : PolarGrid,
    grid_cartesian_phys : CartesianGrid2D,
    image_polar : np.ndarray | torch.Tensor,
    isign: int = 1,
    eps: float = 1.0e-6,
    precision: Precision = Precision.SINGLE,
    use_cuda: bool = True
) -> torch.Tensor:
    use_cuda = _check_cuda_available() and use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Using device:", device)
    (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    eps = precision.set_epsilon(eps)

    if not len(image_polar.shape) in [1, 2]:
        raise ValueError("image array is not 1D (single imagen (n_points,)) or 2D (multiple images, (n_images, n_points)).")
    if image_polar.shape[-1] != grid_fourier_polar.n_points:
        raise ValueError("image_polar.size != grid_fourier_polar.n_points")
    if not isinstance(image_polar, torch.Tensor):
        image_polar = torch.tensor(image_polar, dtype = torch_complex_type)
    else:
        image_polar = image_polar.to(torch_complex_type)
    if len(image_polar.shape) == 1:
        image_polar = image_polar.unsqueeze(0)
    if len(image_polar.shape) != 2:
        raise ValueError("Invalid image_polar shape", image_polar.shape)
    n_images = image_polar.shape[0]
    n_x = grid_cartesian_phys.n_pixels[0]
    n_y = grid_cartesian_phys.n_pixels[1]
    x1 = grid_fourier_polar.x_points
    x2 = grid_fourier_polar.y_points
    if (x1 is None or x2 is None):
        raise ValueError("Polar grid does not have Cartesian-equivalent points set.")
    weight_points = grid_fourier_polar.weight_points
    if use_cuda:
        x1 = torch.tensor(x1, dtype = torch_float_type, device = device)
        x2 = torch.tensor(x2, dtype = torch_float_type, device = device)
        weight_points = torch.tensor(weight_points, dtype = torch_float_type, device = device)
    k1 = x1 * (2.0 * np.pi) * 2.0 / grid_cartesian_phys.n_pixels[0]
    k2 = x2 * (2.0 * np.pi) * 2.0 / grid_cartesian_phys.n_pixels[1]
    n_xy = grid_cartesian_phys.n_pixels_total
    n_modes = (n_x, n_y)
    rescale_factor = 2.0 / np.sqrt(n_xy) * (2 * np.pi)
    # rescale_factor = 2.0 / n_xy * (2 * np.pi)
    if use_cuda:
        if n_images == 1:
            from cufinufft import nufft2d1
            image_polar_gpu = image_polar.to(device) * weight_points
            image_phys_gpu = torch.zeros((1, n_x, n_y), dtype = torch_complex_type, device = device)
            nufft2d1(
                k1, k2, image_polar_gpu,
                out = image_phys_gpu,
                n_modes = n_modes,
                eps = eps,
                isign = isign,
            )
            assert isinstance(image_phys_gpu, torch.Tensor)
            image_phys_gpu *= rescale_factor
            image_phys = image_phys_gpu.detach().cpu()
            return image_phys
        else:
            from cufinufft import nufft2d1 #Plan
            image_polar = image_polar.reshape(n_images, -1)
            image_phys = torch.zeros((n_images, n_x, n_y), dtype = torch_complex_type, device = "cpu")
            # plan = Plan(
            #     nufft_type = 1,
            #     n_modes = n_modes,
            #     n_trans = 1,
            #     dtype = "complex64" if precision == Precision.SINGLE else "complex128",
            #     eps = eps, isign=isign
            # )
            # plan.setpts(k1, k2)
            image_phys_gpu = torch.zeros((n_x, n_y), dtype = torch_complex_type, device = device)
            for i_image in range(n_images):
                image_polar_gpu = image_polar[i_image].to(device) * weight_points
                # plan.execute(image_polar_gpu, image_phys_gpu)
                nufft2d1(
                    k1, k2, image_polar_gpu,
                    out = image_phys_gpu,
                    n_modes = n_modes,
                    eps = eps,
                    isign = isign,
                )
                image_phys_gpu *= rescale_factor
                image_phys[i_image,:,:] = image_phys_gpu.detach().cpu()
            return image_phys
    else:
        numpy_float_type = np.float64
        numpy_complex_type = np.complex128
        image_polar = image_polar.cpu().numpy().astype(numpy_complex_type)
        k1 = k1.astype(numpy_float_type)
        k2 = k2.astype(numpy_float_type)
        image_phys = None
        if len(image_polar.shape) == 1:
            image_polar_weighted = image_polar.flatten() * grid_fourier_polar.weight_points
            image_phys = nufft2d1_cpu(k1, k2, image_polar_weighted, n_modes = n_modes, eps = eps, isign = isign)
        elif len(image_polar.shape) == 2:
            n_images = image_polar.shape[0]
            image_polar_weighted = image_polar.reshape(n_images, -1) * grid_fourier_polar.weight_points[None,:]
            # image_phys = torch.zeros((n_images, n_x, n_y), dtype = torch_complex_type)
            image_phys = np.zeros((n_images, n_x, n_y), dtype = image_polar.dtype)
            for i in range(n_images):
                image_phys[i] = nufft2d1_cpu(k1, k2, image_polar_weighted[i], n_modes = n_modes, eps = eps, isign = isign)
        else:
            raise ValueError("Invalid image_polar shape")
        image_phys *= rescale_factor
        return torch.tensor(image_phys, dtype = torch_complex_type)

def cartesian_phys_to_fourier_polar(
    grid_cartesian_phys : CartesianGrid2D,
    grid_fourier_polar : PolarGrid,
    images_phys : np.ndarray | torch.Tensor,
    isign: int = -1,
    eps: float = 1.0e-6,
    precision: Precision = Precision.SINGLE,
    use_cuda: bool = True
) -> torch.Tensor:
    use_cuda = _check_cuda_available() and use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    eps = precision.set_epsilon(eps)
    if not isinstance(images_phys, torch.Tensor):
        images_phys = torch.tensor(images_phys, dtype = torch_complex_type)
    else:
        images_phys = images_phys.to(torch_complex_type)
    assert isinstance(images_phys, torch.Tensor)
    if len(images_phys.shape) == 2:
        images_phys = images_phys.unsqueeze(0)
    if len(images_phys.shape) != 3:
        raise ValueError("Invalid image_phys shape")
    n_images = images_phys.shape[0]
    n_x = grid_cartesian_phys.n_pixels[0]
    n_y = grid_cartesian_phys.n_pixels[1]
    if images_phys.shape[1] != n_x or images_phys.shape[2] != n_y:
        raise ValueError("Image size does not match grid size. %s != %s" % (images_phys.shape[1:], grid_cartesian_phys.n_pixels))
    n_xy = grid_cartesian_phys.n_pixels_total
    n_modes = (n_x, n_y)
    n_points_polar = grid_fourier_polar.n_points
    rescale_factor: float = 2.0 / np.sqrt(n_xy) * (2 * np.pi)
    images_phys = images_phys.to(torch_complex_type)
    x1 = grid_fourier_polar.x_points
    x2 = grid_fourier_polar.y_points
    if (x1 is None or x2 is None):
        raise ValueError("Polar grid does not have Cartesian-equivalent points set.")
    if use_cuda:
        x1 = torch.tensor(x1, dtype = torch_float_type, device = device)
        x2 = torch.tensor(x2, dtype = torch_float_type, device = device)
    k1 = x1 * (2.0 * np.pi) * 2.0 / grid_cartesian_phys.n_pixels[0]
    k2 = x2 * (2.0 * np.pi) * 2.0 / grid_cartesian_phys.n_pixels[1]
    if use_cuda:
        image_polar = torch.zeros((n_images, n_points_polar), dtype = torch_complex_type, device = "cpu")
        if n_images == 1:
            from cufinufft import nufft2d2
            image_phys_gpu = images_phys[0,:,:].to(device)
            image_polar_gpu = image_polar.flatten().to(device)
            nufft2d2(
                k1, k2, image_phys_gpu,
                out = image_polar_gpu,
                eps = eps,
                isign = isign
            )
            assert isinstance(image_polar_gpu, torch.Tensor)
            image_polar_gpu *= rescale_factor
            image_polar = image_polar_gpu.detach().cpu()
        else:
            from cufinufft import Plan
            plan = Plan(
                nufft_type = 2,
                n_modes = n_modes,
                n_trans = 1,
                dtype = "complex64" if precision == Precision.SINGLE else "complex128",
                eps = eps,
                isign = isign
            )
            plan.setpts(k1, k2)
            image_polar_gpu = torch.zeros(n_points_polar, dtype = torch_complex_type, device = device)
            for i_image in range(n_images):
                plan.execute(images_phys[i_image,:,:].to(device), image_polar_gpu)
                assert isinstance(image_polar_gpu, torch.Tensor)
                image_polar_gpu *= rescale_factor
                image_polar[i_image,:] = image_polar_gpu.detach().cpu()
        return image_polar
    else:
        numpy_float_type = np.float64
        numpy_complex_type = np.complex128
        images_phys = images_phys.cpu().numpy().astype(numpy_complex_type)
        k1 = k1.astype(numpy_float_type)
        k2 = k2.astype(numpy_float_type)
        if n_images == 1:
            image_polar = nufft2d2_cpu(k1, k2, images_phys, eps = eps, isign = isign)
        else:
            image_polar = np.zeros((n_images, n_points_polar), dtype = images_phys.dtype)
            for i in range(n_images):
                image_polar[i,:] = nufft2d2_cpu(k1, k2, images_phys[i], eps = eps, isign = isign)
        image_polar *= rescale_factor
        return torch.tensor(image_polar, dtype = torch_complex_type)

def volume_phys_to_fourier_points(
    volume : Volume,
    fourier_slices : np.ndarray | torch.Tensor,
    isign: int = -1,
    eps: float = 1.0e-6,
    precision: Precision = Precision.SINGLE,
    use_cuda: bool = True,
    output_device: torch.device | None = None,
    verbose: bool = False
):
    use_cuda = _check_cuda_available() and use_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    if not use_cuda:
        precision = Precision.DOUBLE
    output_device = device if output_device is None else output_device
    assert output_device is not None
    (torch_float_type, torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
    eps = precision.set_epsilon(eps)
    if volume.voxel_grid is None:
        raise ValueError("Volume does not have a voxel grid")
    if volume.density_physical is None:
        raise ValueError("Volume does not have a physical density")
    n_voxels = volume.voxel_grid.n_voxels
    eta = (2.0 / n_voxels * (2.0 * np.pi))#.astype(torch_float_type)
    n_modes = (n_voxels[0], n_voxels[1], n_voxels[2])
    n_voxels_total = n_voxels[0] * n_voxels[1] * n_voxels[2]
    if not len(fourier_slices.shape) in [2, 3]:
        raise ValueError("fourier_slices array is not 3D or 2D")
    if not fourier_slices.shape[-1] in [2, 3]:
        raise ValueError("fourier_slices positions are not 3D or 2D")
    z_points = None
    voxel_size_x = volume.voxel_grid.voxel_size[0]
    voxel_size_y = volume.voxel_grid.voxel_size[1]
    voxel_size_z = volume.voxel_grid.voxel_size[2]
    # voxel_size_3d = voxel_size_x * voxel_size_y * voxel_size_z
    if use_cuda:
        if not isinstance(fourier_slices, torch.Tensor):
            fourier_slices = torch.tensor(fourier_slices, dtype = torch_float_type, device = device)
        else:
            fourier_slices = fourier_slices.to(torch_float_type).to(device)
    else:
        if not isinstance(fourier_slices, np.ndarray):
            fourier_slices = fourier_slices.to(torch_float_type).cpu().numpy()
        else:
            fourier_slices = fourier_slices
    if len(fourier_slices.shape) == 2:
        fourier_slices = fourier_slices[None,:,:]
    n_images = fourier_slices.shape[0]
    n_points = fourier_slices.shape[1]
    x_points = fourier_slices[:,:,0] * eta[0]
    y_points = fourier_slices[:,:,1] * eta[1]
    z_points = fourier_slices[:,:,2] * eta[2]
    # x_points = fourier_slices[:,:,0].astype(torch_float_type)
    # y_points = fourier_slices[:,:,1].astype(torch_float_type)
    # if fourier_slices.shape[-1] == 2:
    #     z_points = np.zeros((fourier_slices.shape[0], fourier_slices.shape[1]), dtype = torch_float_type)
    # else:
    #     z_points = fourier_slices[:,:,2].astype(torch_float_type)
    # rescale_factor = 2.0 / (np.power(n_voxels_total, 1.0 / 3.0)) * (2.0 * np.pi)
    rescale_factor = 1.0 / (np.power(n_voxels_total, 1.0 / 3.0)) * (2.0 * np.pi) / 2.0
    density = volume.density_physical.to(torch_complex_type) * rescale_factor
    if use_cuda:
        image_polar = torch.zeros((n_images, n_points), dtype = torch_complex_type, device = output_device)
        from cufinufft import Plan
        # from cufinufft import nufft3d2
        try:
            plan = Plan(
                nufft_type = 2,
                n_modes = n_modes,
                n_trans = 1,
                dtype = "complex64" if precision == Precision.SINGLE else "complex128",
                eps = eps, isign = isign
            )
        except Exception as e:
            print("Error creating plan, possibly not enough GPU memory. Try using a downsampled volume")
            raise e

        density_gpu = density.to(device)
        if verbose:
            from tqdm import trange
            tmp = trange(n_images)
        else:
            tmp = range(n_images)
        for i in tmp:
            image_polar_gpu = torch.zeros(n_points, dtype = torch_complex_type, device = device)
            plan.setpts(x_points[i], y_points[i], z_points[i])
            plan.execute(density_gpu, image_polar_gpu)
            image_polar[i] = image_polar_gpu.to(output_device)
        # n_batch_points = 1
        # batch_size_points = n_points
        # for i in tmp:
        #     try:
        #         for i_batch in range(n_batch_points):
        #             batch_start = i_batch * batch_size_points
        #             batch_end = min((i_batch + 1) * batch_size_points, n_points)
        #             batch_size = batch_end - batch_start
        #             if batch_size == 0:
        #                 break
        #             image_polar_gpu_batch = torch.zeros(batch_size, dtype = torch_complex_type, device = device)
        #             # nufft3d2(
        #             #     x = x_points[i][batch_start:batch_end],
        #             #     y = y_points[i][batch_start:batch_end],
        #             #     z = z_points[i][batch_start:batch_end],
        #             #     data = density_gpu,
        #             #     out = image_polar_gpu_batch,
        #             #     eps = eps,
        #             #     isign = isign)
        #             plan.setpts(x_points[i][batch_start:batch_end], y_points[i][batch_start:batch_end], z_points[i][batch_start:batch_end])
        #             plan.execute(density_gpu, image_polar_gpu_batch)
        #             image_polar[i][batch_start:batch_end] = image_polar_gpu_batch.to(output_device)
        #     except RuntimeError: ## RuntimeError: Error creating plan, possibly memory error (?) TODO: Check this
        #         n_batch_points *= 2
        #         batch_size_points = n_points // n_batch_points
        #         if n_points % n_batch_points:
        #             batch_size_points += 1
        #         print("Error creating plan, retrying with batch size", batch_size_points)
        #         torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    else:
        # raise NotImplementedError("CPU version not implemented")
        # numpy_float_type = np.float32 if precision == Precision.SINGLE else np.float64
        # numpy_complex_type = np.complex64 if precision == Precision.SINGLE else np.complex128
        numpy_float_type = np.float64
        numpy_complex_type = np.complex128
        x_points = x_points.astype(numpy_float_type)
        y_points = y_points.astype(numpy_float_type)
        z_points = z_points.astype(numpy_float_type)
        density = density.cpu().numpy().astype(numpy_complex_type)
        image_polar = np.zeros((n_images, n_points), dtype = numpy_complex_type)
        if verbose:
            from tqdm import trange
            tmp = trange(n_images)
        else:
            tmp = range(n_images)
        for i in tmp:
            image_polar[i] = nufft3d2_cpu(
                x = x_points[i],
                y = y_points[i],
                z = z_points[i],
                f = density,
                isign = isign, eps = eps)
        image_polar = torch.tensor(image_polar, dtype = torch_complex_type, device = output_device)
    return image_polar#.to(output_device)