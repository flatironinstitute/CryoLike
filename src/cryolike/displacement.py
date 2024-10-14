from cryolike.util.types import FloatArrayType
import numpy as np
import torch
from cryolike.polar_grid import PolarGrid
from cryolike.array import to_torch
from cryolike.util.enums import Precision

def get_possible_displacements(
    max_displacement,
    n_displacements,
    has_zero_displacement = True
):
    
    if n_displacements <= 1:
        n_displacements = 1
        x_displacements = np.array([0])
        y_displacements = np.array([0])
        return n_displacements, x_displacements, y_displacements
    
    n_gridpoints = int(1 + np.floor(np.sqrt(n_displacements)))
    continue_flag = True
    X_: np.ndarray
    Y_: np.ndarray
    while continue_flag:
        x_ = np.linspace(- max_displacement, + max_displacement, n_gridpoints)
        y_ = np.linspace(- max_displacement, + max_displacement, n_gridpoints)
        X_, Y_ = np.meshgrid(x_, y_)
        R_ = np.sqrt(X_ ** 2 + Y_ ** 2)
        indices = np.where(R_ <= max_displacement)
        if np.size(indices) >= n_displacements:
            continue_flag = False
            break
        n_gridpoints = n_gridpoints + 1
    n_displacements = indices[0].size
    x_displacements = X_[indices]
    y_displacements = Y_[indices]
    # if not n_gridpoints % 2:
    ## if not already present, add zero displacement
    if has_zero_displacement and not np.any(np.logical_and(x_displacements == 0.0, y_displacements == 0.0)):
        x_displacements = np.concatenate((np.array([0]), x_displacements))
        y_displacements = np.concatenate((np.array([0]), y_displacements))
        n_displacements += 1

    return n_displacements, x_displacements, y_displacements


def get_possible_displacements_grid(
    max_displacement : float,
    n_gridpoints_x : int, 
    n_gridpoints_y : int,
) -> tuple[int, FloatArrayType, FloatArrayType]:
    n_displacements = n_gridpoints_x * n_gridpoints_y
    if n_displacements <= 1:
        n_displacements = 1
        x_displacements = np.array([0])
        y_displacements = np.array([0])
        return n_displacements, x_displacements, y_displacements
    x_ = np.linspace(- max_displacement, + max_displacement, n_gridpoints_x, endpoint=True)
    y_ = np.linspace(- max_displacement, + max_displacement, n_gridpoints_y, endpoint=True)
    X_, Y_ = np.meshgrid(x_, y_)
    n_displacements = X_.size
    x_displacements = X_.flatten()
    y_displacements = Y_.flatten()
    
    return n_displacements, x_displacements, y_displacements


def translation_kernel_fourier(
    polar_grid: PolarGrid,
    x_displacements: np.ndarray | torch.Tensor,
    y_displacements: np.ndarray | torch.Tensor,
    precision: Precision = Precision.DEFAULT,
    device: str | torch.device = "cuda"
) -> torch.Tensor:
    
    if not torch.cuda.is_available():
        device = "cpu"
    
    n_shells = polar_grid.n_shells
    n_inplanes = polar_grid.n_inplanes
    x_displacements = to_torch(x_displacements, precision = precision, device = device)
    y_displacements = to_torch(y_displacements, precision = precision, device = device)
    # x_displacements = x_displacements.to(device)
    # y_displacements = y_displacements.to(device)
    # xy_displacements_unique, indices = np.unique(np.stack((x_displacements, y_displacements), axis = 1), axis = 0, return_index = True)
    # indices = torch.tensor(indices, dtype = torch.long, device = device)
    xy_displacements_unique, indices = torch.unique(torch.stack((x_displacements, y_displacements), dim = 1), dim = 0, return_inverse = True)
    x_displacements_unique = xy_displacements_unique[:, 0]
    y_displacements_unique = xy_displacements_unique[:, 1]
    # x_displacements_unique = to_torch(x_displacements_unique, precision = precision, device = device)
    # y_displacements_unique = to_torch(y_displacements_unique, precision = precision, device = device)
    assert polar_grid.x_points is not None
    assert polar_grid.y_points is not None
    polar_x = to_torch(polar_grid.x_points, precision = precision, device = device) * (-2.0 * np.pi * 1j)
    polar_y = to_torch(polar_grid.y_points, precision = precision, device = device) * (-2.0 * np.pi * 1j)
    translation_kernel = torch.exp(polar_x[None, :] * x_displacements_unique[:, None] + polar_y[None, :] * y_displacements_unique[:, None])
    # weight_points_device = to_torch(polar_grid.weight_points, precision = precision, device = device).unsqueeze(0)
    # translation_kernel /= torch.sum(translation_kernel * weight_points_device, dim = 1, keepdim = True)[:, None]
    translation_kernel = translation_kernel.reshape(-1, n_shells, n_inplanes)
    translation_kernel = translation_kernel[indices]
    return translation_kernel