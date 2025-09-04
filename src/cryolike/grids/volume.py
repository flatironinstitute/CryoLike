import mrcfile
import numpy as np
import torch
from typing import Optional, cast

from .cartesian_grid import CartesianGrid3D
from .sphere_grid import SphereGrid
from .data_transfer_classes import PhysicalVolume, FourierVolume

from cryolike.util import (
    FloatArrayType,
    project_descriptor,
    TargetType,
)


class Volume:
    """Class representing a volume in physical or Fourier space, with methods manipulating them.
    
    Attributes:
        box_size (FloatArrayType): Size of the (Cartesian-space) viewing port.
        voxel_grid (CartesianGrid3D): A grid describing the space in which physical images reside.
        sphere_grid (SphereGrid): A grid describing the space in which Fourier images reside.
        density_physical (torch.Tensor): Cartesian-space density as a voxel-value array of [X-index x Y-index x Z-index].
        density_fourier (torch.Tensor): Fourier-space density as a point-value array of [point-index].
    """
    
    box_size: FloatArrayType
    voxel_grid: CartesianGrid3D
    sphere_grid: SphereGrid
    density_physical: Optional[torch.Tensor]
    density_fourier: Optional[torch.Tensor]
    filename: Optional[str]

    def __init__(
        self,
        density_physical_data: Optional[PhysicalVolume] = None,
        density_fourier_data: Optional[FourierVolume] = None,
        box_size: Optional[float | FloatArrayType] = None,
    ) -> None:
        if (density_physical_data is None) and (density_fourier_data is None):
            raise ValueError("Must pass at least one of Fourier and physical density.")
        
        voxel_grid_set = False
        if density_physical_data is not None:
            if density_physical_data.voxel_grid is None:
                raise ValueError("Can't happen: this should be fixed in the constructor.")
            self.voxel_grid = density_physical_data.voxel_grid
            voxel_grid_set = True
            self.density_physical = density_physical_data.density_physical
        else:
            self.density_physical = None
            
        if density_fourier_data is not None:
            self.sphere_grid = density_fourier_data.sphere_grid
            self.density_fourier = density_fourier_data.density_fourier
        else:
            self.density_fourier = None
            
        if box_size is not None:
            _box_size = cast(FloatArrayType, project_descriptor(box_size, "box_size", 3, TargetType.FLOAT))
            self.box_size = _box_size
        elif voxel_grid_set == True:
            self.box_size = self.voxel_grid.n_voxels * self.voxel_grid.voxel_size
        else:
            self.box_size = np.array([2., 2., 2.], dtype = np.float64)
        
        self._check_density()
        
    
    def _ensure_phys_density(self):
        if self.density_physical is None:
            raise ValueError("Physical density is not set.")
        if self.voxel_grid is None:
            raise ValueError("Voxel grid is not set.")
    
    
    def _ensure_fourier_density(self):
        if self.density_fourier is None:
            raise ValueError("Fourier density is not set.")
        if self.sphere_grid is None:
            raise ValueError("Sphere grid is not set.")
    
    
    def _check_density(self):
        if self.density_physical is None and self.density_fourier is None:
            raise ValueError("Density is not set.")
        if self.density_physical is not None:
            if self.voxel_grid is None:
                raise ValueError("Voxel grid is not defined for physical density.")
            if len(self.density_physical.shape) != 3:
                raise ValueError("Physical density must be a 3D array.")
            if not np.allclose(self.voxel_grid.n_voxels, self.density_physical.shape):
                raise ValueError("Dimension mismatch: n_voxels {self.voxel_grid.n_voxels} but density shape is {self.density_physical.shape}")
            if not np.allclose(self.box_size, self.voxel_grid.box_size):
                print(" %% Warning: Box size does not match the number of voxels and the voxel size. box_size = {self.box_size}, voxel_grid.box_size = {self.voxel_grid.box_size}")
        if self.density_fourier is not None:
            if self.sphere_grid is None:
                raise ValueError("Sphere grid is not defined for Fourier density.")
            if len(self.density_fourier.shape) != 1:
                raise ValueError("Fourier density must be a 1D array.")
            if not np.allclose(self.sphere_grid.n_points, self.density_fourier.shape):
                raise ValueError("Dimension mismatch: n_points {self.sphere_grid.n_points} but density shape is {self.density_fourier.shape}")
            
        pass
    
    
    @classmethod
    def from_mrc(cls, filename: str, voxel_size: float | list[float] | FloatArrayType | None = None, device: str | torch.device = 'cpu'):
        """Create a new physical density from an MRC file.

        Args:
            filename (str): Name of MRC file to load. Must end with a .mrc or .mrcs extension.
            voxel_size (Optional[float | list[float] | FloatArrayType]): Sizes of the volume voxels (Angstrom). If set,
                will override the values in the MRC file; if unset, the file values will be used.
            device (str | torch.device, optional): Device to use for the resulting image Tensor. Defaults to 'cpu'.

        Raises:
            ValueError: If a non-MRC file extension is passed.
            ValueError: If the voxel_size is not set and the existing MRC file has non-positive pixel sizes.

        Returns:
            Images: An Images image collection, with the physical images and grid populated per the saved data.
        """
        if not (filename.endswith(('mrc', 'mrcs', 'map'))):
            raise ValueError("Invalid file format. Only .mrc or .mrcs or .map files are supported.")
        
        with mrcfile.open(filename) as mrc:
            assert isinstance(mrc.data, np.ndarray)
            density_phys = mrc.data.copy()
            if len(density_phys.shape) == 2:
                density_phys = density_phys[None, :, :]
            if voxel_size is not None:
                _voxel_size = cast(FloatArrayType, project_descriptor(voxel_size, "pixel size", 2, TargetType.FLOAT))
            else:
                _voxel_size = cast(FloatArrayType, np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype = float))
                if np.any(_voxel_size <= 0):
                    raise ValueError(f"MRC file {filename} contains non-positive pixel sizes.")
            density_phys = torch.from_numpy(density_phys).to(device)
            density_physical_data = PhysicalVolume(density_physical=density_phys, voxel_size=_voxel_size)
        volume = cls(density_physical_data=density_physical_data)
        volume.filename = filename
        return volume
    
    
    @classmethod
    def from_tensor_physical(cls, density_physical: torch.Tensor, voxel_size: float | list[float] | FloatArrayType):
        """Create a new physical density from an MRC file.

        Args:
            density_physical (torch.Tensor): Physical density data to use in the volume class.
            voxel_size (Optional[float  |  list[float]  |  FloatArrayType]): Sizes of the volume voxels (Angstrom).
        """
        density_physical_data = PhysicalVolume(density_physical=density_physical, voxel_size=voxel_size)
        volume = cls(density_physical_data=density_physical_data)
        return volume
    
    
    @classmethod
    def from_tensor_fourier(cls, density_fourier: torch.Tensor, sphere_grid: SphereGrid):
        """Create a new physical density from an MRC file.

        Args:
            density_physical (torch.Tensor): Physical density data to use in the volume class.
            voxel_size (Optional[float  |  list[float]  |  FloatArrayType]): Sizes of the volume voxels (Angstrom).
        """
        density_fourier_data = FourierVolume(density_fourier=density_fourier, sphere_grid=sphere_grid)
        volume = cls(density_fourier_data=density_fourier_data)
        return volume
    
    
    # def generate_atomic_volume_torch(
    #     self,
    #     atomic_model : AtomicModel = None,
    #     device = "cuda",
    #     precision = "single",
    #     verbose = False
    # ):
        
    #     ## check if device is available
    #     if device != "cpu":
    #         if not torch.cuda.is_available():
    #             device = "cpu"
    #     device = torch.device(device)
        
    #     if precision == "single":
    #         self.torch_float_type = torch.float32
    #         self.torch_complex_type = torch.complex64
    #     elif precision == "double":
    #         self.torch_float_type = torch.float64
    #         self.torch_complex_type = torch.complex128
        
    #     if atomic_model is None:
    #         print("Atomic model not specified. Using default testing set of atomic model.")
    #         atomic_model = AtomicModel()

    #     n_atoms = atomic_model.n_atoms
    #     atom_radii_scaled = torch.tensor(atomic_model.atom_radii, dtype = self.torch_float_type, device = device) / self.box_size * 2.0
    #     atomic_coordinates = torch.tensor(atomic_model.atomic_coordinates, dtype = self.torch_float_type, device = device)
    #     atomic_coordinates_scaled = atomic_coordinates.T / self.box_size * 2.0 * (- 2.0 * np.pi)
    #     pi_atom_radii_sq_times_two = 2.0 * (np.pi * atom_radii_scaled) ** 2
    #     norm = (2 * np.pi) ** (1.5) * atom_radii_scaled ** 3 / n_atoms
    #     # norm = (2 * np.pi) ** (- 1.5) / atom_radii_scaled ** 3 / n_atoms * (np.pi * 2 * atom_radii_scaled ** 2) ** (1.5)
    #     if type(norm) == np.ndarray:
    #         norm = torch.tensor(norm, dtype = self.torch_float_type, device = device)[None, :]
    #     self.n_points = self.sphere_grid.n_points
    #     if verbose:
    #         print("Generating atomic volume with {} atoms.".format(n_atoms))
    #     self.density_torch = torch.zeros(self.n_points, dtype = self.torch_complex_type, device = "cpu")
        
    #     n_batch = 1
    #     batchSize = self.n_points
    #     batch_size_determined = False
    #     while not batch_size_determined:
    #         try:
    #             batchSize = self.n_points // n_batch + (self.n_points % n_batch > 0) * 1
    #             sphere_xyz_points_device = torch.empty(batchSize, dtype = self.torch_float_type, device = device)
    #             rsq_points = torch.empty(batchSize, dtype = self.torch_float_type, device = device)
    #             # if type(atom_radii_scaled) == float:
    #             #     gaussKernelAtoms = torch.empty(batchSize, dtype = self.torch_float_type, device = device)
    #             # elif type(atom_radii_scaled) == np.ndarray:
    #             gaussKernelAtoms = torch.empty((batchSize, n_atoms), dtype = self.torch_float_type, device = device)
    #             kdotr = torch.empty((batchSize, n_atoms), dtype = self.torch_float_type, device = device)
    #             volumeFourierBatch = torch.empty(batchSize, dtype = self.torch_complex_type, device = device)
    #             buffer = torch.empty((8, batchSize, n_atoms), dtype = self.torch_float_type, device = device)
    #             batch_size_determined = True
    #             del buffer
    #         except RuntimeError:
    #             n_batch += 1
    #     if verbose:
    #         print("Memory allocated for number of batch: {} of size: {}".format(n_batch, batchSize))
        
    #     tmp = range(n_batch)
    #     if verbose:
    #         from tqdm import tqdm
    #         tmp = tqdm(tmp)
    #     assert self.sphere_grid.cartesian_points is not None
    #     (_, _, _, xyz_points) = self.sphere_grid.cartesian_points
    #     for ibatch in tmp:
    #         batchStart = ibatch * batchSize
    #         batchEnd = min((ibatch + 1) * batchSize, self.sphere_grid.n_points)
    #         sphere_xyz_points_device = torch.tensor(xyz_points[batchStart:batchEnd], dtype = self.torch_float_type, device = device)
    #         rsq_points = torch.tensor(self.sphere_grid.radius_points[batchStart:batchEnd], dtype = self.torch_float_type, device = device) ** 2
    #         kdotr = torch.mm(sphere_xyz_points_device, atomic_coordinates_scaled)
    #         if type(norm) != float:
    #             rsq_points = rsq_points[:, None]
    #         gaussKernelAtoms = torch.exp(- rsq_points * pi_atom_radii_sq_times_two) * norm
    #         if len(gaussKernelAtoms.shape) == 1:
    #             gaussKernelAtoms = gaussKernelAtoms[:, None]
    #         kdotr = torch.exp(1j * kdotr) * gaussKernelAtoms
    #         volumeFourierBatch = torch.sum(kdotr, dim = 1)
    #         self.density_torch[batchStart:batchEnd] = volumeFourierBatch.cpu()
            
    #     self.density = self.density_torch.numpy()
        
    #     return self.density_torch


    # def project_to_physical(
    #     self,
    #     voxel_grid : CartesianGrid3D = None,
    #     use_cuda : bool = False
    # ):
    #     eps_nufft = 1.0e-6
    #     # eta = np.pi / n_voxels
    #     eta = (2.0 * np.pi) / voxel_grid.n_voxels * 2.0
    #     density_physical = None
    #     assert self.sphere_grid.cartesian_points is not None
    #     (x_pts, y_pts, z_pts, _) = self.sphere_grid.cartesian_points
    #     if use_cuda:
    #         import cupy as cp
    #         from cufinufft import nufft3d1
    #         density_weight = self.density.flatten() * self.sphere_grid.weight_points
    #         density_physical = nufft3d1(
    #             x = cp.asarray(x_pts * eta[0]),
    #             y = cp.asarray(y_pts * eta[1]),
    #             z = cp.asarray(z_pts * eta[2]),
    #             data = cp.asarray(density_weight),
    #             n_modes = (voxel_grid.n_voxels[0], voxel_grid.n_voxels[1], voxel_grid.n_voxels[2]),
    #             isign = +1,
    #             eps = eps_nufft
    #         ) * (2 * np.pi) ** (1.5)
    #         density_physical = density_physical.get()
    #     else:
    #         from finufft import nufft3d1
    #         density_physical = nufft3d1(
    #             x = x_pts * eta[0],
    #             y = y_pts * eta[1],
    #             z = z_pts * eta[2],
    #             c = self.density.flatten() * self.sphere_grid.weight_points,
    #             n_modes = (voxel_grid.n_voxels[0], voxel_grid.n_voxels[1], voxel_grid.n_voxels[2]),
    #             isign = +1,
    #             eps = eps_nufft
    #         ) * (2 * np.pi) ** (1.5)
    #     density_physical = VolumePhysical(voxel_grid = voxel_grid, density = density_physical)
    #     return density_physical


# class VolumePhysical:

#     def __init__(
#         self,
#         box_size = None, ## in physical space, in Angstrom
#         n_voxels = None,
#         voxel_size = None, ## in physical space, in Angstrom
#         voxel_grid : CartesianGrid3D = None,
#         mrc_file : str = None,
#         density = None,
#     ):
#         self.box_size = box_size
#         self.n_voxels = n_voxels
#         self.voxel_size = voxel_size
#         self.voxel_grid = voxel_grid
#         self.density = density
#         if mrc_file is not None:
#             if density is not None:
#                 print(" %% Warning: Density provided. Ignoring mrc file.")
#             else:
#                 self.read_from_mrc(mrc_file)
        
#         num_param_provided = (self.box_size is not None) + (self.n_voxels is not None) + (self.voxel_size is not None)
#         if self.voxel_grid is None:
#             if num_param_provided == 3:
#                 if not np.allclose(self.box_size, self.n_voxels * self.voxel_size, rtol = 1e-3):
#                     print(" %% Warning: Box size does not match the number of voxels and the voxel size. box_size = ", self.box_size, " n_voxels * voxel_size = ", self.n_voxels * self.voxel_size)
#             elif num_param_provided != 2:
#                 raise ValueError(" %% Error: Specify 2 of the 3 parameters: box_size, n_voxels, voxel_size. ")
#             if self.box_size is None:
#                 self.box_size = self.n_voxels * self.voxel_size
#             if self.n_voxels is None:
#                 self.n_voxels = self.box_size / self.voxel_size
#             if self.voxel_size is None:
#                 self.voxel_size = self.box_size / self.n_voxels
#             self.check_grid_param()
#             self.build_grid()
#         else:
#             if num_param_provided > 0:
#                 print(" %% Warning: Grid provided. Ignoring other provided parameters.")
#             self.voxel_grid = voxel_grid
#             self.n_voxels = self.voxel_grid.n_voxels
#             self.voxel_size = self.voxel_grid.voxel_size
#             self.box_size = self.n_voxels * self.voxel_size
#         self.check_grid_param()
#         # self.check_density()
        
#         pass

#     def check_grid_param(self):

#         if self.box_size is None:
#             raise ValueError(" %% Error: Box size is not specified. ")
#         if np.isscalar(self.box_size):
#             self.box_size = np.array([self.box_size, self.box_size, self.box_size], dtype = np.float64)
#         if self.box_size.size != 3:
#             raise ValueError(" %% Error: Box size must have length 3. ")
        
#         if self.n_voxels is None:
#             raise ValueError(" %% Error: Number of voxels is not specified. ")
#         if np.isscalar(self.n_voxels):
#             self.n_voxels = np.array([self.n_voxels, self.n_voxels, self.n_voxels], dtype = np.int64)
#         if self.n_voxels.size != 3:
#             raise ValueError(" %% Error: Number of voxels must have length 3. ")
        
#         if self.voxel_size is None:
#             raise ValueError(" %% Error: Voxel size is not specified. ")
#         if np.isscalar(self.voxel_size):
#             self.voxel_size = np.array([self.voxel_size, self.voxel_size, self.voxel_size], dtype = np.float64)
#         if self.voxel_size.size != 3:
#             raise ValueError(" %% Error: Voxel size must have length 3. ")
        
#         return
    
#     def build_grid(self):
#         self.voxel_grid = CartesianGrid3D(n_voxels = self.n_voxels, voxel_size = self.voxel_size, endpoint = False)
#         pass
    
#     def check_density(self):
#         if self.density is None:
#             raise ValueError("Density is not specified.")
#         if len(self.density.shape) != 3:
#             raise ValueError("Density must be a 3D array.")
#         if not np.allclose(np.array(self.density.shape), self.n_voxels):
#             print("Density size does not match the number of voxel_grid points. Regenerating voxel_grid points...")
#             self.voxel_grid = CartesianGrid3D(n_voxels = np.array(self.density.shape), voxel_size = self.voxel_size, endpoint = False)
#         pass
    
#     def read_from_mrc(
#         self,
#         filename : str
#     ):
#         import mrcfile
#         with mrcfile.open(filename) as mrc:
#             self.density = torch.from_numpy(mrc.data)
#             self.n_voxels = np.array(self.density.shape, dtype = np.int64)
#             self.voxel_size = np.array([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype = np.float64)
#             self.box_size = self.n_voxels * self.voxel_size
#         self.check_density()
#         return
    
#     def change_volume_size(
#         self,
#         n_voxels : np.ndarray = None,
#         padding_mode : str = 'constant'
#     ):
#         if n_voxels is None:
#             raise ValueError(" %% Error: Number of voxels is not specified. ")
#         if np.isscalar(n_voxels):
#             n_voxels = np.array([n_voxels, n_voxels, n_voxels], dtype = np.int64)
#         if type(n_voxels) != np.ndarray:
#             n_voxels = np.array(n_voxels, dtype = np.int64)
#         paddings = []
#         for i in range(3):
#             if n_voxels[i] < self.n_voxels[i]:
#                 self.density = self.density[:n_voxels[i],:n_voxels[i],:n_voxels[i]]
#                 paddings.append((0, 0))
#             elif n_voxels[i] > self.n_voxels[i]:
#                 padbefore = (n_voxels[i] - self.n_voxels[i]) // 2
#                 padafter = n_voxels[i] - self.n_voxels[i] - padbefore
#                 paddings.append((padbefore, padafter))
#             else:
#                 paddings.append((0, 0))
#         self.density = np.pad(self.density, paddings, padding_mode)
#         self.n_voxels = n_voxels
#         self.box_size = n_voxels * self.voxel_size
#         return
        
#     def generate_atomic_volume_torch(
#         self,
#         atomic_model : AtomicModel = None,
#         device = "cuda",
#         precision = "single",
#         verbose = False
#     ):
#         ## check if device is available
#         if device != "cpu":
#             if not torch.cuda.is_available():
#                 device = "cpu"
#         device = torch.device(device)
        
#         if precision == "single":
#             self.torch_float_type = torch.float32
#             self.torch_complex_type = torch.complex64
#         elif precision == "double":
#             self.torch_float_type = torch.float64
#             self.torch_complex_type = torch.complex128
        
#         if atomic_model is None:
#             print("Atomic model not specified. Using default testing set of atomic model.")
#             atomic_model = AtomicModel()

#         atom_radii = torch.tensor(atomic_model.atom_radii, dtype = self.torch_float_type, device = device)
#         atomic_coordinates = torch.tensor(atomic_model.atomic_coordinates, dtype = self.torch_float_type, device = device).T
#         n_atoms = atomic_model.n_atoms
#         atom_radii = atom_radii[None,None,None,:]
#         self.density_torch = torch.zeros((self.n_voxels[0], self.n_voxels[1], self.n_voxels[2]), dtype = self.torch_float_type, device = "cpu")
#         x_axis = torch.tensor(self.voxel_grid.x_axis, dtype = self.torch_float_type, device = device)
#         y_axis = torch.tensor(self.voxel_grid.y_axis, dtype = self.torch_float_type, device = device)
#         z_axis = torch.tensor(self.voxel_grid.z_axis, dtype = self.torch_float_type, device = device)
#         n_batch = 1
#         boxSizeX = self.n_voxels[0]
#         batchSize = boxSizeX
#         batch_size_determined = False
#         while not batch_size_determined:
#             try:
#                 batchSize = boxSizeX // n_batch + (boxSizeX % n_batch > 0) * 1
#                 dx2 = torch.zeros((batchSize, self.n_voxels[1], self.n_voxels[2], n_atoms), dtype = self.torch_float_type, device = device)
#                 dy2 = torch.zeros((batchSize, self.n_voxels[1], self.n_voxels[2], n_atoms), dtype = self.torch_float_type, device = device)
#                 dz2 = torch.zeros((batchSize, self.n_voxels[1], self.n_voxels[2], n_atoms), dtype = self.torch_float_type, device = device)
#                 r2____ = torch.zeros((batchSize, self.n_voxels[1], self.n_voxels[2], n_atoms), dtype = self.torch_float_type, device = device)
#                 g___ = torch.zeros((batchSize, self.n_voxels[1], self.n_voxels[2]), dtype = self.torch_float_type, device = device)
#                 volumePhysicalBatch = torch.zeros((batchSize, self.n_voxels[1], self.n_voxels[2]), dtype = self.torch_float_type, device = device)
#                 buffer = torch.zeros((4, batchSize, self.n_voxels[1], self.n_voxels[2], n_atoms), dtype = self.torch_float_type, device = device)
#                 batch_size_determined = True
#                 del buffer
#             except RuntimeError:
#                 n_batch += 1
#         if verbose:
#             print("Memory allocated for number of batch: {} of size: {}".format(n_batch, batchSize))
#         if n_batch == 1:
#             dx2 = (x_axis[:,None] - atomic_coordinates[0,:][None,:]) ** 2
#             dy2 = (y_axis[:,None] - atomic_coordinates[1,:][None,:]) ** 2
#             dz2 = (z_axis[:,None] - atomic_coordinates[2,:][None,:]) ** 2
#             r2____ = dx2[:,None,None,:] + dy2[None,:,None,:] + dz2[None,None,:,:]
#             g___ = ((2 * np.pi) ** (- 1.5) / atom_radii ** 3 / n_atoms) * torch.exp(- r2____ / (2 * atom_radii ** 2))
#             self.density_torch = torch.sum(g___, axis = 3).cpu()
#         else:
#             tmp = range(n_batch)
#             if verbose:
#                 from tqdm import tqdm
#                 tmp = tqdm(tmp)
#             for ibatch in tmp:
#                 batchStart = ibatch * batchSize
#                 batchEnd = min((ibatch + 1) * batchSize, boxSizeX)
#                 dx2 = (x_axis[batchStart:batchEnd,None] - atomic_coordinates[0,:][None,:]) ** 2
#                 dy2 = (y_axis[:,None] - atomic_coordinates[1,:][None,:]) ** 2
#                 dz2 = (z_axis[:,None] - atomic_coordinates[2,:][None,:]) ** 2
#                 r2____ = dx2[:,None,None,:] + dy2[None,:,None,:] + dz2[None,None,:,:]
#                 g___ = ((2 * np.pi) ** (- 1.5) / atom_radii ** 3 / n_atoms) * torch.exp(- r2____ / (2 * atom_radii ** 2))
#                 volumePhysicalBatch = torch.sum(g___, dim = 3)
#                 self.density_torch[batchStart:batchEnd,:,:] = volumePhysicalBatch.cpu()
#         self.density = self.density_torch.numpy()
#         return self.density_torch

#     def downsample_volume(
#         self,
#         downsample_factor = 2,      # must be an integer
#         coarsening_method = 'sum',  # or 'mean'
#         replace_volume = True
#     ):
        
#         # Downsample a volume by a factor of [downsample_factor] in each dimension.
#         downsampled_n_voxel = self.n_voxels // downsample_factor
#         downsampled_shape = (downsampled_n_voxel, downsampled_n_voxel, downsampled_n_voxel)

#         if coarsening_method == 'sum':
#             coarsening_func = np.sum
#         elif coarsening_method == 'mean' or coarsening_method == 'average':
#             coarsening_func = np.mean
#         else:
#             raise ValueError('Unknown coarsening method: {}'.format(coarsening_method))

#         downsampled_volume = np.zeros(downsampled_shape, dtype = self.density.dtype)
#         for i in range(downsampled_shape[0]):
#             for j in range(downsampled_shape[1]):
#                 for k in range(downsampled_shape[2]):
#                     downsampled_volume[i,j,k] = coarsening_func(self.density[
#                         i*downsample_factor:(i+1)*downsample_factor,
#                         j*downsample_factor:(j+1)*downsample_factor,
#                         k*downsample_factor:(k+1)*downsample_factor,
#                     ])

#         if replace_volume:
#             self.density = downsampled_volume
        
#         return downsampled_volume

#     def project_to_fourier(
#         self,
#         sphere_grid : SphereGrid = None,
#         use_cuda : bool = False
#     ):
#         print("Performing NUFFT on the volume density...")
#         voxel_size_3d = (2.0 / self.n_voxels[0]) * (2.0 / self.n_voxels[1]) * (2.0 / self.n_voxels[2])
#         eps_nufft = 1.0e-6
#         eta = (2.0 * np.pi) / self.n_voxels * 2.0
#         density_fourier = None
#         assert sphere_grid.cartesian_points is not None
#         (x_pts, y_pts, z_pts, _) = sphere_grid.cartesian_points
#         if use_cuda:
#             import cupy as cp
#             from cufinufft import nufft3d2
#             density_fourier = nufft3d2(
#                 x = cp.asarray(x_pts * eta[0]),
#                 y = cp.asarray(y_pts * eta[1]),
#                 z = cp.asarray(z_pts * eta[2]),
#                 data = cp.asarray(self.density.astype(np.complex128) * voxel_size_3d),
#                 isign = -1,
#                 eps = eps_nufft) * (2 * np.pi) ** (-1.5)
#             density_fourier = density_fourier.get()
#         else:
#             from finufft import nufft3d2
#             density_fourier = nufft3d2(
#                 x = x_pts * eta[0],
#                 y = y_pts * eta[1],
#                 z = z_pts * eta[2],
#                 f = self.density.astype(np.complex128) * voxel_size_3d,
#                 isign = -1,
#                 eps = eps_nufft) * (2 * np.pi) ** (-1.5)
#         density_fourier = VolumeFourier(sphere_grid = sphere_grid, density = density_fourier)
#         return density_fourier
