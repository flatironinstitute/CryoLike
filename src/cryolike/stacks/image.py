from typing import Literal, Optional, cast
import numpy as np
import mrcfile
import torch

from cryolike.grids import (
    CartesianGrid2D,
    PolarGrid,
    FourierImages,
    PhysicalImages
)
from cryolike.microscopy import (
    CTF,
    translation_kernel_fourier,
    fourier_polar_to_cartesian_phys,
    cartesian_phys_to_fourier_polar,
    translation_kernel_fourier,
    ViewingAngles,
)
from .template import Templates
from cryolike.util import (
    ensure_positive,
    Cartesian_grid_2d_descriptor,
    ComplexArrayType,
    FloatArrayType,
    get_imgs_max,
    IntArrayType,
    NormType,
    Precision,
    project_descriptor,
    TargetType,
    to_torch,
)


# Nobody is asking for this yet
# def read_image_npy(filename: str) -> FloatArrayType:
#     if not filename.endswith('.npy'):
#         raise ValueError(f'Image file name {filename} must end with .npy extension.')
#     return np.load(filename)


def _project_displacements(displacements: torch.Tensor | FloatArrayType | float) -> torch.Tensor:
    if isinstance(displacements, torch.Tensor):
        return displacements
    if isinstance(displacements, np.ndarray):
        return torch.from_numpy(displacements)
    # return torch.from_numpy(np.array([displacements], dtype = float)[None,:])
    return torch.tensor([[displacements]], dtype = torch.double)


Displacement_raw_type = torch.Tensor | FloatArrayType | float
def _verify_displacements(x_disp_raw: Displacement_raw_type, y_disp_raw: Displacement_raw_type, n_imgs: int) -> tuple[torch.Tensor, torch.Tensor]:
    x_disp = _project_displacements(x_disp_raw)
    y_disp = _project_displacements(y_disp_raw)
    if x_disp.shape[0] != y_disp.shape[0]:
        raise ValueError("x_displacements and y_displacements must have the same length")
    n_disp = x_disp.shape[0]
    if n_disp == n_imgs:
        return (x_disp, y_disp)
    if n_disp == 1: # manually broadcast
        return (x_disp[None,:], y_disp[None,:])
    raise ValueError("Number of displacements must be 1 or equal to the number of images.")


class Images:
    """Class representing a collection of (Cartesian-space or Fourier-space) images, with methods for manipulating them.

    Attributes:
        box_size (FloatArrayType): Size of the (Cartesian-space) viewing port.
        phys_grid (CartesianGrid2D): A grid describing the space in which physical images reside.
        polar_grid (PolarGrid): A grid describing the space in which Fourier images reside.
        images_phys (Optional[torch.Tensor]): Cartesian-space images as a pixel-value array of [image x X-index x Y-index].
        images_fourier (Optional[torch.Tensor]): Fourier-space images as a complex-valued array of [image x ? x ?].
        n_images (int): Count of images in the collection.
        ctf (CTF | None): Contrast transfer function applied to the images, if any.
        viewing_angles (Optional[ViewingAngles]): Optimal viewing angle, if set.
        filename (str | None): The name of the file of origin of these images, if they were read from an MRC file.
    """
    box_size: FloatArrayType
    phys_grid: CartesianGrid2D
    polar_grid: PolarGrid
    images_phys: Optional[torch.Tensor] # if set, should be ndarray of [n_imgs, pixel_column, pixel_row]
    images_fourier: Optional[torch.Tensor]
    n_images: int
    ctf: CTF | None
    viewing_angles: Optional[ViewingAngles]
    filename: str | None

    def __init__(self,
        phys_images_data: Optional[PhysicalImages] = None,
        fourier_images_data: Optional[FourierImages] = None,
        box_size: Optional[float | FloatArrayType] = None,
            # box size trumps if provided, else use physical & derive from physical grid if provided;
            # if no box size (i.e. b/c only passed images_fourier) then default to 2x2
        polar_grid: Optional[PolarGrid] = None,
        phys_grid: Optional[CartesianGrid2D] = None,
        viewing_angles: Optional[ViewingAngles] = None,
        ctf : Optional[CTF] = None,
    ) -> None:
        if (phys_images_data is None and fourier_images_data is None):
            raise ValueError("Must pass at least one of Fourier and cartesian images.")

        phys_grid_set = False
        if phys_images_data is not None:
            if phys_images_data.phys_grid is None:
                raise ValueError("Can't happen: this should be fixed in the constructor.")
            self.phys_grid = phys_images_data.phys_grid
            phys_grid_set = True
            self.images_phys = phys_images_data.images_phys
            self.n_images = self.images_phys.shape[0]
        else:
            self.images_phys = None
        if phys_grid is not None and phys_images_data is None:
            self.phys_grid = phys_grid
            phys_grid_set = True

        if fourier_images_data is not None:
            self.polar_grid = fourier_images_data.polar_grid
            self.images_fourier = fourier_images_data.images_fourier
            self.n_images = self.images_fourier.shape[0]
        else:
            self.images_fourier = None
        if polar_grid is not None and fourier_images_data is None:
            self.polar_grid = polar_grid
        
        if box_size is not None:
            _box_size = cast(FloatArrayType, project_descriptor(box_size, "box_size", 2, TargetType.FLOAT))
            self.box_size = _box_size
        elif phys_grid_set == True:
            self.box_size = self.phys_grid.box_size
        else:
            self.box_size = np.array([2., 2.])

        self._check_image_array()
        self.set_ctf(ctf)
        self.viewing_angles = viewing_angles


    def set_ctf(self, ctf: Optional[CTF]):
        self.ctf = ctf


    def _ensure_phys_images(self):
        if self.images_phys is None:
            raise ValueError("Physical images not found.")
        if self.phys_grid is None:
            raise ValueError("No physical grid found.")
        

    def _ensure_fourier_images(self):
        if self.images_fourier is None:
            raise ValueError("Fourier images not found.")
        if self.polar_grid is None:
            raise ValueError("No polar grid found.")


    def _check_image_array(self):
        if self.images_phys is None and self.images_fourier is None:
            raise ValueError("No image array found")
        if self.images_phys is not None:
            if self.phys_grid is None:
                # can't happen
                raise ValueError("Physical grid is not defined for physical images")
            if len(self.images_phys.shape) == 2:
                self.images_phys = self.images_phys[None,:,:]
            if len(self.images_phys.shape) != 3:
                raise ValueError("Invalid shape for images.")
            if (self.n_images != self.images_phys.shape[0]):
                raise ValueError(f"Images object lists image count of {self.n_images} but the physical images array has {self.images_phys.shape[0]} entries.")
            # TODO: There's probably another consistency check required with the phys grid.
            if (self.phys_grid.n_pixels[0] != self.images_phys.shape[1] or self.phys_grid.n_pixels[1] != self.images_phys.shape[2]):
                raise ValueError('Dimension mismatch: n_pixels {self.phys_grid.n_pixels[0]} x {self.phys_grid.n_pixels[1]} but shape is {self.images_phys.shape}')
            if not np.allclose(self.box_size, self.phys_grid.box_size):
                print(f"WARNING: Images box size {self.box_size} is outside tolerance of physical grid box size {self.phys_grid.box_size}")
        if self.images_fourier is not None:
            if self.polar_grid is None:
                # Can't happen
                raise ValueError("Polar grid is not defined for Fourier images")
            if self.n_images != self.images_fourier.shape[0]:
                raise ValueError(f"Images object lists image count of {self.n_images} but the fourier images array has {self.images_fourier.shape[0]} entries.")


    @classmethod
    def from_templates(cls, templates: Templates):
        """Create an Images object from existing Templates. Mainly for testing.

        Args:
            templates (Templates): Templates to convert to Images.

        Returns:
            Images: The images in the passed templates, as an Images object.
        """
        ## initialize the image class with the template class
        if templates.templates_fourier is not None:
            fourier_templates = templates.templates_fourier
            if not isinstance(fourier_templates, torch.Tensor):
                # This shouldn't happen
                fourier_templates = torch.from_numpy(templates.templates_fourier)
            fourier_im = FourierImages(fourier_templates, templates.polar_grid)
        else:
            fourier_im = None
        if templates.templates_phys is not None:
            phys_im = PhysicalImages(templates.templates_phys, templates.phys_grid.pixel_size)
        else:
            phys_im = None
        
        return cls(
            phys_images_data = phys_im,
            fourier_images_data = fourier_im,
            box_size = templates.box_size,
            viewing_angles = templates.viewing_angles
        )


    @classmethod
    def from_mrc(cls, filename: str, pixel_size: Optional[float | list[float] | FloatArrayType], device: str | torch.device = 'cpu'):
        """Create a new set of physical images from an MRC file.

        Args:
            filename (str): Name of MRC file to load. Must end with a .mrc or .mrcs extension.
            pixel_size (Optional[float  |  list[float]  |  FloatArrayType]): Sizes of the image pixels in Angstroms. If set,
                will override the values in the MRC file; if unset, the file values will be used.
            device (str | torch.device, optional): Device to use for the resulting image Tensor. Defaults to 'cpu'.

        Raises:
            ValueError: If a non-MRC file extension is passed.
            ValueError: If the pixel_size is not set and the existing MRC file has non-positive pixel sizes.

        Returns:
            Images: An Images image collection, with the physical images and grid populated per the saved data.
        """
        if not (filename.endswith('.mrc') or filename.endswith('.mrcs')):
            raise ValueError("Invalid file format. Only .mrc or .mrcs files are supported.")
        
        with mrcfile.open(filename) as mrc:
            assert isinstance(mrc.data, np.ndarray)
            imgs_phys = mrc.data.copy()
            if len(imgs_phys.shape) == 2:
                imgs_phys = imgs_phys[None, :, :]
            if pixel_size is not None:
                _pixel_size = cast(FloatArrayType, project_descriptor(pixel_size, "pixel size", 2, TargetType.FLOAT))
            else:
                _pixel_size = cast(FloatArrayType, np.array([mrc.voxel_size.x, mrc.voxel_size.y], dtype = float))
                if np.any(_pixel_size <= 0):
                    raise ValueError(f"MRC file {filename} contains non-positive pixel sizes.")
            imgs_phys = torch.from_numpy(imgs_phys).to(device)
            phys_imgs_data = PhysicalImages(images_phys=imgs_phys, pixel_size=_pixel_size)
        imgs = cls(phys_images_data=phys_imgs_data)
        imgs.filename = filename
        return imgs


    def save_to_mrc(self, filename: str):
        """Write the physical images to an MRC file. Will overwrite any existing file with that name.

        Args:
            filename (str): Name of the file to save.
        """
        self._ensure_phys_images()
        self._check_image_array()
        with mrcfile.new(filename, overwrite = True) as mrc:
            mrc.set_data(self.images_phys)
            mrc.voxel_size = (self.phys_grid.pixel_size[0], self.phys_grid.pixel_size[1], 1.0)


    def modify_pixel_size(self, pixel_size: float | list[float] | FloatArrayType):
        """Update the pixel sizes describing the physical images in this image collection. Note that physical images
        must exist for this to succeed (or make sense). This will overwrite any existing viewing box size.

        Args:
            pixel_size (float | list[float] | FloatArrayType): New pixel sizes. If scalar, the same value will be
                used for the X and Y dimensions; if a list-like is passed, the 0th index will be the X-dimension
                and the 1st index will be the Y-dimension.
        """
        _pixel_size = cast(FloatArrayType, project_descriptor(pixel_size, "pixel size", 2, TargetType.FLOAT))
        self.phys_grid = CartesianGrid2D(self.phys_grid.n_pixels, _pixel_size)
        self.box_size = self.phys_grid.box_size


    def _pad_images_if_needed(self):
        assert self.images_phys is not None
        device = self.images_phys.device
        pad_width = [(0, 0), (0, 0), (0, 0)]
        # NOTE: I have not checked the math on this.
        for i in range(2):
            if self.phys_grid.n_pixels[i] == self.images_phys.shape[i+1]:
                continue
            if self.phys_grid.n_pixels[i] > self.images_phys.shape[i+1]:
                pad_start = (self.phys_grid.n_pixels[i] - self.images_phys.shape[i+1]) // 2
                pad_end = self.phys_grid.n_pixels[i] - self.images_phys.shape[i+1] - pad_start
                pad_width[i+1] = (pad_start, pad_end)           
            else:
                start = max(0, (self.images_phys.shape[i+1] - self.phys_grid.n_pixels[i]) // 2)
                end = start + self.phys_grid.n_pixels[i]
                if i == 0:
                    self.images_phys = self.images_phys[:,start:end,:]
                else:
                    self.images_phys = self.images_phys[:,:,start:end]
        if np.any(np.array(pad_width) > 0):
            self.images_phys = torch.from_numpy(np.pad(self.images_phys, pad_width, mode = 'constant')).to(device)


    def change_images_phys_size(
        self,
        n_pixels: Optional[int | list[int] | IntArrayType] = None,
        box_size: Optional[float | list[float] | FloatArrayType] = None
    ):
        """Change the sizes of the physical images in this image collection.

        Args:
            n_pixels (Optional[int  |  list[int]  |  IntArrayType], optional): New number of pixels. Defaults to None.
            box_size (Optional[float  |  list[float]  |  FloatArrayType], optional): New viewing box size. Defaults to None.

        Raises:
            ValueError: If both n_pixels and box_size are set, or if neither is set.
            ValueError: If physical images do not actually exist in this collection.
        """
        if box_size is not None and n_pixels is not None:
            raise ValueError("Only one of n_pixels or box_size must be provided.")
        if box_size is None and n_pixels is None:
            raise ValueError("Either n_pixels or box_size must be provided.")
        if self.images_phys is None:
            raise ValueError('Attempt to change physical image size, but no physical images have been set.')
        if box_size is not None:
            _box_size = cast(FloatArrayType, project_descriptor(box_size, "box size", 2, TargetType.FLOAT))
            self.box_size = _box_size
            n_pixels = np.array([int(np.floor(self.box_size[i] / self.phys_grid.pixel_size[i] + 1e-3)) for i in range(2)], dtype = int)
        assert n_pixels is not None   # either it came in, or we just set it
        _n_pixels = cast(IntArrayType, project_descriptor(n_pixels, "n_pixels", 2, TargetType.INT))
        self.phys_grid = CartesianGrid2D(_n_pixels, self.phys_grid.pixel_size)
        self._pad_images_if_needed()
        if box_size is None:
            self.box_size = self.phys_grid.box_size
        self._check_image_array()


    def transform_to_fourier(
        self,
        polar_grid: Optional[PolarGrid] = None,
        nufft_eps: float = 1e-12,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True       # TODO: Better to ask for a device
    ):
        """Transform the physical images in this collection to Fourier-space representation. Existing
        physical images are kept. The new Fourier-space images will be placed on the same device as the
        physical images.

        Args:
            polar_grid (Optional[PolarGrid], optional): The polar grid for the Fourier-space image
                representation. If unset, there must be a pre-existing one. Defaults to None.
            nufft_eps (float, optional): Tolerance for non-uniform FFT. Defaults to 1e-12.
            precision (Precision, optional): Whether to use 64- or 128-bit representation.
                (Note that the Fourier-space representation uses complex numbers, so the bit
                counts are twice the corresponding precision for reals.) Defaults to
                Precision.DEFAULT, which matches the precision of the physical representation.
            use_cuda (bool, optional): Whether to use a cuda device. Defaults to True.

        Raises:
            ValueError: If no polar grid exists on the collection already, and none is passed.
        """
        self._ensure_phys_images()
        # TODO: find way to remove assertion
        assert self.images_phys is not None
        if polar_grid is not None:
            self.polar_grid = polar_grid
        if self.polar_grid is None:
            raise ValueError("No polar grid found")
        if precision == Precision.DEFAULT:
            precision = Precision.SINGLE if self.images_phys.dtype == torch.float32 else Precision.DOUBLE
        self.images_fourier = cartesian_phys_to_fourier_polar(
            grid_cartesian_phys = self.phys_grid,
            grid_fourier_polar = self.polar_grid,
            images_phys = self.images_phys,
            eps = nufft_eps,
            precision = precision,
            use_cuda = use_cuda
        )
        # TODO: Remove this assertion once cartesian_phys_to_fourier_polar is typed
        assert self.images_fourier is not None
        if self.polar_grid.uniform:
            self.images_fourier = self.images_fourier.reshape(self.n_images, self.polar_grid.n_shells, self.polar_grid.n_inplanes)
        self.images_fourier.to(self.images_phys.device)     # TODO: make this user-configurable, they might not want same device


    def transform_to_spatial(
        self,
        grid: CartesianGrid2D | Cartesian_grid_2d_descriptor | None = None,
        nufft_eps: float = 1e-12,
        precision: Precision = Precision.DEFAULT,
        use_cuda: bool = True
    ):
        """Transform the Fourier-space images in this collection to a Cartesian-space representation.
        Existing Fourier-space images are kept. The new images will be placed on the same device as the
        Fourier-space images.

        Args:
            grid (Optional[Cartesian_grid_descriptor], optional): Cartesian grid describing the new physical
                images. Defaults to None. If None is passed, the collection's existing grid will be used.
                If None is passed and no existing grid exists, this operation will fail.
            nufft_eps (float, optional): Tolerance for the non-uniform FFT. Defaults to 1e-12.
            precision (Precision, optional): Whether to use 32- or 64-bit representation.
                Defaults to Precision.DEFAULT (which matches the current Fourier precision).
            use_cuda (bool, optional): Whether to use a cuda device. Defaults to True.

        Raises:
            ValueError: If no existing Cartesian grid was set, and no new one was passed.
        """
        self._ensure_fourier_images()
        assert self.images_fourier is not None
        if grid is None and self.phys_grid is None:
            raise ValueError('No physical grid found, and physical grid parameters were not provided.')
        if grid is not None:
            self.phys_grid = CartesianGrid2D.from_descriptor(grid)
        device = self.images_fourier.device
        images_fourier = self.images_fourier.reshape(self.n_images, -1)
        # images_fourier = cast(ComplexArrayType, images_fourier)
        if not isinstance(images_fourier, torch.Tensor):
            images_fourier = torch.from_numpy(images_fourier)
        if precision == Precision.DEFAULT:
            precision = Precision.SINGLE if images_fourier.dtype == torch.complex64 else Precision.DOUBLE
        else:
            if (precision == Precision.SINGLE and images_fourier.dtype != torch.complex64) or \
               (precision == Precision.DOUBLE and images_fourier.dtype == torch.complex128):
                print("Precision %s provided, overriding the existing precision." % precision.value)
        self.images_phys = fourier_polar_to_cartesian_phys(
            grid_fourier_polar = self.polar_grid,
            grid_cartesian_phys = self.phys_grid,
            image_polar = images_fourier,
            eps = nufft_eps,
            precision = precision,
            use_cuda = use_cuda     # TODO: Better to ask for a device
        ).real
        if isinstance(self.images_phys, torch.Tensor):
            self.images_phys.to(device)     # TODO: Make this configurable
        else:
            raise ValueError("images_phys were not returned as a tensor. Shouldn't happen.")

    
    def center_physical_image_signal(self, norm_type: NormType = NormType.MAX):
        """Converts pixel values to 0-mean and unit norm representation.

        Args:
            norm_type (NormType, optional): Whether to use max-norm or standard-deviation
                norm. Defaults to NormType.MAX.

        Raises:
            ValueError: If physical images do not exist in this collection.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The computed mean and norm of the images.
        """
        if self.images_phys is None:
            raise ValueError("Physical images not generated.")
        mean = torch.mean(self.images_phys, dim = (1,2), keepdim=True)
        self.images_phys -= mean
        if norm_type == NormType.MAX:
            norm = torch.amax(torch.abs(self.images_phys), dim = (1,2), keepdim = True)
        elif norm_type == NormType.STD:
            norm = torch.std(self.images_phys, dim = (1,2), keepdim = True)
        self.images_phys /= norm
        return mean, norm


    def apply_ctf(self, ctf: CTF):
        """Applies a contrast transfer function to the Fourier-space images.

        Args:
            ctf (CTF): CTF to apply.

        Raises:
            NotImplementedError: If the Fourier-space images are using a non-uniform
                polar grid.

        Returns:
            torch.Tensor: The updated images (which will also be persisted to the collection).
        """
        self._ensure_fourier_images()
        if not self.polar_grid.uniform:
            raise NotImplementedError("Non-uniform Fourier images not implemented yet.")
        self.ctf = ctf
        assert self.images_fourier is not None
        self.images_fourier = ctf.apply(self.images_fourier)
        return self.images_fourier


    def add_noise_phys(self, snr: float | FloatArrayType | torch.Tensor = 1.0):
        """Add random ((0,1) Gaussian) noise to the Cartesian-space images in the collection.

        Args:
            snr (float | FloatArrayType | torch.Tensor, optional): Signal-to-noise ratio. Defaults to 1.0.

        Raises:
            ValueError: If Cartesian-space images do not exist.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated images (as also updated in-place), and the
                noise applied to them. 
        """
        ensure_positive(snr, "signal-to-noise ratio")
        if (self.images_phys is None):
            raise ValueError("Atempting to add physical noise, but physical images are not set.")
        device = self.images_phys.device
        # if not torch.cuda.is_available():
        #     device = 'cpu'
        if isinstance(snr, np.ndarray):
            snr = torch.tensor(snr).to(device)
        power_image = torch.mean(torch.abs(self.images_phys) ** 2, dim = (1,2))
        sigma_noise = torch.sqrt(power_image / snr).unsqueeze(1).unsqueeze(2)
        noise = sigma_noise * torch.randn_like(self.images_phys)
        self.images_phys = self.images_phys + noise
        return self.images_phys, sigma_noise.flatten().cpu().numpy()


    # def add_noise_fourier(self, snr: float | FloatArrayType | torch.Tensor = 1.0, device: str = 'cuda'):
    def add_noise_fourier(self, snr: float | FloatArrayType | torch.Tensor = 1.0):
        """Add random ((0,1) complex normal) noise to Fourier-space images in the collection.

        Args:
            snr (float | FloatArrayType | torch.Tensor, optional): Signal-to-noise ratio. Defaults to 1.0.

        Raises:
            ValueError: If Fourier-space images do not exist.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The updated images (as also updated in-place), and the
                noise applied to them. 
        """
        ensure_positive(snr, "signal-to-noise ratio")
        if (self.images_fourier is None):
            raise ValueError("Attempting to add fourier noise, but fourier images are not set.")
        device = self.images_fourier.device
        if isinstance(snr, np.ndarray):
            snr = torch.tensor(snr).to(device)
        power_image = self.polar_grid.integrate(self.images_fourier.abs().pow(2))
        power_image = power_image.unsqueeze(1).unsqueeze(2)
        sigma_noise = torch.sqrt(power_image / snr).unsqueeze(1).unsqueeze(2)
        noise = torch.randn_like(self.images_fourier) * sigma_noise
        self.images_fourier = self.images_fourier + noise
        return self.images_fourier, sigma_noise.flatten().cpu().numpy()


    def displace_images_fourier(
        self,
        x_displacements: FloatArrayType | torch.Tensor | float,
        y_displacements: FloatArrayType | torch.Tensor | float,
        precision: Precision = Precision.DEFAULT,
    ):
        """Apply a displacement to the Fourier-space images in the collection.

        Args:
            x_displacements (FloatArrayType | torch.Tensor | float): Displacements to apply. May be constant
                for each pixel, or a set of displacements to apply per-pixel for each image, or a complete
                tensor of displacements.
            y_displacements (FloatArrayType | torch.Tensor | float): Displacements to apply. May be constant
                for each pixel, or a set of displacements to apply per-pixel for each image, or a complete
                tensor of displacements.
            precision (Precision, optional): Whether to use single or double precision for the resulting
                representation. Defaults to keeping current image precision.
        """
        self._ensure_fourier_images()
        assert self.images_fourier is not None
        if precision == Precision.DEFAULT:
            precision = Precision.SINGLE if self.images_fourier.dtype == torch.complex64 else Precision.DOUBLE
        device = self.images_fourier.device
        (x_disp, y_disp) = _verify_displacements(x_displacements, y_displacements, self.images_fourier.shape[0])
        x_disp = x_disp * 2.0 / self.box_size[0]
        y_disp = y_disp * 2.0 / self.box_size[1]
        translation_kernel = translation_kernel_fourier(self.polar_grid, x_disp, y_disp, precision, str(device))
        image_fourier_device = to_torch(self.images_fourier, precision, device)
        self.images_fourier = (image_fourier_device * translation_kernel)

    
    def normalize_images_phys(
        self,
        ord: int = 1,
        use_max: bool = False
    ):
        """Normalize the Cartesian-space images in the collection.

        Args:
            ord (int, optional): Degree of norm to apply. Defaults to 1.
            use_max (bool, optional): Whether to use the max in place of an LP norm. Defaults to False.

        Returns:
            torch.Tensor: The norm applied to the images (which are modified in-place).
        """
        self._ensure_phys_images()
        assert self.images_phys is not None
        if use_max:
            maxval = get_imgs_max(self.images_phys)
            self.images_phys /= maxval
            return maxval
        else:
            lpnorm = torch.norm(self.images_phys, dim = (1,2), p = ord, keepdim = False)
            self.images_phys /= lpnorm[:,None,None]
            return lpnorm


    def normalize_images_fourier(
        self,
        ord: int = 1,
        use_max: bool = False,
    ):
        """Normalize the Fourier-space images in the collection.

        Args:
            ord (int, optional): Degree of norm to apply. Defaults to 1.
            use_max (bool, optional): Whether to use the max in place of an LP norm. Defaults to False.

        Returns:
            torch.Tensor: The norm applied to the images (which are modified in-place).
        """
        self._ensure_fourier_images()
        assert self.images_fourier is not None
        if use_max:
            maxval = get_imgs_max(self.images_fourier)
            self.images_fourier /= maxval
            return maxval
        else:
            lpnorm = self.polar_grid.integrate(self.images_fourier.abs().pow(ord)).pow(1.0 / ord)
            for i in range(len(self.images_fourier.shape) - 1):
                # lpnorm = np.expand_dims(lpnorm, axis = -1)
                lpnorm = torch.unsqueeze(lpnorm, dim = -1)
            self.images_fourier /= lpnorm
            return lpnorm


    def rotate_images_fourier(
        self,
        inplane_rotations : np.ndarray | torch.Tensor | float
    ):
        """Apply an in-plane rotation to the Fourier-space images.

        Args:
            inplane_rotations (np.ndarray | torch.Tensor | float): Rotation to apply.

        Raises:
            ValueError: If the inplane_rotations is multi-dimensional, or cannot be
                obviously broadcast to the number of images.
        """
        self._ensure_fourier_images()
        assert self.images_fourier is not None
        if np.isscalar(inplane_rotations):
            inplane_rotations = torch.tensor([inplane_rotations], dtype = torch.double)
        if isinstance(inplane_rotations, np.ndarray):
            inplane_rotations = torch.from_numpy(inplane_rotations)
        assert isinstance(inplane_rotations, torch.Tensor)
        if inplane_rotations.ndim > 1:
            raise ValueError("inplane_rotations must be a 1D array.")
        inplane_rotations.to(self.images_fourier.device)

        if inplane_rotations.shape[0] != self.images_fourier.shape[0]:
            if inplane_rotations.shape[0] == 1:
                # Manually broadcast rotations
                inplane_rotations = inplane_rotations * torch.ones(self.images_fourier.shape[0])
            else:
                raise ValueError("Number of rotations must be equal to the number of images.")
        inplane_rotations_step = 2 * np.pi / self.polar_grid.n_inplanes
        inplane_rotations_discrete = cast(torch.Tensor, -np.round(inplane_rotations / inplane_rotations_step))
        # TODO: See if this can be vectorized better
        for i in range(self.n_images):
            self.images_fourier[i] = torch.roll(self.images_fourier[i], int(inplane_rotations_discrete[i]), dims = 1)

    
    def filter_padded_images(self, rtol = 1e-1):
        """Restrict the image collection to the set of images which do not have padding.
        """
        ## this is to remove the artifacted images in some dataset on EMPIAR
        ## find padded physical images and remove them
        if self.images_phys is None:
            return
        not_padded = np.ones(self.images_phys.shape[0], dtype=bool)
        for i in range(self.images_phys.shape[0]):
            if np.allclose(self.images_phys[i,0,:], self.images_phys[i,1,:], rtol = rtol) or \
                np.allclose(self.images_phys[i,:,0], self.images_phys[i,:,1], rtol = rtol) or \
                np.allclose(self.images_phys[i,-1,:], self.images_phys[i,-2,:], rtol = rtol) or \
                np.allclose(self.images_phys[i,:,-1], self.images_phys[i,:,-2], rtol = rtol):
                not_padded[i] = False
        # print(f"Number of not padded images: {np.sum(not_padded)}")
        self.images_phys = self.images_phys[not_padded]
        self.n_images = self.images_phys.shape[0]
        return not_padded


    # TODO: Consider returning Tensor instead of ndarray? Or is this only used for plotting?
    def get_power_spectrum(self):
        """Gets the power spectrum of the (Fourier-space) images.

        Raises:
            ValueError: If no Fourier-space images exist in the collection.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of power-spectrum values and the resolutions.
        """
        if self.images_fourier is None:
            raise ValueError("Fourier images not found. Please transform the images to Fourier domain before calculating the power spectrum.")
        resolutions = np.amax(self.box_size) / (2.0 * self.polar_grid.radius_shells)
        images_fourier = self.images_fourier
        power_spectrum: ComplexArrayType = torch.mean(torch.abs(images_fourier) ** 2, dim = (0, 2)).cpu().numpy()
        return power_spectrum, resolutions
    

    # TODO: Figure out the right type hint for tensor indexing
    def select_images(self, indices):
        """Restrict this collection's images to the specified indices.

        Args:
            indices (_type_): The indices to keep.
        """
        if self.images_phys is not None:
            self.images_phys = self.images_phys[indices]
        if self.images_fourier is not None:
            self.images_fourier = self.images_fourier[indices]
        self.n_images = len(indices)


    def downsample_images_phys(self,
            downsample_factor: int = 2,                
            type: Literal['mean'] | Literal['max'] = 'mean'
    ):
        """Downsample the physical images in this collection.

        Args:
            downsample_factor (int, optional): Factor by which to downsample the images. Defaults to 2.
            type (Literal['mean'] | Literal['max']): Whether to take mean or max of the pool for
                downsampling. Defaults to mean.

        Raises:
            ValueError: If physical images do not exist in this collection.
        """
        self._ensure_phys_images()
        if downsample_factor == 1:
            return
        ## downsample factor must be an integer of multiple of 2
        if downsample_factor % 2 != 0:
            raise ValueError("Downsample factor must be a multiple of 2.")
        assert self.images_phys is not None
        self.images_phys = self.images_phys.unsqueeze(1)
        if type == 'mean':
            self.images_phys = torch.nn.functional.avg_pool2d(self.images_phys, downsample_factor, downsample_factor)
        elif type == 'max':
            self.images_phys = torch.nn.functional.max_pool2d(self.images_phys, downsample_factor, downsample_factor)
        self.images_phys = self.images_phys.squeeze(1)
        _n_pixels_downsampled = np.array([self.images_phys.shape[1], self.images_phys.shape[2]], dtype = int)
        _pixel_size_downsampled = self.phys_grid.pixel_size * downsample_factor
        self.phys_grid = CartesianGrid2D(_n_pixels_downsampled, _pixel_size_downsampled)
        self.box_size = self.phys_grid.box_size
        return self.images_phys
