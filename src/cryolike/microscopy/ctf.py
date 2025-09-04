import numpy as np
import torch

from cryolike.grids import PolarGrid
from cryolike.metadata import LensDescriptor
from cryolike.util import FloatArrayType, Precision, to_torch

h =  6.62607015e-34 # Planck constant [Js] = [kgm^2/s]
e =  1.60217663e-19 # electron charge [C]; note that [CV] = [J]
c =  2.99792458e8   # speed of light [m/s]
m0 = 9.1093837015e-31 # electron rest mass [kg]

def _ctf_relion(
    lens: LensDescriptor,
    grid: PolarGrid,
    box_size: float = 1.0,
    anisotropy: bool = True,
    cs_corrected: bool = False
):
    defocus = 0.5 * (lens.defocusU + lens.defocusV)
    astigmatism = 0.5 * np.abs(lens.defocusU - lens.defocusV) # abs not necessary
    sphericalAberration = lens.sphericalAberration * 1e7 # mm to Angstrom
    voltage = lens.voltage * 1e3 # kV to V

    wavelength = h / np.sqrt(2 * m0 * e * voltage) / np.sqrt(1 + e * voltage / (2 * m0 * c * c)) * 1e10 # in Angstrom
    wavelength3 = wavelength ** 3
    coef4 = np.pi / 2 * sphericalAberration * wavelength3
    coef2 = np.pi * wavelength

    r_shell_scaled_ = grid.radius_shells * 2.0 / box_size
    r_shell_2 = r_shell_scaled_ ** 2
    r_shell_4 = r_shell_2 ** 2
    
    gamma = None

    if anisotropy:
        theta_ = np.linspace(0, 2 * np.pi, grid.n_inplanes, endpoint = False)
        local_defocus = defocus[:, None] + astigmatism[:, None] * np.cos(2 * (theta_[None,:] - lens.defocusAngle[:, None]))
        if cs_corrected:
            gamma = - coef2 * r_shell_2[None,:,None] * local_defocus[:,None,:] - lens.phaseShift[:, None, None]
        else:
            gamma = - coef2 * r_shell_2[None,:,None] * local_defocus[:,None,:] + coef4 * r_shell_4[None,:, None] - lens.phaseShift[:, None, None]
    else:
        if cs_corrected:
            gamma = - coef2 * r_shell_2[None, :] * defocus[:, None] - lens.phaseShift[:, None]
        else:
            gamma = - coef2 * r_shell_2[None, :] * defocus[:, None] + coef4 * r_shell_4[None, :] - lens.phaseShift[:, None]
    ctf: FloatArrayType = - np.sqrt(1 - lens.amplitudeContrast * lens.amplitudeContrast) * np.sin(gamma) + lens.amplitudeContrast * np.cos(gamma)

    return ctf


class CTF:
    """Class representing a contrast transfer function.

    Attributes:
        box_size (float): Side length of a (square) viewing box, in Angstroms
        anisotropy (bool): Whether the described CTF is anisotropic
        cs_corrected (bool): Whether the described CTF is CS-corrected
        n_CTF (int): Number of CTFs, which determines whether the CTF can
            effectively describe an image stack
        ctf (torch.Tensor): The value of the CTF function as a Numpy array.
            It is indexed as [n_images, n_shells, n_inplanes]
        lens_descriptor (LensDescriptor | None): If the CTF was computed, this
            field stores the lens description which was used to compute the CTF.
    """
    box_size: float
    anisotropy: bool
    cs_corrected: bool
    n_CTF: int
    ctf: torch.Tensor
    # Not sure if we actually need to keep this around
    lens_descriptor: LensDescriptor | None


    def __init__(
        self,
        ctf_descriptor: torch.Tensor | LensDescriptor,
        polar_grid: PolarGrid | None = None,
        box_size: float = -1.,
        anisotropy: bool = True,
        cs_corrected: bool = False,
        precision: Precision = Precision.DOUBLE
    ):
        """Constructor for class representing a contrast transfer function.

        The function can be instantiated from known function values, or a
        description of the properties of the apparatus.

        Args:
            ctf_descriptor (torch.Tensor | LensDescriptor): The CTF itself, If
                represented as a Tensor, the values will be used directly.
                Otherwise, this should be a descriptor of the apparatus so that
                the CTF can be computed as per relion.
            polar_grid (PolarGrid): The Fourier-space grid in which the CTF lives
            box_size (float): The side length of the (square) viewing port, in Angstrom
            anisotropy (bool, optional): Whether the described CTF is anisotropic. Defaults to True.
            cs_corrected (bool, optional): Whether the described CTF is CS-corrected. Defaults to False.
        """
        self.anisotropy = anisotropy
        self.cs_corrected = cs_corrected

        if isinstance(ctf_descriptor, torch.Tensor):
            self.ctf = ctf_descriptor
            self._check_explicit_ctf_isotropy_and_shape()
            return
        
        assert polar_grid is not None

        if np.any(box_size < 0.0):
            raise ValueError('Box size must be positive')
        self.box_size = box_size
        ctf = _ctf_relion(ctf_descriptor, polar_grid, box_size, anisotropy, cs_corrected)
        self.ctf = to_torch(ctf, precision=precision)
        if not self.anisotropy:
            # add back the undifferentiated inplane-rotation dimension
            self.ctf = self.ctf.unsqueeze(-1)


    def _check_explicit_ctf_isotropy_and_shape(self):
        # Ensure that all explicitly-set CTFs have 3 dimensions
        # (i.e. image, by radius, by inplane-rotation)
        # 1-d CTF must be isotropic and should have null dimensions
        # in the image and inplane indices;
        # 2-d CTF is either a per-image isotropic CTF (and so should
        # get a null inplane dimension) or a singleton anisotropic
        # one (and so should get a null per-image dimension)
        # 3-d CTF must be anisotropic, per-image.
        if len(self.ctf.shape) == 1:
            if self.anisotropy:
                raise ValueError('Invalid shape for anisotropic ctf')
            else:
                self.n_CTF = 1
                self.ctf = self.ctf[None,:,None]
        elif len(self.ctf.shape) == 2:
            if self.anisotropy:
                self.n_CTF = 1
                self.ctf = self.ctf[None,:,:]
            else:
                self.n_CTF = self.ctf.shape[0]
                self.ctf = self.ctf[:,:,None]
        elif len(self.ctf.shape) == 3:
            if self.anisotropy:
                self.n_CTF = self.ctf.shape[0]
            else:
                raise ValueError('Invalid shape for isotropic ctf')
        else:
            raise ValueError('Invalid shape for ctf')


    @classmethod
    def from_file(cls, filename: str, polar_grid: PolarGrid, box_size: float):
        if filename.endswith('.star'):
            lens = LensDescriptor.from_starfile(filename)
        elif filename.endswith('.cs'):
            lens = LensDescriptor.from_cryosparc_file(filename)
        else:
            raise ValueError('Invalid filename')
        return cls(
            ctf_descriptor=lens,
            polar_grid=polar_grid,
            box_size=box_size,
            # anisotropy? cs_corrected?
        )

    
    def apply(self, image_fourier: torch.Tensor) -> torch.Tensor:
        ctf = self.ctf.to(dtype = image_fourier.dtype, device = image_fourier.device)
        if self.anisotropy:
            return image_fourier * ctf[:,:,:]
        else:
            return image_fourier * ctf[:,:,None]
