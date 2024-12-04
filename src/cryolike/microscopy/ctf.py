import numpy as np
import torch
from typing import TypeVar, cast

from cryolike.grids import PolarGrid
from cryolike.util import ComplexArrayType, FloatArrayType

h =  6.62607015e-34 # Planck constant [Js] = [kgm^2/s]
e =  1.60217663e-19 # electron charge [C]; note that [CV] = [J]
c =  2.99792458e8   # speed of light [m/s]
m0 = 9.1093837015e-31 # electron rest mass [kg]

T = TypeVar("T", bound=ComplexArrayType | torch.Tensor)


class LensDescriptor():
    """Class describing the properties of a device, to be used to compute
    the relevant contrast transfer function (CTF).
    
    Attributes:
        defocusU (FloatArrayType): Defocus in U-dimension, in Angstrom
        defocusV (FloatArrayType): Defocus in V-dimension, in Angstrom
        defocusAng (FloatArrayType): Defocus angle, in radians
        sphericalAberration (FloatArrayType): Spherical aberration, in mm
        voltage (FloatArrayType): Voltage, in kV
        amplitudeContrast (FloatArrayType): Amplitude contrast
        phaseShift (FloatArrayType): phase shift, in radians
    """
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAng: FloatArrayType
    sphericalAberration: FloatArrayType
    voltage: FloatArrayType
    amplitudeContrast: FloatArrayType
    phaseShift: FloatArrayType

    def __init__(self,
        defocusU: np.ndarray = np.array([10200.0]),
        defocusV: np.ndarray = np.array([9800.0]),
        defocusAng: np.ndarray = np.array([90.0]),
        defocusAng_degree: bool = True,
        sphericalAberration: float | FloatArrayType = 2.7,
        voltage: float | FloatArrayType = 300.0,
        amplitudeContrast: float | FloatArrayType = 0.1,
        phaseShift: np.ndarray = np.array([0.0]),
        phaseShift_degree: bool = True
    ):
        """Constructor for device properties used to compute a CTF.

        Args:
            defocusU (np.ndarray, optional): In Angstroms. Defaults to np.array([10200.0]).
            defocusV (np.ndarray, optional): In Angstroms. Defaults to np.array([9800.0]).
            defocusAng (np.ndarray, optional): Defocus angle, in degrees unless otherwise
                specified. Defaults to np.array([90.0]).
            defocusAng_degree (bool, optional): If True (the default), defocus angle is
                presumed to be in degrees; if False, defocus angle is treated as radians.
            sphericalAberration (float | FloatArrayType, optional): Spherical aberration.
                Defaults to 2.7.
            voltage (float | FloatArrayType, optional): Voltage, in kV. Defaults to 300.0.
            amplitudeContrast (float | FloatArrayType, optional): Amplitude contrast.
                Defaults to 0.1.
            phaseShift (np.ndarray, optional): Phase shift, in degrees unless otherwise
                specified. Defaults to np.array([0.0]).
            phaseShift_degree (bool, optional): If True, phase shift is presumed to be in
                degrees; if False, phase shift is in radians. Defaults to True.
        """
        self.defocusU = _to_float_flatten_np_array(defocusU)  # in Angstrom
        self.defocusV = _to_float_flatten_np_array(defocusV)  # in Angstrom
        if self.defocusU.size != self.defocusV.size:
            raise ValueError('defocusU and defocusV must have the same size')
        n_CTF = len(defocusU)
        self.defocusAng = _to_float_flatten_np_array(defocusAng) # in degrees, defocus angle
        if defocusAng_degree:
            self.defocusAng = np.radians(self.defocusAng)
        self.sphericalAberration = np.array([sphericalAberration])  # in mm, spherical aberration
        self.voltage = np.array([voltage])  # in kV
        self.amplitudeContrast = np.array([amplitudeContrast])
        self.phaseShift = _to_float_flatten_np_array(phaseShift)
        if phaseShift_degree:
            self.phaseShift = np.radians(phaseShift)
        if self.phaseShift.size == 1:
            self.phaseShift = self.phaseShift * np.ones(n_CTF)


    @classmethod
    def read_from_cryosparc(cls, filename: str):
        ctf_data = np.load(filename)

        return cls(
            defocusU = ctf_data['ctf/df1_A'],
            defocusV = ctf_data['ctf/df2_A'],
            defocusAng = ctf_data['ctf/df_angle_rad'],
            sphericalAberration = ctf_data['ctf/cs_mm'][0],
            voltage = ctf_data['ctf/accel_kv'][0],
            amplitudeContrast = ctf_data['ctf/amp_contrast'][0],
            phaseShift = ctf_data['ctf/phase_shift_rad'],
        )


    @classmethod
    def read_from_star(cls, filename: str):
        from cryolike.microscopy.star_file import read_star_file
        dataBlock, _ = read_star_file(filename) # 2nd return is the param list
        for param in dataBlock.keys():
            y = dataBlock[param]
            y_unique = np.unique(y)
            if len(y_unique) == 1:
                dataBlock[param] = y_unique[0]

        return cls(
            defocusU = np.array(dataBlock["DefocusU"]),
            defocusV = np.array(dataBlock["DefocusV"]),
            defocusAng = np.array(dataBlock["DefocusAngle"]),
            sphericalAberration = np.array(dataBlock["SphericalAberration"]),
            voltage = np.array(dataBlock["Voltage"]),
            amplitudeContrast = np.array(dataBlock["AmplitudeContrast"]),
            phaseShift = np.array(dataBlock["PhaseShift"]),
        )
        

def _to_float_flatten_np_array(x):
    if isinstance(x, float):
        return np.array([x], dtype = np.float64)
    if not isinstance(x, np.ndarray):
        raise ValueError('Invalid type for x')
    if len(x.shape) > 1:
        x = x.flatten()
    return x


def _ctf_relion(lens: LensDescriptor, grid: PolarGrid, box_size: float = 1.0, anisotropy: bool = True, cs_corrected: bool = False):
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
        local_defocus = defocus[:, None] + astigmatism[:, None] * np.cos(2 * (theta_[None,:] - lens.defocusAng[:, None]))
        if cs_corrected:
            gamma = - coef2 * r_shell_2[None,:,None] * local_defocus[:,None,:] - lens.phaseShift[:, None, None]
        else:
            gamma = - coef2 * r_shell_2[None,:,None] * local_defocus[:,None,:] + coef4 * r_shell_4[None,:, None] - lens.phaseShift[:, None, None]
    else:
        if cs_corrected:
            gamma = - coef2 * r_shell_2[None, :] * defocus[:, None] - lens.phaseShift[:, None]
        else:
            gamma = - coef2 * r_shell_2[None, :] * defocus[:, None] + coef4 * r_shell_4[None, :] - lens.phaseShift[:, None]
    ctf = - np.sqrt(1 - lens.amplitudeContrast * lens.amplitudeContrast) * np.sin(gamma) + lens.amplitudeContrast * np.cos(gamma)

    return ctf


class CTF:
    """Class representing a contrast transfer function.

    Attributes:
        box_size (float): Side length of a (square) viewing box, in Angstroms
        anisotropy (bool): Whether the described CTF is anisotropic
        cs_corrected (bool): Whether the described CTF is CS-corrected
        n_CTF (int): Number of CTFs, which determines whether the CTF can
            effectively describe an image stack
        ctf (FloatArrayType): The value of the CTF function as a Numpy array.
            It is indexed as [image_number, ??, ??]
        lens_descriptor (LensDescriptor | None): If the CTF was computed, this
            field stores the lens description which was used to compute the CTF.
    """
    box_size: float
    anisotropy: bool
    cs_corrected: bool
    n_CTF: int
    ctf: FloatArrayType
    # Not sure if we actually need to keep this around
    lens_descriptor: LensDescriptor | None


    def __init__(
        self,
        ctf_descriptor: np.ndarray | LensDescriptor,
        polar_grid: PolarGrid | None = None,
        box_size: float = -1.,
        anisotropy: bool = True,
        cs_corrected: bool = False
    ):
        """Constructor for class representing a contrast transfer function.

        The function can be instantiated from known function values, or a
        description of the properties of the apparatus.

        Args:
            ctf_descriptor (np.ndarray | LensDescriptor): The CTF itself, If
                represented as a Numpy array, the values will be used directly.
                Otherwise, this should be a descriptor of the apparatus so that
                the CTF can be computed as per relion.
            polar_grid (PolarGrid): The Fourier-space grid in which the CTF lives
            box_size (float): The side length of the (square) viewing port, in Angstrom
            anisotropy (bool, optional): Whether the described CTF is anisotropic. Defaults to True.
            cs_corrected (bool, optional): Whether the described CTF is CS-corrected. Defaults to False.
        """
        self.anisotropy = anisotropy
        self.cs_corrected = cs_corrected
        if isinstance(ctf_descriptor, np.ndarray):
            self.ctf = ctf_descriptor
            if len(self.ctf.shape) == 1:
                if self.anisotropy:
                    raise ValueError('Invalid shape for anisotropic ctf')
                else:
                    self.n_CTF = 1
                    self.ctf = self.ctf[None,:]
            elif len(self.ctf.shape) == 2:
                if self.anisotropy:
                    self.n_CTF = 1
                    self.ctf = self.ctf[None,:,:]
                else:
                    self.n_CTF = self.ctf.shape[0]
            elif len(self.ctf.shape) == 3:
                if self.anisotropy:
                    self.n_CTF = self.ctf.shape[0]
                else:
                    raise ValueError('Invalid shape for isotropic ctf')
            else:
                raise ValueError('Invalid shape for ctf')
            return
        assert polar_grid is not None
        
        # TODO: There's code for handling the case where box_size is a
        # numpy array, but it's not clear that that's used or allowed?
        if np.any(box_size < 0.0):
            raise ValueError('Box size must be positive')
        # 
        # if hasattr(box_size, '__len__') and len(box_size) > 1:
        #     if np.abs(box_size[0] - box_size[1]) > 1e-6:
        #         raise ValueError('Box size must be isotropic')
        #     self.box_size = box_size[0]
        # else:
        self.box_size = box_size
        self.ctf = _ctf_relion(ctf_descriptor, polar_grid, box_size, anisotropy, cs_corrected)


    @classmethod
    def from_file(cls, filename: str, polar_grid: PolarGrid, box_size: float):
        if filename.endswith('.star'):
            lens = LensDescriptor.read_from_star(filename)
        elif filename.endswith('.cs'):
            lens = LensDescriptor.read_from_cryosparc(filename)
        else:
            raise ValueError('Invalid filename')
        return cls(
            ctf_descriptor=lens,
            polar_grid=polar_grid,
            box_size=box_size,
            # anisotropy? cs_corrected?
        )

    
    def apply(self, image_fourier: T) -> T:
        ctf = self.ctf
        if type(image_fourier) is np.ndarray:
            ctf = np.array(self.ctf, dtype = image_fourier.dtype)
        if type(image_fourier) is torch.Tensor and type(self.ctf) is np.ndarray:
            ctf = torch.tensor(self.ctf, dtype = image_fourier.dtype, device = image_fourier.device)
        if self.anisotropy:
            return cast(T, image_fourier * ctf[:,:,:])
        else:
            return cast(T, image_fourier * ctf[:,:,None])
