from scipy.special import jv
from torch import Tensor
import torch
import numpy as np
from numpy import pi
from numpy import conj
from scipy.special import loggamma as lgamma

from cryolike.grids import PolarGrid
from cryolike.stacks import Templates
from cryolike.metadata import ViewingAngles
from cryolike.microscopy import CTF
from cryolike.util import Precision, to_torch, absq

from cross_correlation_fixtures import (
    get_difference_wavevector_images_templates,
    planewave_planar_planewave_planar,
)


def planar_planewave_planewave(
    wavevector_planewave_templates: Tensor,
    wavevector_planewave_images: Tensor,
    grid_inplanes: Tensor,
    searched_displacements: Tensor,
    angle_planar_ctf_template: Tensor,
    grid_max_radius_K: float
) -> Tensor:
    """
    Analytic solution for integral of planar fn x planewave x planewave,
    returning a vector of [image_count] dimension.
    
    Args:
        wavevector_planewave_templates (Tensor): Tensor representation of the planewave
            Templates (in Fourier space)
        wavevector_planewave_images (Tensor): Tensor representation of the planewave
            Images (in Fourier space)
        grid_inplanes (Tensor): The inplane rotational angles of the quadrature points.
            Corresponds to gamma.
        searched_displacements (Tensor): The set of displacements (in Cartesian space)
            which will be used for matching
        angle_planar_ctf_template (Tensor): Angle by which the template CTF's planar
            function has been rotated about the +Z axis
        grid_max_radius_K (float): Upper bound for the integration
        
    Returns:
        Tensor: ...
    
    """
    _device = wavevector_planewave_templates.device
    offset_delta_T, offset_radius, offset_angle_omega_t = get_difference_wavevector_images_templates(
        wavevector_planewave_templates,
        wavevector_planewave_images,
        grid_inplanes,
        searched_displacements
    )
    offset_radius = offset_radius.to(_device)
    offset_angle_omega_t = offset_angle_omega_t.to(_device)
    phi = angle_planar_ctf_template - offset_angle_omega_t
    two_pi_K = grid_max_radius_K * 2 * pi
    tK = offset_radius * two_pi_K
    bessel_1 = jv(1, tK).to(_device)
    bessel_3 = jv(3, tK).to(_device)
    result = - pi * (grid_max_radius_K ** 3) * torch.cos(phi) * (bessel_1 + bessel_3)
    return result
    
    
def planewave_planewave(
    wavevector_planewave_templates: Tensor,
    wavevector_planewave_images: Tensor,
    grid_inplanes: Tensor,
    searched_displacements: Tensor,
    grid_max_radius_K: float
) -> Tensor:
    """
    Analytic solution for integral of planewave x planewave,
    returning a vector of [image_count] dimension.
    
    Args:
        wavevector_planewave_templates (Tensor): Tensor representation of the planewave
            Templates (in Fourier space)
        wavevector_planewave_images (Tensor): Tensor representation of the planewave
            Images (in Fourier space)
        grid_inplanes (Tensor): The inplane rotational angles of the quadrature points.
            Corresponds to gamma.
        searched_displacements (Tensor): The set of displacements (in Cartesian space)
            which will be used for matching
        grid_max_radius_K (float): Upper bound for the integration
        
    Returns:
        Tensor: ...
    """
    offset_delta_T, offset_radius, offset_angle_omega_t = get_difference_wavevector_images_templates(
        wavevector_planewave_templates,
        wavevector_planewave_images,
        grid_inplanes,
        searched_displacements
    )
    two_pi_K = grid_max_radius_K * 2 * pi
    tK = offset_radius * two_pi_K
    bessel_0 = jv(0, tK).to(wavevector_planewave_templates.device)
    bessel_2 = jv(2, tK).to(wavevector_planewave_templates.device)
    result = pi * (grid_max_radius_K ** 2) * (bessel_0 + bessel_2)
    return result


class LogLikelihoodPlanarCTFPlanewaves():
    
    wavevector_planewave_templates: Tensor
    wavevector_planewave_images: Tensor
    wavevector_planewave_identity: Tensor
    angle_planar_ctf_template: Tensor
    angle_planar_ctf_image: Tensor
    gamma: Tensor
    displacements: Tensor
    polar_grid: PolarGrid
    precision: Precision
    torch_float_type: torch.dtype
    torch_complex_type: torch.dtype
    device: torch.device
    
    def __init__(
        self,
        wavevector_planewave_templates: torch.Tensor,
        wavevector_planewave_images: torch.Tensor,
        wavevector_planewave_identity: torch.Tensor,
        angle_planar_ctf_template: float,
        angle_planar_ctf_image: float,
        gamma: torch.Tensor,
        displacements: torch.Tensor,
        polar_grid: PolarGrid,
        n_pixels: int,
        precision: Precision = Precision.DOUBLE,
        # device: torch.device = torch.device("cpu"),
    ):
        """
        """

        device = torch.device("cpu")

        (self.torch_float_type, self.torch_complex_type, _) = precision.get_dtypes(default=Precision.SINGLE)
        self.device = device
        angle_planar_ctf_template_tensor = torch.tensor(angle_planar_ctf_template, dtype=self.torch_float_type, device=device)
        angle_planar_ctf_image_tensor = torch.tensor(angle_planar_ctf_image, dtype=self.torch_float_type, device=device)
        self.n_pixels = n_pixels

        # wavevector_planewave_images = wavevector_planewave_images[:-1]
         
        self.n_templates = wavevector_planewave_templates.shape[0]
        self.n_images = wavevector_planewave_images.shape[0]
        # self.n_displacements = displacements.shape[0]

        wavevector_planewave_templates = wavevector_planewave_templates.to(device)
        wavevector_planewave_images = wavevector_planewave_images.to(device)
        wavevector_planewave_identity = wavevector_planewave_identity.to(device)

        self.I_T_T_form = planewave_planar_planewave_planar(
            wavevector_planewave_templates,
            wavevector_planewave_templates,
            torch.tensor([0.0], dtype=self.torch_float_type, device=device),
            torch.tensor([0.0, 0.0], dtype=self.torch_float_type, device=device),
            angle_planar_ctf_template_tensor,
            angle_planar_ctf_template_tensor,
            polar_grid.radius_max
        )[range(self.n_templates), range(self.n_templates), :, :] ## cross correlation of CTF * templates and CTF * templates for this version of PPPP
        self.I_M_M_form = planewave_planar_planewave_planar(
            wavevector_planewave_images,
            wavevector_planewave_images,
            torch.tensor([0.0], dtype=self.torch_float_type, device=device),
            torch.tensor([0.0, 0.0], dtype=self.torch_float_type, device=device),
            angle_planar_ctf_image_tensor,
            angle_planar_ctf_image_tensor,
            polar_grid.radius_max
        )[range(self.n_images), range(self.n_images), :, :] ## cross correlation of CTF images and CTF images for this version of PPPP
        self.I_E_E_form = planewave_planewave(
            wavevector_planewave_identity,
            wavevector_planewave_identity,
            torch.tensor([0.0], dtype=self.torch_float_type, device=device),
            torch.tensor([0.0, 0.0], dtype=self.torch_float_type, device=device),
            polar_grid.radius_max
        )
        self.I_M_T_form = planewave_planar_planewave_planar(
            wavevector_planewave_templates,
            wavevector_planewave_images,
            gamma,
            displacements, # n_displacements x 2
            angle_planar_ctf_template_tensor,
            angle_planar_ctf_image_tensor,
            polar_grid.radius_max
        ) ## cross correlation of CTF * templates and CTF images for this version of PPPP
        self.I_E_T_form = planar_planewave_planewave(
            wavevector_planewave_templates,
            wavevector_planewave_identity,
            gamma,
            displacements, # n_displacements x 2
            angle_planar_ctf_template_tensor,
            polar_grid.radius_max
        ) ## cross correlation of templates and identity for this version of PP
        self.I_E_M_form = planar_planewave_planewave(
            wavevector_planewave_images,
            wavevector_planewave_identity,
            gamma,
            displacements, # n_displacements x 2
            angle_planar_ctf_image_tensor,
            polar_grid.radius_max
        ) ## cross correlation of images and identity for this version of PP

        self.I_T_M_form = self.I_M_T_form.transpose(0, 1)
        self.I_T_E_form = self.I_E_T_form.transpose(0, 1)
        self.I_M_E_form = self.I_E_M_form.transpose(0, 1)

        # self.I_E_E_form = torch.clamp(self.I_E_E_form, min=1e-12)

        ## print the shape of things
        print(f'I_T_T_form shape: {self.I_T_T_form.shape}, type: {self.I_T_T_form.dtype}, max: {self.I_T_T_form.max()}, min: {self.I_T_T_form.min()}, absmin: {self.I_T_T_form.abs().min()}')
        print(f'I_M_M_form shape: {self.I_M_M_form.shape}, type: {self.I_M_M_form.dtype}, max: {self.I_M_M_form.max()}, min: {self.I_M_M_form.min()}, absmin: {self.I_M_M_form.abs().min()}')
        print(f'I_E_E_form shape: {self.I_E_E_form.shape}, type: {self.I_E_E_form.dtype}, max: {self.I_E_E_form.max()}, min: {self.I_E_E_form.min()}, absmin: {self.I_E_E_form.abs().min()}')
        print(f'I_T_M_form shape: {self.I_T_M_form.shape}, type: {self.I_T_M_form.dtype}, max: {self.I_T_M_form.max()}, min: {self.I_T_M_form.min()}, absmin: {self.I_T_M_form.abs().min()}')
        print(f'I_T_E_form shape: {self.I_T_E_form.shape}, type: {self.I_T_E_form.dtype}, max: {self.I_T_E_form.max()}, min: {self.I_T_E_form.min()}, absmin: {self.I_T_E_form.abs().min()}')
        print(f'I_M_E_form shape: {self.I_M_E_form.shape}, type: {self.I_M_E_form.dtype}, max: {self.I_M_E_form.max()}, min: {self.I_M_E_form.min()}, absmin: {self.I_M_E_form.abs().min()}')
        print(f'I_M_T_form shape: {self.I_M_T_form.shape}, type: {self.I_M_T_form.dtype}, max: {self.I_M_T_form.max()}, min: {self.I_M_T_form.min()}, absmin: {self.I_M_T_form.abs().min()}')
        print(f'I_E_T_form shape: {self.I_E_T_form.shape}, type: {self.I_E_T_form.dtype}, max: {self.I_E_T_form.max()}, min: {self.I_E_T_form.min()}, absmin: {self.I_E_T_form.abs().min()}')
        print(f'I_E_M_form shape: {self.I_E_M_form.shape}, type: {self.I_E_M_form.dtype}, max: {self.I_E_M_form.max()}, min: {self.I_E_M_form.min()}, absmin: {self.I_E_M_form.abs().min()}')
        # exit()

        self.I_Tcen_Tcen_form = self.I_T_T_form
        self.I_Mcen_Mcen_form = self.I_M_M_form
        self.I_Tcen_Mcen_form = self.I_T_M_form
        self.I_Mcen_Tcen_form = self.I_M_T_form

        # self.I_Tcen_Tcen_form = self.I_T_T_form - (self.I_T_E_form[:,0,:,:] ** 2) / self.I_E_E_form[:,0,:,:]
        # self.I_Mcen_Mcen_form = self.I_M_M_form - (self.I_M_E_form[:,0,:,:] ** 2) / self.I_E_E_form[:,0,:,:]
        # self.I_Tcen_Mcen_form = self.I_T_M_form - (self.I_T_E_form[:,0,:,:].unsqueeze(1) * self.I_M_E_form[:,0,:,:].unsqueeze(0)) / self.I_E_E_form
        # self.I_Mcen_Tcen_form = self.I_M_T_form - (self.I_M_E_form[:,0,:,:].unsqueeze(1) * self.I_T_E_form[:,0,:,:].unsqueeze(0)) / self.I_E_E_form
        
        print(f'I_Tcen_Tcen_form shape: {self.I_Tcen_Tcen_form.shape}, type: {self.I_Tcen_Tcen_form.dtype}, max: {self.I_Tcen_Tcen_form.max()}, min: {self.I_Tcen_Tcen_form.min()}, absmin: {self.I_Tcen_Tcen_form.abs().min()}')
        print(f'I_Mcen_Mcen_form shape: {self.I_Mcen_Mcen_form.shape}, type: {self.I_Mcen_Mcen_form.dtype}, max: {self.I_Mcen_Mcen_form.max()}, min: {self.I_Mcen_Mcen_form.min()}, absmin: {self.I_Mcen_Mcen_form.abs().min()}')
        print(f'I_Tcen_Mcen_form shape: {self.I_Tcen_Mcen_form.shape}, type: {self.I_Tcen_Mcen_form.dtype}, max: {self.I_Tcen_Mcen_form.max()}, min: {self.I_Tcen_Mcen_form.min()}, absmin: {self.I_Tcen_Mcen_form.abs().min()}')
        print(f'I_Mcen_Tcen_form shape: {self.I_Mcen_Tcen_form.shape}, type: {self.I_Mcen_Tcen_form.dtype}, max: {self.I_Mcen_Tcen_form.max()}, min: {self.I_Mcen_Tcen_form.min()}, absmin: {self.I_Mcen_Tcen_form.abs().min()}')

        # self.I_Tnrm_Tnrm_form = self.I_Tcen_Tcen_form / torch.clamp(torch.sqrt(self.I_Tcen_Tcen_form * self.I_Tcen_Tcen_form), min=1e-12)
        self.I_Tnrm_Mnrm_form = self.I_Tcen_Mcen_form / torch.sqrt(self.I_Tcen_Tcen_form.unsqueeze(0) * self.I_Mcen_Mcen_form.unsqueeze(1))
        self.I_Mnrm_Tnrm_form = self.I_Mcen_Tcen_form / torch.sqrt(self.I_Mcen_Mcen_form.unsqueeze(1) * self.I_Tcen_Tcen_form.unsqueeze(0))
        # self.I_Mnrm_Mnrm_form = self.I_Mcen_Mcen_form / torch.clamp(torch.sqrt(self.I_Mcen_Mcen_form * self.I_Mcen_Mcen_form), min=1e-12)

        # print(f'I_Tnrm_Tnrm_form shape: {self.I_Tnrm_Tnrm_form.shape}, type: {self.I_Tnrm_Tnrm_form.dtype}')
        print(f'I_Tnrm_Mnrm_form shape: {self.I_Tnrm_Mnrm_form.shape}, type: {self.I_Tnrm_Mnrm_form.dtype}, max: {self.I_Tnrm_Mnrm_form.max()}, min: {self.I_Tnrm_Mnrm_form.min()}, absmin: {self.I_Tnrm_Mnrm_form.abs().min()}')
        print(f'I_Mnrm_Tnrm_form shape: {self.I_Mnrm_Tnrm_form.shape}, type: {self.I_Mnrm_Tnrm_form.dtype}, max: {self.I_Mnrm_Tnrm_form.max()}, min: {self.I_Mnrm_Tnrm_form.min()}, absmin: {self.I_Mnrm_Tnrm_form.abs().min()}')
        # print(f'I_Mnrm_Mnrm_form shape: {self.I_Mnrm_Mnrm_form.shape}, type: {self.I_Mnrm_Mnrm_form.dtype}')

        # self.I_costheta = 0.5 * (self.I_Mnrm_Tnrm_form + self.I_Tnrm_Mnrm_form.transpose(0,1))  # similarity measure.
        self.I_costheta = self.I_Mnrm_Tnrm_form
        self.I_sinthetasquared = 1 - self.I_costheta ** 2  # dis-similarity measure.

        print(f'I_costheta shape: {self.I_costheta.shape}, type: {self.I_costheta.dtype}, max: {self.I_costheta.max()}, min: {self.I_costheta.min()}')
        print(f'I_sinthetasquared shape: {self.I_sinthetasquared.shape}, type: {self.I_sinthetasquared.dtype}, max: {self.I_sinthetasquared.max()}, min: {self.I_sinthetasquared.min()}, absmin: {self.I_sinthetasquared.abs().min()}')

    
    def _ssnll_function(self):
        return 0.5 * self.I_Tcen_Tcen_form.unsqueeze(0) * self.I_sinthetasquared


    def log_likelihood_final(self):
        """
        -(3/2 - n_pixel/2) * log(2*pi) ...
        +(1/2) * log(max(1e-12,I_Mcen_Mcen_form)) ...
        +(1.0) * log(2) ...
        -(2 - n_pixel/2) * log(max(1e-12,ssnll_function)) ...
        - gammaln(n_pixel/2 - 2) ...
        + log(max(1e-12,I_E_E_form)) ...
        """
        # return - torch.log(self._ssnll_function())

        ###
        ### Marginalizing over COMPLEX mass (offset identity kernel)
        ###
        log_lik = + (1.5 - self.n_pixels / 2.0) * np.log(2 * pi) \
               - 0.5 * torch.log(self.I_Mcen_Mcen_form.unsqueeze(1)) \
               - 1.0 * np.log(2) \
               + (2.0 - self.n_pixels / 2.0) * torch.log(self._ssnll_function()) \
               + lgamma(self.n_pixels / 2.0 - 2.0) \
               - torch.log(self.I_E_E_form) \
        
        ###
        ### Marginalizing over REAL mass (offset identity kernel)
        ###
        # log_lik = + (1.0 - self.n_pixels / 2.0) * np.log(2 * pi) \
        #        - 0.5 * torch.log(self.I_Mcen_Mcen_form.unsqueeze(1)) \
        #        - 1.0 * np.log(2) \
        #        + (1.5 - self.n_pixels / 2.0) * torch.log(self._ssnll_function()) \
        #        + lgamma(self.n_pixels / 2.0 - 1.5) \
        #        - torch.log(self.I_E_E_form)

        print(f'log_likelihood_final shape: {log_lik.shape}, type: {log_lik.dtype}, max: {log_lik.max()}, min: {log_lik.min()}, absmin: {log_lik.abs().min()}')
        return log_lik
        