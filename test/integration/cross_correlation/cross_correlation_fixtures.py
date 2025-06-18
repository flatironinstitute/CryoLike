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

from cryolike.util import (
    Precision,
    to_torch,
    absq
)


class parameters():
    device: str
    n_pixels: int
    precision: Precision
    wavevector: torch.Tensor
    max_displacement: float  # this is max per axis; real max will be this * root 2
    abs_tolerance_cross_correlation: float
    rel_tolerance_cross_correlation: float

    def __init__(
        self,
        device: str,
        n_pixels: int,
        precision: Precision,
        wavevector: torch.Tensor,
        max_displacement: float,
        abs_tolerance_cross_correlation: float,
        rel_tolerance_cross_correlation: float,
        abs_tolerance_log_likelihood: float,
        rel_tolerance_log_likelihood: float,
    ):
        self.device = device
        self.n_pixels = n_pixels
        self.precision = precision
        self.wavevector = wavevector
        self.max_displacement = max_displacement
        self.abs_tolerance_cross_correlation = abs_tolerance_cross_correlation
        self.rel_tolerance_cross_correlation = rel_tolerance_cross_correlation
        self.abs_tolerance_log_likelihood = abs_tolerance_log_likelihood
        self.rel_tolerance_log_likelihood = rel_tolerance_log_likelihood


    def duplicate(self, *,
        device: str | None = None,
        n_pixels: int | None = None,
        precision: Precision | None = None,
        wavevector: torch.Tensor | None = None,
        max_displacement: float | None = None,
        abs_tolerance_cross_correlation: float | None = None,
        rel_tolerance_cross_correlation: float | None = None,
        abs_tolerance_log_likelihood: float | None = None,
        rel_tolerance_log_likelihood: float | None = None,
    ):
        (float_type, _, _) = self.precision.get_dtypes(default=Precision.DOUBLE)
        if precision is not None:
            (float_type, _, _) = precision.get_dtypes(default=Precision.DOUBLE)
        wv = wavevector if wavevector is not None else self.wavevector
        return parameters(
            device=device if device is not None else self.device,
            n_pixels=n_pixels if n_pixels is not None else self.n_pixels,
            precision=precision if precision is not None else self.precision,
            wavevector=wv.to(float_type),
            max_displacement=max_displacement if max_displacement is not None else self.max_displacement,
            abs_tolerance_cross_correlation=abs_tolerance_cross_correlation if abs_tolerance_cross_correlation is not None else self.abs_tolerance_cross_correlation,
            rel_tolerance_cross_correlation=rel_tolerance_cross_correlation if rel_tolerance_cross_correlation is not None else self.rel_tolerance_cross_correlation,
            abs_tolerance_log_likelihood=abs_tolerance_log_likelihood if abs_tolerance_log_likelihood is not None else self.abs_tolerance_log_likelihood,
            rel_tolerance_log_likelihood=rel_tolerance_log_likelihood if rel_tolerance_log_likelihood is not None else self.rel_tolerance_log_likelihood,
        )


    @staticmethod
    def default():
        (default_float, _, _) = Precision.DOUBLE.get_dtypes(default=Precision.DOUBLE)
        return parameters(
            device='cpu',
            n_pixels=128,
            precision=Precision.DOUBLE,
            wavevector=torch.tensor([-0.17, -0.03, 0.07], dtype=default_float),
            max_displacement=0.08,
            abs_tolerance_cross_correlation=1e-6,
            rel_tolerance_cross_correlation=1e-6,
            abs_tolerance_log_likelihood=1e-3,
            rel_tolerance_log_likelihood=1e-3,
        )
    

# TODO: Parameterize based on: wavevector_planewave, max_displacement,
    # this basically simulates the effect of a displaced atom
    # So maybe also try -.8 to +.8
    # try 3 to 5 possible values in that range? Further away = higher values = lower accuracy

def make_cases() -> list[parameters]:
    cases = [parameters.default()]
    with_single_precision = [x.duplicate(
        precision=Precision.SINGLE, 
        abs_tolerance_cross_correlation=1e-5, rel_tolerance_cross_correlation=1e-3, 
        abs_tolerance_log_likelihood=1e-3, rel_tolerance_log_likelihood=1e-3
    ) for x in cases]
    cases.extend(with_single_precision)
    with_cuda = [x.duplicate(device='cuda') for x in cases]
    cases.extend(with_cuda)
    low_pixel = [x.duplicate(n_pixels=64) for x in cases]
    high_pixel = [x.duplicate(n_pixels=256) for x in cases]
    cases.extend(low_pixel)
    cases.extend(high_pixel)

    return cases


def make_polar_grid(n_pixels: int) -> PolarGrid:
    radius_max = n_pixels / (2.0 * pi) * pi / 2.0
    dist_radii = 0.5 / (2.0 * pi) * pi / 2.0
    n_inplanes = n_pixels * 4
    polar_grid = PolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        uniform = True,
        return_cartesian = True
    )
    return polar_grid


def make_planewave_templates(
    wavevector_planewave: Tensor,
    viewing_angles: ViewingAngles,
    polar_grid: PolarGrid,
    precision: Precision,
    device: torch.device = torch.device("cpu"),
) -> Templates:
    (_, complex_type, _) = precision.get_dtypes(default=Precision.DOUBLE)
    def wavevector_function(fourier_slice: Tensor) -> Tensor:
        fs = fourier_slice.to(wavevector_planewave.dtype)
        return torch.exp(2 * pi * 1j * torch.matmul(fs, wavevector_planewave)).to(complex_type)
    templates = Templates.generate_from_function(
            wavevector_function, viewing_angles, polar_grid,
            compute_device=device, output_device="cpu", precision=precision
        )
    return templates


def make_viewing_angles(device: torch.device, dtype: torch.dtype):
    polars = 1 * pi * torch.tensor([ 0.28, 0.09, 0.72, 0.00 ], dtype=dtype).to(device)
    azimus = 2 * pi * torch.tensor([ 0.10, 0.32, 0.85, 0.00 ], dtype=dtype).to(device)
    gammas = 2 * pi * torch.tensor([ 0.71, 0.14, 0.48, 0.00 ], dtype=dtype).to(device)
    return ViewingAngles(polars, azimus, gammas)


def viewing_angles_to_cartesian_displacements(
    viewing_angles: ViewingAngles,
    wave_vector_delta_a: torch.Tensor,
) -> torch.Tensor:
    """Convert viewing angles to Cartesian displacements on the wave vector.

    The general formula used here is as follows.

    let sa and ca be sin(polar_a) and cos(polar_a), respectively.
    let sb and cb be sin(azimu_b) and cos(azimu_b), respectively.
    let sc and cc be sin(gamma_z) and cos(gamma_z), respectively.
    And rotation by azimu_b about the +z-axis is represented as:
    Rz(azimu_b) =
    [ +cb -sb 0 ]
    [ +sb +cb 0 ]
    [  0   0  1 ]
    And rotation by polar_a about the +y-axis is represented as:
    Ry(polar_a) =
    [ +ca 0 +sa ]
    [  0  1  0  ]
    [ -sa 0 +ca ]
    And rotation by gamma_z about the +z-axis is represented as:
    Rz(gamma_z) =
    [ +cc -sc 0 ]
    [ +sc +cc 0 ]
    [  0   0  1 ]
    Which, collectively, implies that under the transform:
    Rz(azimu_b) * Ry(polar_a) * Rz(gamma_z),
    Which is the same as:
    [ +cb -sb 0 ] [ +ca*cc -ca*sc +sa ]   [ +cb*ca*cc - sb*sc , -cb*ca*sc -sb*cc , +cb*sa ]
    [ +sb +cb 0 ] [ +sc    +cc    0   ] = [ +sb*ca*cc + cb*sc , -sb*ca*sc +cb*cc , +sb*sa ]
    [  0   0  1 ] [ -sa*cc +sa*sc +ca ]   [ -sa*cc            , +sa*sc           , +ca    ]
    the point [1;0;0] is mapped to:
    [ template_k_c_0    ; template_k_c_1    ; template_k_c_2 ] =
    [ +cb*ca*cc - sb*sc ; +sb*ca*cc + cb*sc ; -sa*cc         ]
    
    Given the convention above, we associated the template-value
    S_k_p_wkS__(1+na,1+nS) with the indices:
    nS = template-index determining viewing angles.
    na = multi-index nw + n_w_csum_(1+nk_p_r)
    nw =angle psi = 2*pi*nw/max(1,n_w_(1+nk_p_r)),
    nk_p_r = radius k = k_p_r_(1+nk_p_r).
    with the location Rz(+psi)*[1;0;0] under the above map in 3-dimensional k_p_ space.

    Similarly, we can consider reconstructing the template S_k_p_wkS__(:,1+nS)
    by first rotating the volume a_k_p by the transformation:
    inv( Rz(azimu_b) * Ry(polar_a) * Rz(gamma_z) ) = Rz(-gamma_z) * Ry(-polar_a) * Rz(-azimu_b)
    and then taking the equatorial-slice: (k,psi) = [k*cos(psi);k*sin(psi);0] in 3-dimensions.

    Thus, if:
    a_k_p = exp(+2*pi*i * transpose(k_p_) * delta_a_ )
    we can see that:
    R__ * a_k_p = exp(+2*pi*i * transpose((inv(R) * k_p_)) * delta_a_ )
    R__ * a_k_p = exp(+2*pi*i * transpose(k_p_) * R__ * delta_a_ )

    and thus we can reconstitute the template:
    S_k_p_wkS__(:,1+nS)
    by applying the transformation:
    Rz(-gamma_z) * Ry(-polar_a) * Rz(-azimu_b)
    to the wave-vector delta_a_,
    and then restricting the result to the equatorial-plane (i.e., ignoring the final coordinate).


    Args:
        viewing_angles (ViewingAngles): The viewing angles used throughout the test
        wave_vector_delta_a (FloatArrayType): The 3-d wave vector
    """
    device = viewing_angles.gammas.device

    def _rotate_about_plus_z_axis(angle: Tensor):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        rotation_matrix = torch.stack((
            torch.stack(( cos_angle, -1. * sin_angle, zeros)),
            torch.stack(( sin_angle,       cos_angle, zeros)),
            torch.stack((     zeros,           zeros,  ones))
        )).to(device).permute(2, 0, 1)
        return rotation_matrix
    
    def _rotate_about_plus_y_axis(angle: Tensor):
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        zeros = torch.zeros_like(angle)
        ones = torch.ones_like(angle)
        rotation_matrix = torch.stack((
            torch.stack((      cos_angle, zeros, sin_angle)),
            torch.stack((          zeros,  ones,     zeros)),
            torch.stack((-1. * sin_angle, zeros, cos_angle))
        )).to(device).permute(2, 0, 1)
        return rotation_matrix

    Rz_gam = _rotate_about_plus_z_axis(-1. * viewing_angles.gammas)
    Ry     = _rotate_about_plus_y_axis(-1. * viewing_angles.polars)
    Rz_azi = _rotate_about_plus_z_axis(-1. * viewing_angles.azimus)

    rotation = torch.matmul(Rz_gam, torch.matmul(Ry, Rz_azi)) # n_angles x 3 x 3
    # below: n x 2
    rotated_wavevector: torch.Tensor = torch.tensordot(rotation, wave_vector_delta_a.to(device), dims=([2], [0]))[:, 0:2] # type: ignore
    return rotated_wavevector


def get_planar_ctf(grid: PolarGrid, phi_S: float, box_size: float, precision: Precision, device: torch.device) -> CTF:
    ctf = (2.0 * to_torch(grid.radius_points, precision=precision, device=device) \
            * torch.cos(to_torch(grid.theta_points, precision=precision, device=device) - phi_S)) \
          .reshape(1, grid.n_shells, grid.n_inplanes)
    return CTF(
        polar_grid=grid,
        box_size = box_size,
        anisotropy = True,
        ctf_descriptor = ctf
    )

def get_difference_wavevector_images_templates(
    wavevector_planewave_templates: Tensor,
    wavevector_planewave_images: Tensor,
    grid_inplanes: Tensor,
    searched_displacements: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:

    sin_gamma = torch.sin(grid_inplanes) # (n_inplanes)
    cos_gamma = torch.cos(grid_inplanes) # (n_inplanes)
    gamma_z_rotation = torch.stack((torch.stack((cos_gamma, - sin_gamma)),
                                    torch.stack((sin_gamma,   cos_gamma)))).permute(2,0,1)  # n_inplanes * 2 * 2
    _wavevector_planewave_templates = wavevector_planewave_templates.to(searched_displacements.device)
    _wavevector_planewave_images = wavevector_planewave_images.to(searched_displacements.device)

    searched_displacements_plus_wavevector_planewave = _wavevector_planewave_templates.unsqueeze(1) - searched_displacements.unsqueeze(0) # (viewing_angles * n_displacements * 2)
    searched_displacements_plus_wavevector_planewave_templates_rotated = torch.tensordot(gamma_z_rotation, searched_displacements_plus_wavevector_planewave, dims=([2], [2])).permute(2,3,0,1) # type: ignore # viewing_angles * n_displacements * n_inplanes * 2
    searched_displacements_plus_wavevector_planewave_templates_rotated = searched_displacements_plus_wavevector_planewave_templates_rotated.unsqueeze(0) # unsqueeze for images dimension
    
    offset_delta_T = _wavevector_planewave_images.unsqueeze(1).unsqueeze(2).unsqueeze(3) - searched_displacements_plus_wavevector_planewave_templates_rotated
    offset_radius = torch.norm(offset_delta_T, dim=-1, p="fro")
    offset_angle_omega_t = torch.atan2(offset_delta_T[:,:,:,:,1], offset_delta_T[:,:,:,:,0])
    
    return offset_delta_T, offset_radius, offset_angle_omega_t


def planewave_planar_planewave_planar(
    wavevector_planewave_templates: Tensor,
    wavevector_planewave_images: Tensor,
    grid_inplanes: Tensor,
    searched_displacements: Tensor,
    angle_planar_ctf_template: Tensor,
    angle_planar_ctf_image: Tensor,
    grid_max_radius_K: float
) -> Tensor:
    """Analytic solution for integral of plane wave x planar fn x plane wave x planar fn,
    returning a vector of [image_count] dimension.

    This corresponds to equation 0.6 in the explanatory note.

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
        angle_planar_ctf_image (Tensor): Angle by which the image CTF's planar
            function has been rotated about the +Z axis
        grid_max_radius_K (float): Upper bound for the integration

    Returns:
        Tensor: A tensor directly comparable with the non-aggregated cross-correlation
            likelihood result returned from
            CrossCorrelationLikelihood._compute_cross_correlation_likelihood()
    """

    two_pi_K = grid_max_radius_K * 2 * pi
    _device = wavevector_planewave_templates.device
    
    offset_delta_T, offset_radius, offset_angle_omega_t = get_difference_wavevector_images_templates(
        wavevector_planewave_templates,
        wavevector_planewave_images,
        grid_inplanes,
        searched_displacements
    )
    offset_angle_omega_t = offset_angle_omega_t.to(_device)
    two_pi_K_delta_T = (offset_radius * two_pi_K).cpu()

    bessel_0 = jv(0, two_pi_K_delta_T).to(_device)
    bessel_2 = jv(2, two_pi_K_delta_T).to(_device)
    bessel_4 = jv(4, two_pi_K_delta_T).to(_device)

    phi_pos = angle_planar_ctf_template + angle_planar_ctf_image ## scalar
    phi_neg = angle_planar_ctf_template - angle_planar_ctf_image ## scalar

    mode_0: Tensor = torch.cos(phi_neg) * (3. * bessel_0 + 2. * bessel_2 - bessel_4) / 12.0
    mode_2: Tensor = torch.cos(phi_pos - 2. * offset_angle_omega_t) * (bessel_2 + bessel_4) / 6.0
    result = 4 * (mode_0 - mode_2) # * pi * grid_max_radius_K ** 4

    return result