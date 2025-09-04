from scipy.special import jv
from torch import Tensor
import torch
from numpy import pi

from cryolike.grids import PolarGrid
from cryolike.stacks import Templates
from cryolike.metadata import ViewingAngles
from cryolike.microscopy import CTF

from cryolike.util import (
    Precision,
    to_torch
)


def make_polar_grid(n_pixels: int) -> PolarGrid:
    radius_max = n_pixels / (2.0 * pi) * pi / 2.0
    dist_radii = 0.5 / (2.0 * pi) * pi / 2.0
    n_inplanes = n_pixels * 4
    polar_grid = PolarGrid(
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        uniform = True
    )
    return polar_grid


def make_planewave_templates(
    wavevector_planewave: Tensor,
    viewing_angles: ViewingAngles,
    polar_grid: PolarGrid,
    precision: Precision,
    device: str = "cpu"
) -> Templates:
    (_, complex_type, _) = precision.get_dtypes(default=Precision.DOUBLE)
    def wavevector_function(fourier_slice: Tensor) -> Tensor:
        fs = fourier_slice.to(wavevector_planewave.dtype)
        # There is surely a better way to do this
        # but for now we're just going to iterate over the wave vectors
        res = torch.zeros(fourier_slice.shape[0:-1], dtype=complex_type, device=device)
        for i in range(wavevector_planewave.shape[0]):
            res += torch.exp(2 * pi * 1j * torch.matmul(fs, wavevector_planewave[i]))
        return res
    templates = Templates.generate_from_function(
                    wavevector_function,
                    viewing_angles,
                    polar_grid, 
                    device = device,
                    output_device = "cpu",
                    precision = precision
                )
    return templates


def make_viewing_angles(device: str, dtype: torch.dtype):
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
    rotated_wavevector: torch.Tensor = \
        torch.tensordot(rotation,
                        wave_vector_delta_a.to(device),
                        dims=([2], [-1]) # type: ignore
                       ).permute(0,2,1)[..., 0:2]
    return rotated_wavevector


def get_planar_ctf(grid: PolarGrid, phi_ctf: float, box_size: float, precision: Precision, device: str) -> CTF:
    r_pts = to_torch(grid.radius_points, precision=precision, device=device)
    thetas = to_torch(grid.theta_points, precision=precision, device=device)
    ctf = (2.0 * r_pts * torch.cos(thetas - phi_ctf)) \
        .reshape(1, grid.n_shells, grid.n_inplanes)

    return CTF(
        polar_grid=grid,
        box_size = box_size,
        anisotropy = True,
        ctf_descriptor = ctf
    )


def _pairwise_difference(a: Tensor, b: Tensor, dim: int = -1):
    a_cnt = a.shape[dim]
    b_cnt = b.shape[dim]
    a_expanded_shape = [1] * len(a.shape)
    a_expanded_shape[dim] *= b_cnt

    delta = a.repeat(a_expanded_shape) - b.repeat_interleave(a_cnt, dim = dim)
    return delta


def _rotation_matrix(gamma_rotations: Tensor):
    sin_gamma = torch.sin(gamma_rotations)
    cos_gamma = torch.cos(gamma_rotations)
    rotations = torch.stack((torch.stack((cos_gamma, -sin_gamma)),
                             torch.stack((sin_gamma,  cos_gamma)))
                           ).permute(2, 0, 1)
    return rotations


def _displace_templates(
    templates: Tensor,
    displacements: Tensor,
    rotations: Tensor
):
    """Apply displacements and rotations to template wavevectors, resulting in
    a tensor of [n_viewings, n_displacements, n_inplanes, n_sources, [x,y]].

    Args:
        templates (Tensor): A 3D template of [n_viewings x n_sources_per_template x [x,y]]
        displacements (Tensor): Displacement grid of [n_displacements x [x,y]]
        rotations (Tensor): Inplane rotation matrix of [n_inplanes x 2 x 2, where the 
            2x2 dimensions constitute a rotation matrix per inplane

    Returns:
        Tensor: Fully realized set of template planewave sources, indexed as
            [n_viewing, n_displacement, n_inplane, n_source, [x,y]] ([x,y] being a
            plane wave source location)
    """
    # to displace the templates, apply each displacement to a copy of each template, adding a dimension.
    # so expand [T x s x 2] -> [T, :, source, 2]
    #  and from [D x 2]     -> [:, D,   :,    2]
    displaced = templates.unsqueeze(1) - displacements.unsqueeze(1).unsqueeze(0)
    # Yields [T x D x source x 2]. Now apply rotation tensor ([n_inplanes x 2 x2]) to each source.
    # (for each n_inplane of the rot, we have a 2x2 matrix, and we want to multiply that
    # against the 2-vector in the last place of the template tensor to yield a rotated 2-vector.)
    # Since we're left-multiplying by the rotations, tensordot will give us that tensor's indices as
    # the major ones, i.e. [n_inplanes, [rotated x,y], n_viewing, n_displacement, n_source],
    # so permute it back out to [n_viewing, n_displacement, n_rotation, n_source, (x,y)]
    rotated = torch.tensordot(rotations, displaced, dims=([-1], [-1])).permute(2, 3, 0, 4, 1) # type: ignore
    # TODO: Figure out a way to do this switching the two tensors so we don't have to permute so much.
    return rotated


def p_xx_p_kernel_multiwave(
    wavevectors_a: Tensor,
    wavevectors_b: Tensor,
    angle_a: float,
    angle_b: float,
    max_radius_K: float
):
    bessel_coeff = 2 * pi * max_radius_K
    return_coeff = pi * max_radius_K ** 4

    # recall delta is m - s, phi is s - m
    # _pairwise_difference must assume that the vectors might have
    # different wave counts, and anyway we need the order of the
    # pairwise differences to match between the displacements and
    # the angles.
    delta_t = _pairwise_difference(wavevectors_a, wavevectors_b, dim = -2)
    # # raise ValueError(f"dt new method slice 1 2 12 200: {delta_t[1,2,12,200]}")
    # # raise ValueError(f"delta t new method: {delta_t.shape}")

    # phi_neg = _pairwise_difference(-1. * angles_a, -1. * angles_b)
    phi_neg: Tensor = torch.tensor(angle_b - angle_a)
    phi_pos: Tensor = torch.tensor(angle_b + angle_a)
    delta_t_norm: Tensor = torch.norm(delta_t, dim=-1)
    omega_t = torch.atan2(delta_t[...,1], delta_t[...,0])
    
    x = (bessel_coeff * delta_t_norm).cpu().numpy()
    bessel_0: Tensor = torch.tensor(jv(0, x), device=wavevectors_a.device)
    bessel_2: Tensor = torch.tensor(jv(2, x), device=wavevectors_a.device)
    bessel_4: Tensor = torch.tensor(jv(4, x), device=wavevectors_a.device)

    mode_0: Tensor = torch.cos(phi_neg) * (3. * bessel_0 + 2. * bessel_2 - bessel_4) / 12.
    mode_2: Tensor = torch.cos(phi_pos - 2. * omega_t) * (bessel_2 + bessel_4) / 6.

    return 4 * return_coeff * torch.sum((mode_0 - mode_2), dim=-1)


def p_xx_p_multiwave_vectorized(
    t_wv: Tensor,
    i_wv: Tensor,
    template_ctf_angle: float,
    image_ctf_angle: float,
    gamma_rotations: Tensor,
    displacement_grid: Tensor,
    grid_max_radius_K: float
):
    """Analytical computation of normalized cross-correlation between all (displaced
    and rotated) templates and all images, using vector operations.

    Args:
        t_wv (Tensor): Template wavevectors. A set of planewave templates, with
            each template defined as a stack of 2-d displacements defining a plane
            wave whose wavelength is the displacement between the origin and
            the displacement. We assume that the templates are all different
            projections of a common parent planewave, viewed from different
            viewing angles (computed elsewhere).
            Vector should be indexed as [n_template, n_wave_vectors, [x, y]].
        i_wv (Tensor): Image wavevectors. As with the template wave vectors, but
            there is no expectation of any relationship to viewing angles.
            Vector should be indexed as [n_img, n_wave_vectors, [x, y]].
        template_ctf_angles (Tensor): A stack of stacks of angles defining the
            planar ctf function applied to each plane wave of the templates.
            Every plane wave displacement has a single ctf angle, i.e. the
            n_wave_vector dimension must match the one in template_wavevectors.
            Indexed as [n_template x n_wave_vector].
        image_ctf_angles (Tensor): A stack of stacks of angles defining the
            planar ctf function applied to each plane wave of the images.
            Indexed as [n_image x n_wave_vector].
        gamma_rotations (Tensor): Vector of inplane rotations from the
            viewing angles via the cross-correlation-likelihood object.
            Used to apply rotations to the templates.
        displacement_grid (Tensor): Grid of x- and y-displacements to apply to
            each template before comparing to the images
        grid_max_radius_K (float): Maximum radius of the polar quadrature grid

    Returns:
        Tensor: A tensor of normalized cross-correlation likelihoods for each
            base template-image pair, times all possible displacements, and
            all possible inplane rotations. Should match the output of the
            cross-correlation likelihood calculation's FULL_TENSOR return.
            Indexed as [n_template, n_image, n_displacement, n_inplane].
    """
    if len(t_wv.shape) < 3:
        t_wv = t_wv.unsqueeze(1)
    if len(i_wv.shape) < 3:
        i_wv = i_wv.unsqueeze(1)
    rotations = _rotation_matrix(gamma_rotations)
    # realized_templates are now [S, d, w, n_sources, [x,y]]
    realized_templates = _displace_templates(t_wv, displacement_grid, rotations)

    rt_norms = p_xx_p_kernel_multiwave(
        realized_templates,
        realized_templates,
        template_ctf_angle,
        template_ctf_angle,
        grid_max_radius_K
    )
    image_norms = p_xx_p_kernel_multiwave(
        i_wv,
        i_wv,
        image_ctf_angle,
        image_ctf_angle,
        grid_max_radius_K
    )

    # realized_templates should now be S,d,w,n_sources,[x,y].
    # image_wavevectors remain M,n_sources,[x,y].
    # We want the pairwise differences to look like S,M,d,w,n_src,[x,y].
    # So expand realized_templates to S,:,d,w,n_src,2 and
    # images to :,M,:,:,n_src,2
    # # realized_templates = realized_templates.unsqueeze(1)
    # TODO: This matches the current implementation code, which has S and M indices swapped
    realized_templates = realized_templates.unsqueeze(0)
    i_shape = i_wv.shape
    # TODO: Should be subtracting images from templates but it matches implementation code
    i_wv = i_wv.reshape(i_shape[0], 1, 1, 1, i_shape[1], i_shape[2])

    # TODO: Implementation code has swapped image and template indices, swap back once fixed
    # (raw_integral, d_t_norm) = p_xx_p_kernel_multiwave(realized_templates, i_wv, template_ctf_angles, image_ctf_angles, grid_max_radius_K)
    raw_integral = p_xx_p_kernel_multiwave(i_wv, realized_templates, image_ctf_angle, template_ctf_angle, grid_max_radius_K)
    # TODO: denominator will also need to be swapped when implementation is fixed
    denominator = torch.sqrt(rt_norms.unsqueeze(0) * image_norms.reshape(i_shape[0], 1, 1, 1))

    return raw_integral / denominator


# # ## NOTE: The following two functions are only a check that the vectorized
# # ## multi-wave-vector analytic formula was implemented correctly.
# # ## We should keep them for future reference but we are not actually
# # ## testing anything with them in an automated sense.

# # def _p_xx_p_minikernel(
# #     delta_t_norm: Tensor,
# #     phi_pos: Tensor,
# #     phi_neg: Tensor,
# #     omega_t: Tensor,
# #     bessel_coeff: float,
# #     return_coeff: float
# # ):
# #     x = (bessel_coeff * delta_t_norm).cpu().numpy()
# #     bessel_0: Tensor = torch.tensor(jv(0, x), device=delta_t_norm.device)
# #     bessel_2: Tensor = torch.tensor(jv(2, x), device=delta_t_norm.device)
# #     bessel_4: Tensor = torch.tensor(jv(4, x), device=delta_t_norm.device)

# #     mode_0: Tensor = torch.cos(phi_neg) * (3. * bessel_0 + 2. * bessel_2 - bessel_4) / 12
# #     mode_2: Tensor = torch.cos(phi_pos - 2. * omega_t) * (bessel_2 + bessel_4) / 6.

# #     return 4 * return_coeff * (mode_0 - mode_2)


# # def p_xx_p_multiwave_iterative(
# #     t_wv: Tensor,
# #     i_wv: Tensor,
# #     template_ctf_angle: float,
# #     image_ctf_angle: float,
# #     gamma_rotations: Tensor,
# #     displacement_grid: Tensor,
# #     grid_max_radius_K: float
# # ):
# #     # 0: Constants
# #     bessel_coeff = 2 * pi * grid_max_radius_K
# #     return_coeff = pi * grid_max_radius_K ** 4

# #     # 1: Compute image norms.
# #     img_count = i_wv.shape[0]
# #     i_src_count = i_wv.shape[1]
# #     _img_norm_phi_pos = torch.tensor(2 * image_ctf_angle)
# #     _img_norm_phi_neg = torch.tensor(0)
# #     i_norms = []
# #     for img in range(img_count):
# #         i_norm = 0
# #         for i in range(i_src_count):
# #             m_src = i_wv[img, i]
# #             for j in range(i_src_count):
# #                 s_src = i_wv[img, j]
# #                 delta_t = m_src - s_src
# #                 delta_t_norm = torch.norm(delta_t)
# #                 omega_t = torch.arctan2(delta_t[1], delta_t[0])
# #                 i_norm += _p_xx_p_minikernel(delta_t_norm, _img_norm_phi_pos, _img_norm_phi_neg, omega_t, bessel_coeff, return_coeff)
# #         i_norms.append(i_norm)
# #     i_norms = torch.stack(i_norms, dim=0)

# #     # 2: Rotate and displace templates.
# #     rotations = _rotation_matrix(gamma_rotations)
# #     realized_templates = _displace_templates(t_wv, displacement_grid, rotations)

# #     # 3: Compute fully-realized-template norms.
# #     # Realized_templates s.b. [Template, displacement, rotation, wave-vector]
# #     assert len(realized_templates.shape) == 5
# #     t_count = realized_templates.shape[0]
# #     d_count = realized_templates.shape[1]
# #     w_count = realized_templates.shape[2]
# #     t_src_cnt = realized_templates.shape[3]

# #     _tp_norm_phi_pos = torch.tensor(2 * template_ctf_angle)
# #     _tp_norm_phi_neg = torch.tensor(0)
# #     t_norms = []
# #     for t in range(t_count):
# #         td_norms_buffer = []
# #         for d in range(d_count):
# #             tdw_norms_buffer = []
# #             for w in range(w_count):
# #                 t_norm = 0
# #                 for i in range(t_src_cnt):
# #                     m_src = realized_templates[t,d,w,i]
# #                     assert len(m_src.shape) == 1
# #                     assert m_src.shape[0] == 2
# #                     for j in range(t_src_cnt):
# #                         s_src = realized_templates[t,d,w,j]
# #                         assert len(s_src.shape) == 1
# #                         assert s_src.shape[0] == 2
# #                         delta_t = m_src - s_src
# #                         delta_t_norm = torch.norm(delta_t)
# #                         omega_t = torch.arctan2(delta_t[1], delta_t[0])
# #                         t_norm += _p_xx_p_minikernel(delta_t_norm, _tp_norm_phi_pos, _tp_norm_phi_neg, omega_t, bessel_coeff, return_coeff)
# #                 tdw_norms_buffer.append(t_norm)
# #             tdw_norms = torch.stack(tdw_norms_buffer, dim=0)
# #             td_norms_buffer.append(tdw_norms)
# #         td_norms = torch.stack(td_norms_buffer, dim=0)
# #         t_norms.append(td_norms)
# #     t_norms = torch.stack(t_norms, dim=0)


# #     # 4: Get individual x-corrs, file appropriately
# #     phi_pos = torch.tensor(image_ctf_angle + template_ctf_angle)
# #     phi_neg = torch.tensor(template_ctf_angle - image_ctf_angle)  # needs to be opposite of wv subtraction

# #     raw_xcorr = []
# #     norm_xcorr = []
# #     for i in range(img_count):
# #         it_buffer = []
# #         it_buffer_raw = []
# #         for t in range(t_count):
# #             itd_buffer = []
# #             itd_buffer_raw = []
# #             for d in range(d_count):
# #                 itdw_buffer = []
# #                 itdw_buffer_raw = []
# #                 for w in range(w_count):
# #                     integral = 0
# #                     for i_src in range(i_src_count):
# #                         m_src = i_wv[i, i_src]
# #                         for t_src in range(t_src_cnt):
# #                             s_src = realized_templates[t, d, w, t_src]
# #                             delta_t = m_src - s_src
# #                             assert len(delta_t.shape) == 1
# #                             assert delta_t.shape[0] == 2
# #                             dt_norm = torch.norm(delta_t)
# #                             omega_t = torch.arctan2(delta_t[1], delta_t[0])
# #                             integral += _p_xx_p_minikernel(dt_norm, phi_pos, phi_neg, omega_t, bessel_coeff, return_coeff)
# #                     itdw_buffer_raw.append(integral)
# #                     normed = integral / torch.sqrt(t_norms[t, d, w] * i_norms[i]).item()
# #                     itdw_buffer.append(normed)
# #                 itd_buffer.append(torch.stack(itdw_buffer, dim=0))
# #                 itd_buffer_raw.append(torch.stack(itdw_buffer_raw, dim=0))
# #             it_buffer.append(torch.stack(itd_buffer, dim=0))
# #             it_buffer_raw.append(torch.stack(itd_buffer_raw, dim=0))
# #         raw_xcorr.append(torch.stack(it_buffer_raw, dim=0))
# #         norm_xcorr.append(torch.stack(it_buffer, dim=0))
# #     raw_xcorr = torch.stack(raw_xcorr, dim=0)
# #     norm_xcorr = torch.stack(norm_xcorr, dim=0)

# #     return norm_xcorr
