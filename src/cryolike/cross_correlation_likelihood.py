from collections.abc import Callable
from itertools import product
from math import lgamma
import numpy as np
from tqdm import trange
from typing import Literal, NamedTuple, NoReturn, overload
import torch

from cryolike.microscopy import get_possible_displacements_grid, translation_kernel_fourier
from cryolike.stacks import Templates
from cryolike.grids import PolarGrid
from cryolike.likelihood import _ensure_identity_kernel
from cryolike.util import (
    absq,
    fourier_bessel_transform,
    get_device,
    CrossCorrelationReturnType,
    Precision,
    to_torch,
    FloatArrayType
)


class WeightedTemplates(NamedTuple):
    sqrtweighted_fourier_templates_mnw: torch.Tensor
    sqrtweighted_fourier_templates_bessel_mdnq: torch.Tensor


class OptimalPoseReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal pose (template, displacement, and rotation) for each image.

    Attributes:
        cross_correlation_S (torch.Tensor): Best cross-correlation for each
            image. Indexed by image number.
        optimal_template_S (torch.Tensor): The template generating the best
            cross-correlation for each image, indexed by image number.
        optimal_displacement_x_S (torch.Tensor): The optimal x-displacement
            for each image, indexed by image number
        optimal_displacement_y_S (torch.Tensor): The optimal y-displacement
            for each image, indexed by image number
        optimal_inplane_rotation_S (torch.Tensor): The optimal inplane
            rotation for each image, indexed by image number
    """
    cross_correlation_S: torch.Tensor
    optimal_template_S: torch.Tensor
    optimal_displacement_x_S: torch.Tensor
    optimal_displacement_y_S: torch.Tensor
    optimal_inplane_rotation_S: torch.Tensor


class OptimizedDisplacementAndRotationReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal displacement and rotation for each image.

    Attributes:
        cross_correlation_SM (torch.Tensor): Cross-correlation for each
            image-template pair. Indexed by [image, template].
        optimal_displacement_x_SM (torch.Tensor): The optimal x-displacement
            for each image-template pair, indexed by [image, template]
        optimal_displacement_y_SM (torch.Tensor): The optimal y-displacement
            for each image-template pair, indexed by [image, template]
        optimal_inplane_rotation_SM (torch.Tensor): The optimal inplane
            rotation for each image-template pair, indexed by [image, template]
    """
    cross_correlation_SM: torch.Tensor
    optimal_displacement_x_SM: torch.Tensor
    optimal_displacement_y_SM: torch.Tensor
    optimal_inplane_rotation_SM: torch.Tensor


class OptimizedDisplacementReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal displacement for each image.

    Attributes:
        cross_correlation_SMw (torch.Tensor): Cross-correlation for each
            image-template pair at each angle. Indexed by [image, template, angle].
        optimal_displacement_x_SMw (torch.Tensor): The optimal x-displacement
            for each image-template pair at each angle, indexed by [image, template, angle]
        optimal_displacement_y_SMw (torch.Tensor): The optimal y-displacement
            for each image-template pair at each angle, indexed by [image, template, angle]
    """
    cross_correlation_SMw: torch.Tensor
    optimal_displacement_x_SMw: torch.Tensor
    optimal_displacement_y_SMw: torch.Tensor


class OptimizedRotationReturn(NamedTuple):
    """Class representing the cross-correlation value
    and optimal rotation angle for each image.

    Attributes:
        cross_correlation_SMd (torch.Tensor): Cross-correlation for each
            image-template pair at each displacement. Indexed by [image, template, displacement].
        optimal_inplane_rotation_SMd(torch.Tensor): The optimal inplane
            rotation for each image-template pair at each displacement,
            indexed by [image, displacement]
    """
    cross_correlation_SMd: torch.Tensor
    optimal_inplane_rotation_SMd: torch.Tensor


class CrossCorrelationReturn(NamedTuple):
    """Class representing the cross-correlation values for each image, template
    pair at each possible displacement and inplane rotation angle.

    Attributes:
        cross_correlation_SMdw (torch.Tensor): Cross-correlation between each
            image-template pair, at each displacement and rotation. Indexed by
            [image, template, displacement, rotation].
    """
    cross_correlation_SMdw: torch.Tensor

integrated_log_likelihood_type = torch.Tensor
cross_correlation_result_type = OptimalPoseReturn | OptimizedDisplacementAndRotationReturn | OptimizedDisplacementReturn | OptimizedRotationReturn | CrossCorrelationReturn

IllKernelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor | None]
"""Signature for integrated log likelihood computation callback.

Args:
    integration_weights_points (torch.Tensor):
    integration_weights_points_sqrt (torch.Tensor):
    CTF_weighted_fourier_templates_smnw (torch.Tensor):
    images_fourier_snw (torch.Tensor):
    Ixx (torch.Tensor):
    Ixy (torch.Tensor):
    Iyy (torch.Tensor):

Returns:
    (torch.Tensor | None): The integrated log likelihood for each (image, template) pair in
        this batch; or None, if ill computation was not actually requested.
"""

def _compute_cross_correlation(
    n_inplanes: int,
    sqrtweighted_premultiplied_CTF_image_fourier_bessel_conj_snq: torch.Tensor,
    sqrtweighted_displaced_fourier_bessel_templates_mdnq: torch.Tensor
) -> torch.Tensor:
    ## Compute cross correlation between sample and template
    ## This is the most computationally expensive part of the code
    cross_correlation_smdq = torch.einsum(
        "snq,mdnq->smdq",
        sqrtweighted_premultiplied_CTF_image_fourier_bessel_conj_snq,
        sqrtweighted_displaced_fourier_bessel_templates_mdnq
    )
    ## Fourier bessel transform, irfft output only the real part
    cross_correlation_smdw = torch.fft.irfft(
        cross_correlation_smdq,
        n=n_inplanes,
        dim=3,
        norm="forward",
    )
    return cross_correlation_smdw


def _cross_images_and_templates(
    device: torch.device,
    n_inplanes: int,
    images_fourier_snw: torch.Tensor,
    sqrtweighted_fourier_templates_mnw: torch.Tensor,
    sqrtweighted_fourier_templates_bessel_mdnq: torch.Tensor,
    ctf_snw: torch.Tensor,       # should be preshaped, of appropriate precision, etc.
    integration_weights_points: torch.Tensor,
    integration_weights_points_sqrt: torch.Tensor,
    compute_ill: IllKernelFn
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute cross correlation and, optionally, integrated log likelihood, over a supplied
    subset of images and templates.

    Args:
        device (torch.device): Device to use for the computation
        n_inplanes (int): Number of in-plane angles
        images_fourier_snw (torch.Tensor): Images in Fourier space, indexed as
            [image, distance, angle]
        fourier_templates_mnw (torch.Tensor): Templates in Fourier space, indexed as
            [template, distance, angle]
        fourier_templates_bessel_mdnq (torch.Tensor): Templates in Fourier space
            after application of Bessel transform, indexed as [template, distance,
            shell number, Bessel coefficient]
        ctf_snw (torch.Tensor): CTF vector in Fourier space, indexed as
            [image, distance, angle]. If the same CTF is used for every image,
            the first dimension will be 1 and the CTF will be broadcast.
        integration_weights_points (torch.Tensor): Weighted integration points
            from the template polar grid.
        integration_weights_points_sqrt (torch.Tensor): Square root of integration
            points.
        compute_ill (IllKernelFn): A function that computes per-patch integrated
            log likelihood, if requested; otherwise, a no-op function.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]: The first member is the cross correlation,
            indexed as [image, template, distance, angle]; the second member is the
            integrated log likelihood, if requested, else None.
    """
    # Sending these to the device should be redundant at this point
    images_fourier_snw = images_fourier_snw.to(device)
    sqrtweighted_fourier_templates_mnw = sqrtweighted_fourier_templates_mnw.to(device)
    sqrtweighted_fourier_templates_bessel_mdnq = sqrtweighted_fourier_templates_bessel_mdnq.to(device)
    ctf_snw = ctf_snw.to(device)
    integration_weights_points = integration_weights_points.to(device)
    integration_weights_points_sqrt = integration_weights_points_sqrt.to(device)

    # Apply CTF to images
    # Normally I favor using the function, but this is essentially a one-liner, & I think inlining may
    # let torch chain the operation better.
    sqrtweighted_image_fourier_bessel_conj_snq: torch.Tensor = torch.fft.fft(
        images_fourier_snw * ctf_snw * integration_weights_points_sqrt[None, :, :],
        dim = 2, norm = "ortho"
    ).conj()

    # Apply CTF to templates
    CTF_sqrtweighted_fourier_templates_smnw = (sqrtweighted_fourier_templates_mnw * ctf_snw.unsqueeze(1))

    Ixx_sm = torch.sum(absq(CTF_sqrtweighted_fourier_templates_smnw), dim = (2, 3))
    Iyy_s = torch.sum(absq(images_fourier_snw) * integration_weights_points, dim = (1,2))
    Ixy_smdw = _compute_cross_correlation(
        n_inplanes,
        sqrtweighted_image_fourier_bessel_conj_snq,
        sqrtweighted_fourier_templates_bessel_mdnq
    )

    Ixx = Ixx_sm.unsqueeze(2).unsqueeze(3)
    Iyy = Iyy_s.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    Ixy = Ixy_smdw
    cross_correlation_smdw = Ixy / torch.sqrt(Ixx * Iyy)

    # This fn should be a no-op returning None if ILL computation was not requested
    log_likelihood_smdw = compute_ill(
        integration_weights_points,
        integration_weights_points_sqrt,
        CTF_sqrtweighted_fourier_templates_smnw,
        images_fourier_snw,
        Ixx,
        Ixy,
        Iyy
    )
    return (cross_correlation_smdw, log_likelihood_smdw)


# TODO: Move this functionality into the CTF object
def conform_ctf(ctf: torch.Tensor, anisotropic: bool) -> torch.Tensor:
    """Ensure the CTF has appropriate dimensions to be applied to image/template sets.

    Args:
        ctf (torch.Tensor): CTF function, as tensor.
        anisotropic (bool): Whether the described CTF is anisotropic.

    Returns:
        torch.Tensor: The described CTF function, expanded in the last dimension if anisotropic.
    """
    if not anisotropic:
        return ctf.unsqueeze(-1)
    return ctf


# TODO: Be more memory-aware
def _get_sqrtweighted_templates(templates_fourier: torch.Tensor, sqrtweighted_translation_kernel: torch.Tensor, device: torch.device) -> WeightedTemplates:
    """Applies known weighted translation kernel to the full template set before computing cross-correlation.
    Doing this to the whole template set at once, rather than batching, keeps the translation kernel in
    memory, improving performance.

    Args:
        templates_fourier (torch.Tensor): Full set of templates, in Fourier space.
        sqrtweighted_translation_kernel (torch.Tensor): Translation kernel to apply
        device (torch.device): The device to use for the operation.

    Returns:
        WeightedTemplates: A tuple of the resulting templates, in Fourier space, with and without application
            of the Bessel transform.
    """
    templates_fourier = templates_fourier.to(device)
    sqrtweighted_translation_kernel = sqrtweighted_translation_kernel.to(device)

    sqrtweighted_displaced_fourier_templates_mdnw = templates_fourier.unsqueeze(1) * sqrtweighted_translation_kernel
    sqrtweighted_displaced_fourier_templates_bessel_mdnq = fourier_bessel_transform(sqrtweighted_displaced_fourier_templates_mdnw, 3, "ortho")
    sqrtweighted_displaced_fourier_templates_mnw = sqrtweighted_displaced_fourier_templates_mdnw[:,0,:,:]

    return WeightedTemplates(
        sqrtweighted_fourier_templates_mnw=sqrtweighted_displaced_fourier_templates_mnw,
        sqrtweighted_fourier_templates_bessel_mdnq=sqrtweighted_displaced_fourier_templates_bessel_mdnq,
    )


def _ill_kernel_factory(
    Iss: float,
    n_pixels_phys: int,
    s_points: torch.Tensor
):
    def _kernel(
        integration_weights_points: torch.Tensor,
        integration_weights_points_sqrt: torch.Tensor,
        CTF_sqrtweighted_fourier_templates_smnw: torch.Tensor,
        images_fourier_snw: torch.Tensor,
        Ixx: torch.Tensor,
        Ixy: torch.Tensor,
        Iyy: torch.Tensor
    ):
        weighted_s_points = s_points * integration_weights_points
        sqrt_weighted_s_points = s_points * integration_weights_points_sqrt
        Isx_sm = torch.sum(sqrt_weighted_s_points * CTF_sqrtweighted_fourier_templates_smnw, dim = (2,3))
        Isy_s = torch.sum(weighted_s_points * images_fourier_snw, dim=(1,2))

        Isx = Isx_sm.unsqueeze(2).unsqueeze(3)
        Isy = Isy_s.unsqueeze(1).unsqueeze(2).unsqueeze(3)

        A = - absq(Isx) + Ixx * Iss
        B = - Isx.real * Isy.real - Isx.imag * Isy.imag + Ixy * Iss
        C =   absq(Isy) - Iyy * Iss
        D = - (B ** 2 / A + C)
        p = n_pixels_phys / 2.0 - 2.0
        constant = (3.0 - n_pixels_phys) / 2.0 * np.log(2 * np.pi) - np.log(2) - 0.5 * np.log(Iss) + lgamma(n_pixels_phys / 2.0 - 2.0) + p * np.log(2 * Iss)
        log_likelihood_smdw = -p * torch.log(D) - 0.5 * torch.log(A) + constant
        return log_likelihood_smdw
    return _kernel


def _make_batches(n_imgs_per_batch, n_imgs, n_templates_per_batch, n_templates):
    img_batch_ranges = _safe_rangify_batch(n_imgs_per_batch, n_imgs)
    template_batch_ranges =  _safe_rangify_batch(n_templates_per_batch, n_templates)
    return product(template_batch_ranges, img_batch_ranges)


# TODO: Check inclusivity!
def _safe_rangify_batch(n_per_batch: int, n_max: int) -> np.ndarray:
    """For batches of m elements each with total list length l, return n ndarrays
    corresponding to the start and (exclusive) end index of every batch, without
    exceeding the total length. n is computed as ceil(n_max / n_per_batch)

    E.g. for 15 items in batches of 6, return array of [0, 5], [6, 11], [12, 15].

    Args:
        n_per_batch (int): Max item count per batch.
        n_max (int): Maximum length of the list.

    Returns:
        np.ndarray: Array of n arrays with start and (exclusive) end points of each range.
    """
    n_batches = int(np.ceil(n_max / n_per_batch))
    return np.array([
        (a, min(a + n_per_batch, n_max))
        for a in np.array(range(n_batches)) * n_per_batch
    ], dtype=np.int64)


## Utility functions for init
def _get_integration_weights_points(templates: Templates, precision: Precision):
    iwp = to_torch(templates.polar_grid.weight_points, precision, "cpu")
    iwp = iwp.reshape(templates.polar_grid.n_shells, templates.polar_grid.n_inplanes)
    return iwp * (2.0 * np.pi) ** 2


def _get_log_weights_viewing(templates: Templates, precision: Precision, device: torch.device):
    weights_viewing = templates.viewing_angles.weights
    if weights_viewing is None:
        weights_viewing = torch.ones(templates.viewing_angles.n_angles) / templates.viewing_angles.n_angles
    weights_viewing = to_torch(weights_viewing, precision, device)
    weights_viewing /= weights_viewing.sum()
    return torch.log(weights_viewing).unsqueeze(0)


class _Displacements(NamedTuple):
    n_displacements: int
    x_displacements: FloatArrayType
    y_displacements: FloatArrayType
    x_disp_expt_scale: FloatArrayType
    y_disp_expt_scale: FloatArrayType


def _get_displacements(
    max_displacement: float,
    n_disp_x: int,
    n_disp_y: int,
    box_edge_size: float
):
    if (n_disp_x < 1):
        n_disp_x = n_disp_y
    if (n_disp_y < 1):
        n_disp_y = n_disp_x
    if n_disp_x < 1 and n_disp_y < 1:
        raise ValueError("Number of displacements must be set and non-negative.")      

    _max_disp = max_displacement * 2.0 / box_edge_size
    n_disp, _x_disp, _y_disp = get_possible_displacements_grid(_max_disp, n_disp_x, n_disp_y)
    print(f"n_displacements: {n_disp}")
    _x_disp_expt = _x_disp / 2.0 * box_edge_size
    _y_disp_expt = _y_disp / 2.0 * box_edge_size
    return _Displacements(n_disp, _x_disp, _y_disp, _x_disp_expt, _y_disp_expt)


def _get_s_points(polar_grid: PolarGrid, precision: Precision):
    if polar_grid.x_points is None or polar_grid.y_points is None:
        (x_pts, y_pts) = polar_grid.get_cartesian_points()
    else:
        (x_pts, y_pts) = (polar_grid.x_points, polar_grid.y_points)
    x_points = to_torch(x_pts, precision, "cpu")
    y_points = to_torch(y_pts, precision, "cpu")
    s_points = torch.sinc(2.0 * x_points) * torch.sinc(2.0 * y_points) * 4.0
    return s_points.reshape(polar_grid.n_shells, polar_grid.n_inplanes)


def _get_translation_kernel(
    sqrt_weights: torch.Tensor,
    disp: _Displacements,
    polar_grid: PolarGrid,
    precision: Precision,
    device: torch.device
):
    _integration_weights_points_sqrt = sqrt_weights.to(device)

    _x_displacements = to_torch(disp.x_displacements, precision, device)
    _y_displacements = to_torch(disp.y_displacements, precision, device)
    translation_kernel = translation_kernel_fourier(polar_grid, _x_displacements, _y_displacements, precision, device)
    sqrtweighted_translation_kernel = (translation_kernel * _integration_weights_points_sqrt[None, :, :]).unsqueeze(0)
    return sqrtweighted_translation_kernel


class CrossCorrelationLikelihood:
    """Class storing a set of templates for cross-correlation, with needed values precomputed.

    Note the general naming scheme for likelihood tensors indicates the values which can be
    used to index into them. Tensors named with _smdw are indexed left to right by image,
    template, displacement, and rotational angle; _SM indicates a tensor indexed only by
    image and template. Capital SM indicate that the tensor should contain information about
    the entire image/template stack, while lowercase sm indicates that the numbering corresponds
    to only the current subset/batch.

    Attributes:
        torch_float_type (torch.dtype): dtype representing the precision for computation
        torch_complex_type (torch.dtype): dtype representing the precision to use for complex-number
            computation.
        templates_fourier (torch.Tensor): Set of template images in Fourier space.
        integration_weights_points (torch.Tensor): Weighted integration points from the
            templates' underlying polar grid.
        integration_weights_points_sqrt (torch.Tensor): Square root of the weighted
            integration points from the templates' underlying polar grid.
        log_weights_viewing (torch.Tensor): log of the normalized weights for the templates'
            viewing angles. Used in computing integrated log likelihood.
        n_displacements (int): number of displacements considered in checking optimal displacement.
            Should be the length of the x_ and y_displacement vectors.
        x_displacements_expt_scale (torch.Tensor): Vector of x_displacements to consider in matching
        y_displacements_expt_scale (torch.Tensor): Vector of y_displacements to consider in matching
        s_points (torch.Tensor): Matrix of x- and y-points from the template grid, after
            application of sinc function. Used in computing integrated log likelihood.
        Iss (float): Integral of s_points, used in computing integrated log likelihood
        _log_likelihood_SM (torch.Tensor): Tensor allocated as scratch space if integrated
            log likelihood computation was requested.
        _gamma (torch.Tensor): List of angles to consider in finding optimal rotation
        _result_collection (cross_correlation_result_type): Scratch space for collecting
            the requested cross-correlation likelihoods
    """
    torch_float_type: torch.dtype
    torch_complex_type: torch.dtype
    templates_fourier: torch.Tensor
    integration_weights_points: torch.Tensor
    integration_weights_points_sqrt: torch.Tensor
    log_weights_viewing: torch.Tensor
    n_displacements: int
    x_displacements_expt_scale: torch.Tensor
    y_displacements_expt_scale: torch.Tensor
    s_points: torch.Tensor
    Iss: float
    _log_likelihood_SM: torch.Tensor
    _gamma: torch.Tensor
    _result_collection: cross_correlation_result_type


    def __init__(
        self,
        templates: Templates,
        max_displacement: float = 0,
        n_displacements_x: int = -1,
        n_displacements_y: int = -1,
        identity_kernel: Callable[[PolarGrid, Precision], torch.Tensor] = _get_s_points,
        precision: Precision = Precision.DEFAULT,
        device: str | torch.device = "cpu",
        verbose: bool = False
    ) -> None:
        """Initializer for CrossCorrelationLikelihood

        Args:
            templates (Templates): Stack of Templates to use for cross-correlation
            max_displacement (float, optional): Largest displacement to consider. Defaults to 0.
            n_displacements_x (int, optional): Number of displacements in the x-direction. At least one of
                n_displacements_x and n_displacements_y must be set. If only one is set, the other
                dimension will be assumed to be the same.
            n_displacements_y (int, optional): Number of displacements in the y-direction.  At least one of
                n_displacements_x and n_displacements_y must be set. If only one is set, the other
                dimension will be assumed to be the same.
            flip_template_sign (bool, optional): If True, the Template images' sign
                will be flipped. Defaults to False.
            precision (Precision, optional): Precision (single or double) at which to do
                comnputations. Single will result in Float32/Complex64 computations,
                while Double will result in Float64/Complex128 computations.
                Defaults to Precision.DEFAULT, which will use the native precision of the
                input Templates object for initialization.
            device (str | torch.device, optional): Device to use for pre-computation of
                values in setting up the cross correlation. Does not affect subsequent
                computations, which have their own device parameter. Defaults to "cpu".
            verbose (bool, optional): If set, will be chattier about initialization. Defaults to False.
        """
        _device = get_device(device, verbose)
        (self.torch_float_type, self.torch_complex_type, _) = precision.get_dtypes(default=Precision.DOUBLE)
        CrossCorrelationLikelihood._validate_templates(templates)
        box_edge_size: float = templates.box_size[0]

        self.n_inplanes = templates.polar_grid.n_inplanes
        self.n_templates = templates.images_fourier.shape[0]
        self.templates_fourier = to_torch(templates.images_fourier, precision, "cpu")

        disp = _get_displacements(max_displacement, n_displacements_x, n_displacements_y, box_edge_size)
        self.n_displacements = disp.n_displacements
        self.x_displacements_expt_scale = to_torch(disp.x_disp_expt_scale, precision, _device)
        self.y_displacements_expt_scale = to_torch(disp.y_disp_expt_scale, precision, _device)

        self.integration_weights_points = _get_integration_weights_points(templates, precision).to(_device)
        self.integration_weights_points_sqrt = torch.sqrt(self.integration_weights_points)

        self.s_points = _ensure_identity_kernel(templates.polar_grid, identity_kernel, precision, _device)
        self.Iss = torch.sum(self.s_points.abs() ** 2 * self.integration_weights_points).item()
        self._gamma = torch.linspace(0, 2*np.pi, self.n_inplanes+1, dtype=self.torch_float_type, device=_device)[:-1]
        self.log_weights_viewing = _get_log_weights_viewing(templates, precision, _device)

        self.sqrtweighted_translation_kernel = _get_translation_kernel(
            self.integration_weights_points_sqrt,
            disp,
            templates.polar_grid,
            precision,
            _device
        )

        ## probably not enough memory to do this
        # self.templates_fourier = _get_weighted_templates(_templates_fourier, sqrtweighted_translation_kernel, _device)


    @staticmethod
    def _validate_templates(templates: Templates):
        if not templates.polar_grid.uniform:
            raise NotImplementedError("Non-uniform polar grid is not yet supported.")
        if not np.isclose(templates.box_size[0], templates.box_size[1], rtol=1e-6):
            raise NotImplementedError("Box size must be same in both dimensions")


    def _initialize_collector(self, n_images: int, return_type: CrossCorrelationReturnType):
        collect_batch = self._no_op_collector
        if return_type == CrossCorrelationReturnType.FULL_TENSOR:
            self._initialize_full_cross_correlation(n_images)
            collect_batch = self._collect_full_cross_correlation
        elif return_type in [CrossCorrelationReturnType.OPTIMAL_POSE, CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION]:
            self._initialize_optimal_pose(n_images)
            collect_batch = self._collect_optimal_pose
        elif return_type == CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT:
            self._initialize_optimized_displacement(n_images)
            collect_batch = self._collect_optimized_displacement
        elif return_type == CrossCorrelationReturnType.OPTIMAL_ROTATION:
            self._initialize_optimized_rotation(n_images)
            collect_batch = self._collect_optimized_rotation
        else:
            if return_type != CrossCorrelationReturnType.NONE:
                raise ValueError("Unreachable: Unknown cross-correlation return type requested.")

        return collect_batch


    # NOTE: We don't *really* need to do this. We *could* just rely on the caller to figure it out.
    # However, this will provide stronger typing, without assertions, to all callers. At the expense of
    # a bunch of extra code that doesn't do much.
    @overload
    def compute_optimal_pose(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[False]
    ) -> OptimalPoseReturn:
        ... # pragma: no cover
    @overload
    def compute_optimal_pose(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimalPoseReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    def compute_optimal_pose(self,
        device: torch.device,
        images_fourier: torch.Tensor,
        ctf: torch.Tensor,
        n_pixels_phys: int,
        n_images_per_batch: int,
        n_templates_per_batch: int,
        return_integrated_likelihood: bool
    ):
        """Compute cross-correlation between templates and images, returning optimal pose
        and (optionally) the integrated log likelihood for each template-image pair.

        Args:
            device (torch.device): Device to use for computation
            images_fourier (torch.Tensor): Fourier-space images
            ctf (torch.Tensor): Contrast transfer function, either singleton (applied to
                every image) or a vector (one per image).
            n_pixels_phys (int): Number of pixels in the physical image, i.e. the
                physical image source's Cartesian grid
            n_images_per_batch (int): Number of images to use per batch
            n_templates_per_batch (int): Number of templates to use per batch
            return_integrated_likelihood (bool): Whether to include integrated log likelihood
                in the return

        Returns:
            OptimalPoseReturn | tuple(OptimalPoseReturn, integrated_log_likelihood_type): The
                optimal pose (if no ILL requested) or a tuple of (optimal pose, ILL). ILL is
                represented as a matrix of scores indexed as [image, template].
        """
        if return_integrated_likelihood:
            return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_POSE, return_integrated_likelihood=True)
        return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_POSE, return_integrated_likelihood=False)


    @overload
    def compute_optimized_displacement_and_rotation(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[False]
    ) -> OptimizedDisplacementAndRotationReturn:
        ... # pragma: no cover
    @overload
    def compute_optimized_displacement_and_rotation(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimizedDisplacementAndRotationReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    def compute_optimized_displacement_and_rotation(self,
        device: torch.device,
        images_fourier: torch.Tensor,
        ctf: torch.Tensor,
        n_pixels_phys: int,
        n_images_per_batch: int,
        n_templates_per_batch: int,
        return_integrated_likelihood: bool
    ):
        """Compute cross-correlation between templates and images, returning optimal
        displacement and rotation, and (optionally) the integrated log likelihood for
        each template-image pair.

        Args:
            device (torch.device): Device to use for computation
            images_fourier (torch.Tensor): Fourier-space images
            ctf (torch.Tensor): Contrast transfer function, either singleton (applied to
                every image) or a vector (one per image).
            n_pixels_phys (int): Number of pixels in the physical image, i.e. the
                physical image source's Cartesian grid
            n_images_per_batch (int): Number of images to use per batch
            n_templates_per_batch (int): Number of templates to use per batch
            return_integrated_likelihood (bool): Whether to include integrated log likelihood
                in the return

        Returns:
            OptimizedDisplacementAndRotationReturn | tuple(OptimizedDisplacementAndRotationReturn, integrated_log_likelihood_type): The
                optimal displacement and rotation for each image (if no ILL requested) or a tuple of (same, ILL). ILL is
                represented as a matrix of scores indexed as [image, template].
        """
        if return_integrated_likelihood:
            return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION, return_integrated_likelihood=True)
        return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION, return_integrated_likelihood=False)


    @overload
    def compute_optimized_displacement(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[False]
    ) -> OptimizedDisplacementReturn:
        ... # pragma: no cover
    @overload
    def compute_optimized_displacement(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimizedDisplacementReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    def compute_optimized_displacement(self,
        device: torch.device,
        images_fourier: torch.Tensor,
        ctf: torch.Tensor,
        n_pixels_phys: int,
        n_images_per_batch: int,
        n_templates_per_batch: int,
        return_integrated_likelihood: bool
    ):
        """Compute cross-correlation between templates and images, returning optimal
        displacement, and (optionally) the integrated log likelihood for
        each template-image pair.

        Args:
            device (torch.device): Device to use for computation
            images_fourier (torch.Tensor): Fourier-space images
            ctf (torch.Tensor): Contrast transfer function, either singleton (applied to
                every image) or a vector (one per image).
            n_pixels_phys (int): Number of pixels in the physical image, i.e. the
                physical image source's Cartesian grid
            n_images_per_batch (int): Number of images to use per batch
            n_templates_per_batch (int): Number of templates to use per batch
            return_integrated_likelihood (bool): Whether to include integrated log likelihood
                in the return

        Returns:
            OptimizedDisplacementReturn | tuple(OptimizedDisplacementReturn, integrated_log_likelihood_type): The
                optimal displacement for each image (if no ILL requested) or a tuple of (same, ILL). ILL is
                represented as a matrix of scores indexed as [image, template].
        """
        if return_integrated_likelihood:
            return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT, return_integrated_likelihood=True)
        return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT, return_integrated_likelihood=False)


    @overload
    def compute_optimized_rotation(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[False]
    ) -> OptimizedRotationReturn:
        ... # pragma: no cover
    @overload
    def compute_optimized_rotation(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimizedRotationReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    def compute_optimized_rotation(self,
        device: torch.device,
        images_fourier: torch.Tensor,
        ctf: torch.Tensor,
        n_pixels_phys: int,
        n_images_per_batch: int,
        n_templates_per_batch: int,
        return_integrated_likelihood: bool
    ):
        """Compute cross-correlation between templates and images, returning optimal
        rotation, and (optionally) the integrated log likelihood for
        each template-image pair.

        Args:
            device (torch.device): Device to use for computation
            images_fourier (torch.Tensor): Fourier-space images
            ctf (torch.Tensor): Contrast transfer function, either singleton (applied to
                every image) or a vector (one per image).
            n_pixels_phys (int): Number of pixels in the physical image, i.e. the
                physical image source's Cartesian grid
            n_images_per_batch (int): Number of images to use per batch
            n_templates_per_batch (int): Number of templates to use per batch
            return_integrated_likelihood (bool): Whether to include integrated log likelihood
                in the return

        Returns:
            OptimizedRotationReturn | tuple(OptimizedRotationReturn, integrated_log_likelihood_type): The
                optimal rotation for each image (if no ILL requested) or a tuple of (same, ILL). ILL is
                represented as a matrix of scores indexed as [image, template].
        """    
        if return_integrated_likelihood:
            return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_ROTATION, return_integrated_likelihood=True)
        return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.OPTIMAL_ROTATION, return_integrated_likelihood=False)


    @overload
    def compute_cross_correlation_complete(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[False]
    ) -> CrossCorrelationReturn:
        ... # pragma: no cover
    @overload
    def compute_cross_correlation_complete(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_integrated_likelihood: Literal[True]
    ) -> tuple[CrossCorrelationReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    def compute_cross_correlation_complete(self,
        device: torch.device,
        images_fourier: torch.Tensor,
        ctf: torch.Tensor,
        n_pixels_phys: int,
        n_images_per_batch: int,
        n_templates_per_batch: int,
        return_integrated_likelihood: bool
    ):
        """Compute cross-correlation between templates and images, returning a
        tensor identifying the cross-correlation for each displacement and rotation,
        and (optionally) the integrated log likelihood, for each template-image pair.

        Args:
            device (torch.device): Device to use for computation
            images_fourier (torch.Tensor): Fourier-space images
            ctf (torch.Tensor): Contrast transfer function, either singleton (applied to
                every image) or a vector (one per image).
            n_pixels_phys (int): Number of pixels in the physical image, i.e. the
                physical image source's Cartesian grid
            n_images_per_batch (int): Number of images to use per batch
            n_templates_per_batch (int): Number of templates to use per batch
            return_integrated_likelihood (bool): Whether to include integrated log likelihood
                in the return

        Returns:
            CrossCorrelationReturn | tuple(CrossCorrelationReturn, integrated_log_likelihood_type): The
                full cross correlation matrix for each image (if no ILL requested) or a tuple of (same, ILL). ILL is
                represented as a matrix of scores indexed as [image, template].
        """    
        if return_integrated_likelihood:
            return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.FULL_TENSOR, return_integrated_likelihood=True)
        return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.FULL_TENSOR, return_integrated_likelihood=False)


    def compute_integrated_log_likelihood(self,
        device: torch.device,
        images_fourier: torch.Tensor,
        ctf: torch.Tensor,
        n_pixels_phys: int,
        n_images_per_batch: int,
        n_templates_per_batch: int,
    ) -> integrated_log_likelihood_type:
        return self._compute_cross_correlation_likelihood(device, images_fourier, ctf, n_pixels_phys, n_images_per_batch, n_templates_per_batch, return_type=CrossCorrelationReturnType.NONE, return_integrated_likelihood=True)


    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.NONE],
        return_integrated_likelihood: Literal[True]
    ) -> integrated_log_likelihood_type:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.NONE],
        return_integrated_likelihood: Literal[False]
    ) -> NoReturn:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.FULL_TENSOR],
        return_integrated_likelihood: Literal[True]
    ) -> tuple[CrossCorrelationReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.FULL_TENSOR],
        return_integrated_likelihood: Literal[False]
    ) -> CrossCorrelationReturn:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT],
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimizedDisplacementReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT],
        return_integrated_likelihood: Literal[False]
    ) -> OptimizedDisplacementReturn:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION],
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimizedDisplacementAndRotationReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_DISPLACEMENT_AND_ROTATION],
        return_integrated_likelihood: Literal[False]
    ) -> OptimizedDisplacementAndRotationReturn:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_POSE],
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimalPoseReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_POSE],
        return_integrated_likelihood: Literal[False]
    ) -> OptimalPoseReturn:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_ROTATION],
        return_integrated_likelihood: Literal[True]
    ) -> tuple[OptimizedRotationReturn, integrated_log_likelihood_type]:
        ... # pragma: no cover
    @overload
    def _compute_cross_correlation_likelihood(self, device: torch.device, images_fourier: torch.Tensor, ctf: torch.Tensor, n_pixels_phys: int, n_images_per_batch: int, n_templates_per_batch: int,
        return_type: Literal[CrossCorrelationReturnType.OPTIMAL_ROTATION],
        return_integrated_likelihood: Literal[False]
    ) -> OptimizedRotationReturn:
        ... # pragma: no cover
    def _compute_cross_correlation_likelihood(self,
        device: torch.device,
        images_fourier: torch.Tensor, # TODO: accommodate a data loader
        ctf: torch.Tensor,  # Whatever unsqueeze(-1)ing is needed for not-anisotropic should have been done already
        n_pixels_phys: int,     # Caller to compute as Images.phys_grid.n_pixels[0] * Images.phys_grid.n_pixels[1]
        n_images_per_batch: int,
        n_templates_per_batch: int,
        return_type: CrossCorrelationReturnType,
        return_integrated_likelihood: bool,
        __ill_kernel_factory: Callable = _ill_kernel_factory,
        log_likelihood_keep_displacement_and_rotation: bool = False,
        # TODO: Verbose?
    ) -> integrated_log_likelihood_type | cross_correlation_result_type | tuple[cross_correlation_result_type, integrated_log_likelihood_type]:
        # Assume that the device is available.
        # Note that an improved version would allow multiple devices, or launching parallel jobs, possibly on
        # multiple machines, here.

        if return_type == CrossCorrelationReturnType.NONE and not return_integrated_likelihood:
            raise ValueError("Current settings do not request anything to be returned.")

        ctf_is_singleton = ctf.shape[0] == 1
        # (fourier_templates_mnw, fourier_templates_bessel_mdnq) = self.templates_fourier
        n_images = images_fourier.shape[0]

        if return_integrated_likelihood:
            if log_likelihood_keep_displacement_and_rotation:
                self._initialize_log_likelihood_keep_displacement_and_rotation(n_images)
            else:
                self._initialize_log_likelihood(n_images, device)
            s_points_cuda = self.s_points.to(device)
            ill_kernel = __ill_kernel_factory(self.Iss, n_pixels_phys, s_points_cuda)
        else:
            ill_kernel = lambda _1, _2, _3, _4, _5, _6, _7: None

        collect_batch = self._initialize_collector(n_images, return_type)
        # batches = _make_batches(n_images_per_batch, n_images, n_templates_per_batch, self.n_templates)

        # for (t_batch_rng, i_batch_rng) in tqdm(batches):
            
        #     assert isinstance(t_batch_rng, np.ndarray)
        #     assert isinstance(i_batch_rng, np.ndarray)

        #     (i_start, i_end) = (i_batch_rng[0], i_batch_rng[1])
        #     (t_start, t_end) = (t_batch_rng[0], t_batch_rng[1])

        for t_start in trange(0, self.n_templates, n_templates_per_batch):
            
            t_end = min(t_start + n_templates_per_batch, self.n_templates)
            
            sqrtweighted_templates = _get_sqrtweighted_templates(self.templates_fourier[t_start:t_end, :, :], self.sqrtweighted_translation_kernel, device = device)
            f_sqrtweighted_templates_mnw = sqrtweighted_templates.sqrtweighted_fourier_templates_mnw
            f_sqrtweighted_templates_bessel_mdnq = sqrtweighted_templates.sqrtweighted_fourier_templates_bessel_mdnq
            
            for i_start in range(0, n_images, n_images_per_batch):
                
                i_end = min(i_start + n_images_per_batch, n_images)
            
                f_imgs = images_fourier[i_start:i_end, :, :].to(device)
                _ctf = ctf if ctf_is_singleton else ctf[i_start:i_end, :, :].to(device)

                (cross_correlation_smdw, log_likelihood_smdw) = _cross_images_and_templates(
                    device,
                    self.n_inplanes,
                    f_imgs,
                    f_sqrtweighted_templates_mnw,
                    f_sqrtweighted_templates_bessel_mdnq,
                    _ctf,
                    self.integration_weights_points.to(device),
                    self.integration_weights_points_sqrt.to(device),
                    ill_kernel
                )
                if torch.isnan(cross_correlation_smdw).any():
                    raise ValueError("NaN detected in cross_correlation_smdw")
                collect_batch(i_start, i_end, t_start, t_end, cross_correlation_smdw)
                if (log_likelihood_smdw is not None):
                    if log_likelihood_keep_displacement_and_rotation:
                        self._collect_log_likelihood_keep_displacement_and_rotation(i_start, i_end, t_start, t_end, log_likelihood_smdw)
                    else:
                        self._collect_log_likelihood(i_start, i_end, t_start, t_end, log_likelihood_smdw)

        if return_type == CrossCorrelationReturnType.OPTIMAL_POSE:
            self._finalize_optimal_pose(n_images)
        if return_integrated_likelihood:
            if log_likelihood_keep_displacement_and_rotation:
                # self._log_likelihood_SM = torch.logsumexp(self._log_likelihood_SMDW, dim=(2, 3))
                integrated_log_likelihood = self._log_likelihood_SMDW
            else:
                integrated_log_likelihood = self._finalize_log_likelihood()
            if return_type == CrossCorrelationReturnType.NONE:
                return integrated_log_likelihood
            return (self._result_collection, integrated_log_likelihood)
        return self._result_collection


    def _initialize_log_likelihood(self, n_images, device):
        self._log_likelihood_SM = torch.zeros((n_images, self.n_templates), dtype=self.torch_float_type, device=device)#"cpu")
    

    def _initialize_log_likelihood_keep_displacement_and_rotation(self, n_images: int):
        self._log_likelihood_SMDW = torch.zeros((n_images, self.n_templates, self.n_displacements, self.n_inplanes), dtype=self.torch_float_type, device="cpu")


    def _collect_log_likelihood(self, s_start: int, s_end: int, m_start: int, m_end: int, log_likelihood_smdw: torch.Tensor):
        self._log_likelihood_SM[s_start:s_end, m_start:m_end] = torch.logsumexp(log_likelihood_smdw, dim=(2,3))


    def _collect_log_likelihood_keep_displacement_and_rotation(self, s_start: int, s_end: int, m_start: int, m_end: int, log_likelihood_smdw: torch.Tensor):
        self._log_likelihood_SMDW[s_start:s_end, m_start:m_end, :, :] = log_likelihood_smdw#.cpu()
        

    def _finalize_log_likelihood(self) -> torch.Tensor:
        log_likelihood_S = torch.logsumexp(self._log_likelihood_SM + self.log_weights_viewing, dim=1) - np.log(self.n_displacements) - np.log(self.n_inplanes)
        return log_likelihood_S.cpu()


    def _initialize_optimal_pose(self, n_images: int):
        cross_correlation_SM = torch.zeros((n_images, self.n_templates), dtype=self.torch_float_type, device="cpu")
        optimal_displacement_x_SM = torch.zeros((n_images, self.n_templates), dtype=self.torch_float_type, device="cpu")
        optimal_displacement_y_SM = torch.zeros((n_images, self.n_templates), dtype=self.torch_float_type, device="cpu")
        optimal_inplane_rotation_SM = torch.zeros((n_images, self.n_templates), dtype=self.torch_float_type, device="cpu")
        self._result_collection = OptimizedDisplacementAndRotationReturn(
            cross_correlation_SM,
            optimal_displacement_x_SM,
            optimal_displacement_y_SM,
            optimal_inplane_rotation_SM
        )


    def _collect_optimal_pose(self, s_start: int, s_end: int, m_start: int, m_end: int, cross_correlation_smdw: torch.Tensor):
        assert isinstance(self._result_collection, OptimizedDisplacementAndRotationReturn)
        shape_dw = cross_correlation_smdw.shape[2:]
        cross_correlation_smdw_cuda_optdw = cross_correlation_smdw.flatten(start_dim=2, end_dim=3)
        values, indices_ravel = torch.max(cross_correlation_smdw_cuda_optdw, dim=2)
        self._result_collection.cross_correlation_SM[s_start:s_end, m_start:m_end] = values
        indices = torch.unravel_index(indices_ravel, shape_dw)
        nd = indices[0]
        nw = indices[1]
        self._result_collection.optimal_displacement_x_SM[s_start:s_end, m_start:m_end] = self.x_displacements_expt_scale[nd]
        self._result_collection.optimal_displacement_y_SM[s_start:s_end, m_start:m_end] = self.y_displacements_expt_scale[nd]
        self._result_collection.optimal_inplane_rotation_SM[s_start:s_end, m_start:m_end] = self._gamma[nw]


    def _finalize_optimal_pose(self, n_images: int):
        assert isinstance(self._result_collection, OptimizedDisplacementAndRotationReturn)
        imgs_range = torch.arange(n_images).cpu()
        cross_correlation_S, optimal_template_S = torch.max(self._result_collection.cross_correlation_SM, dim = 1)
        optimal_displacement_x_S = self._result_collection.optimal_displacement_x_SM[imgs_range, optimal_template_S].cpu()
        optimal_displacement_y_S = self._result_collection.optimal_displacement_y_SM[imgs_range, optimal_template_S].cpu()
        optimal_inplane_rotation_S = self._result_collection.optimal_inplane_rotation_SM[imgs_range, optimal_template_S].cpu()
        self._result_collection = OptimalPoseReturn(
            cross_correlation_S.cpu(), optimal_template_S.cpu(), 
            optimal_displacement_x_S, optimal_displacement_y_S, optimal_inplane_rotation_S
        )


    def _initialize_optimized_displacement(self, n_images: int):
        cross_correlation_SMw = torch.zeros((n_images, self.n_templates, self.n_inplanes), dtype=self.torch_float_type, device="cpu")
        optimal_displacement_x_SMw = torch.zeros((n_images, self.n_templates, self.n_inplanes), dtype=self.torch_float_type, device="cpu")
        optimal_displacement_y_SMw = torch.zeros((n_images, self.n_templates, self.n_inplanes), dtype=self.torch_float_type, device="cpu")
        self._result_collection = OptimizedDisplacementReturn(
            cross_correlation_SMw,
            optimal_displacement_x_SMw,
            optimal_displacement_y_SMw,
        )


    def _collect_optimized_displacement(self, s_start: int, s_end: int, m_start: int, m_end: int, cross_correlation_smdw: torch.Tensor):
        assert isinstance(self._result_collection, OptimizedDisplacementReturn)
        values, nd = torch.max(cross_correlation_smdw, dim=2)
        nd = nd.cpu()
        self._result_collection.cross_correlation_SMw[s_start:s_end, m_start:m_end] = values.cpu()
        self._result_collection.optimal_displacement_x_SMw[s_start:s_end, m_start:m_end] = self.x_displacements_expt_scale[nd].cpu()
        self._result_collection.optimal_displacement_y_SMw[s_start:s_end, m_start:m_end] = self.y_displacements_expt_scale[nd].cpu()


    def _initialize_optimized_rotation(self, n_images: int):
        cross_correlation_SMd = torch.zeros((n_images, self.n_templates, self.n_displacements), dtype=self.torch_float_type, device="cpu")
        optimal_inplane_rotation_SMd = torch.zeros((n_images, self.n_templates, self.n_displacements), dtype=self.torch_float_type, device="cpu")
        self._result_collection = OptimizedRotationReturn(cross_correlation_SMd, optimal_inplane_rotation_SMd)


    def _collect_optimized_rotation(self, s_start: int, s_end: int, m_start: int, m_end: int, cross_correlation_smdw: torch.Tensor):
        assert isinstance(self._result_collection, OptimizedRotationReturn)
        values, nw = torch.max(cross_correlation_smdw, dim=3)
        nw = nw.cpu()
        self._result_collection.cross_correlation_SMd[s_start:s_end, m_start:m_end] = values.cpu()
        self._result_collection.optimal_inplane_rotation_SMd[s_start:s_end, m_start:m_end] = self._gamma[nw].cpu()


    def _initialize_full_cross_correlation(self, n_images):
        cross_correlation_SMdw = torch.zeros(
            (n_images, self.n_templates, self.n_displacements, self.n_inplanes),
            dtype=self.torch_float_type,
            device="cpu")
        self._result_collection = CrossCorrelationReturn(cross_correlation_SMdw)


    def _collect_full_cross_correlation(self, s_start: int, s_end: int, m_start: int, m_end: int, cross_correlation_smdw: torch.Tensor):
        assert isinstance(self._result_collection, CrossCorrelationReturn)
        self._result_collection.cross_correlation_SMdw[s_start:s_end, m_start:m_end, :, :] = cross_correlation_smdw.cpu()
    

    def _no_op_collector(self, s_start: int, s_end: int, m_start: int, m_end: int, x: torch.Tensor):
        ...
