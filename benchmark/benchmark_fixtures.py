import torch
import numpy as np
import pickle
from pathlib import Path

from torch import Tensor
import numpy as np
from numpy import pi

from cryolike.grids import PolarGrid
from cryolike.stacks import Templates
from cryolike.metadata import ViewingAngles
from cryolike.microscopy import CTF

from cryolike.util import Precision, to_torch


class Parameters:

    device: str
    n_pixels: int
    n_templates: int
    n_templates_per_batch: int
    n_images_per_batch: int
    n_displacements: int
    precision: Precision

    n_shells: int
    n_inplanes: int

    t_ms : float | None

    def __init__(
        self,
        device: str,
        n_pixels: int,
        n_templates: int,
        n_displacements_x: int,
        n_displacements_y: int,
        n_templates_per_batch: int,
        n_images_per_batch: int,
        precision: Precision,
    ):
        self.device = device
        self.n_pixels = n_pixels
        self.n_templates = n_templates
        self.n_displacements_x = n_displacements_x
        self.n_displacements_y = n_displacements_y
        self.n_templates_per_batch = n_templates_per_batch
        self.n_images_per_batch = n_images_per_batch
        self.precision = precision
        self.t_ms = None


    def duplicate(self, *,
        device: str | None = None,
        n_pixels: int | None = None,
        n_templates: int | None = None,
        n_templates_per_batch: int | None = None,
        n_images_per_batch: int | None = None,
        n_displacements_x: int | None = None,
        n_displacements_y: int | None = None,
        precision: Precision | None = None
    ):
        return Parameters(
            device=device if device is not None else self.device,
            n_pixels=n_pixels if n_pixels is not None else self.n_pixels,
            n_templates=n_templates if n_templates is not None else self.n_templates,
            n_displacements_x=n_displacements_x if n_displacements_x is not None else self.n_displacements_x,
            n_displacements_y=n_displacements_y if n_displacements_y is not None else self.n_displacements_y,
            n_templates_per_batch=n_templates_per_batch if n_templates_per_batch is not None else self.n_templates_per_batch,
            n_images_per_batch=n_images_per_batch if n_images_per_batch is not None else self.n_images_per_batch,
            precision=precision if precision is not None else self.precision,
        )


    @staticmethod
    def default():
        return Parameters(
            device='cuda',
            n_pixels=64,#128,
            n_templates=1024,
            n_displacements_x=8,
            n_displacements_y=8,
            n_templates_per_batch=128,
            n_images_per_batch=128,
            precision=Precision.SINGLE,
        )


    def save(self, filename: Path):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    
    @staticmethod
    def load(filename: str):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    


def make_batch_size_params() -> list[Parameters]:
    params = [Parameters.default()]
    with_image_batch_sizes = [x.duplicate(n_images_per_batch=i) for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] for x in params]
    params.extend(with_image_batch_sizes)
    # TODO CHANGE BACK
    # with_template_batch_sizes = [x.duplicate(n_templates_per_batch=i) for i in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] for x in params]
    with_template_batch_sizes = [x.duplicate(n_templates_per_batch=i) for i in [1, 2, 4, 8, 16, 32, 64] for x in params]
    params.extend(with_template_batch_sizes)
    return params


def make_n_pixels_params() -> list[Parameters]:
    params = [Parameters.default()]
    with_n_pixels = [x.duplicate(n_pixels=i) for i in [64, 128, 256, 512, 1024] for x in params]
    params.extend(with_n_pixels)
    return params


def make_n_displacements_params() -> list[Parameters]:
    params = [Parameters.default()]
    with_n_displacements = [x.duplicate(n_displacements_x=i,n_displacements_y=i) for i in [4, 8, 16, 32] for x in params]
    params.extend(with_n_displacements)
    return params



def make_default_polar_grid(n_pixels: int) -> PolarGrid:
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


def make_default_viewing_angles(n_im: int, precision: Precision = Precision.SINGLE) -> ViewingAngles:
    (torch_float_type, _, _) = precision.get_dtypes(default=Precision.SINGLE)
    azimus = torch.linspace(0, 2 * np.pi, n_im, dtype=torch_float_type)
    polars = torch.linspace(0, np.pi, n_im, dtype=torch_float_type)
    gammas = torch.linspace(0, 2 * np.pi, n_im, dtype=torch_float_type)
    return ViewingAngles(azimus, polars, gammas)


def make_default_ctf(polar_grid: PolarGrid, anisotropy: bool = False, precision: Precision = Precision.SINGLE) -> CTF:
    _radius_shells = to_torch(polar_grid.radius_shells, precision, "cuda")
    _ctf_tensor = _radius_shells.unsqueeze(0)
    if anisotropy:
        _ctf_tensor = _ctf_tensor.unsqueeze(-1).expand(-1, -1, polar_grid.n_inplanes)
    ctf = CTF(
        ctf_descriptor=_ctf_tensor,
        polar_grid=polar_grid,
        box_size=2.0,
        anisotropy=anisotropy,
        cs_corrected=False
    )
    return ctf


def make_default_templates(polar_grid: PolarGrid, viewing_angles: ViewingAngles, precision: Precision = Precision.SINGLE) -> Templates:
    def _generator_function(x: Tensor) -> Tensor:
        return torch.amax(x, dim=-1) + torch.amin(x, dim=-1) * 1j
    templates = Templates.generate_from_function(
        function=_generator_function,
        viewing_angles=viewing_angles,
        polar_grid=polar_grid,
        precision=precision,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    return templates
