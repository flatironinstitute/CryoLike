import numpy as np
import matplotlib as mpl
import matplotlib.axes as mpl_axes
from matplotlib import pyplot as plt
import torch
from collections.abc import Iterable

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
cmap = plt.get_cmap("gray")

from cryolike.stacks import Images, Templates
from cryolike.grids import FourierImages, PolarGrid, CartesianGrid2D
from cryolike.util import FloatArrayType

def plot_images(
    images: torch.Tensor | np.ndarray,
    grid: PolarGrid | CartesianGrid2D,
    n_plots: int = 16,
    filename: str = '',
    show: bool = False,
    with_ctf: bool = False,
    label_angles: bool = False,
) -> None:
    if isinstance(grid, PolarGrid) and not grid.uniform:
        raise NotImplementedError("Non-uniform fourier images not implemented yet.")

    if filename == '' and not show:
        # raise ValueError("Provide either a filename or set show")
        show = True     # TODO: This ok?
    n_images = images.shape[0]
    if n_images < n_plots:
        # raise ValueError("Requested number of plots exceeds the number of images available.")
        n_plots = n_images ## TODO: This ok?

    use_polar = True if isinstance(grid, PolarGrid) else False
    axs = _setup_figure(n_plots, use_polar)

    for im, ax in zip(images, axs.flatten()):
        real_img = _extract_real_image_as_np(im)
        if isinstance(grid, PolarGrid):
            _plot_polar_image_uniform_grid(grid, ax, real_img)
        else:
            _plot_physical_image(grid, ax, real_img)
    _finalize_plot(filename, show)


def _setup_figure(n_plots: int, use_polar: bool = False) -> Iterable[mpl_axes.Axes]:
    n_axis_x = int(np.ceil(np.sqrt(n_plots)))
    n_axis_y = int(np.ceil(n_plots / n_axis_x))
    subplot_kw = {'projection': 'polar'} if use_polar else None
    _, axs = plt.subplots(n_axis_x, n_axis_y, figsize = (12, 12), subplot_kw = subplot_kw)
    if n_plots == 1:
        assert isinstance(axs, mpl_axes.Axes)
        return [axs]
    assert isinstance(axs, np.ndarray)
    return axs


def _plot_polar_image_uniform_grid(polar_grid: PolarGrid, ax: mpl_axes.Axes, image: np.ndarray):
    theta = polar_grid.theta_shell
    radius = polar_grid.radius_shells
    ax.pcolormesh(theta, radius, image, cmap = cmap)
    # if label_angles:
    #     label = ""
    #     if images_or_templates.polars_viewing is not None:
    #         label += "P " + "%.3f" % images_or_templates.polars_viewing[i].item() + " "
    #     if images_or_templates.azimus_viewing is not None:
    #         label += "A " + "%.3f" % images_or_templates.azimus_viewing[i].item() + " "
    #     if images_or_templates.gammas_viewing is not None:
    #         label += "G " + "%.3f" % images_or_templates.gammas_viewing[i].item() + " "
    #     ax.set_title(label, fontsize = 12)
    ax.axis('off')


def _extract_real_image_as_np(im: torch.Tensor | np.ndarray):
    real_img = im.real
    if isinstance(real_img, torch.Tensor):
        return real_img.detach().cpu().numpy()
    return real_img


def _plot_physical_image(phys_grid: CartesianGrid2D, ax: mpl_axes.Axes, image: np.ndarray):
    ax.imshow(image, cmap = cmap, origin = "lower")
    ax.axis('off')


def _save_to_file(filename: str):
    if filename == '':
        return
    plt.savefig(filename)


def _finalize_plot(filename: str, show: bool):
    plt.tight_layout()
    _save_to_file(filename)
    if show:
        plt.show()
    plt.close()


PowerSpectrumTuple = tuple[torch.Tensor, PolarGrid, float | FloatArrayType | None]
def plot_power_spectrum(
    source: Images | Templates | PowerSpectrumTuple,
    filename_plot: str = '',
    show: bool = False
):
    if filename_plot is None and not show:
        print("Filename not provided and show is set to False. No plot will be generated.")
    if isinstance(source, tuple):
        (images_fourier, polar_grid, box_size) = source
        source = Images(fourier_images_data=FourierImages(images_fourier, polar_grid), box_size=box_size)
    power_spectrum, resolutions = source.get_power_spectrum()
    if isinstance(power_spectrum, torch.Tensor):
        power_spectrum = power_spectrum.detach().cpu().numpy()
    if filename_plot is not None or show:
        fig = plt.figure(figsize = (8, 6))
        plt.plot(resolutions, power_spectrum)
        plt.xscale("log")
        plt.yscale("log")
        plt.gca().invert_xaxis()
        plt.xlabel("Resolution (Angstrom)")
        plt.ylabel("Power Spectrum")
        if filename_plot is not None:
            plt.savefig(filename_plot)
        if show:
            plt.show()
        plt.close()
    return power_spectrum, resolutions


def plot_polar_density_on_axis(
    ax: mpl_axes.Axes,
    polar_grid: PolarGrid,
    density: np.ndarray,
    cmap = cmap,
    title: str = '',
    label_angles: bool = False
):
    theta = polar_grid.theta_shell
    radius = polar_grid.radius_shells
    density = density.reshape(polar_grid.n_shells, polar_grid.n_inplanes)
    if density.shape != (polar_grid.n_shells, polar_grid.n_inplanes):
        raise ValueError("Density shape does not match the polar grid.", density.shape, (polar_grid.n_shells, polar_grid.n_inplanes))
    ax.pcolormesh(theta, radius, density, cmap = cmap)
    if title != '':
        ax.set_title(title)
    if label_angles:
        label = ""
        # NOTE: polar_grid doesn't have viewing angles in any form.
        # if polar_grid.polars_viewing is not None:
        #     label += "P " + "%.3f" % polar_grid.polars_viewing + " "
        # if polar_grid.azimus_viewing is not None:
        #     label += "A " + "%.3f" % polar_grid.azimus_viewing + " "
        # if polar_grid.gammas_viewing is not None:
        #     label += "G " + "%.3f" % polar_grid.gammas_viewing + " "
        ax.set_title(label)
    ax.axis('off')
