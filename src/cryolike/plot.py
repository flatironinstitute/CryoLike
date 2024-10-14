import numpy as np
import matplotlib as mpl
import matplotlib.axes as mpl_axes
from matplotlib import pyplot as plt
import torch
from typing import Union

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 16
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
cmap = plt.get_cmap("gray")

from cryolike.template import Templates
from cryolike.image import Images, FourierImages
from cryolike.polar_grid import PolarGrid
from cryolike.cartesian_grid import CartesianGrid2D

def plot_images(
    images: np.ndarray | torch.Tensor | None = None,
    polar_grid: PolarGrid | None = None,
    phys_grid: CartesianGrid2D | None = None,
    n_plots : int = 16,
    filename : str | None = None,
    show : bool = False,
    with_ctf: bool = False,
    label_angles: bool = False,
) -> None:
    if filename is None and not show:
        raise ValueError("Provide either a filename or set show")
    if images is None:
        raise ValueError("Images not provided.")
    if polar_grid is None and phys_grid is None:
        raise ValueError("Provide either a polar grid or a physical grid.")
    if polar_grid is not None:
        n_images = images.shape[0]
        if n_images < n_plots:
            raise ValueError("Number of images to plot exceeds the number of images available.")
        theta = polar_grid.theta_shell
        radius = polar_grid.radius_shells
        n_axis_x = int(np.ceil(np.sqrt(n_plots)))
        n_axis_y = int(np.ceil(n_plots / n_axis_x))
        fig, axs = plt.subplots(n_axis_x, n_axis_y, figsize = (12, 12), subplot_kw = {'projection': 'polar'})
        for i in range(n_plots):
            if n_plots == 1:
                ax = axs
            else:
                ax = axs[i // n_axis_y, i % n_axis_y]
            image_this = None
            if polar_grid.uniform:
                image_this = images[i].real
            else:
                raise ValueError("Non-uniform fourier images not implemented yet.")
            if isinstance(image_this, torch.Tensor):
                image_this = image_this.detach().cpu().numpy()
            ax.pcolormesh(theta, radius, image_this, cmap = cmap)
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
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()
        return
    if phys_grid is not None:
        n_images = images.shape[0]
        if n_images < n_plots:
            raise ValueError("Number of images to plot exceeds the number of images available.")
        # fig = plt.figure(figsize = (12, 12))
        n_axis_x = int(np.ceil(np.sqrt(n_plots)))
        n_axis_y = int(np.ceil(n_plots / n_axis_x))
        fig, axs = plt.subplots(n_axis_x, n_axis_y, figsize = (12, 12))
        for i in range(n_plots):
            if n_plots == 1:
                ax = axs
            else:
                ax = axs[i // n_axis_y, i % n_axis_y]
            image_this = images[i].real
            if isinstance(image_this, torch.Tensor):
                image_this = image_this.detach().cpu().numpy()
            ax.imshow(image_this, cmap = cmap, origin = "lower")
            ax.axis('off')
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()
        return
    return

def plot_power_spectrum(
    image_or_template: Union[Images, Templates] | None = None,
    images_fourier: np.ndarray | torch.Tensor | None = None,
    polar_grid: PolarGrid | None = None,
    box_size: float | None = None,
    filename_plot: str | None = None,
    show: bool = False
):
    if filename_plot is None and not show:
        print("Filename not provided and show is set to False. No plot will be generated.")
    if image_or_template is None:
        if box_size is None:
            raise ValueError("Box size not provided.")
        if polar_grid is None:
            raise ValueError("Polar grid not provided.")
        if images_fourier is None:
            raise ValueError("Fourier images must be passed if no Image or Template object was passed.")
        if isinstance(images_fourier, np.ndarray):
            images_fourier = torch.tensor(images_fourier)
        image_or_template = Images(fourier_images_data=FourierImages(images_fourier, polar_grid), box_size=box_size)
    power_spectrum, resolutions = image_or_template.get_power_spectrum()
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
    ax : mpl_axes.Axes,
    polar_grid : PolarGrid,
    density : np.ndarray,
    cmap = cmap,
    title : str = '',
    label_angles : bool = False
):
    if density is None:
        raise ValueError("Density not provided.")
    if polar_grid is None:
        raise ValueError("Polar grid not provided.")
    if ax is None:
        raise ValueError("Axes not provided.")
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
    return