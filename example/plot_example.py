import os
import torch
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt

mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24
mpl.rcParams['legend.fontsize'] = 20
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams['xtick.major.width'] = 4
mpl.rcParams['ytick.major.width'] = 4
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['legend.facecolor'] = 'white'
mpl.rcParams['legend.edgecolor'] = 'white'
mpl.rcParams['legend.frameon'] = True

PLOT_ALPHA = 0.8

colors = [
    "#000000", "#E69F00", "#56B4E9",
    "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7",
]
linestyles = [
    "-", "--", "-.", ":"
]

folder_output = "./output/likelihood/"
folder_output_log_likelihood_1 = os.path.join(folder_output, 'template0', 'log_likelihood')
folder_output_log_likelihood_0 = os.path.join(folder_output, 'template1', 'log_likelihood')

bins = np.linspace(-200, 1200, 351)

## integrated fourier
log_likelihood_P0 = torch.load(os.path.join(folder_output_log_likelihood_0, "log_likelihood_integrated_fourier_stack_000000.pt"), weights_only=True).detach().numpy()
log_likelihood_P1 = torch.load(os.path.join(folder_output_log_likelihood_1, "log_likelihood_integrated_fourier_stack_000000.pt"), weights_only=True).detach().numpy()
log_likelihood_ratio_integrated = log_likelihood_P1 - log_likelihood_P0
hist, bin_edges = np.histogram(log_likelihood_ratio_integrated, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
cumhist_integrated = np.cumsum(hist) * (bin_centers[1] - bin_centers[0])
n_images = log_likelihood_ratio_integrated.shape[0]

# ## optimal pose physical
# log_likelihood_phys_P0 = torch.load(os.path.join(folder_output_log_likelihood_0, "log_likelihood_optimal_physical_stack_000000.pt"), weights_only=True).detach().numpy()
# log_likelihood_phys_P1 = torch.load(os.path.join(folder_output_log_likelihood_1, "log_likelihood_optimal_physical_stack_000000.pt"), weights_only=True).detach().numpy()
# log_likelihood_ratio_phys = log_likelihood_phys_P1 - log_likelihood_phys_P0
# hist, bin_edges = np.histogram(log_likelihood_ratio_phys, bins=bins, density=True)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
# cumhist_phys = np.cumsum(hist) * (bin_centers[1] - bin_centers[0])

## optimal pose fourier
log_likelihood_fourier_P0 = torch.load(os.path.join(folder_output_log_likelihood_0, "log_likelihood_optimal_fourier_stack_000000.pt"), weights_only=True).detach().cpu().numpy()
log_likelihood_fourier_P1 = torch.load(os.path.join(folder_output_log_likelihood_1, "log_likelihood_optimal_fourier_stack_000000.pt"), weights_only=True).detach().cpu().numpy()
log_likelihood_ratio_fourier = log_likelihood_fourier_P1 - log_likelihood_fourier_P0
log_likelihood_ratio_fourier = np.sort(log_likelihood_ratio_fourier)
hist, bin_edges = np.histogram(log_likelihood_ratio_fourier, bins=bins, density=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
cumhist_fourier = np.cumsum(hist) * (bin_centers[1] - bin_centers[0])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(bin_centers, cumhist_integrated, label="Integrated Fourier", color=colors[0], linestyle=linestyles[0], alpha=PLOT_ALPHA)
# ax.plot(bin_centers, cumhist_phys, label="Optimal physical", color=colors[1], linestyle=linestyles[1], alpha=PLOT_ALPHA)
ax.plot(bin_centers, cumhist_fourier, label="Optimal Fourier", color=colors[2], linestyle=linestyles[2], alpha=PLOT_ALPHA)
## vertical line at 0
ax.plot([0, 0], [0, 1], 'k--', alpha = 0.2)
ax.set_xlim(-200, 1200)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(-200, 1201, 200))
ax.set_xlabel("Log likelihood ratio")
ax.set_ylabel("Cumulative probability")
plt.legend(
    frameon=False,
    loc="lower right",
)
plt.tight_layout()
# plt.show()
plt.savefig(os.path.join(folder_output, "log_likelihood_ratio.png"))