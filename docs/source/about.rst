Mathematical Framework
======================

.. _mathframework:
        :title: Math Framework

CryoLike is a GPU-accelerated software package for efficiently computing 
image-to-structure likelihood in cryo-electron microscopy (cryo-EM). 
It is upon a mathematical framework using Fourier-Bessel representations, and draws on prior work in cross-correlation computation
for ab initio reconstruction [Rangan, Greengard 2023].
Cryolike extends this method to support full likelihood-based comparisons
in a user-friendly Python interface and enables scalable image-to-structure likelihood evaluation across large image datasets.
It allows researchers to assess how well candidate structures 
explain cryo-EM data without the overhead of full reconstruction pipelines. 

The CryoLike workflow includes modules for template generation, particle image conversion, 
cross-correlation computation, and log-likelihood evaluation, 
all optimized for modern GPUs. 


Main objective
----------------------

Given a cryo-EM image, and a 3D structure or map, cryoLike computes the image-to-structure likelihood. 
Representing the image and templates (2D projections of the 3D structure) in a Fourier-Bessel basis set, 
it searches for  the optimal cross-correlation between the image and templates,
i.e. finds the pose with highest cross-correlation to the image. 
CryoLike then approximates the image-to-structure likelihood by using the image-to-template likelihood
evaluated at the optimal pose obtained by finding the best cross-correlation. 

Cross-Correlation in Fourier Space
################################

At its core, the cross-correlation :math:`\mathcal{C}(\theta; f, g)` between two functions :math:`f(\psi)` and :math:`g(\psi)` is defined as:

.. math::

    \mathcal{C}(\theta; f, g) = \int_{\Omega} f^*(\psi) \cdot g(\psi + \theta) \, d\psi,

where :math:`f^*` denotes the complex conjugate and :math:`\Omega` is the domain of integration.

Using the convolution theorem, this cross-correlation can be evaluated via the inverse Fourier transform of the element-wise product of the Fourier transforms:

.. math::

    \mathcal{C}(\cdot; f, g) = \mathcal{F}^{-1}\left[ \mathcal{F}[f]^* \odot \mathcal{F}[g] \right],

where :math:`\mathcal{F}` represents the 1D Fourier transform and :math:`\odot` denotes element-wise multiplication.

CryoLike leverages this principle to compute cross-correlations in the angular coordinate of 2D Fourier space. 
Specifically, for an image :math:`\tilde{I}(k, \psi)` and a template :math:`\tilde{T}_{\phi}(k, \psi)`, the 
frequency-space cross-correlation :math:`\mathcal{C}_{\text{freq}}(\theta; \tilde{T}, \tilde{I})` is calculated 
by combining radial integration with angular convolution. In practice, this involves transforming both the image and 
template into a Fourier-Bessel basis via a 1D angular Fourier transform, multiplying their coefficients, and 
then applying an inverse transform to obtain the final cross-correlation with respect to in-plane rotation :math:`\theta`.


Image-to-Template Likelihood
################################

This section outlines a simplified formulation of the main cryoLike output: the image-to-template likelihood.
For the sake of simplicity, we formulate it in physical space but it can be easily extended to Fourier space. 
A full derivation of both formulations is provided in the paper.

We assume a Gaussian white-noise model in physical space. 
Each cryo-EM image :math:`I(\mathbf{x})` is modeled as a scaled projection template :math:`T_{\phi}(\mathbf{x})` 
at pose :math:`\phi`, with image-specific intensity :math:`\alpha`, a constant offset :math:`\beta`, and additive noise:

.. math::

    I(\mathbf{x}) \sim \alpha T_{\phi}(\mathbf{x}) + \beta \mathbb{1}(\mathbf{x}) + \epsilon(\mathbf{x}),

where :math:`\epsilon(\mathbf{x})` is drawn from a zero-mean Gaussian noise with constant pixel variance :math:`\lambda^2`.

The likelihood of observing image :math:`I` given the template and parameters is:

.. math::

    P(I \mid T_{\phi}, \alpha, \beta, \lambda) = \frac{1}{(2\pi)^{N/2} \lambda^N} \exp\left\{ -\frac{\ell_{\text{phys}}(I, T_{\phi}, \alpha, \beta)}{\lambda^2} \right\},

where :math:`\ell_{\text{phys}}` is the squared L2-norm between the modeled and observed image:

.. math::

    \ell_{\text{phys}}(I, T_{\phi}, \alpha, \beta) = \int_{\Omega} \left| \alpha T_{\phi}(\mathbf{x}) + \beta \mathbb{1}(\mathbf{x}) - I(\mathbf{x}) \right|^2 d\mathbf{x},

where :math:`\Omega` is the image space. We can write the last expression as a function of the cross-correlation 
.. math::
    missing

