Run likelihood
=================
This example demonstrates how to run the likelihood computation
using the
:py:func:`run_likelihood_optimal_pose() <cryolike.run_likelihood.run_likelihood_optimal_pose>`
function from the :py:mod:`cryolike.run_likelihood` module. Please
see the linked documentation for a brief overview of the possible
parameters.

The
:py:func:`run_likelihood_optimal_pose() <cryolike.run_likelihood.run_likelihood_optimal_pose>`
function computes the likelihood of observing the cryo-EM images given
a set of 3D structures or maps.
The 3D structure is represented as a set of 2D template images,
observed from different viewing angles, for example, created in
:doc:`/examples/make_templates`.

Example usage:

.. literalinclude:: ../../../example/run_likelihood_example.py
    :language: python
