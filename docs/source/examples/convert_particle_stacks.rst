Convert Particles
=================

This script demonstrates the usage of the
:py:func:`convert_particle_stacks_from_star_files()
<cryolike.convert_particle_stacks.particle_stacks_conversion.convert_particle_stacks_from_star_files>`
function. Please see the linked documentation for a brief description
of the possible parameters, and the
:doc:`documentation on image conversion</concepts/imageConversion>` for a more
complete description of image conversion capabilities in CryoLike.

The ``convert_particle_stacks_from_star_files()`` function converts particle
stacks from ``mrc`` files (containing the image) and ``star`` files (containing
the image metadata). These are passed as two lists of files. The function
assumes that each star file describes all of the images in the
corresponding MRC file, in order.

Example usage:

.. literalinclude:: ../../../example/convert_particle_stacks_example.py
    :language: python
