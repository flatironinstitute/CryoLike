Converting particles from STAR file formats
========================================================

This tutorial shows how to import cryo-EM image data into CryoLike from
MRC files described by STAR-formatted files.

When CryoLike loads STAR file data, we expect the actual image values
to be stored in MRC, MRCS, or MAP files, with the descriptive values
stored in one or more STAR files.

For more background on STAR files in CryoLike, see the
:ref:`file formats documentation<starfile-format-background-info>`


STAR file data representations
-------------------------------

CryoLike can import STAR file data using two possible modalities:
**indexed files** and **parallel file lists**.

In both cases, the STAR file describes how to interpret the images,
and the settings/parameters of the image capture apparatus, while
the image data itself is in accompanying MRC files.

For more complete background on file conversion, see
:ref:`the image conversion documentation<star-file-general-info>`.


Parallel lists
****************

We use the term "parallel lists" to refer to the
:py:func:`cryolike.file_conversions.particle_stacks_wrappers.convert_particle_stacks_from_paired_star_and_mrc_files`
conversion function, whose inputs include two separate lists:
one list of STAR files, and one list of MRC files.

This function assumes that the two lists match: that is, that
every STAR file describes exactly the images in the corresponding
MRC file, and that the STAR file describes those images in the
same order as they appear in the MRC file.

Because the user passes the paths to all the STAR and MRC files
individually, this function makes no assumptions about directory
structure. It ignores any MRC file indexing that is present in the
STAR files.


Indexed files
****************

We use the term "indexed file" to refer to a collection of images
described by a single STAR file (even though they may reside in
many MRC files). We assume that indexed STAR files will follow
the ReLION naming convention for the descriptive fields.

In particular, we assume that images will be described in an
``ImageName`` section, where each row has an initial field
following the pattern of ``12345@path/to/file.mrc``--i.e.,
a numeric index, followed by an ``@`` sign, followed by
a valid path to the MRC file. The other data block contains
descriptive information in the same order as the files
appear in the ``ImageName`` block.

.. admonition:: Example:

    So, a file whose ``ImageName`` block contained the following:

    - ``0004@my/file.mrc``
    - ``0002@my/otherfile.mrc``

    would be describing two images. The first image described
    would be the 4th image stored in ``my/file.mrc``, and the
    second image described would be the 2nd image stored in
    ``my/otherfile.mrc``.

CryoLike allows the user to indicate that the MRC files are
actually located in a different directory, in which case the
path information (but not the filename or index number) is
ignored.

Common parameters
-------------------

For a description of the parameters common to all image
conversion functions, see the
:ref:`image conversion documentation<image-file-conversion-common-parameters>`.

Examples
------------


Parallel lists example
*************************

.. admonition:: Suppose:

    - ``MyFile.star`` is a STAR file which describes 2 images
    - ``File2.star`` is a STAR file which describes 3 images
    - ``images.mrc`` is an MRC file containing 2 images
    - ``images2.mrc`` is an MRC file containing 3 images

The following function call would create a single CryoLike image
stack with 5 images in the ``OUTDIR`` directory:

.. code-block:: python

    convert_particle_stacks_from_paired_star_and_mrc_files(
        params_input="my_params_file.npz",
        particle_file_list=["images.mrc", "images2.mrc"],
        star_file_list=["MyFile.star", "File2.star"],
        folder_output='OUTDIR'
    )

Note that this function assumes that the first row of
data in ``MyFile.star`` describes the first image in
``images.mrc``, the second row of data describes the
second image, and so on: **any indexing information**
**in the STAR file will be ignored.**

By default this would assume that the defocus angle and
phase shift information was given in degrees, not radians.

By default, existing output files in the ``OUTDIR`` directory
will be left in place and file conversion will stop if there
would be a naming conflict. Setting the ``overwrite`` parameter
to ``True`` will suppress this behavior.


Indexed files examples
**************************

.. admonition:: Suppose:

    - ``MyFile.star`` is a STAR file which describes 5 images
    - The file's ``ImageName`` section contains:

      - ``0003@somedir/file1.mrc``
      - ``0001@somedir/file1.mrc``
      - ``0010@somedir/file2.mrc``
      - ``10@otherdir/file3.mrc``
      - ``0002@somedir/file1.mrc``

    ``MyFile.star`` reflects the following intention:

    - The first entry in its data block describes the
      third image in ``somedir/file1.mrc``
    - The second entry describes the first image in ``somedir/file1.mrc``
    - The third entry describes the 10th image in ``somedir/file2.mrc``
    - The fourth entry describes the 10th image in ``otherdir/file3.mrc``
    - The fifth entry describes the second image in ``somedir/file1.mrc``


The following function call would create a single CryoLike image
stack with 5 images in the ``OUTDIR`` directory:

.. code-block:: python

    convert_particle_stacks_from_indexed_star_files(
        params_input="my_params_file.npz",
        star_file='MyFile.star',
        folder_output="OUTDIR",
    )

assuming that the ``somedir`` and ``otherdir`` directories exist
in the directory where CryoLike is being run.

If, instead, the user has moved ``file1.mrc``, ``file2.mrc``,
and ``file3.mrc`` into the ``~/my_research/my_mrc_files/``
directory, then this call would achieve the same result:

.. code-block:: python

    convert_particle_stacks_from_indexed_star_files(
        params_input="my_params_file.npz",
        star_file='MyFile.star',
        folder_mrc='~/my_research/my_mrc_files/',
        folder_output="OUTDIR",
    )

In this case, CryoLike will honor the indexing information
but ignore the path information (other than the MRC files'
basenames).

The following call would result in two output stacks (one with
the first 3 images and one with the remaining 2 images). It would
also downsample each image by a factor of 2 during the image conversion,
taking the mean of the downsampled pixel ranges:

.. code-block:: python

    convert_particle_stacks_from_indexed_star_files(
        params_input="my_params_file.npz",
        star_file='MyFile.star',
        folder_output="OUTDIR",
        batch_size=3,
        downsample_factor=2,
        downsample_type='mean'
    )
