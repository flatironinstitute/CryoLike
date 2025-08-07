Converting particles from CryoSPARC file formats
========================================================

**TODO: CHANGE NAME BASED ON UPDATED API NAME**

This tutorial shows how to import cryo-EM data into CryoLike from
CryoSPARC files.

When CryoLike loads CryoSPARC data, we expect to load one CryoSPARC
file that will describe image data held in one or more MRC files.


CryoSPARC data representations
------------------------------

CryoLike can import CryoSPARC data from two possible
data storage paradigms: **indexed files** and **job directories**.

In both cases, the CryoSPARC file describes how to interpret
the images, and the settings/parameters of the image capture
apparatus.

See :ref:`cryosparc-needed-fields` for information about
expected fields in CryoSPARC files.

Indexed files
****************

We use the term "indexed file" to refer to a collecton of images
described by a single CryoSPARC file. When image data is stored
in this format, we expect that the CryoSPARC file will have an
``fs`` section with ``path`` and ``idx`` members. These should
hold two lists: a list of *paths on the file system*, and
a list of *index numbers*. Each *path* specifies where to find
an MRC file holding the actual image data. The *index number*
tells us which one of the images in the MRC file is being
described by the corresponding CryoSPARC data.

The user can specify the parent directory of the MRC files,
in which case we assume that all MRC files are located
immediately under that directory. If the parent directory
is not specified, we assume that the CryoSPARC file contains
a valid path (either absolute, or relative to the directory
where CryoLike is being run) to the location of each MRC file.

**TODO: DOUBLE CHECK AFTER RENAMING**
The function that processes images indexed by a CryoSPARC
file is
:py:func:`cryolike.convert_particle_stacks.particle_stacks_conversion.convert_particle_stacks_from_cryosparc`


Job directories
*****************

We use the term "job directory" to refer to a collection of
images stored in a CryoSPARC job directory. In this setting,
we do not use any filesystem or index data from the CryoSPARC
file--we expect that the rows of the CryoSPARC file describe
the images in the MRC files in the job directory, in sequence.

This version makes particular strong assumptions about the layout of
the job directory. The expected layout is discussed in detail in the
:ref:`image conversion <cryosparc-job-folder-full-description>` documentation.

**TODO: UPDATE NAME AFTER RENAMING**
The function that processes images from a CryoSPARC job directory is
:py:func:`cryolike.convert_particle_stacks.particle_stacks_conversion.convert_particle_stacks_from_cryosparc_restack`


Common parameters
--------------------

For a description of the parameters common to all image conversion
functions, see the
:ref:`image conversion documentation<image-file-conversion-common-parameters>`.


Examples
----------------

Indexed files examples
************************

.. admonition:: Suppose:

    - ``MyFile.cs`` is a CryoSPARC file which describes 5 images
    - The file's ``blob/path`` member contains:

      - ``somedir/file1.mrc``
      - ``somedir/file1.mrc``
      - ``somedir/file2.mrc``
      - ``otherdir/file3.mrc``
      - ``somedir/file1.mrc``

    - The file's ``blob/idx`` member contains:

      - ``3``
      - ``1``
      - ``10``
      - ``10``
      - ``2``

    ``MyFile.cs`` thus reflects the following intention:

    - Its first entry describes the third image in ``somedir/file1.mrc``
    - Its second entry describes the first image in ``somedir/file1.mrc``
    - Its third entry describes the 10th image in ``somedir/file2.mrc``
    - Its fourth entry describes the 10th image in ``otherdir/file3.mrc``
    - Its fifth entry describes the second image in ``somedir/file1.mrc``

In this case, the following function call would put all 5 images into
a single output file in ``OUTDIR``:

.. code-block:: python

    convert_particle_stacks_from_cryosparc(
        params_input="my_params_file.npz",
        file_cs="MyFile.cs",
        folder_output='OUTDIR',
    )

assuming that it is run from a directory where ``somedir`` and
``otherdir`` exist.

If, however, you had moved ``file1.mrc``, ``file2.mrc``,
and ``file3.mrc`` into the ``~/my_research/my_mrc_files/``
directory, then this call would achieve the same result:

.. code-block:: python

    convert_particle_stacks_from_cryosparc(
        params_input="my_params_file.npz",
        file_cs="MyFile.cs",
        folder_cryosparc='~/my_research/my_mrc_files/',
        folder_output='OUTDIR',
    )


The following call would create 2 image stacks in the
current directory. The first stack would have the first 3
images from ``MyFile.cs`` and the second stack would hold
the remaining 2 images:

.. code-block:: python

    convert_particle_stacks_from_cryosparc(
        params_input="my_params_file.npz",
        file_cs="MyFile.cs",
        batch_size=3
    )



Job directory examples
***********************

.. admonition:: Suppose:

    - The CryoSPARC job folder is located at ``./cryosparc/J4``
    - ``./cryosparc/J4/J4_passthrough_particles.cs`` exists, and has
      data describing at least 12 images
    - ``./cryosparc/J4/restack/`` exists and contains:

      - ``batch_000000_restacked.mrc`` with 4 images
      - ``batch_000001_restacked.mrc`` with 4 images
      - ``batch_000002_restacked.mrc`` with 4 images

    - ``./cryosparc/J4/downsample/`` exists and contains:

      - ``batch_000000_downsample.mrc`` with 4 images
      - ``batch_000001_downsample.mrc`` with 4 images
      - ``batch_000002_downsample.mrc`` with 4 images
      - ``batch_000004_downsample.mrc`` with 4 images (note that
        ``...000003...`` has been deliberately skipped)

The following call would convert all 12 images from the ``restack``
directory into a single image stack placed in the ``OUTDIR``
directory:

.. code-block:: python

    convert_particle_stacks_from_cryosparc_restack(
        params_input="my_params.npz",
        folder_cryosparc= 'cryosparc',
        job_number=4,
        folder_output='OUTDIR'
    )

**If the** ``cryosparc/j4/restack/``
**directory did not exist**, then
the MRC files from the ``downsample/`` directory would be used.
The file ``batch_000004_downsample.mrc`` would never be read,
because image conversion would stop when the program looked for
``batch_000003_downsample.mrc`` and could not find it.

The following call would stop processing after emitting 2
stacks of 4 images each:

.. code-block:: python

    convert_particle_stacks_from_cryosparc_restack(
        params_input="my_params.npz",
        folder_cryosparc= 'cryosparc',
        job_number=4,
        folder_output='OUTDIR',
        batch_size=4,
        n_stacks_max=2
    )

The following call would downsample the imported images by
a factor of 2 using the mean value over the affected pixel
range:

.. code-block:: python

    convert_particle_stacks_from_cryosparc_restack(
        params_input="my_params.npz",
        folder_cryosparc= 'cryosparc',
        job_number=4,
        folder_output='OUTDIR',
        downsample_factor=2,
        downsample_type='mean'
    )

Note that this would be *independent* of any
downsampling already done to the image files.
