File formats for image conversion
##########################################

Image conversion (from mrc or cryoSPARC particle files) requires both an
:doc:`image descriptor</usage/imageSettings>` and capture apparatus
metadata. The image descriptor defines the grids and scales to use
to interpret the image and is usually constructed from base values.

Capture apparatus metadata, however, is expected to be delivered
in either Starfile or CryoSparc format. Since these file formats
are potentially quite loosely defined, here we discuss the exact
formats currently supported.

Note that, regardless of format, we always expect that there will one CTF per
image in the stack. Therefore, defocus information (and
phase shift information, if present) need to be available for
each captured image.


Starfile
=========

The base Starfile file reader can be found at TODO: API XREF

We expect that the Starfile can be read by the ``starfile`` library
and will return either a Pandas dataframe or a dictionary of string
keys pointing to the Pandas dataframe.

If the dictionary is returned, we expect it to contain the keys ``optics``
and ``particles``. These two dataframes will be joined together on
the ``rlnOpticsGroup`` field.

Any valid Starfile must contain at least the ``Voltage`` and
``SphericalAberration`` fields.


ReLion-formatted
----------------

For an example of a supported ReLion-formatted file, see the
``relion_style_particles.star`` file in the ``test/data`` directory
of the CryoLike github repository.

This format is specifically used for "indexed" metadata files, where
the single Starfile describes a large number of particle images which
may be found in other files across the filesystem.

We expect ReLion-formatted Starfiles to define ``optics`` and ``particles``
sections
which can be joined on the ``rlnOpticsGroup`` column. Once this join
is done, we will read all fields from the result. (Note that any leading
``rln`` or ``_rln`` will be stripped from the field names.)

The following fields MUST be defined:

 - ``DefocusU``
 - ``DefocusV``
 - ``DefocusAngle``
 - ``SphericalAberration``
 - ``Voltage``

The following fields will be given default values or ignored, if missing:

 - ``PhaseShift`` -- defaults to 0 if missing
 - ``AmplitudeContrast`` -- defaults to 0.1 if missing
 - ``AngleRot`` -- ignored if not present
 - ``AngleTilt`` -- ignored if not present
 - ``AnglePsi`` -- ignored if not present
 - ``CtfBfactor`` -- defaults to 0 if missing
 - ``CtfScalefactor`` -- defaults to 1 if missing
 - ``ImagePixelSize`` -- size of pixels in the image

Any other field will be ignored.

We expect every row to consist of entries of the form::

    000001@filename.mrcs    [fields...]

where ``000001`` (the part before the ``@`` sign) indicates the 1-based
index of the image within the MRC file, and ``filename.mrcs`` (the part after
the ``@`` sign) indicates the source MRC file. The parameters passed to
the image conversion function will determine how the mrc filename is
interpreted: if a folder parameter is passed, we will take only the
filename (ignoring any path) and look for the file within the specified
directory; otherwise we will keep the path from the Starfile and assume
it is a valid relative path from the working directory
(where the script was called).


Non-ReLion-formatted
--------------------

We also support a more generic Starfile format. The same basic conditions
(must be readable as a Pandas dataframe or dictionary of dataframes with
``optics`` and ``particles`` keys) apply. However, instead of trying to
parse the remaining rows, we simply treat the rows as a list of metadata
values to be applied in sequence to images from an MRC file.

This is the format expected in the
``convert_particle_stacks_from_star_files()``
wrapper
(see :doc:`the image conversion documentation</usage/imageConversion>`).

The expected fields are the same as for the ReLion case, above, except
that we will ignore the ``AngleRot``, ``AngleTilt``, ``AnglePsi``, and
``ImagePixelSize`` fields in this case, even if they are present.


CryoSparc
==========

CryoSparc files do not require a special library to read; they are assumed
to be implemented as Numpy array files. We expect the following fields
to be defined:

 - ``ctf/df1_A`` as the "DefocusU"
 - ``ctf/df2_A`` as the "DefocusV"
 - ``ctf/df_angle_rad`` as the defocus angle
 - ``ctf/cs_mm`` as the spherical aberration value (note
   that this is assumed to be the same for all described images)
 - ``ctf/accel_kv`` as the voltage value (assumed consistent for all images)
 - ``ctf/amp_constrast`` as the amplitude contrast (assumed consistent
   for all images)
 - ``ctf/phase_shift_rad`` as the phase shift value

With internal index
-------------------

If the CryoSparc file defines an internal index of particle files, we will
also look for the following fileds:

 - ``blob/path``: defines the path to the MRC file containing
   each particle image
 - ``blob/idx``: states the index, within the MRC file, of the
   image being described
 - ``blob/psize_A``: optional. If defined, states the pixel size of the image

Note that all these values are being read by the same index. So for an indexed
CryoSparc file, looking at index ``i``, we would expect:

 - ``ctf/df1_A[i]`` to give the defocus U value for that image
 - ``ctf/phase_shift_rad[i]`` to give the phase shift value for that image
 - ``blob/path[i]`` to be the path to the MRC file storing that particle image
 - ``blob/idx[i]`` to be the index within ``blob/path[i]`` of that image

etc.


Without internal index
-----------------------

If the internal index fields are not present, we assume that the
records are correctly-ordered descriptors of the images in the
MRC files in the job directory. See the
:doc:`image conversion documentation</usage/imageConversion>`
for more details (``convert_particle_stacks_from_cryosparc_restack()``).

