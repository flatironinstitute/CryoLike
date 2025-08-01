Full cryoLike pipeline from models and images to Likelihood matrix
=====================================================================

Overview
-----------
Below we describe all the steps needed to go from a set of structures and
images to obtain a CryoLike likehood matrix.

Before starting, ensure you have a Python environment set up with all
the required dependencies, as per the
:doc:`installation section</installation>`. For the commands in this
tutorial, we assume that the ``python`` command will invoke the
Python interpreter associated with this environment.

Step 1 is only performed once. Steps 2 and 3 are performed when
there are new models or particles. Step 4 is the one that actually
calculates the likelihood, comparing all converted particles to all
templates.


Step 1: Set parameters
-----------------------

Prepare the parameters.

Here we define the image and likelihood
parameters and search ranges:

  .. code-block:: console

   (venv) $ python3 set_image_parameters_example.py

See :doc:`/concepts/imageSettings` for details about the parameters,
and :doc:`/examples/set_image_parameters` for the example code.
The full set of parameters are discussed further in
**TODO: LINK API DOC**

The optimal choice of parameters depends on the specific system
being studied and is often selected empirically, based on an
understanding of the structural differences involved and the
available computational resources.

For systems where large conformational changes are expected—such
as highly flexible molecules that undergo substantial structural
rearrangements—it may be appropriate to use lower-resolution settings,
especially if the goal is to track broad conformational transitions
rather than fine structural details.

The parameters are written to a file in the output folder.


Step 2:  Make Templates from maps or PDBs
--------------------------------------------------

Example k-centers are distributed with this repository in the
``example/data/map`` folder. We can make templates
from these models using:

  .. code-block:: console

   (venv) $ python make_templates_from_inputs_example.py

The resulting Templates objects are saved in the ``output/templates``
folder.

See :doc:`/examples/make_templates` for more information about
this example, and **TODO LINK** for detailed documentation
about the template conversion functions.


Step 3: Convert images
------------------------------

We have the original particles as MRC files and STAR files.
These are distributed with the repository in the
``example/data/particles`` directory.

To convert the particles in that folder, run

  .. code-block:: console

   (venv) $ pyhton convert_particle_stacks_example.py

Converted particles are stored in the ``output/particles`` folder.

See :doc:`/concepts/imageConversion` for an overview of image
conversion, and :doc:`/tutorials/read_cryosparc_file`,
:doc:`/tutorials/read_cryosparc_restack`, and
:doc:`/tutorials/read_star_file_indexed` for tutorials on reading
different image input file formats.


Step 4: Run likelihood
------------------------------

Now we have all the inputs needed to run the likelihood calculations
on a gpu node:

  .. code-block:: console

   (venv) $ python run_likelihood_example.py

The log-likelihood for each template and image batch is stored in a
pytorch file. These outputs are stored in the ``output/likelihood`` folder.


Step 4: Collect likelihood matrix from output folders
-------------------------------------------------------------

 .. code-block:: console

   (venv) $ python get_loglike_example.py

The output is the log likelhood matrix, which is a pytorch tensor with the
shape (n_images,n_templates), saved as a text file.


Further Analysis
---------------------

- Compute log likelihood ratio.
- The log-likelihood matrix can be used as input for the ensemble
  reweighting to compute the structure weights (see
  https://github.com/flatironinstitute/Ensemble-reweighting-using-Cryo-EM-particles)
