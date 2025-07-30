Full cryoLike pipeline from models and images to Likelihood matrix
=========================================

Overview
-----------
Below we describe all steps to go from a set of structures and images to obatin cryoLike likehood matrix. 

We assume that the user has set a python environment with all the dependencies and that ``python3`` is associated with it.

The first step is only performed once. Step 2 and 3 are only performed when there are new models or particles. 
Step 4 is the one that actually calculates the likelihood on a gpu node, comparing all templates to all converted particles.


Step 1: set parameters
----------

Prepare the parameters. Here we define the image and likelihood parameters and search ranges:

  .. code-block:: console

   (venv) $ python3 set_image_parameters_example.py

See :doc:`/usage/imageSettings` for parameters details. 

The optimal choice of parameters depends on the specific system being studied and is often selected 
empirically, based on an understanding of the structural differences involved and the available computational resources.

For systems where large conformational changes are expected—such as highly flexible molecules 
that undergo substantial structural rearrangements—it may be appropriate to use lower-resolution settings, 
especially if the goal is to track broad conformational transitions rather than fine structural details.

The params are stored in the output folder.

Step 2:  Make Templates from maps or PDBs
----------

Example k-centers are in the models folders. We now make the templates from these models using:
  .. code-block:: console

   (venv) $ python3 make_templates_from_inputs_example.py

Templates are stored in the output folder.


Step 3: Convert Images 
----------

We have the original particles as  MRC files and STAR files. 
So to convert the particles in that folder run

  .. code-block:: console

   (venv) $ pyhton convert_particle_stacks_example.py

Converted particles are stored in the output folder. 
See :doc:`/tutorials/read_cryosparc_file` , :doc:`/tutorials/read_cryosparc_restack`, and 
:doc:`/tutorials/read_star_file_indexed` for details on reading different image input files.

Step 4: Run Likelihood 
----------

Now we have all the inputs to run the likelihood calculations on a gpu node:
  .. code-block:: console

   (venv) $ python run_likelihood_example.py

The log-likelihood for each template and image batch is stored in a pytorch file. 
These outputs are stored in the output folder. 

Step 4: Collect Likelihood Matrix from output folders
-----------

 .. code-block:: console

   (venv) $ python3 get_loglike_example.py 

  
The output is the log likelhood matrix, which is an array with the shape (n_images,n_templates), 
saved as a text file.
   
   
Further Analysis 
-----------

- Compute log likelihood ratio. 
- The log-likelihood matrix can be used as input for the ensemble reweighting to compute the structure weights (see https://github.com/flatironinstitute/Ensemble-reweighting-using-Cryo-EM-particles)