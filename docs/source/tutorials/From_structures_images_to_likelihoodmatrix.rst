Full cryoLike pipeline from models and images to Likelihood matrix
=========================================

Overview
-----------
Below we describe all steps to go from a set of structures and images to obatin cryoLike likehood matrix. 

We assume that the user has set a python environment with all the dependencies and that ``python3`` is associated with it.

The first step is only performed once. Step 2 and 3 only when there are new models or particles. Step 4 is the one that actually calculates the likelihood on a gpu node, comparing all templates to all converted particles.


Step 1: set parameters
----------

Prepare the parameters. Here we define the image and likelihood parameters and search ranges:

  .. code-block:: console

   (venv) $ python3 set_image_params.py

See XXX for parameters details. 

We note that the optimal parameter choice depends on the system of interest and be chosen more or less empirically
by knowing what are the  differences between structures and how much compute time you can spend. 

For a system where I expect large conformational changes between the structures 
(i.e. quite flexible - large conformational changes that i can track with low resolution) low resolution etc XX 

The params are stored in the output folder.

Step 2:  Make Templates from maps or PDBs
----------

Example k-centers are in the models folders. We now make the templates from these models using:
  .. code-block:: console

   (venv) $ python3 make_templates_from_inputs.py

Templates are stored in the output folder.


Step 3: Convert Images 
----------

We have the original_particles from cryoSPARC jobs with mrc and star files in subfolders. These subfolders are named after the cryoSPARC job e.g. J1412.
So to convert the particles in that folder (up to batch 20) run with the CS job number

  .. code-block:: console

   (venv) $ pyhton convert_particle_stack.py 1412

Converted particles are stored in the output folder.

Step 4: Run Likelihood 
----------

Now we have all the inputs to run the likelihood calculations on a gpu node:
  .. code-block:: console

   (venv) $ sbatch run_likelihood.sh 1412


This job performs the likelihood calculation for each model indexed by $SLURM_ARRAY_TASK_ID to images in jobid=1412

  .. code-block:: console

   (venv) $ python3 likelihood_1mod_images.py $jobid $SLURM_ARRAY_TASK_ID tagtemplate



Note that when the number of models changes you have to change #SBATCH --array=0-49


Step 4: Collect Likelihood Matrix from output folders
-----------

 .. code-block:: console

   (venv) $ python3 
   
   
Further Analysis 
-----------

- compute log likelihood rotatio
- computes weights with ensemble reweighting 