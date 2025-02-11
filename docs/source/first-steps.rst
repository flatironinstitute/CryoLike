First Steps
======================

CryoLike computes the likelihood of a match between cryo-EM image captures ("Images") and known pose configurations
("Templates").

Using CryoLike requires three steps:

- Create known-pose Templates from input data files (in ``mrc``/``mrcs``/``maps`` or ``pdb`` format)
- Create experimentally-captured Images from cryo-em radiographs (in ``CryoSparc`` or ``Starfile`` format)
- Run the cross-correlation between Templates and Images

CryoLike provides functions for each of these three tasks. The full range of options will be discussed in detail in
[LINK APPROPRIATE DETAILED DOCUMENTATION HERE].

This page walks you through running a few basic usage example scripts. The scripts themselves can be found
in the ``example`` direcvtory of the CryoLike repository. They operate on a small dataset that is distributed
with this repository in the ``example/data`` directory.


Example Usage
----------------

To try your first CryoLike processing, run the following scripts in order. Before running, ``cd`` into the
``example`` directory. If you installed CryoLike in a virtual environment, be sure to activate it first.
In the command lines below, we assume your environment's python interpreter is invoked through the
``python3`` command; this may vary slightly on your system.

.. code-block:: console

   (venv) $ python3 set_image_parameters_example.py
   (venv) $ python3 make_templates_from_inputs_example.py
   (venv) $ python3 convert_particle_stacks_example.py
   (venv) $ python3 run_likelihood_example.py

This runs four scripts.

- The first script creates an image parameters file and stores it in ``example/output/templates/parameters.npz``. This file
  stores information about the image dimensions and discretization grid used to interpret both template and image files. The
  parameters are needed for creating both template and image files. Template and image data need to have compatible dimensions
  and discretizations for a comparison to be well-founded.
- The second script reads the PDB files distributed with the repository in ``example/data/pdb``, converts them into
  Templates, and saves the result to a defined output directory ``example/output/templates/``.
- The third script reads particle stacks from ``particles.mrcs`` and ``particle_data.star`` (distributed in
  ``example/data/particles/``), converts them into Image stacks, and saves the result in a defined output
  directory ``example/output/particles/``.
- The final script compares the image stacks and templates, and records the likelihoods in the ``example/output/likelihood``
  directory tree.

If the examples run successfully, you can find all output files under the ``example/output`` directory.

Our example also includes a script to plot results. To create the plot, run the following command:

.. code-block:: console

   (venv) $ python3 plot_example.py

Output plots will also be located under the ``output/likelihood`` directory.


Data Sources
---------------

.. _data_sources:

Data used in the examples and test cases are retrieved from the following sources:

- EMPIAR-10026
- EMD: 2788
- PDB ID: 1UAO
- PDB ID: 4V1W

Detailed Tutorials
---------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   tutorials/read_star_file
   tutorials/read_cryosparc_file


