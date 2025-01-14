Usage
=====

.. _usage:

Example Usage
----------------

For an example usage of the CryoLike algorithm, please refer to the `example` directory.

To run the examples, run the following command:
.. code-block:: console

   (venv) $ python3 make_templates_from_inputs_example.py
   (venv) $ python3 convert_particle_stacks_example.py
   (venv) $ python3 run_likelihood_example.py

which makes templates from the input files, converts particle stacks to templates, and runs the likelihood calculation, respectively. If the examples run successfully, you can find the output files in the `output` directory.

To plot the results, run the following command:
.. code-block:: console

   (venv) $ python3 plot_example.py

then you can find the plots in the `output/likelihood` directory.

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


