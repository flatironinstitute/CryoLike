Usage
=====

.. _dependencies:

Dependencies
------------

The Python library requires the following dependencies:

- numpy
- scipy
- matplotlib    ## for plotting
- pytorch       ## for GPU acceleration
- finufft       ## for non-uniform FFT (https://finufft.readthedocs.io/en/latest/)
- cufinufft     ## GPU version of finufft
- tqdm          ## for progress bar
- mrcfile       ## for reading MRC files
- mdtraj        ## for reading PDB files and atom selection

If you are using virtual environment,
.. code-block:: console
   python3 -m venv ./venv/
   source ./venv/bin/activate
   pip install numpy scipy matplotlib tqdm mrcfile mdtraj
   pip3 install torch torchvision
   pip install finufft
   pip install cufinufft

Note that the torch installation might depend on your system, please refer to the pytorch official website for the installation guide https://pytorch.org/get-started/locally/.
For detailed installation of finufft and cufinufft, please refer to the official website https://finufft.readthedocs.io/en/latest/.

.. _installation:

Installation
------------

To install the CryoLike Python library, clone the repository and run the following command in the root directory of the repository:
.. code-block:: console
   pip install .

To test the installation, run the following commands:
.. code-block:: console
   pip install pytest
   pytest

.. _usage:

Usage
----------------

For usage of the CryoLike algorithm, please refer to the `example` directory.

To run the examples, run the following command:
.. code-block:: console
   python3 make_templates_from_inputs_example.py
   python3 convert_particle_stacks_example.py
   python3 run_likelihood_example.py

which makes templates from the input files, converts particle stacks to templates, and runs the likelihood calculation, respectively. If the examples run successfully, you can find the output files in the `output` directory.

To plot the results, run the following command:
.. code-block:: console
   python3 plot_example.py
then you can find the plots in the `output/likelihood` directory.

.. _data_sources:
Data Sources
------------

Data used in the examples and test cases are retrieved from the following sources:
- EMPIAR-10026
- EMD: 2788
- PDB ID: 1UAO
- PDB ID: 4V1W