Installation
=====

.. _installation:
    :title: Installation

Before installing the library, make sure you have the following dependencies installed:

The Python library requires the following dependencies:

- numpy
- scipy
- matplotlib: for plotting
- pytorch: for GPU acceleration
- finufft: for non-uniform FFT (https://finufft.readthedocs.io/en/latest/)
- cufinufft: GPU version of finufft
- tqdm: for progress bar
- mrcfile: for reading MRC files
- mdtraj: for reading PDB files and atom selection
- pydantic: for data validation

If you are using virtual environment,
.. code-block:: console

   $ python3 -m venv ./venv/
   $ source ./venv/bin/activate
   (venv) $ pip install numpy scipy matplotlib tqdm mrcfile mdtraj pydantic
   (venv) $ pip3 install torch torchvision
   (venv) $ pip install finufft
   (venv) $ pip install cufinufft

Note that the torch installation might depend on your system, please refer to the pytorch official website for the installation guide https://pytorch.org/get-started/locally/.
For detailed installation of finufft and cufinufft, please refer to the official website https://finufft.readthedocs.io/en/latest/.

To install the CryoLike Python library, clone the repository and run the following command in the root directory of the repository:
.. code-block:: console

   (venv) $ pip install .

To test the installation, run the following commands:
.. code-block:: console

   (venv) $ pip install pytest
   (venv) $ pytest