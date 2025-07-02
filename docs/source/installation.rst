Installation
============

.. _installation:
    :title: Installation

Before installing the library, make sure you have the following
dependencies installed:

- ``numpy``
- ``scipy``
- ``pytorch``: for GPU acceleration
- ``matplotlib``: for plotting
- ``finufft``: for non-uniform FFT
- ``cufinufft``: GPU version of finufft
- ``tqdm``: for progress bar
- ``mrcfile``: for reading MRC files
- ``mdtraj``: for reading PDB files and atom selection
- ``pydantic``: for data validation
- ``starfile``: for reading/writing starfiles

If you are using a ``venv`` virtual environment:

.. code-block:: console

   $ python3 -m venv ./venv/
   $ source ./venv/bin/activate
   (venv) $ pip install numpy scipy matplotlib tqdm mrcfile mdtraj pydantic starfile
   (venv) $ pip install torch torchvision
   (venv) $ pip install finufft
   (venv) $ pip install cufinufft

Note that the torch installation might depend on your system, please refer to the
pytorch official website for the `pytorch installation guide <https://pytorch.org/get-started/locally/>`_.
For detailed installation of finufft and cufinufft, please refer to `the official FINUFFT website <https://finufft.readthedocs.io/en/latest/>`_.

To install the CryoLike Python library itself, clone the repository and
run the following command in the ``/python`` directory of the repository:

.. code-block:: console

   (venv) $ pip install .

To test the installation, run the following commands:

.. code-block:: console

   (venv) $ pip install pytest pytest-cov
   (venv) $ pytest

Note that this requires installing ``pytest`` and ``pytest-cov``
in addition to the dependencies above.
