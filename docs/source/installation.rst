Installation
============

.. _installation:
    :title: Installation

Before installing the library, make sure you have the following
dependencies installed:

- ``numpy``
- ``scipy``
- ``torch``: for GPU acceleration
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

Note that the torch installation might depend on
your system, please refer to the
``pytorch`` official website for the
`pytorch installation guide <https://pytorch.org/get-started/locally/>`_.
For detailed installation of ``finufft`` and ``cufinufft``, please refer to
`the official FINUFFT website <https://finufft.readthedocs.io/en/latest/>`_.
Some environments recommend using the ``--system-site-packages`` flag with
the virtual environment; if in doubt, consult your local system administrator.

Next, install the CryoLike Python library itself. Clone the repository and
run the following command in the ``/python`` directory of the repository:

.. code-block:: console

   (venv) $ pip install .

To test the installation, run the following commands:

.. code-block:: console

   (venv) $ pip install pytest pytest-cov
   (venv) $ pytest

Note that this requires installing ``pytest`` and ``pytest-cov``
in addition to the dependencies above; these are not necessary during
normal usage. If you wish to avoid running tests locally, you can
also just step through the examples distributed with the repository
(see the :doc:`examples section</examples/examples>`).
