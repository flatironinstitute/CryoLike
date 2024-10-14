# CryoLike Python library

This repo is dedicated to the development of a Python library for CryoLike, a fast and accurate algorithm for computing the likelihood of images given a cryo-EM map and atomic model. The algorithm is described in the following paper:
...

## Dependencies for Python verion

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
```
python3 -m venv ./venv/
source ./venv/bin/activate
pip install numpy scipy matplotlib tqdm mrcfile mdtraj
pip3 install torch torchvision torchaudio
pip install finufft
pip install cufinufft
```
the torch installation depends on your system, please refer to the official website for the installation guide https://pytorch.org/get-started/locally/.

## Installation

To install the Python library, clone the repository and run the following command in the root directory of the repository:
```
pip install -e .
```
To test the installation, run the following command:
```
pytest
```

## Usage

For usage of the CryoLike algorithm, please refer to the `examples` directory.

To run the examples, run the following command:
```
python3 make_templates_from_inputs_example.py
python3 convert_particle_stacks_example.py
python3 run_likelihood_example.py
```
which makes templates from the input files, converts particle stacks to templates, and runs the likelihood calculation, respectively. If the examples run successfully, you can find the output files in the `output` directory.

To plot the results, run the following command:
```
python3 plot_example.py
```
then you can find the plots in the `output/likelihood` directory.

## Data

Data used in the examples and test cases are retrieved from the following sources:
- EMPIAR-10026
- EMD: 2788
- PDB ID: 1UAO
- PDB ID: 4V1W

## TODO

- [ ] Add documentation for the library
- [ ] Add more test cases to cover all the functionalities
- [ ] Add more examples to demonstrate the usage of the library
- [ ] features to be added...