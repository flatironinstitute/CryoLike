import os
import numpy as np

from typing import NamedTuple, cast

from cryolike.util import (
    AtomShape,
    FloatArrayType,
    IntArrayType,
    Precision,
    project_descriptor,
    TargetType,
)

class ParsedParameters(NamedTuple):
    """Class representing metadata parameters describing an image set.

    Attributes:
        n_voxels (IntArrayType): Number of voxels in each dimension
        voxel_size (FloatArrayType): Size of each dimension of
            each voxel, in Angstroms
        box_size (FloatArrayType): Bounds of the space in which the
            images reside
        radius_max (float): Maximum radius of shells in a (Fourier-space)
            polar grid
        dist_radii (float): Distance between two radial shells of a
            (Fourier-space) polar grid
        n_inplanes (int): Number of points in each radial shell
        precision (Precision): Single or Double precision, as enum
        viewing_distance (float): Angular distance between two viewing
            angles. Defaults to 1.0 / (4.0 * np.pi).
        atom_radii (float | None): Radii of atoms in an atomic model.
            Defaults to None.
        atom_selection (str | None): Selected atoms from atomic model.
            Defaults to None.
        use_protein_residue_model (bool): Whether to interpret an
            atomic model using standard protein residue sizes.
            Defaults to True.
        atom_shape (AtomShape): How to interpret atoms in an atomic
            model. Defaults to "default" which will be interpreted
            appropriately by downstream applications.
    """
    n_voxels: IntArrayType
    voxel_size: FloatArrayType
    box_size: FloatArrayType
    radius_max: float
    dist_radii: float
    n_inplanes: int
    precision: Precision = Precision.DEFAULT
    viewing_distance: float = 1.0 / (4.0 * np.pi)
    atom_radii: float | None = None
    atom_selection: str | None = None
    use_protein_residue_model: bool = True
    atom_shape: AtomShape = AtomShape.DEFAULT


def ensure_parameters(params: str | ParsedParameters) -> ParsedParameters:
    """Given an input, ensures that it is a valid
    set of parameters, or an on-disk representation
    of a valid set of parameters.

    Args:
        params (str | ParsedParameters): Either a file path
            to an on-disk representation of parameters, or
            a valid parameter object.

    Raises:
        ValueError: If the parameters are not in fact valid.

    Returns:
        ParsedParameters: The input parameter set, as a valid
            object.
    """
    if type(params) == str:
        params = load_parameters(params)
    if type(params) != ParsedParameters:
        raise ValueError("Error: Invalid parameters")
    return params


def parse_parameters(
    n_voxels: int,
    voxel_size: float,
    resolution_factor: float = 1.0,
    precision: str | Precision = 'single',
    viewing_distance: float | None = None,
    n_inplanes: int | None = None,
    atom_radii: float | None = None,
    atom_selection: str = "name CA",
    use_protein_residue_model: bool = True,
    atom_shape: str | AtomShape = "gaussian",
) -> ParsedParameters:
    """Given a set of individual parameter values, convert into a formal
    object.

    Args:
        n_voxels (int): Number of voxels in each dimension. (The volume described
            is assumed to have the same number of voxels in each dimension.)
        voxel_size (float): Size of each dimension of each (cubic) voxel,
            in Angstroms.
        resolution_factor (float, optional): 1.0 (the default) for full resolution at
            Nyquist frequency, or 0.5 for half resolution at Nyquist frequency.
        precision (str | Precision, optional): Whether to use single or double precision.
            Defaults to 'single'.
        viewing_distance (float | None, optional): Angular distance between two
            viewing angles. If unset, will default to 1/(4pi) over the resolution
            factor.
        n_inplanes (int | None, optional): Number of points in each radial shell.
            If unset, will default to twice the number of voxels.
        atom_radii (float | None, optional): Radius of atoms, in Angstroms. Must be
            set unless the model is treated as a protein residue model.
        atom_selection (str, optional): Which atoms to use from the model file.
            Defaults to "name CA" (which assumes a protein residue model).
        use_protein_residue_model (bool, optional): If True (the default), we assume
            the parameters describe a PDB input and set several interpretation values
            accordingly. See the AtomicModel class for more details.
        atom_shape (str | AtomShape, optional): Type of topology for modeling atomic
            density. Defaults to "gaussian".

    Returns:
        ParsedParameters: A complete parameter record, with defaults populated.
    """
    if atom_radii is None and not use_protein_residue_model:
        raise ValueError("Atom radii must be set")
    if type(precision) == str:
        if precision == 'single':
            precision = Precision.SINGLE
        elif precision == 'double':
            precision = Precision.DOUBLE
        else:
            raise ValueError("Invalid precision value")
    if type(atom_shape) == str:
        if atom_shape == 'hard-sphere':
            atom_shape = AtomShape.HARD_SPHERE
        elif atom_shape == 'gaussian':
            atom_shape = AtomShape.GAUSSIAN
        else:
            raise ValueError("Invalid atom shape value")
    assert isinstance(precision, Precision)
    assert isinstance(atom_shape, AtomShape)
    box_size = n_voxels * voxel_size
    radius_max = n_voxels * np.pi / 2.0 / (2.0 * np.pi) * resolution_factor
    dist_radii = np.pi / 2.0 / (2.0 * np.pi)
    if n_inplanes is None:
        n_inplanes = n_voxels * 2
    assert n_inplanes is not None
    if viewing_distance is None:
        viewing_distance = 1.0 / (4.0 * np.pi) / resolution_factor
    assert viewing_distance is not None
    # _box_size = cast(FloatArrayType, project_descriptor(box_size, "box_size", 3, TargetType.FLOAT))
    # _n_voxels = cast(IntArrayType, project_descriptor(n_voxels, "n_voxels", 3, TargetType.INT))
    # _voxel_size = cast(FloatArrayType, project_descriptor(voxel_size, "voxel_size", 3, TargetType.FLOAT))
    _n_voxels = int(n_voxels)
    _voxel_size = float(voxel_size)
    _box_size = float(box_size)
    return ParsedParameters(
        n_voxels = _n_voxels,
        voxel_size = _voxel_size,
        box_size = _box_size,
        radius_max = radius_max,
        dist_radii = dist_radii,
        n_inplanes = n_inplanes,
        precision = precision,
        viewing_distance = viewing_distance,
        atom_radii = atom_radii,
        atom_selection = atom_selection,
        use_protein_residue_model = use_protein_residue_model,
        atom_shape = atom_shape,
    )


def save_parameters(parameters: ParsedParameters, file_output: str):
    """Persist a set of parameters to a file.

    Args:
        parameters (ParsedParameters): The parameters to save
        file_output (str): Name of the file to write to. Will be overwritten
            if it already exists.
    """
    np.savez_compressed(
        file_output,
        n_voxels = parameters.n_voxels,
        voxel_size = parameters.voxel_size,
        box_size = parameters.box_size,
        radius_max = parameters.radius_max,
        dist_radii = parameters.dist_radii,
        n_inplanes = parameters.n_inplanes,
        precision = parameters.precision.value,
        viewing_distance = parameters.viewing_distance,
        atom_radii = '' if parameters.atom_radii is None else parameters.atom_radii,
        atom_selection = '' if parameters.atom_selection is None else parameters.atom_selection,
        use_protein_residue_model = parameters.use_protein_residue_model, # use residue model
        atom_shape = parameters.atom_shape.value,
    )


def load_parameters(file_output: str) -> ParsedParameters:
    """Loads a previously saved set of parameters from the specified .npz file.

    Args:
        file_output (str): Path to NPZ file to load

    Returns:
        ParsedParameters: Loaded parameter object from file.
    """
    if isinstance(file_output, str):
        if not os.path.exists(file_output):
            raise ValueError("Error: file not found: ", file_output)
        if not file_output.endswith('.npz'):
            raise ValueError("Error: invalid file format: ", file_output, ", expecting .npz")
    with np.load(file_output, allow_pickle=True) as data:
        _radii = data['atom_radii']
        radii = None if _radii == '' else cast(float, _radii)
        _selection = data['atom_selection']
        selection = None if _selection == '' else _selection
        if isinstance(data['precision'], str):
            precision = Precision(data['precision'])
        elif isinstance(data['precision'], Precision):
            precision = data['precision']
        else:
            precision = Precision.DEFAULT
            print("Warning: Invalid precision value, using default")
        if isinstance(data['atom_shape'], str):
            atom_shape = AtomShape(data['atom_shape'])
        elif isinstance(data['atom_shape'], AtomShape):
            atom_shape = data['atom_shape']
        else:
            atom_shape = AtomShape.DEFAULT
            print("Warning: Invalid atom shape value, using default")

        return ParsedParameters(
            n_voxels = cast(IntArrayType, data['n_voxels']),
            voxel_size = cast(FloatArrayType, data['voxel_size']),
            box_size = cast(FloatArrayType, data['box_size']),
            radius_max = cast(float, data['radius_max']),
            dist_radii = cast(float, data['dist_radii']),
            n_inplanes = cast(int, data['n_inplanes']),
            precision = precision,
            viewing_distance = cast(float, data['viewing_distance']),
            atom_radii = radii,
            atom_selection = selection,
            use_protein_residue_model = cast(bool, data['use_protein_residue_model']),
            atom_shape = atom_shape
        )


def print_parameters(parameters: ParsedParameters) -> None:
    """Prints a set of parameters to the command line.

    Args:
        parameters (ParsedParameters): The parameters to print.
    """
    print("Parameters:")
    print("n_voxels:", parameters.n_voxels)
    print("voxel_size:", parameters.voxel_size)
    print("box_size:", parameters.box_size)
    print("radius_max:", parameters.radius_max)
    print("dist_radii:", parameters.dist_radii)
    print("n_inplanes:", parameters.n_inplanes)
    print("precision:", parameters.precision)
    print("viewing_distance:", parameters.viewing_distance)
    print("atom_radii:", parameters.atom_radii)
    print("atom_selection:", parameters.atom_selection)
    print("use_protein_residue_model:", parameters.use_protein_residue_model)
    print("atom_shape:", parameters.atom_shape)
