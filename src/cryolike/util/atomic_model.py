import os
import numpy as np
import torch
from typing import Union

from .types import FloatArrayType
from .enums import Precision

_ATOMIC_RADIUS_RESNAME = {
    "CYS": 2.75, "PHE": 3.20, "LEU": 3.10, "TRP": 3.40,
    "VAL": 2.95, "ILE": 3.10, "MET": 3.10, "HIS": 3.05,
    "TYR": 3.25, "ALA": 2.50, "GLY": 2.25, "PRO": 2.80,
    "ASN": 2.85, "THR": 2.80, "SER": 2.60, "ARG": 3.30,
    "GLN": 3.00, "ASP": 2.80, "LYS": 3.20, "GLU": 2.95,
}

_DEFAULT_ATOM_RADII = 0.1


def _random_coordinates(
    n_source: int = 16,
    radius: float = 1.0,
    random_seed: int = 0,
    precision: Precision = Precision.DEFAULT
) -> torch.Tensor:
    torch.manual_seed(random_seed)
    torch_float_type, _, _ = precision.get_dtypes(Precision.SINGLE)
    atomic_coordinates = torch.randn((n_source, 3), dtype=torch_float_type) * radius
    delta_f_ = torch.norm(atomic_coordinates, dim=1)
    tmp_r = radius / np.sqrt(2)
    tmp_filter_ = delta_f_ > tmp_r
    if torch.any(tmp_filter_):
        atomic_coordinates[tmp_filter_, :] /= (
            delta_f_[tmp_filter_][:, None] * tmp_r
        )
    return atomic_coordinates.unsqueeze(0)  # Add a frame dimension


class AtomicModel:
    """Class representing a particle model based on known atomic/protein residue positions.

    Attributes:
        atomic_coordinates (np.ndarray): Locations of the atoms in the model.
        atom_radii (np.ndarray): Size of each atom in the model.
        pdb_file (str): The path to the file from which this model was loaded, if any.
            (Otherwise empty string.)
        box_size (float): The side length of the (square) viewing box in which the atoms
            reside.
    """
    atomic_coordinates: torch.Tensor
    atom_radii: torch.Tensor
    top_file: str
    trj_file: str
    box_size: float
    n_frames: int
    n_atoms: int

    def __init__(
        self,
        atomic_coordinates: torch.Tensor | FloatArrayType | None = None,
        atom_radii: torch.Tensor | FloatArrayType | float | None = None,
        box_size: float | None = None,
        precision: Precision = Precision.DEFAULT,
    ) -> None:
        """Constructor for atomic/protein residue particle model.

        Args:
            atomic_coordinates (np.ndarray | None, optional): If set, contains the coordinates
                of the atoms in the model. First index is the index of the atom, second
                index is a 3-vector of the (x, y, z) position. If None, random coordinates
                will be generated; this is intended mainly for testing.
            atom_radii (Union[np.ndarray, float], optional): Radii of the atoms (in Angstroms),
                either as an array indexed by atom, or a scalar value applied to all
                atoms. Defaults to 0.1.
            box_size (float | None, optional): Side length of (square) viewing box for the
                atomic model. Defaults to 2.0.
        """
        self.torch_float_type, _, _ = precision.get_dtypes(Precision.SINGLE)
        self.precision = precision
        self.set_atomic_coordinates(atomic_coordinates)
        self.set_atom_radii(atom_radii)
        self.set_box_size(box_size)
        self.check_inbound()

        self.top_file = ""
        self.trj_file = ""


    def set_atomic_coordinates(self, atomic_coordinates: torch.Tensor | FloatArrayType | None) -> None:
        if atomic_coordinates is None:
            print("Atomic coordinates not specified. Using default random set of coordinates.")
            self.atomic_coordinates = _random_coordinates(radius = self.box_size / 4, precision=self.precision)
        elif isinstance(atomic_coordinates, torch.Tensor):
            self.atomic_coordinates = atomic_coordinates
        else: # assume numpy array
            self.atomic_coordinates = torch.tensor(atomic_coordinates, dtype=self.torch_float_type)
        if self.atomic_coordinates.ndim == 2:
            # If 2D, assume a single frame and add a frame dimension
            self.atomic_coordinates = self.atomic_coordinates.unsqueeze(0)
        self.n_frames = self.atomic_coordinates.shape[0]
        self.n_atoms = self.atomic_coordinates.shape[1]
        assert isinstance(self.atomic_coordinates, torch.Tensor)
        assert self.atomic_coordinates.ndim == 3, "Atomic coordinates must be a 3D array with shape (n_frames, n_atoms, 3)."
        assert self.atomic_coordinates.shape[2] == 3, "Atomic coordinates must have shape (n_frames, n_atoms, 3)."


    def set_box_size(self, box_size: float | None) -> None:
        if box_size is None:
            print("Box size not specified. Using default box_size = 2.0.")
            box_size = 2.0
        assert isinstance(box_size, float) and box_size > 0, "Box size must be a positive float."
        self.box_size = box_size


    def set_atom_radii(self, atom_radii: torch.Tensor | FloatArrayType | float | None) -> None:
        if np.isscalar(atom_radii):
            assert isinstance(atom_radii, float)
            self.atom_radii = torch.ones(self.n_atoms) * float(atom_radii)
        elif isinstance(atom_radii, np.ndarray):
            self.atom_radii = torch.tensor(atom_radii, dtype=self.torch_float_type)
        elif isinstance(atom_radii, torch.Tensor):
            self.atom_radii = atom_radii
        assert isinstance(self.atom_radii, torch.Tensor)
        assert self.atom_radii.ndim == 1, "Atomic radii must be an 1D array."
        assert self.atom_radii.shape[0] == self.n_atoms, "Atomic radii must match number of atoms."


    def check_inbound(self) -> None:
        """Check if all atomic coordinates and radii are within the defined box."""
        assert isinstance(self.atomic_coordinates, torch.Tensor)
        assert self.atomic_coordinates.ndim == 3, "Atomic coordinates must be a 3D array."
        assert torch.all(self.atomic_coordinates >= -self.box_size / 2), "Atomic coordinates must be greater than or equal to -box_size/2."
        assert torch.all(self.atomic_coordinates <= self.box_size / 2), "Atomic coordinates must be less than or equal to box_size/2."
        assert isinstance(self.atom_radii, torch.Tensor)
        assert self.atom_radii.ndim == 1, "Atomic radii must be a 1D array."
        assert self.atom_radii.shape[0] == self.n_atoms, "Atomic radii must match number of atoms."
        assert torch.all(self.atom_radii > 0), "Atomic radii must be positive."
        assert torch.all(self.atom_radii < self.box_size / 2), "Atomic radii must be less than box_size/2."


    @classmethod
    def read_from_traj(cls,
        top_file: str,
        trj_file: str = "",
        stride: int = 1,
        in_nanometer: bool = True,
        box_size: float | None = None,
        atom_radii: torch.Tensor | FloatArrayType | float | None = None,
        atom_selection: str | None = None,
        centering: bool = False,
        use_protein_residue_model: bool = True,
        precision: Precision = Precision.DEFAULT
    ):
        """Build an atomic model from a trajectory file.

        Args:
            top_file (str): Path to the topology file (PDB or other) to load
            trj_file (str): Path to the trajectory file to load
            box_size (float | None, optional): Size of the viewing box. If None (the default),
                a default box size defined in the AtomicModel constructor will be used.
            atom_radii (Union[np.ndarray, float], optional): Radii of the atoms, either
                as a per-atom array of values or a single value for all atoms. Defaults to 0.1.
            atom_selection (str | None, optional): Which atoms to choose from the model.
                If using a protein residue model, will be set automatically. Otherwise,
                it should be a valid index of the PDB file's Topology. Defaults to None.
            centering (bool, optional): Whether to center the coordinates to a zero mean.
                Defaults to True.
            use_protein_residue_model (bool, optional): If True, will use the 'name CA'
                atom selection and read atomic radii from known amino acid sizes.
                Defaults to True.

        Returns:
            AtomicModel: Instantiated atomic model from the PDB file.
        """
        torch_float_type, _, _ = precision.get_dtypes(Precision.SINGLE)
        from mdtraj import load, Trajectory, Topology
        assert os.path.exists(top_file), f"Topology file {top_file} does not exist."
        if trj_file == "":
            u = load(top_file, frame=0)
            _atomic_coordinates = torch.tensor(u.xyz, dtype=torch_float_type)
        else:
            assert os.path.exists(trj_file), f"Trajectory file {trj_file} does not exist."
            u = load(trj_file, top=top_file, stride=stride)
            _atomic_coordinates = torch.tensor(u.xyz, dtype=torch_float_type)
        if _atomic_coordinates.ndim == 2:
            # If 2D, assume a single frame and add a frame dimension
            _atomic_coordinates = _atomic_coordinates.unsqueeze(0)
        if atom_radii is None:
            assert use_protein_residue_model, "If atom_radii is None, use_protein_residue_model must be True."
            atom_selection = "name CA"
            res = u.topology.residue
            n_residues = u.topology.n_residues
            assert isinstance(n_residues, int) and n_residues > 0, "Number of residues must be a positive integer."
            _atom_radii = torch.zeros(n_residues, dtype=torch_float_type)
            for i in range(n_residues):
                resname = res(i).name
                _atom_radii[i] = _ATOMIC_RADIUS_RESNAME.get(resname, 3.0)
            print("atomic radii = ", _atom_radii)
        else:
            _atom_radii = atom_radii
        assert _atom_radii is not None, "Atomic radii must be specified."
        if atom_selection is not None:
            indices = u.topology.select(atom_selection)
            if len(indices) == 0:
                raise ValueError("No atoms selected.")
            _atomic_coordinates = _atomic_coordinates[:,indices,:]
        if in_nanometer:
            _atomic_coordinates = _atomic_coordinates * 10.0
        print(f"Atomic coordinates shape: {_atomic_coordinates.shape}")
        assert _atomic_coordinates.ndim == 3, "Atomic coordinates must be a 3D array with shape (n_frames, n_atoms, 3)."

        if centering:
            _atomic_coordinates -= torch.mean(_atomic_coordinates, dim=1, keepdim=True)
        atomic_model = cls(_atomic_coordinates, _atom_radii, box_size, precision)
        atomic_model.top_file = top_file
        atomic_model.trj_file = trj_file
        return atomic_model


    @classmethod
    def clone(cls, atomic_model):
        return cls(
            atomic_coordinates=atomic_model.atomic_coordinates.clone(),
            atom_radii=atomic_model.atom_radii.clone(),
            box_size=atomic_model.box_size,
            precision=atomic_model.precision
        )
    

    def repeat_frames(self, n_frames: int) -> "AtomicModel":
        """Repeat the atomic model frames to create a new model with n_frames.

        Args:
            n_frames (int): Number of frames to repeat.

        Returns:
            AtomicModel: New atomic model with repeated frames.
        """
        if n_frames <= 0:
            raise ValueError("n_frames must be a positive integer.")
        new_atomic_coordinates = self.atomic_coordinates.repeat(n_frames, 1, 1)
        return AtomicModel(
            atomic_coordinates=new_atomic_coordinates,
            atom_radii=self.atom_radii.clone(),
            box_size=self.box_size,
            precision=self.precision
        )