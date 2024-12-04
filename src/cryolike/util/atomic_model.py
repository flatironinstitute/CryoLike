import numpy as np
from typing import Union

from .types import FloatArrayType

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
) -> FloatArrayType:
    np.random.seed(random_seed)
    atomic_coordinates = np.random.normal(0, 1, (n_source, 3)) * radius
    delta_f_ = np.linalg.norm(atomic_coordinates, axis = 1)
    tmp_r = radius / np.sqrt(2)
    tmp_filter_ = delta_f_ > tmp_r
    if np.any(tmp_filter_):
        atomic_coordinates[tmp_filter_, :] = atomic_coordinates[tmp_filter_, :] / delta_f_[tmp_filter_][:,None] * tmp_r
    return atomic_coordinates


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
    atomic_coordinates: FloatArrayType
    atom_radii: np.ndarray
    pdb_file: str
    box_size: float


    def __init__(
        self,
        atomic_coordinates: FloatArrayType | None = None,
        atom_radii: Union[np.ndarray, float] = 0.1,
        box_size: float | None = None
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
        if box_size is None:
            print("Box size not specified. Using default box_size = 2.0.")
            box_size = 2.0
        # if not np.isscalar(box_size):
        #     ## not supported yet
        #     box_size = box_size[0] ## TODO: replace with proper handling
        self.box_size = float(box_size)
        if atomic_coordinates is None:
            # raise ValueError("Atomic coordinates or pdb_file must be specified.")
            print("Atomic coordinates or pdb_file not specified. Using default random set of coordinates.")
            atomic_coordinates = _random_coordinates(radius = self.box_size / 4)
        self.atomic_coordinates = atomic_coordinates
        self.n_atoms = self.atomic_coordinates.shape[0]

        if np.isscalar(atom_radii):
            assert isinstance(atom_radii, float)
            self.atom_radii = np.ones(self.n_atoms) * float(atom_radii)
        else:
            assert isinstance(atom_radii, np.ndarray)
            self.atom_radii = atom_radii
        self.pdb_file = ""
        self.check_model()

    
    def set_atom_radii(self, atom_radii: np.ndarray) -> None:
        self.atom_radii = atom_radii
        if np.issubdtype(type(self.atom_radii), np.integer) or np.issubdtype(type(self.atom_radii), np.floating):
            self.atom_radii = np.ones(self.n_atoms, dtype=np.float32) * self.atom_radii
        self.check_model()


    def check_model(self) -> None:
        if self.atomic_coordinates is None:
            print("Atomic coordinates not specified. Defaulting to randomly chosen coordinates.")
            self.atomic_coordinates = _random_coordinates(radius = self.box_size / 4)
        if len(self.atomic_coordinates.shape) != 2:
            raise ValueError("Atomic coordinates must be a 2D array.")
        if self.atomic_coordinates.shape[1] != 3:
            raise ValueError("atomic_coordinates.shape[1] != 3")
        self.n_atoms = self.atomic_coordinates.shape[0]
        ### Add check if atomic coordinates are within the box_size
        ### ...
        ###
        if self.atom_radii is None:
            # NOTE: This is different from the default in the constructor
            print("Atomic radii not specified. Using default radii = 3.0.")
            self.atom_radii = np.ones(self.n_atoms) * 3.0
        if isinstance(self.atom_radii, np.ndarray):
            if self.atom_radii.size != self.n_atoms:
                raise ValueError("Number of atomic radii does not match the number of atomic coordinates.")
            if np.any(self.atom_radii < 0):
                raise ValueError("Atomic radii must be greater than 0.")
        else:
            if np.issubdtype(type(self.atom_radii), np.integer) or np.issubdtype(type(self.atom_radii), np.floating):
                self.set_atom_radii(self.atom_radii)
            else:
                raise ValueError("Atomic radii must be a numpy array.")


    @classmethod
    def read_from_pdb(cls,
        pdb_file: str,
        box_size: float | None = None,
        atom_radii: Union[np.ndarray, float] = 0.1,
        atom_selection: str | None = None,
        centering: bool = True,
        use_protein_residue_model: bool = True,
    ):
        """Build an atomic model from a PDB file.

        Args:
            pdb_file (str): Path to the PDB file to load
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
        from mdtraj import load_pdb, Trajectory, Topology
        u: Trajectory = load_pdb(pdb_file, frame=0, no_boxchk=True)
        assert u.xyz is not None
        positions = u.xyz[0,:,:] * 10.0 ## convert from nanometer to Angstrom
        if use_protein_residue_model:
            atom_selection = "name CA"
            assert isinstance(u.topology, Topology)
            res = u.topology.residue
            atom_radii = np.zeros(u.topology.n_residues, dtype = np.float32)
            for i in range(u.topology.n_residues):
                resname = res(i).name
                atom_radii[i] = _ATOMIC_RADIUS_RESNAME.get(resname, 3.0)    
            print("atomic radii = ", atom_radii)
        if atom_selection is not None:
            assert isinstance(u.topology, Topology)
            indices = u.topology.select(atom_selection)
            if len(indices) == 0:
                raise ValueError("No atoms selected.")
            atomic_coordinates = positions[indices,:]
        else:
            atomic_coordinates = positions
        atomic_coordinates = np.array(atomic_coordinates, dtype = np.float32)
        if centering:
            atomic_coordinates = atomic_coordinates - np.mean(atomic_coordinates, axis = -2)
        instance = cls(atomic_coordinates, atom_radii, box_size)
        instance.pdb_file = pdb_file
        return instance
