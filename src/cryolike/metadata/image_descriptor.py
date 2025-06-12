import numpy as np
from typing import Literal
from pydantic import BaseModel, ConfigDict

from cryolike.util import (
    AtomShape,
    FloatArrayType,
    IntArrayType,
    Precision,
    project_descriptor,
    TargetType,
    save_descriptors,
    load_file,
    extract_unique_float,
    extract_unique_str
)
from cryolike.grids import CartesianGrid2D, PolarGrid
from .viewing_angles import ViewingAngles, SerializedViewingAngles


# TODO: Support for non-uniform grids and viewing angle sets

class SerializedImageDescriptor(BaseModel):
    n_pixels: IntArrayType | int
    pixel_size: FloatArrayType | float
    radius_max: float
    dist_radii: float
    n_inplanes: int
    precision: Precision
    viewing_distance: float | None
    viewing_angles: SerializedViewingAngles | None
    atom_radii: float | FloatArrayType | None
    atom_selection: str | np.ndarray | None
    use_protein_residue_model: bool
    atom_shape: AtomShape

    model_config = ConfigDict(use_enum_values=True, arbitrary_types_allowed=True)

    # TODO: May need some custom code for handling
    # enums to avoid needing use_pickle = True
    # TODO: Serialization currently assumes regularity, i.e.
    # uniform polar grids and square/cubic cartesian pixels/boxes.


class ImageDescriptor():
    precision: Precision
    cartesian_grid: CartesianGrid2D
    polar_grid: PolarGrid
    viewing_angles: ViewingAngles
    viewing_distance: float | None
    atom_radii: float | None
    atom_selection: str | None
    use_protein_residue_model: bool
    atom_shape: AtomShape


    def __init__(
        self,
        precision: Precision,
        cartesian_grid: CartesianGrid2D,
        polar_grid: PolarGrid,
        viewing_angles: ViewingAngles | None = None,
        viewing_distance: float | None = None,
        atom_radii: float | FloatArrayType | None = None,
        atom_selection: str | np.ndarray | None = "name CA",
        use_protein_residue_model: bool = True,
        atom_shape: AtomShape = AtomShape.GAUSSIAN
    ):
        self.precision = Precision.from_str(precision)
        self.cartesian_grid = cartesian_grid
        self.polar_grid = polar_grid
        if (viewing_angles is None):
            if (viewing_distance is None):
                raise ValueError("One of viewing angles and viewing distance must be set.")
            self.viewing_angles = ViewingAngles.from_viewing_distance(viewing_distance)
            self.viewing_distance = viewing_distance
        else:
            self.viewing_angles = viewing_angles
            self.viewing_distance = None
        self.atom_radii = None if atom_radii is None else extract_unique_float(atom_radii, "atom radii")
        self.atom_selection = None if atom_selection is None else extract_unique_str(atom_selection, "atom selection")
        self.use_protein_residue_model = use_protein_residue_model
        self.atom_shape = AtomShape.from_str(atom_shape)


    def get_3d_box_size(self):
        curr_size = self.cartesian_grid.box_size
        if curr_size[0] != curr_size[1]:
            raise NotImplementedError("Need 3rd dimension to make 3d box size if we can't assume cubic")
        return project_descriptor(curr_size[0], "3d box size", 3, TargetType.FLOAT)


    def is_compatible_with_imagestack(self, other: 'Images') -> bool: # type: ignore
        assert self.polar_grid is not None
        assert self.cartesian_grid is not None

        if other.polar_grid is not None:
            if (self.polar_grid.radius_max != other.polar_grid.radius_max
                or self.polar_grid.dist_radii != other.polar_grid.dist_radii
                or self.polar_grid.n_inplanes != other.polar_grid.n_inplanes
            ):
                return False
        if other.phys_grid is not None:
            if (not np.array_equal(self.cartesian_grid.n_pixels, other.phys_grid.n_pixels)
                or not np.array_equal(self.cartesian_grid.pixel_size, other.phys_grid.pixel_size)
            ):
                return False
        return True


    def is_compatible_with(self, other: 'ImageDescriptor') -> bool:
        # TODO: Expand on this
        # Currently, compatibility is defined as "uses the same grids".
        if self.polar_grid is not None and other.polar_grid is not None:
            if (self.polar_grid.radius_max != other.polar_grid.radius_max
                or self.polar_grid.dist_radii != other.polar_grid.dist_radii
                or self.polar_grid.n_inplanes != other.polar_grid.n_inplanes
            ):
                return False
        if self.cartesian_grid is not None and other.cartesian_grid is not None:
            if (not np.array_equal(self.cartesian_grid.n_pixels, other.cartesian_grid.n_pixels)
                or not np.array_equal(self.cartesian_grid.pixel_size, other.cartesian_grid.pixel_size)
            ):
                return False
        return True
    

    def is_compatible_with_pdb(self):
        if self.atom_radii is not None:
            return self.atom_radii > 0
        return self.use_protein_residue_model


    def serialize(self) -> SerializedImageDescriptor:
        if self.viewing_distance is None:
            raise NotImplementedError("Image descriptor with no explicit viewing distance cannot be serialized.")
        return SerializedImageDescriptor(
            n_pixels=self.cartesian_grid.n_pixels,
            pixel_size=self.cartesian_grid.pixel_size,
            radius_max=self.polar_grid.radius_max,
            dist_radii=self.polar_grid.dist_radii,
            n_inplanes=self.polar_grid.n_inplanes,
            precision=self.precision,
            viewing_distance=self.viewing_distance,
            viewing_angles=self.viewing_angles.serialize() if self.viewing_angles is not None else None,
            atom_radii=self.atom_radii,
            atom_selection=self.atom_selection,
            use_protein_residue_model=self.use_protein_residue_model,
            atom_shape=self.atom_shape
        )


    def to_dict(self):
        return self.serialize().model_dump()


    def save(self, filename: str, overwrite: bool = False):
        """Create NPZ file named `filename` from the to_dict() representations of
        the data in *self*.

        Args:
            filename (str): NPZ file to use as output. Operation will be canceled if
                this named file already exists unless *overwrite* is **True**

        Kwargs:
            overwrite (bool): whether to allow overwriting existing files. Default False

        Raises:
            ValueError: If the requested output filename already exists unless *overwrite* is **True**
        """
        save_descriptors(filename, self.to_dict(), overwrite=overwrite)


    @classmethod
    def load(cls, filename: str):
        return cls.from_dict(load_file(filename))


    def print(self):
        """Prints a set of parameters to the command line.
        """
        # TODO: probably ought to implement this in terms of the serialized dict
        # Or use Pydantic's features
        print("Parameters:")
        print("n_voxels:", self.cartesian_grid.n_pixels)
        print("voxel_size:", self.cartesian_grid.pixel_size)
        print("box_size:", self.cartesian_grid.box_size)
        print("radius_max:", self.polar_grid.radius_max)
        print("dist_radii:", self.polar_grid.dist_radii)
        print("n_inplanes:", self.polar_grid.n_inplanes)
        print("precision:", self.precision)
        print("viewing_distance:", self.viewing_distance)
        print("atom_radii:", self.atom_radii)
        print("atom_selection:", self.atom_selection)
        print("use_protein_residue_model:", self.use_protein_residue_model)
        print("atom_shape:", self.atom_shape)


    @classmethod
    def from_individual_values(
        cls,
        n_pixels: int,
        pixel_size: float,
        precision: Literal['single'] | Literal['double'] | Precision = Precision.SINGLE,
        resolution_factor: float = 1.0,
        viewing_distance: float | None = None,
        n_inplanes: int | None = None,
        atom_radii: float | None = None,
        atom_selection: str | None = "name CA",
        use_protein_residue_model: bool = True,
        atom_shape: AtomShape | str = AtomShape.GAUSSIAN
    ):
        """Create an ImageDescriptor from a small set of values.
        This will produce a square Cartesian grid and uniform polar grid.
        Only the Cartesian-grid dimensions are required: other values will be
        set to defaults if not provided.
        As a reminder, values other than the grids have no effect on Images,
        and only impact Templates during the template-creation process.
        Args:
            n_pixels (int): Number of pixels per side of the grid
            pixel_size (float): Size of each pixel in Angstrom
            precision ('single' | 'double' | Precision, optional): Precision at which
                to carry out computations. Defaults to Precision.SINGLE.
            resolution_factor (float, optional): The resolution factor for template 
                generation. Defaults to 1.0.
            viewing_distance (float | None, optional): Viewing distance of the
                image capture device. Used to compute viewing angles. Defaults to None.
            n_inplanes (int | None, optional): Number of points per ring of the
                polar quadrature grid. Defaults to None.
            atom_radii (float | None, optional): Radius of atoms in the model being
                interpreted. Defaults to None.
            atom_selection (str | None, optional): Which atoms from the PDB file
                to read. Ignored for non-PDB files. Defaults to "name CA".
            use_protein_residue_model (bool, optional): Whether to use the default
                sizes for proteins in PDB. Defaults to True.
            atom_shape (AtomShape | str, optional): Whether to treat atoms as hard
                spheres or Gaussian clouds. Defaults to AtomShape.GAUSSIAN.
        Returns:
            ImageDescriptor: A fully-populated ImageDescriptor object.
        """
        _cartesian_grid = CartesianGrid2D(n_pixels, pixel_size)

        _n_inplanes = n_pixels * 2 if n_inplanes is None else n_inplanes
        # this had been written as np.pi / 2.0 / (2.0 * np.pi)
        # but that's equivalent to np.pi / (4. * np.pi), i.e. a constant 0.25.
        # TODO: CHECK THIS
        radius_max = resolution_factor * n_pixels * 0.25
        _polar_grid = PolarGrid(
            radius_max=radius_max,
            dist_radii=np.pi / 2.0 / (2.0 * np.pi),
            n_inplanes=_n_inplanes,
            uniform=True
        )

        view_dist = 1.0 / (4.0 * np.pi) / resolution_factor if viewing_distance is None else viewing_distance
        
        return cls(
            Precision.from_str(precision),
            _cartesian_grid,
            _polar_grid,
            viewing_angles=None,
            viewing_distance=view_dist,
            atom_radii=atom_radii,
            atom_selection=atom_selection,
            use_protein_residue_model=use_protein_residue_model,
            atom_shape=AtomShape.from_str(atom_shape)
        )


    @classmethod
    def from_dict(cls, data: dict):
        _data = SerializedImageDescriptor.model_validate(data)
        _cartesian_grid = CartesianGrid2D(_data.n_pixels, _data.pixel_size)
        _polar_grid = PolarGrid(
            radius_max=_data.radius_max,
            dist_radii=_data.dist_radii,
            n_inplanes=_data.n_inplanes,
            uniform=True # TODO: support non-uniform polar grids
        )
        # TODO: Support non-uniform viewing angles

        try:
            precision = Precision.from_str(_data.precision)
        except NotImplementedError:
            # I think this is now unreachable
            precision = Precision.DEFAULT
            print("Warning: Invalid precision value, using default")
        try:
            atom_shape = AtomShape.from_str(_data.atom_shape)
        except NotImplementedError:
            atom_shape = AtomShape.DEFAULT
            print("Warning: Invalid atom shape value, using default")

        return cls(
            precision,
            _cartesian_grid,
            _polar_grid,
            viewing_angles=_data.viewing_angles.deserialize() if _data.viewing_angles is not None else None,
            viewing_distance=_data.viewing_distance,
            atom_radii=_data.atom_radii,
            atom_selection=_data.atom_selection,
            use_protein_residue_model=_data.use_protein_residue_model,
            atom_shape=atom_shape
        )
    

    # TODO: Eliminate this
    @classmethod
    def ensure(cls, input: 'str | ImageDescriptor') -> 'ImageDescriptor':
        if isinstance(input, ImageDescriptor):
            return input
        return cls.from_dict(load_file(input))


    # TODO: Query: is this actually sufficient?
    def update_pixel_size(self, pixel_size: float | FloatArrayType):
        self.cartesian_grid = CartesianGrid2D(self.cartesian_grid.n_pixels, pixel_size)
    