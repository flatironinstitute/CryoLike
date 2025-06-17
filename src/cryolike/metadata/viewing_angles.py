# from torch import torch.Tensor, tensor, zeros_like, ones_like, float32, concatenate, rand, arccos, device
import torch
from numpy import pi
from pydantic import BaseModel, ConfigDict

from cryolike.grids import SphereShell
from cryolike.util import SamplingStrategy, Precision, to_torch, get_device

class SerializedViewingAngles(BaseModel):
    """Serialized version of ViewingAngles for use in serialization/deserialization."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    azimus: torch.Tensor
    polars: torch.Tensor
    gammas: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    def deserialize(self) -> "ViewingAngles":
        """Converts the serialized data back to a ViewingAngles object."""
        return ViewingAngles(
            azimus=self.azimus,
            polars=self.polars,
            gammas=self.gammas if self.gammas is not None else None,
            weights=self.weights if self.weights is not None else None
        )

class ViewingAngles:
    """Class storing the viewing angles, with weights, for a particular template/image stack.

    Viewing angles define the orientation of the particle relative to the imaging device. Thus,
    a ViewingAngles object is usually tied to a particular stack of observed images (Images)
    or reference images (Templates), with one value for each member for each image in the stack.
    However, if the ViewingAngles members are unit length in the highest dimension (i.e.
    len(azimus[0] == 1) then the angles can be broadcast to a whole image stack.

    Attributes:
        azimus (torch.Tensor): Azimuthal angle for each image
        polars (torch.Tensor): Polar angle for each image
        gammas (torch.Tensor | None): 2D inplane rotation angle for each image, or None if not used.
        weights_viewing (torch.Tensor | None): Weights to apply to each angle, per-image
        n_angles (int): Number of angles in the stack. For the ViewingAngles object
            to be paired with a Templates or Images object, the number of angles
            must match the number of images in that object, or be 1 (indicating that
            the angles will be broadcast over each image). The outermost length
            of each of the four tensors in the class must match the number of angles.
    """
    azimus: torch.Tensor
    polars: torch.Tensor
    gammas: torch.Tensor | None
    weights: torch.Tensor | None
    n_angles: int


    def __init__(self, 
        azimus: torch.Tensor, 
        polars: torch.Tensor, 
        gammas: torch.Tensor | None = None,
        weights: torch.Tensor | None = None
    ):
        """Constructor for stack of viewing angles.

        Args:
            azimus (FloatArrayType | torch.Tensor): A torch.Tensor or Numpy array describing the azimuthal angles
                for the stack
            polars (FloatArrayType | torch.Tensor): A torch.Tensor or Numpy array describing the polar angles
                for the stack
            gammas (FloatArrayType | torch.Tensor | None): A torch.Tensor or Numpy array describing the gamma
                angles for the stack. If unset, will default to a torch.Tensor of 0s matching the shape
                of the other parameters.
        """
        self.azimus = azimus
        self.polars = polars
        self.gammas = gammas
        self.weights = weights

        assert isinstance(self.azimus, torch.Tensor), "Azimus should be a torch.Tensor."
        assert isinstance(self.polars, torch.Tensor), "Polars should be a torch.Tensor."

        if self.azimus.ndim != 1 or self.polars.ndim != 1:
            raise ValueError("Viewing angle vectors should be 1d.")
        if self.gammas is not None:
            if self.gammas.ndim != 1:
                raise ValueError("Gamma viewing angle vector should be 1d.")

        self.n_angles = len(self.azimus)
        if len(self.polars) != self.n_angles:
            raise ValueError("Azimus and Polars viewing angle tensors should be of the same length.")
        if self.gammas is not None:
            assert isinstance(self.gammas, torch.Tensor), "Gammas should be a torch.Tensor."
            if len(self.gammas) != self.n_angles:
                raise ValueError("Gammas viewing angle tensor should be of the same length as the azimus and polars tensors.")
        if self.weights is not None:
            assert isinstance(self.weights, torch.Tensor), "Weights should be a torch.Tensor."
            if len(self.weights) != self.n_angles:
                raise ValueError("Weights tensor should be of the same length as the azimus, polars, and gammas tensors.")


    @classmethod
    def from_viewing_distance(cls, viewing_distance: float, precision: Precision = Precision.DEFAULT) -> "ViewingAngles":
        """Constructs a set of viewing angles from a regular viewing distance.

        Args:
            viewing_distance (float): The desired difference between two angles

        Returns:
            ViewingAngles: The set of viewing angles computed from this viewing distance
        """
        viewing_shell = SphereShell(radius=1.0, dist_eq=viewing_distance, azimuthal_sampling=SamplingStrategy.ADAPTIVE, compute_cartesian=False)
        _azimus = to_torch(viewing_shell.azimu_points, precision=precision)
        _polars = to_torch(viewing_shell.polar_points, precision=precision)
        obj = cls(azimus=_azimus, polars=_polars, gammas=None)
        obj.weights = to_torch(viewing_shell.weight_points, precision=precision)
        return obj


    @classmethod
    def from_random(cls, n_angles: int, precision: Precision = Precision.DEFAULT) -> "ViewingAngles":
        """Constructs a set of random viewing angles.

        Args:
            n_angles (int): The number of angles to generate

        Returns:
            ViewingAngles: The set of random viewing angles
        """
        float_type, _, _ = precision.get_dtypes(default=Precision.SINGLE)
        _azimus = torch.rand(n_angles, dtype=float_type) * 2 * pi
        _cos_polars = torch.rand(n_angles, dtype=float_type) * 2 - 1
        _polars = torch.arccos(_cos_polars)
        _gammas = torch.rand(n_angles, dtype=float_type) * 2 * pi
        return cls(_azimus, _polars, _gammas, None)


    def clone(self) -> "ViewingAngles":
        """Returns a copy of the ViewingAngles object.

        Returns:
            ViewingAngles: A copy of the ViewingAngles object
        """
        _gammas = self.gammas.clone() if self.gammas is not None else None
        _weights = self.weights.clone() if self.weights is not None else None
        return ViewingAngles(self.azimus.clone(), self.polars.clone(), _gammas, _weights)


    def concatenate(self, other: "ViewingAngles") -> "ViewingAngles":
        """Concatenates two ViewingAngles objects.

        Args:
            other (ViewingAngles): The other ViewingAngles object to concatenate

        Returns:
            ViewingAngles: A new ViewingAngles object with the concatenated angles
        """
        if self.gammas is None and other.gammas is None:
            gammas = None
        elif self.gammas is not None and other.gammas is not None:
            gammas = torch.concatenate([self.gammas, other.gammas], dim=0)
        else:
            raise ValueError("Both ViewingAngles objects must have gammas defined or both must be None.")

        if self.weights is None and other.weights is None:
            weights = None
        elif self.weights is not None and other.weights is not None:
            weights = torch.concatenate([self.weights, other.weights], dim=0)
        else:
            raise ValueError("Both ViewingAngles objects must have weights defined or both must be None.")

        azimus = torch.concatenate([self.azimus, other.azimus], dim=0)
        polars = torch.concatenate([self.polars, other.polars], dim=0)
        return ViewingAngles(azimus, polars, gammas, weights)


    def serialize(self) -> SerializedViewingAngles:
        """Serializes the ViewingAngles object for storage or transmission.

        Returns:
            SerializedViewingAngles: A serialized version of the ViewingAngles object
        """
        return SerializedViewingAngles(
            azimus=self.azimus,
            polars=self.polars,
            gammas=self.gammas,
            weights=self.weights
        )


    def to(self, precision: Precision | None = None, device: torch.device | str | None = None):
        """Moves the ViewingAngles tensors to a specified device.

        Args:
            device (torch.device): The device to move the tensors to

        Returns:
            ViewingAngles: A new ViewingAngles object with tensors moved to the specified device
        """
        if precision is not None:
            float_type, _, _ = precision.get_dtypes(default=Precision.SINGLE)
            self.azimus = self.azimus.to(dtype=float_type)
            self.polars = self.polars.to(dtype=float_type)
            if self.gammas is not None:
                self.gammas = self.gammas.to(dtype=float_type)
            if self.weights is not None:
                self.weights = self.weights.to(dtype=float_type)
        if device is not None:
            _device = get_device(device)
            self.azimus = self.azimus.to(_device)
            self.polars = self.polars.to(_device)
            if self.gammas is not None:
                self.gammas = self.gammas.to(_device)
            if self.weights is not None:
                self.weights = self.weights.to(_device)
        return self


    def get_slice(self, start: int, end: int) -> "ViewingAngles":
        """Returns a slice of the ViewingAngles object.

        Args:
            start (int): The start index of the slice
            end (int): The end index of the slice

        Returns:
            ViewingAngles: A new ViewingAngles object containing the sliced angles
        """
        if start < 0 or end > self.n_angles or start >= end:
            raise ValueError("Invalid slice indices for ViewingAngles.")
        _azimus = self.azimus[start:end]
        _polars = self.polars[start:end]
        _gammas = None
        if self.gammas is not None:
            _gammas = self.gammas[start:end]
        _weights = None
        if self.weights is not None:
            _weights = self.weights[start:end]
        return ViewingAngles(
            azimus=_azimus,
            polars=_polars,
            gammas=_gammas,
            weights=_weights
        )