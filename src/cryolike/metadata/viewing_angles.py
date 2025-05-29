from torch import Tensor, tensor, zeros_like, ones_like, float32, concatenate, rand, arccos
from numpy import pi

from cryolike.grids import SphereShell
from cryolike.util import FloatArrayType, SamplingStrategy

_Viewing_angle_type = FloatArrayType | Tensor

class ViewingAngles:
    """Class storing the viewing angles, with weights, for a particular template/image stack.

    Viewing angles define the orientation of the particle relative to the imaging device. Thus,
    a ViewingAngles object is usually tied to a particular stack of observed images (Images)
    or reference images (Templates), with one value for each member for each image in the stack.
    However, if the ViewingAngles members are unit length in the highest dimension (i.e.
    len(azimus[0] == 1) then the angles can be broadcast to a whole image stack.

    Attributes:
        azimus (Tensor): Azimuthal angle for each image
        polars (Tensor): Polar angle for each image
        gammas (Tensor): Gamma angle (grayscale correction factor) for each image
        weights_viewing (Tensor): Weights to apply to each angle, per-image
        n_angles (int): Number of angles in the stack. For the ViewingAngles object
            to be paired with a Templates or Images object, the number of angles
            must match the number of images in that object, or be 1 (indicating that
            the angles will be broadcast over each image). The outermost length
            of each of the four tensors in the class must match the number of angles.
    """
    azimus: Tensor
    polars: Tensor
    gammas: Tensor
    weights_viewing: Tensor
    n_angles: int


    def __init__(self, azimus: _Viewing_angle_type, polars: _Viewing_angle_type, gammas: _Viewing_angle_type | None):
        """Constructor for stack of viewing angles.

        Args:
            azimus (FloatArrayType | Tensor): A Tensor or Numpy array describing the azimuthal angles
                for the stack
            polars (FloatArrayType | Tensor): A Tensor or Numpy array describing the polar angles
                for the stack
            gammas (FloatArrayType | Tensor | None): A Tensor or Numpy array describing the gamma
                angles for the stack. If unset, will default to a Tensor of 0s matching the shape
                of the other parameters.
        """
        self.azimus = azimus if isinstance(azimus, Tensor) else tensor(azimus, dtype=float32)
        self.polars = polars if isinstance(polars, Tensor) else tensor(polars, dtype=float32)

        if gammas is None:
            self.gammas = zeros_like(self.azimus)
        else:
            self.gammas = gammas if isinstance(gammas, Tensor) else tensor(gammas, dtype=float32)

        if len(self.azimus.shape) != 1 or len(self.polars.shape) != 1 or len(self.gammas.shape) != 1:
            raise ValueError("Viewing angle vectors should be 1d.")

        self.n_angles = len(self.azimus)
        if len(self.polars) != self.n_angles or len(self.gammas) != self.n_angles:
            raise ValueError("Azimus, Polars, and Gammas viewing angle tensors should be of the same length.")
        self.weights_viewing = ones_like(self.azimus) / self.n_angles


    @classmethod
    def from_viewing_distance(cls, viewing_distance: float) -> "ViewingAngles":
        """Constructs a set of viewing angles from a regular viewing distance.

        Args:
            viewing_distance (float): The desired difference between two angles

        Returns:
            ViewingAngles: The set of viewing angles computed from this viewing distance
        """
        viewing_shell = SphereShell(radius=1.0, dist_eq=viewing_distance, azimuthal_sampling=SamplingStrategy.ADAPTIVE, compute_cartesian=False)
        obj = cls(azimus=viewing_shell.azimu_points, polars=viewing_shell.polar_points, gammas=None)
        obj.weights_viewing = tensor(viewing_shell.weight_points)
        return obj

    
    @classmethod
    def from_random(cls, n_angles: int) -> "ViewingAngles":
        """Constructs a set of random viewing angles.

        Args:
            n_angles (int): The number of angles to generate

        Returns:
            ViewingAngles: The set of random viewing angles
        """
        _azimus = rand(n_angles) * 2 * pi
        _cos_polars = rand(n_angles) * 2 - 1
        _polars = arccos(_cos_polars)
        _gammas = rand(n_angles) * 2 * pi
        _weights_viewing = ones_like(_azimus) / n_angles
        return cls(azimus=_azimus, polars=_polars, gammas=_gammas)


    def clone(self) -> "ViewingAngles":
        """Returns a copy of the ViewingAngles object.

        Returns:
            ViewingAngles: A copy of the ViewingAngles object
        """
        return ViewingAngles(self.azimus.clone(), self.polars.clone(), self.gammas.clone())


    def concatenate(self, other: "ViewingAngles") -> "ViewingAngles":
        """Concatenates two ViewingAngles objects.

        Args:
            other (ViewingAngles): The other ViewingAngles object to concatenate

        Returns:
            ViewingAngles: A new ViewingAngles object with the concatenated angles
        """
        azimus = concatenate([self.azimus, other.azimus], dim=0)
        polars = concatenate([self.polars, other.polars], dim=0)
        gammas = concatenate([self.gammas, other.gammas], dim=0)
        weights_viewing = concatenate([self.weights_viewing, other.weights_viewing], dim=0)
        viewing_angles = ViewingAngles(azimus=azimus, polars=polars, gammas=gammas)
        viewing_angles.weights_viewing = weights_viewing
        return viewing_angles