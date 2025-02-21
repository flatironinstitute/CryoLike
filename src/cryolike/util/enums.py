from enum import Enum
from torch import dtype, int32, int64, float32, float64, complex64, complex128
from numpy import ndarray

class QuadratureType(Enum):
    """Quadrature point selection algorithm for polar grids."""
    GAUSS_JACOBI_BETA_1 = 'gauss-jacobi'
    GAUSS_JACOBI_BETA_2 = 'gauss-jacobi-beta-2'
    GAUSS_LEGENDRE = 'gauss-legendre'


class SamplingStrategy(Enum):
    """Strategy for sampling points for spheres."""
    UNIFORM = 'uniform'
    ADAPTIVE = 'adaptive'


class Precision(Enum):
    """Precision definitions for requests."""
    SINGLE = 'single'
    DOUBLE = 'double'
    DEFAULT = 'default'

    @classmethod
    def from_str(cls, label: 'Precision | str | ndarray') -> 'Precision':
        if isinstance(label, Precision):
            return label
        if isinstance(label, ndarray):
            label = label.item()
        if not isinstance(label, str):
            raise ValueError('Input value was neither a string nor an instance of Precision enum.')
        if label.lower() in ['single']:
            return Precision.SINGLE
        if label.lower() in ['double']:
            return Precision.DOUBLE
        if label.lower() in ['default', '']:
            return Precision.DEFAULT
        raise NotImplementedError('Unsupported precision value.')


    def get_dtypes(self, default: 'Precision') -> tuple[dtype, dtype, dtype]:
        """Interprets this Precision to return the desired dtypes.

        Args:
            default (Precision): Precision level to use if DEFAULT is set.

        Returns:
            tuple[dtype, dtype, dtype]: Torch float-type, complex-type, and int-type
        for the requested precision.
        """
        if default == Precision.DEFAULT:
            raise ValueError("The 'default' parameter cannot also be Default: you must say what the default is.")
        if self == Precision.DEFAULT:
            return default.get_dtypes(default)
        if self == Precision.SINGLE:
            return (float32, complex64, int32)
        elif self == Precision.DOUBLE:
            return (float64, complex128, int64)
        else:
            raise ValueError("Unreachable: unrecognized precision type.")
       

    def set_epsilon(self, requested: float, default: 'Precision | None' =None) -> float:
        """Return the desired epsilon value, or the machine precision of the chosen dtype,
        whichever is larger.

        Args:
            requested (float): The value of epsilon requested
            default (Precision | None, optional): Precision to use if the current precision
                is default. If not specified, we assume single precision.

        Returns:
            float: If the current precision allows the requested value of epsilon, that will
                be returned unmodified. If the requested epsilon is too small for the current
                precision level (e.g. caller requests epsilon of 1e-10 for single-precision,
                where machine precision is only 1e-6) then the finest value of epsilon supported
                by machine precision of this dtype will be returned instead.
        """
        if default is None or default == Precision.DEFAULT:
            default = Precision.SINGLE
        if self == Precision.DEFAULT:
            return default.set_epsilon(requested, default)
        if self == Precision.SINGLE:
            target_epsilon = 1.0e-6
        if self == Precision.DOUBLE:
            target_epsilon = 1.0e-12
        return max(requested, target_epsilon)


class NormType(Enum):
    """Types of norms for centering signals."""
    MAX = 'max'
    STD = 'std'


class Basis(Enum):
    PHYS = 'phys'
    FOURIER = 'fourier'


class CrossCorrelationReturnType(Enum):
    NONE = 0,
    OPTIMAL_POSE = 1
    OPTIMAL_DISPLACEMENT_AND_ROTATION = 2
    OPTIMAL_DISPLACEMENT = 3
    OPTIMAL_ROTATION = 4
    FULL_TENSOR = 10


class AtomShape(Enum):
    """Type of topology for modelling atomic density"""
    HARD_SPHERE = 'hard-sphere'
    GAUSSIAN = 'gaussian'
    DEFAULT = 'default'

    @classmethod
    def from_str(cls, label: 'AtomShape | str | ndarray') -> 'AtomShape':
        if isinstance(label, AtomShape):
            return label
        if isinstance(label, ndarray):
            label = label.item()
        if not isinstance(label, str):
            raise ValueError('Input value was neither a string nor an instance of AtomShape enum.')
        if label.lower() in ['hard_sphere', 'hard-sphere', 'hard']:
            return AtomShape.HARD_SPHERE
        if label.lower() in ['gaussian', 'gauss']:
            return AtomShape.GAUSSIAN
        if label.lower() in ['default']:
            return AtomShape.DEFAULT
        raise NotImplementedError('Invalid atom shape value.')
