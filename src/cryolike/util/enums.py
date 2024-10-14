from enum import Enum

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
