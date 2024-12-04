from typing import TypeVar, cast
from numpy import ndarray
import numpy as np
import torch

from .particle_stacks_metadata import _Metadata
from cryolike.util import ComplexArrayType, FloatArrayType


T = TypeVar("T", bound=FloatArrayType | ComplexArrayType | torch.Tensor)
def _pop_batch(u: T, batch_size: int) -> tuple[T, T]:
    head = cast(T, u[:batch_size])
    tail = cast(T, u[batch_size:])
    return (head, tail)

class _MetadataBuffer(_Metadata):
    defocusU: FloatArrayType
    defocusV: FloatArrayType
    defocusAngle: FloatArrayType
    phaseShift: FloatArrayType
    sphericalAberration: float
    voltage: float
    amplitudeContrast: float
    stack_size: int


    def __init__(self, meta: _Metadata):
        self.stack_size = 0
        self.defocusU = np.array([])
        self.defocusV = np.array([])
        self.defocusAngle = np.array([])
        self.phaseShift = np.array([])

        _abb = float(meta.sphericalAberration[0].item()) if isinstance(meta.sphericalAberration, ndarray) else float(meta.sphericalAberration)
        _volt = float(meta.voltage[0].item()) if isinstance(meta.voltage, ndarray) else float(meta.voltage)
        _ampl = float(meta.amplitudeContrast[0].item()) if isinstance(meta.amplitudeContrast, ndarray) else float(meta.amplitudeContrast)
        self.sphericalAberration = _abb
        self.voltage = _volt
        self.amplitudeContrast = _ampl
        self.defocus_is_degree = meta.defocus_is_degree
        self.phase_shift_is_degree = meta.phase_shift_is_degree
        self.ctfBfactor = meta.ctfBfactor
        self.ctfScalefactor = meta.ctfScalefactor


    def _ensure_size_consistency(self):
        if (self.defocusU.shape[0] != self.stack_size or
            self.defocusV.shape[0] != self.stack_size or
            self.defocusAngle.shape[0] != self.stack_size or
            self.phaseShift.shape[0] != self.stack_size
        ):
            raise ValueError("Inconsistent size in metadata buffers.")


    def make_copy(self, defU: FloatArrayType, defV: FloatArrayType, defA: FloatArrayType, phase: FloatArrayType):
        copy = _MetadataBuffer(self)
        copy.defocusU = defU
        copy.defocusV = defV
        copy.defocusAngle = defA
        copy.phaseShift = phase
        copy.stack_size = defU.shape[0]
        return copy


    def append_batch(self, defocusU: FloatArrayType, defocusV: FloatArrayType, defocusAng: FloatArrayType, phaseShift: FloatArrayType):
        if self.stack_size == 0:
            self.defocusU = defocusU
            self.defocusV = defocusV
            self.defocusAngle = defocusAng
            self.phaseShift = phaseShift
        else:
            self.defocusU = np.concatenate((self.defocusU, defocusU), axis = 0)
            self.defocusV = np.concatenate((self.defocusV, defocusV), axis = 0)
            self.defocusAngle = np.concatenate((self.defocusAngle, defocusAng), axis = 0)
            self.phaseShift = np.concatenate((self.phaseShift, phaseShift), axis = 0)
        self.stack_size = self.defocusU.shape[0]
        self._ensure_size_consistency()


    def pop_batch(self, batch_size: int):
        _b = min(batch_size, self.stack_size)
        (head_defocusU, tail_defocusU) = _pop_batch(self.defocusU, _b)
        (head_defocusV, tail_defocusV) = _pop_batch(self.defocusV, _b)
        (head_defocusAngle, tail_defocusAngle) = _pop_batch(self.defocusAngle, _b)
        (head_phaseShift, tail_phaseShift) = _pop_batch(self.phaseShift, _b)
        batch_meta = self.make_copy(head_defocusU, head_defocusV, head_defocusAngle, head_phaseShift)
        self.defocusU = tail_defocusU
        self.defocusV = tail_defocusV
        self.defocusAngle = tail_defocusAngle
        self.phaseShift = tail_phaseShift
        self.stack_size = self.defocusU.shape[0]
        self._ensure_size_consistency()
        batch_meta._ensure_size_consistency()
        return batch_meta


class ImgBuffer():
    images_phys: torch.Tensor
    images_fourier: torch.Tensor
    stack_size: int

    def __init__(self, ):
        self.images_phys = torch.tensor([])
        self.images_fourier = torch.tensor([])
        self.stack_size = 0


    def append_imgs(self, phys: torch.Tensor | None, fourier: torch.Tensor | None):
        if (phys is None or fourier is None):
            return
        if self.stack_size == 0:
            self.images_phys = phys
            self.images_fourier = fourier
        else:
            self.images_phys = torch.concatenate((self.images_phys, phys), dim = 0)
            self.images_fourier = torch.concatenate((self.images_fourier, fourier), dim = 0)
        self.stack_size = self.images_phys.shape[0]
        if (self.images_fourier.shape[0] != self.stack_size):
            raise ValueError("Physical and Fourier image buffers have differing lengths.")


    def pop_imgs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        _b = min(batch_size, self.stack_size)
        assert self.images_phys is not None
        assert self.images_fourier is not None
        (phys_head, phys_tail) = _pop_batch(self.images_phys, _b)
        (fourier_head, fourier_tail) = _pop_batch(self.images_fourier, _b)
        self.images_phys = phys_tail
        self.images_fourier = fourier_tail
        self.stack_size = self.images_phys.shape[0]
        return (phys_head, fourier_head)
