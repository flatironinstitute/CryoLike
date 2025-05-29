import torch

from cryolike.util import pop_batch
from cryolike.stacks import Images


class ImgBuffer():
    images_phys: torch.Tensor
    images_fourier: torch.Tensor
    stack_size: int

    def __init__(self, ):
        self.images_phys = torch.tensor([])
        self.images_fourier = torch.tensor([])
        self.stack_size = 0


    def append_imgs(self, imgs: Images): #phys: torch.Tensor, fourier: torch.Tensor):
        if (not imgs.has_physical_images() and not imgs.has_fourier_images()):
            return
        if (imgs.images_phys.shape[0] != imgs.images_fourier.shape[0]):
            raise ValueError("Attempt to load buffer with Image stack with mismatched fourier and cartesian image counts.")
        if self.stack_size == 0:
            self.images_phys = imgs.images_phys
            self.images_fourier = imgs.images_fourier
        else:
            self.images_phys = torch.concatenate((self.images_phys, imgs.images_phys), dim = 0)
            self.images_fourier = torch.concatenate((self.images_fourier, imgs.images_fourier), dim = 0)
        self.stack_size = self.images_phys.shape[0]
        if (self.images_fourier.shape[0] != self.stack_size):
            raise ValueError("Physical and Fourier image buffers have differing lengths.")


    def pop_imgs(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        _b = min(batch_size, self.stack_size)
        (phys_head, phys_tail) = pop_batch(self.images_phys, _b)
        (fourier_head, fourier_tail) = pop_batch(self.images_fourier, _b)
        self.images_phys = phys_tail
        self.images_fourier = fourier_tail
        self.stack_size = self.images_phys.shape[0]
        return (phys_head, fourier_head)
