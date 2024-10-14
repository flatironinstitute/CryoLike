from torch import Tensor, amax, abs

def get_imgs_max(imgs: Tensor) -> Tensor:
    maxval = amax(abs(imgs), dim = (1,2), keepdim = True)
    return maxval
