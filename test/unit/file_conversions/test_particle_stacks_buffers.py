from pytest import raises
from unittest.mock import Mock
import torch.testing as tt
import torch

from cryolike.file_conversions.particle_stacks_buffers import (
    ImgBuffer
)


def fix_make_tensors(*, length: int = 4, base: int = 0):
    phys = base + torch.arange(length)
    four = base + length + torch.arange(length)
    im = Mock()
    im.images_phys = phys
    im.images_fourier = four
    im.has_physical_images = lambda: len(im.images_phys) > 0
    im.has_fourier_images = lambda: len(im.images_fourier) > 0

    return im


def test_imgbuffer_init():
    sut = ImgBuffer()
    assert sut.stack_size == 0
    assert len(sut.images_phys) == 0
    assert len(sut.images_fourier) == 0


def test_imgbuffer_append_imgs_when_empty():
    im = fix_make_tensors()
    sut = ImgBuffer()
    assert sut.stack_size == 0

    sut.append_imgs(im)
    assert sut.stack_size == len(im.images_phys)
    tt.assert_close(sut.images_phys, im.images_phys)
    tt.assert_close(sut.images_fourier, im.images_fourier)


def test_imgbuffer_append_imgs_when_not_empty():
    im = fix_make_tensors()
    sut = ImgBuffer()
    sut.images_fourier = im.images_fourier
    sut.images_phys = im.images_phys
    sut.stack_size = len(im.images_phys)

    im2 = fix_make_tensors(length=2, base=10)
    sut.append_imgs(im2)
    assert sut.stack_size == 6
    tt.assert_close(sut.images_phys, torch.concatenate((im.images_phys, im2.images_phys), dim=0))
    tt.assert_close(sut.images_fourier, torch.concatenate((im.images_fourier, im2.images_fourier), dim=0))


def test_imgbuffer_append_imgs_skips_appends_with_an_empty_tensor():
    im = fix_make_tensors()
    sut = ImgBuffer()
    sut.images_fourier = im.images_fourier
    sut.images_phys = im.images_phys
    sut.stack_size = len(im.images_phys)

    stack_length = sut.stack_size
    empty_im = fix_make_tensors(length=0)
    assert not empty_im.has_physical_images()
    assert not empty_im.has_fourier_images()
    sut.append_imgs(empty_im)
    assert sut.stack_size == stack_length


def test_imgbuffer_append_imgs_wrong_lengths():
    im = fix_make_tensors()
    sut = ImgBuffer()
    im.images_fourier = torch.concatenate((im.images_fourier, torch.arange(2)), dim=0)
    assert im.images_phys.shape[0] != im.images_fourier.shape[0]
    
    with raises(ValueError, match="mismatched fourier and cartesian image counts"):
        sut.append_imgs(im)


def test_imgbuffer_append_imgs_length_postcondition_check():
    # It should not be possible to have an error here unless we had one going in
    im = fix_make_tensors()
    sut = ImgBuffer()
    sut.images_fourier = im.images_fourier
    sut.images_phys = torch.tensor([])
    sut.stack_size = len(sut.images_fourier)

    im2 = fix_make_tensors(length=2, base=10)
    with raises(ValueError, match="have differing lengths"):
        sut.append_imgs(im2)



def test_imgbuffer_pop_imgs():
    im = fix_make_tensors(length=6)
    sut = ImgBuffer()
    sut.append_imgs(im)
    assert sut.stack_size == 6

    (phys_head, four_head) = sut.pop_imgs(4)
    assert sut.stack_size == 2
    assert len(phys_head) == 4
    assert len(four_head) == 4
    tt.assert_close(im.images_phys, torch.concatenate((phys_head, sut.images_phys), dim=0))
    tt.assert_close(im.images_fourier, torch.concatenate((four_head, sut.images_fourier), dim=0))
