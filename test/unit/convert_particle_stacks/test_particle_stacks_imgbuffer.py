from pytest import raises
import torch.testing as tt
import torch

from cryolike.convert_particle_stacks.particle_stacks_buffers import (
    ImgBuffer
)


def fix_make_tensors(*, length: int = 4, base: int = 0):
    phys = base + torch.arange(length)
    four = base + length + torch.arange(length)
    return (phys, four)


def test_imgbuffer_init():
    sut = ImgBuffer()
    assert sut.stack_size == 0
    assert len(sut.images_phys) == 0
    assert len(sut.images_fourier) == 0


def test_imgbuffer_append_imgs_when_empty():
    (phys, four) = fix_make_tensors()
    sut = ImgBuffer()
    assert sut.stack_size == 0

    sut.append_imgs(phys, four)
    assert sut.stack_size == len(phys)
    tt.assert_close(sut.images_phys, phys)
    tt.assert_close(sut.images_fourier, four)


def test_imgbuffer_append_imgs_when_not_empty():
    (phys, four) = fix_make_tensors()
    sut = ImgBuffer()
    sut.images_fourier = four
    sut.images_phys = phys
    sut.stack_size = len(phys)

    (phys2, four2) = fix_make_tensors(length=2, base=10)
    sut.append_imgs(phys2, four2)
    assert sut.stack_size == 6
    tt.assert_close(sut.images_phys, torch.concatenate((phys, phys2), dim=0))
    tt.assert_close(sut.images_fourier, torch.concatenate((four, four2), dim=0))


def test_imgbuffer_append_imgs_with_none():
    (phys, four) = fix_make_tensors()
    sut = ImgBuffer()
    sut.images_fourier = four
    sut.images_phys = phys
    sut.stack_size = len(phys)

    (phys2, four2) = fix_make_tensors(length=5)
    sut.append_imgs(phys2, None)
    assert sut.stack_size == len(phys)
    sut.append_imgs(None, four2)
    assert sut.stack_size == len(phys)


def test_imgbuffer_append_imgs_wrong_lengths():
    (phys, four) = fix_make_tensors()
    sut = ImgBuffer()
    four = torch.concatenate((four, torch.arange(2)), dim=0)
    with raises(ValueError, match="have differing lengths"):
        sut.append_imgs(phys, four)


def test_imgbuffer_pop_imgs():
    (phys, four) = fix_make_tensors(length=6)
    sut = ImgBuffer()
    sut.append_imgs(phys, four)
    assert sut.stack_size == 6

    (phys_head, four_head) = sut.pop_imgs(4)
    assert sut.stack_size == 2
    assert len(phys_head) == 4
    assert len(four_head) == 4
    tt.assert_close(phys, torch.concatenate((phys_head, sut.images_phys), dim=0))
    tt.assert_close(four, torch.concatenate((four_head, sut.images_fourier), dim=0))
