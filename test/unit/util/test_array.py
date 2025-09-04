import torch
import numpy as np
from cryolike.util.array import (
    pop_batch
)

def assert_pop_batch(orig, head, tail, batch: int):
    assert type(head) == type(orig)
    assert type(head) == type(tail)
    assert len(head) + len(tail) == len(orig)
    assert len(head) == min(len(orig), batch)
    for i, val in enumerate(head):
        assert orig[i] == val
    for i, val in enumerate(tail):
        assert orig[i + len(head)] == val

def test_pop_batch():
    batch_sizes = [0, 4, 8, 10, 11]
    torch_list = torch.arange(10)
    float_list = np.array(np.arange(10), dtype=np.float32)
    cmplx_list = np.array(np.arange(10), dtype=np.complex128)

    for src_list in [torch_list, float_list, cmplx_list]:
        for batch_length in batch_sizes:
            (head, tail) = pop_batch(src_list, batch_length)
            assert_pop_batch(src_list, head, tail, batch_length)
