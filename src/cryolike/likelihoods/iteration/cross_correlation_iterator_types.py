import torch
from typing import Generator, NamedTuple


class CrossCorrelationYieldType(NamedTuple):
    """Return type yielded by a cross-correlation generator.
    These generators are designed to provide an interface between
    the cross-correlation likelihood computation, on the one hand
    (which should not have to worry about managing tensor memory
    or device locations), and collection/summarization on the
    other hand (which aggregates across different iterations
    through the large template/image stacks).

    Attributes:
        t_start (int): Index (from larger stack) of first template
            used in this set of comparisons
        t_end (int): 1 + index (from larger stack) of last template
            used in this set of comparisons. Together, the range
            [t_start:t_end] should be the range, from a larger
            aggregation tensor, into which this set of comparisons
            could be copied.
        i_start (int): Index (from larger stack) of first image
            used in this set of comparisons
        i_end (int): 1 + index (from larger stack) of last image
            used in this set of comparisons (range as with templates)
        cross_correlation_msdw (torch.Tensor): The full results of
            cross-correlation likelihood, properly normalized by
            image and template norms, indexed as [image, template,
            displacement, inplane-rotation]. The m- and s-dimensions
            should match the sizes suggested by t_start/t_end and
            i_start/i_end, above.
        log_likelihood_ms (torch.Tensor | None): If set, the result
            of computing the integrated log likelihood of this image
            range against the tensors. Will be None if ill computation
            was not requested.
    
    """
    t_start: int
    t_end: int
    i_start: int
    i_end: int
    cross_correlation_msdw: torch.Tensor
    log_likelihood_ms: torch.Tensor | None

GeneratorType = Generator[CrossCorrelationYieldType, None, None]
