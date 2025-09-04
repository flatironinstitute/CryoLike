import numpy as np

from cryolike.util import pop_batch
from .lens_descriptor import LensDescriptor, BatchableLensFields

BATCH_FIELDS = BatchableLensFields._fields

BATCH_FIELDS = BatchableLensFields._fields

class LensDescriptorBuffer(LensDescriptor):
    """Buffered version of the LensDescriptor class, which accumulates the per-image
    quantities (defocus U, defocus V, defocus angle, and phase shift) and has methods
    for returning them in regular-sized chunks.

    Since every LensDescriptor (or child class) has some fields which do not need
    to be buffered, we also keep a reference to the (non-buffered) parent instance.
    When we export our data as a dict, this parent instance's to_dict is used as a
    base before overwriting the buffered fields with the values from this object.
    This ensures that a more-specific child class of LensDescriptor will still be
    able to preserve its custom fields, as well.

    Attributes:
        stack_size (int): Length of current buffer
        parent_descriptor (LensDescriptor): Non-buffered parent instance
    """
    stack_size: int
    parent_descriptor: LensDescriptor


    def __init__(self, parent_descriptor: LensDescriptor):
        self.stack_size = 0
        self.parent_descriptor = parent_descriptor
        for fieldname in BATCH_FIELDS:
            parent_val = getattr(parent_descriptor, fieldname)
            new_val = np.array([]) if parent_val is not None else None
            setattr(self, fieldname, new_val)

        # Note: these are never actually used; we should always look at the
        # corresponding value on self.parent_descriptor.
        self.sphericalAberration = -1.
        self.voltage = -1.
        self.amplitudeContrast = -1.


    def _ensure_size_consistency(self):
        for fieldname in BATCH_FIELDS:
            val = getattr(self, fieldname)
            if val is None: continue
            if val.shape[0] != self.stack_size:
                raise ValueError(f"Inconsistent size in lens descriptor buffers ({fieldname}).")
        optionals = [self.angleRotation, self.angleTilt, self.anglePsi]
        if any(x is None for x in optionals) and any(x is not None for x in optionals):
            raise ValueError("Some, but not all, optional buffers are set.")


    def update_parent(self, new_parent: LensDescriptor):
        if self.parent_descriptor == new_parent:
            return
        # TODO: More generalized test for equivalence/compatibility?
        if self.stack_size > 0:
            raise ValueError("Cannot reset parent lens descriptor when the buffer is not empty.")
        self.parent_descriptor = new_parent


    def enqueue(self, fields: BatchableLensFields):
        if self.stack_size == 0:
            for fieldname in BATCH_FIELDS:
                setattr(self, fieldname, getattr(fields, fieldname))
        else:
            for fieldname in BATCH_FIELDS:
                in_val = getattr(fields, fieldname)
                curr_val = getattr(self, fieldname)
                if in_val is not None and curr_val is not None:
                    new_val = np.concatenate((curr_val, in_val), axis = 0)
                    setattr(self, fieldname, new_val)

        self.stack_size = self.defocusU.shape[0]
        self._ensure_size_consistency()


    def pop_batch(self, batch_size: int) -> 'LensDescriptorBuffer':
        _b = min(batch_size, self.stack_size)
        copy = LensDescriptorBuffer(self.parent_descriptor)

        for fieldname in BATCH_FIELDS:
            vector = getattr(self, fieldname)
            (head, tail) = pop_batch(vector, _b)
            setattr(copy, fieldname, head)
            setattr(self, fieldname, tail)
        self.stack_size = self.defocusU.shape[0]
        copy.stack_size = copy.defocusU.shape[0]
        self._ensure_size_consistency()
        copy._ensure_size_consistency()
        return copy


    def is_empty(self):
        return self.stack_size == 0


    def _to_dict_internal(self):
        fields_dict = {}
        for fieldname in BATCH_FIELDS:
            fields_dict[fieldname] = getattr(self, fieldname)

        return fields_dict


    def to_dict(self):
        base_dict = self.parent_descriptor.to_dict()
        # TODO: Some kind of trimming for the non-buffered fields?
        base_dict.update(self._to_dict_internal())

        return base_dict
