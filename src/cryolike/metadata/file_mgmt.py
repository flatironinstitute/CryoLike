from cryolike.util import save_descriptors, load_file
from .lens_descriptor import LensDescriptor
from .image_descriptor import ImageDescriptor



def save_combined_params(
    fn: str,
    img_desc: ImageDescriptor,
    lens_desc: LensDescriptor,
    n_imgs_this_stack: int,
    overall_batch_start: int | None = None
):
    if n_imgs_this_stack <= 0:
        raise ValueError("Request to store image stack with nonpositive image count.")
    counts = { "n_images": n_imgs_this_stack }
    if overall_batch_start is not None:
        counts.update({
            "stack_start": overall_batch_start,
            "stack_end": overall_batch_start + n_imgs_this_stack
        })

    save_descriptors(fn, img_desc.to_dict(), lens_desc.to_dict(), counts)


def load_combined_params(fn: str):
    data = load_file(fn)
    img_desc = ImageDescriptor.from_dict(data)
    lens_desc = LensDescriptor.from_dict(data)

    return (img_desc, lens_desc)
