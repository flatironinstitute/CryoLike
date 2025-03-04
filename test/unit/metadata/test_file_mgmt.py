from pytest import raises
from unittest.mock import patch, Mock

from cryolike.metadata.file_mgmt import (
    save_combined_params,
    load_combined_params
)

PKG = "cryolike.metadata.file_mgmt"


@patch(f"{PKG}.save_descriptors")
def test_save_combined_params(save: Mock):
    filename = "my/output/file"
    img_desc = Mock()
    lens_desc = Mock()
    n_imgs = 25

    save_combined_params(filename, img_desc, lens_desc, n_imgs)

    save.assert_called_once()
    calls = save.call_args[0]
    assert calls[0] == filename
    assert calls[1] == img_desc.to_dict.return_value
    assert calls[2] == lens_desc.to_dict.return_value

    assert len(calls[3].keys()) == 1
    assert calls[3]["n_images"] == n_imgs

    img_desc.to_dict.assert_called_once()
    lens_desc.to_dict.assert_called_once()


@patch(f"{PKG}.save_descriptors")
def test_save_combined_params_adds_stack_data(save: Mock):
    filename = "my/output/file"
    img_desc = Mock()
    lens_desc = Mock()
    n_imgs = 25
    overall_start = 55

    save_combined_params(filename, img_desc, lens_desc, n_imgs, overall_batch_start=overall_start)

    save.assert_called_once()
    calls = save.call_args[0]
    assert len(calls[3].keys()) == 3
    assert calls[3]["n_images"] == n_imgs
    assert calls[3]["stack_start"] == overall_start
    assert calls[3]["stack_end"] == overall_start + n_imgs


@patch(f"{PKG}.save_descriptors")
def test_save_combined_params_throws_on_no_images(save: Mock):
    filename = "my/output/file"
    img_desc = Mock()
    lens_desc = Mock()

    with raises(ValueError, match="nonpositive"):
        save_combined_params(filename, img_desc, lens_desc, n_imgs_this_stack=0)
    save.assert_not_called()

@patch(f"{PKG}.os.path.exists")
def test_save_descriptors_raises_on_existing_file(exists: Mock):
    filename = "my/output/file"
    img_desc = Mock()
    lens_desc = Mock()
    n_imgs = 25
    overall_start = 55
    exists.return_value = True
    with raises(ValueError, match="already exists"):
        save_combined_params(filename, img_desc, lens_desc, n_imgs,
                             overall_batch_start=overall_start)

@patch(f"{PKG}.os.path.exists")
def test_save_descriptors_no_raises_on_existing_file_overwrite(exists: Mock):
    filename = "my/output/file"
    img_desc = Mock()
    lens_desc = Mock()
    n_imgs = 25
    overall_start = 55
    exists.return_value = True
    save_combined_params(filename, img_desc, lens_desc, n_imgs, overall_batch_start=overall_start,
                         overwrite=True)

@patch(f"{PKG}.LensDescriptor")
@patch(f"{PKG}.ImageDescriptor")
@patch(f"{PKG}.load_file")
def test_load_combined_params(load: Mock, img_desc: Mock, lens_desc: Mock):
    # Not much to test here other than that we dispatched correctly
    filename = "my/file/to/load"

    (ret_img_desc, ret_lens_desc) = load_combined_params(filename)
    assert ret_img_desc == img_desc.from_dict.return_value
    assert ret_lens_desc == lens_desc.from_dict.return_value
    load.assert_called_once_with(filename)
    img_desc.from_dict.assert_called_once_with(load.return_value)
    lens_desc.from_dict.assert_called_once_with(load.return_value)
