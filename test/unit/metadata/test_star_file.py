from pathlib import Path
from cryolike.metadata.star_file import read_star_file, write_star_file


def _check_equivalence(dataList: dict, paramsList: list, path: Path):
    dataList2, paramsList2 = read_star_file(path)
    for j in range(len(paramsList)):
        for k in range(len(dataList[paramsList[j]])):
            assert dataList[paramsList[j]][k] == dataList2[paramsList2[j]][k]


def test_star_file(tmp_path):
    file_dir = Path(__file__).resolve()
    filename_in = file_dir.parent.parent.parent.joinpath("data").joinpath("particle_data.star")
    filename_out = tmp_path / "particles_out.star"
    dataList, paramsList = read_star_file(filename_in)
    write_star_file(filename_out, dataList, paramsList)
    _check_equivalence(dataList, paramsList, filename_out)

    filename_in = file_dir.parent.parent.parent.joinpath("data").joinpath("relion_style_particles.star")
    filename_out = tmp_path / "particles_out.star"
    dataList, paramsList = read_star_file(filename_in)
    write_star_file(filename_out, dataList, paramsList)
    _check_equivalence(dataList, paramsList, filename_out)


if __name__ == "__main__":
    from os import remove
    file_dir = Path(__file__).resolve()
    dir_out = file_dir.parent.parent.parent.joinpath("data")
    test_star_file(dir_out)
    print("STAR File test passed")
    remove(dir_out.joinpath("particles_out.star"))
