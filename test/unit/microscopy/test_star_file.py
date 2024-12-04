from pathlib import Path
from cryolike.microscopy.star_file import read_star_file, write_star_file

def test_star_file():
    
    # TODO: Don't actually interact with the file system
    file_dir = Path(__file__).resolve()
    filename_in = file_dir.parent.parent.parent.joinpath("data").joinpath("particle_data.star")
    filename_out = file_dir.parent.parent.parent.joinpath("data").joinpath("particles_out.star")
    dataList, paramsList = read_star_file(filename_in)
    write_star_file(filename_out, dataList, paramsList)
    
    ## check if the two files are the equivalent
    dataList2, paramsList2 = read_star_file(filename_out)
    for j in range(len(paramsList)):
        for k in range(len(dataList[paramsList[j]])):
            assert dataList[paramsList[j]][k] == dataList2[paramsList2[j]][k]
    
if __name__ == "__main__":
    test_star_file()
    print("STAR File test passed")
