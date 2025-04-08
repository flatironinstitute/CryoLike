import numpy as np
from pathlib import Path
import starfile
import pandas as pd
import re


def read_star_file(filename: str | Path, stop = None):
    filepath = Path(filename)
    starfile_dataframe = starfile.read(filepath)
    if not isinstance(starfile_dataframe, pd.DataFrame):
        # type annotations in the package are a little off--starfile.read()
        # is annotated to return DataBlock | Dict[DataBlock], the latter of which
        # is mis-specified (it should be Dict[str, DataBlock]) resulting in misinterpretation
        # in some IDEs. Hence I've re-annotated to an alias variable for this case.
        # A DataBlock itself is an alias in the starfile library for a Union of a
        # (pandas) DataFrame or a dict[str, str | int | float].
        sfd: dict[str, pd.DataFrame] = starfile_dataframe # type: ignore
        if set(sfd.keys()) == {'optics','particles'}:
            optics_block = sfd['optics']
            images_block = sfd['particles']

            # Merge the optics and images blocks on the "rlnOpticsGroup" column
            if 'rlnOpticsGroup' in optics_block.columns and 'rlnOpticsGroup' in images_block.columns:
                combined_block = pd.merge(images_block, optics_block, on='rlnOpticsGroup', suffixes=('_image', '_optics'))
            else:
                raise ValueError("rlnOpticsGroup column is missing in either optics or images block.")
            starfile_dataframe = combined_block
        else:
            # Our example returns a DataFrame from parsing a non-Relion-formatted file,
            # but the starfile library could also return a raw dict.
            # Since we don't have any plan for handling this case, raise an error.
            raise ValueError("Unsupported starfile format.")

    starfile_dataframe = starfile_dataframe.rename(columns=lambda x: re.sub(r'^_?rln', '', x))  # Remove the rln key from everything. Add cistem?

    if not ("Voltage" in list(starfile_dataframe.keys())):
        raise ValueError ('voltage missing from starfile')
    if not ("SphericalAberration" in in list(starfile_dataframe.keys()):
            raise ValueError ('spherical abberation missing from starfile')

    paramsList: list[str] = list(starfile_dataframe.keys())
    dataList: dict[str, np.ndarray] = {}
    for key in paramsList:
        dataList[key] = np.array(list(starfile_dataframe[key].values))
        if not isinstance(dataList[key][0],str):
            dataList[key] = np.array([float(val) for val in starfile_dataframe[key]])

    return dataList, paramsList


# NOTE: This covers the case where we're writing out a (potentially modified)
# starfile dataframe; it does not handle writing our internal data format as a starfile.
def write_star_file(
        filename: str,
        dataList: dict[str, np.ndarray] = {},
        paramsList: list[str] = [],
        ld: 'LensDescriptor | None' = None, # type: ignore
    ):

    filepath = Path(filename)
    df = pd.DataFrame()
    for i in paramsList:
        df[i] = dataList[i]

    starfile.write(df, filepath)
    return


# pragma: no cover
if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    dataList, paramsList = read_star_file(filename)
    print(dataList)
    print(paramsList)
