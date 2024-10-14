def read_star_file(filename, stop = None):

    dataList = []
    paramsList = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        if stop is not None:
            lines = lines[:stop]
        modeRead = False
        data = {}
        nParams = 0
        Params = []
        for line in lines:
            if line.startswith("_rln"):
                line = line.strip()
                line = line.split()[0][4:]
                data[line] = []
                Params.append(line)
                nParams += 1
                modeRead = True
            elif modeRead:
                line = line.strip()
                line = line.split()
                if len(line) == nParams:
                    for i in range(nParams):
                        # check type
                        try:
                            value = float(line[i])
                        except ValueError:
                            value = line[i]
                        data[Params[i]].append(value)
                else:
                    dataList.append(data)
                    paramsList.append(Params)
                    modeRead = False
                    data = {}
                    nParams = 0
                    Params = []
            elif len(data) > 0:
                dataList.append(data)
                paramsList.append(Params)
                modeRead = False
                data = {}
                nParams = 0
                Params = []

    if modeRead:
        dataList.append(data)
        paramsList.append(Params)
      
    if len(dataList) > 1:
        dataListOut = {}
        paramsListOut = []
        for i in range(len(dataList)):
            dataListOut.update(dataList[i])
            for paramItem in paramsList[i]:
                paramsListOut.append(paramItem)
        return dataListOut, paramsListOut
    else:
        dataList = dataList[0]
        paramsList = paramsList[0]
        return dataList, paramsList


def write_star_file(
        filename,
        dataList = None,
        paramsList = None,
        ctf = None,
    ):
    
    if ctf is not None:
        import numpy as np
        dataList = {
            "DefocusU": ctf.defocusU,
            "DefocusV": ctf.defocusV,
            "DefocusAngle": ctf.defocusAng,
            "SphericalAberration": ctf.sphAber * np.ones_like(ctf.defocusU),
            "Voltage": ctf.voltage * np.ones_like(ctf.defocusU),
            "AmplitudeContrast": ctf.ampCon * np.ones_like(ctf.defocusU),
            "PhaseShift": ctf.phaseShift
        }      
        paramsList = [
            "DefocusU", "DefocusV", "DefocusAngle", "SphericalAberration", "Voltage", "AmplitudeContrast", "PhaseShift"
        ]
        
    with open(filename, 'w') as f:
        f.write("\n")
        for j in range(len(paramsList)):
            f.write("_rln" + paramsList[j] + " #" + str(j+1) + "\n")
        for j in range(len(dataList[paramsList[0]])):
            for param in paramsList:
                f.write(str(dataList[param][j]) + " ")
            f.write("\n")
        f.write("\n")
    return


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    dataList, paramsList = read_star_file(filename)
    print(dataList)
    print(paramsList)