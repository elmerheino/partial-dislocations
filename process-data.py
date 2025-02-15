from pathlib import Path
import json
from plots import *
import numpy as np

def loadDepinningDumps(folder):
    vCm = list()
    stresses = None
    for file in Path(folder).iterdir():
        with open(file, "r") as fp:
            depinning = json.load(fp)

            if stresses == None:
                stresses = depinning["stresses"]

            vCm_i = depinning["relaxed_velocities_total"]
            vCm.append(vCm_i)
    
    return (stresses, np.array(vCm))

if __name__ == "__main__":
    stresses, vCm = loadDepinningDumps('results/results-triton/depinning-dumps')
    makeDepinningPlot(stresses, vCm, 10000, 100,folder_name="./")