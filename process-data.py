from pathlib import Path
import json
from plots import *
import numpy as np
from simulation import PartialDislocationsSimulation

def loadResults(file_path):
    # Load the results from a file at file_path to a new simulation object for further processing

    loaded  = np.load(file_path)

    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff = loaded["params"]

    res = PartialDislocationsSimulation(bigN=bigN.astype(int), bigB=bigB, length=length,
                                        time=time,timestep_dt=dt, deltaR=deltaR, 
                                        smallB=smallB, b_p=b_p, cLT1=cLT1, cLT2=cLT2,
                                        mu=mu, tauExt=tauExt, c_gamma=c_gamma, d0=d0, seed=seed.astype(int))
    
    # Modify the internal state of the object according to loaded parameters
    res.tau_cutoff = tau_cutoff
    res.has_simulation_been_run = True

    res.y1 = loaded["y1"]
    res.y2 = loaded["y2"]

    return res

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
    makeDepinningPlotAvg(stresses, vCm, 10000, 100,folder_name="./")

    sim = loadResults("results/15-feb-1/pickle-dumps/seed-100/sim-3.0000.npz")
    makeVelocityPlot(sim, "results/15-feb-1/")
    # makeGif(sim, "results/15-feb-1/")