from pathlib import Path
import json
from plots import *
import numpy as np
from simulation import PartialDislocationsSimulation

def dumpResults(sim: PartialDislocationsSimulation, folder_name: str):
    # Dumps the results of a simulation to a pickle file
    if not sim.has_simulation_been_run:
        raise Exception("Simulation has not been run.")
    
    parameters = np.array([
        sim.bigN, sim.length, sim.time, sim.dt,
        sim.deltaR, sim.bigB, sim.smallB, sim.b_p,
        sim.cLT1, sim.cLT2, sim.mu, sim.tauExt, sim.c_gamma,
        sim.d0, sim.seed, sim.tau_cutoff
        ]) # From these parameters you should be able to replicate the simulation
    

    dump_path = Path(folder_name).joinpath("simulation-dumps")
    dump_path = dump_path.joinpath(f"seed-{sim.seed}")
    dump_path.mkdir(exist_ok=True, parents=True)

    dump_path = dump_path.joinpath(f"sim-{sim.tauExt:.4f}.npz")
    np.savez(str(dump_path), params=parameters, y1=sim.y1, y2=sim.y2)

    pass

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

def loadDepinningDumps(folder, partial:bool):
    vCm = list()
    stresses = None
    for file in Path(folder).iterdir():
        with open(file, "r") as fp:
            try:
                depinning = json.load(fp)
            except:
                print(f"File {file} is corrupted.")

            if stresses == None:
                stresses = depinning["stresses"]
            
            if partial:
                vCm_i = depinning["relaxed_velocities_total"] # TODO: Unify this for single and partial dislocation
            else:
                vCm_i = depinning["v_rel"]
            vCm.append(vCm_i)
    
    return (stresses, np.array(vCm))

if __name__ == "__main__":
    stresses, vCm = loadDepinningDumps('results/triton/single-dislocation/depinning-dumps/', partial=False)
    stresses1, vCm_partial = loadDepinningDumps('results/triton/depinning-dumps-partial/', partial=True)

    makeDepinningPlotAvg(10000, 100, [stresses, stresses1], [vCm[0:100], vCm_partial[0:100]], ["single", "partial"], 
                         folder_name="results/16-feb", colors=["red", "blue"])

    # sim = loadResults("results/15-feb-1/pickle-dumps/seed-100/sim-3.0000.npz")
    # makeVelocityPlot(sim, "results/15-feb-1/")
    # makeGif(sim, "results/15-feb-1/")