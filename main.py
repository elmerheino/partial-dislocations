import numpy as np
from simulation import PartialDislocationsSimulation
from pathlib import Path
import pickle
from tqdm import tqdm
import json
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser
from plots import *
# import time

def dumpResults(sim: PartialDislocationsSimulation, folder_name: str):
    # Dumps the results of a simulation to a json file
    if not sim.has_simulation_been_run:
        raise Exception("Simulation has not been run.")
    
    parameters = [
        sim.bigN, sim.length, sim.time, sim.dt,
        sim.deltaR, sim.bigB, sim.smallB, sim.b_p,
        sim.cLT1, sim.cLT2, sim.mu, sim.tauExt, sim.c_gamma,
        sim.d0, sim.seed, sim.tau_cutoff
        ] # From these parameters you should be able to replicate the simulation
    
    y1_as_list = [list(row) for row in sim.y1]  # Convert the ndarray to a list of lists
    y2_as_list = [list(row) for row in sim.y2]

    result = {
        "parameters":parameters,
        "y1":sim.y1,
        "y2":sim.y2
    }

    dump_path = Path(folder_name).joinpath("pickle-dumps")
    dump_path = dump_path.joinpath(f"seed-{sim.seed}")
    dump_path.mkdir(exist_ok=True, parents=True)

    dump_path = dump_path.joinpath(f"sim-{sim.tauExt:.4f}.pickle")
    with open(str(dump_path), "wb") as fp:
        pickle.dump(result,fp)
    pass

def loadResults(file_path):
    # Load the results from a file at file_path to a new simulation object for further processing

    with open(file_path, "rb") as fp:
        res_dict = pickle.load(fp)

    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff = res_dict["parameters"]
    res = PartialDislocationsSimulation(bigN=bigN, bigB=bigB, length=length,
                                        time=time,timestep_dt=dt, deltaR=deltaR, 
                                        smallB=smallB, b_p=b_p, cLT1=cLT1, cLT2=cLT2,
                                        mu=mu, tauExt=tauExt, c_gamma=c_gamma, d0=d0, seed=seed)
    
    # Modify the internal state of the object according to preferences
    res.tau_cutoff = tau_cutoff
    res.has_simulation_been_run = True

    res.y1 = res_dict["y1"]
    res.y2 = res_dict["y2"]

    return res

def studyConstantStress(tauExt,
                        timestep_dt,
                        time, seed=None, folder_name="results",):
    # Run a simulation with a specified constant stress

    simulation = PartialDislocationsSimulation(tauExt=tauExt, bigN=1024, length=1024, 
                                              timestep_dt=timestep_dt, time=time, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1, seed=seed)
    # print(" ".join(simulation.getParamsInLatex()))

    simulation.run_simulation()

    # dumpResults(simulation, folder_name)

    rV1, rV2, totV2 = simulation.getRelaxedVelocity(time_to_consider=1000) # The velocities after relaxation

    # makeStressPlot(simulation, folder_name)

    return (rV1, rV2, totV2)

def studyDepinning_mp(tau_min:float, tau_max:float, points:int,
                   timestep_dt:float, time:float, seed:int=None, folder_name="results", cores=1):
    # Multiprocessing compatible version of a single depinning study, here the studies
    # are distributed between threads by stress letting python mp library determine the best way
    
    stresses = np.linspace(tau_min, tau_max, points)
    
    with mp.Pool(cores) as pool:
        results = pool.map(partial(studyConstantStress, folder_name=folder_name, timestep_dt=timestep_dt, time=time, seed=seed), stresses)
    
    
    v1_rel = [i[0] for i in results]
    v2_rel = [i[1] for i in results]
    v_cm = [i[2] for i in results]

    makeDepinningPlot(stresses, v_cm, time, seed, folder_name=folder_name)

    results_json = {
        "stresses":stresses.tolist(),
        "v1_rel": v1_rel,
        "v2_rel": v2_rel,
        "relaxed_velocities_total":v_cm,
        "seed":seed,
        "time":time,
        "dt":timestep_dt
    }

    depining_path = Path(folder_name)
    depining_path = depining_path.joinpath("depinning-dumps")
    depining_path.mkdir(exist_ok=True, parents=True)
    depining_path = depining_path.joinpath(f"depinning-{tau_min}-{tau_max}-{points}-{time}-{seed}.json")
    with open(str(depining_path), 'w') as fp:
        json.dump(results_json,fp)

def studyDepinning(tau_min, tau_max, points,
                   timestep_dt, time,folder_name="results", seed=None): # Fix the stress field for different tau_ext
    
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    stresses = np.linspace(tau_min, tau_max, points)
    r_velocities = list()

    for s,p in zip(stresses, tqdm(range(points), desc="Running simulations", unit="simulation")):
        rel_v1, rel_v2, rel_tot = studyConstantStress(tauExt=s, folder_name=folder_name, timestep_dt=timestep_dt, time=time, seed=seed)
        # print(f"Simulation finished with : v1_cm = {rel_v1}  v2_cm = {rel_v2}  v2_tot = {rel_tot}")
        r_velocities.append(rel_tot)
    
    makeDepinningPlot(stresses, r_velocities, time, seed, folder_name=folder_name)

    results = {
        "stresses":stresses.tolist(),
        "relaxed_velocities":r_velocities,
        "seed":seed
    }

    with open(f"{folder_name}/depinning-{tau_min}-{tau_max}-{points}-{time}-{seed}.json", 'w') as fp:
        json.dump(results,fp)
    
def multiple_depinnings_mp(seed):
    # Helper function to run depinning studies one study per thread
    studyDepinning(folder_name="results/11-feb-n2", tau_min=2, tau_max=4, timestep_dt=0.05, time=10000, seed=seed)

def jotain_roskaa():
    # Run multiple such depinning studies with varying seeds
    time = 10000
    dt = 0.5
    studyDepinning_mp(tau_min=2.25, tau_max=2.75, points=50, time=time, timestep_dt=dt, seed=2, folder_name="/Volumes/Tiedostoja/dislocationData/14-feb-n1")
    seeds = range(11,21)
    for seed,_ in zip(seeds, tqdm(range(len(seeds)), desc="Running depinning studies", unit="study")):
        studyDepinning_mp(tau_min=2.25, tau_max=2.75, points=50, time=time, timestep_dt=dt, seed=seed, folder_name="results/13-feb-n2")

    loadedSim = loadResults("joku-simulaatio.pickle")
    makeGif(loadedSim, "results/gifs")

def triton():
    # k = time.time()
    parser = ArgumentParser(prog="Dislocation simulation")
    parser.add_argument('-s', '--seed', help='Specify seed for the individual depinning study. If not specified, seed will be randomized between stresses.', required=True)
    parser.add_argument('-f', '--folder', help='Specify the output folder for all the dumps and results.', required=True)
    parser.add_argument('-tmin', '--tau-min', help='Start value for stress.', required=True)
    parser.add_argument('-tmax', '--tau-max', help='End value for stress.', required=True)
    parser.add_argument('-p', '--points', help='How many points to consider between tau_min and tau_max', required=True)
    parser.add_argument('-dt', '--timestep', help='Timestep size in (s).', required=True)
    parser.add_argument('-t', '--time', help='Total simulation time in (s).', required=True)
    parser.add_argument('-c', '--cores', help='Cores to use in multiprocessing pool.', required=True)

    parsed = parser.parse_args()

    estimate = (int(parsed.time)/float(parsed.timestep))*1024*2*4*1e-6
    # input(f"One simulation will take up {estimate:.1f} MB disk space totalling {estimate*int(parsed.points)*1e-3:.1f} GB")

    studyDepinning_mp(tau_min=float(parsed.tau_min), tau_max=float(parsed.tau_max), points=int(parsed.points),
                       time=float(parsed.time), timestep_dt=float(parsed.timestep), seed=int(parsed.seed), 
                       folder_name=parsed.folder, cores=int(parsed.cores))
    # time_elapsed = time.time() - k
    # print(f"It took {time_elapsed} for the script to run.")
    pass

if __name__ == "__main__":
    triton()