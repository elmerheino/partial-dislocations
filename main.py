import numpy as np
from simulation import PartialDislocationsSimulation
from singleDislocation import *
from pathlib import Path
from tqdm import tqdm
import json
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser
from plots import *
from processData import *
# import time

def studyConstantStress(tauExt,
                        timestep_dt,
                        time, seed=None, folder_name="results",):
    # Run a simulation with a specified constant stress

    simulation = PartialDislocationsSimulation(tauExt=tauExt, bigN=1024, length=1024, 
                                              timestep_dt=timestep_dt, time=time, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1, seed=seed)
    
    simulation.run_simulation()
    # dumpResults(simulation, folder_name)
    rV1, rV2, totV2 = simulation.getRelaxedVelocity(time_to_consider=1000) # The velocities after relaxation

    # makeVelocityPlot(simulation, folder_name)

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

    # makeDepinningPlot(stresses, v_cm, time, seed, folder_name=folder_name)

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

def studyConstantStressSingle(tauExt:float, timestep_dt:float, time:float, seed:int=None, folder_name="results"):
    # Study the velocity of a single relaxed dislocation.
    sim = DislocationSimulation(tauExt=tauExt, bigN=1024, length=1024, 
                                              timestep_dt=timestep_dt, time=time, d0=39, c_gamma=20, 
                                              cLT1=0.1, seed=seed)
    sim.run_simulation()
    v_rel = sim.getRelaxedVelocity()
    return v_rel

def studyDepinnningSingle_mp(tau_min:float, tau_max:float, points:int,
                   timestep_dt:float, time:float, seed:int=None, folder_name="results", cores=1):
    # Run a depinning study for a single dislocation

    stresses = np.linspace(tau_min, tau_max, points)

    with mp.Pool(cores) as pool:
        velocities = pool.map(partial(studyConstantStressSingle, folder_name=folder_name, timestep_dt=timestep_dt, time=time, seed=seed), stresses)
    
    # makeDepinningPlot(stresses, velocities, time, seed=seed, folder_name=folder_name)

    results_json = {
        "stresses":stresses.tolist(),
        "v_rel":velocities,
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

    pass

    
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
    parser.add_argument('-c', '--cores', help='Cores to use in multiprocessing pool. Is not specified, use all available.')
    parser.add_argument('--partial', help='Simulate a partial dislocation.', action="store_true")
    parser.add_argument('--single', help='Simulate a single dislocation.', action="store_true")

    parsed = parser.parse_args()

    estimate = (int(parsed.time)/float(parsed.timestep))*1024*2*4*1e-6
    # input(f"One simulation will take up {estimate:.1f} MB disk space totalling {estimate*int(parsed.points)*1e-3:.1f} GB")

    if parsed.cores == None:
        cores = mp.cpu_count()
    else:
        cores = int(parsed.cores)

    if parsed.partial:
        studyDepinning_mp(tau_min=float(parsed.tau_min), tau_max=float(parsed.tau_max), points=int(parsed.points),
                        time=float(parsed.time), timestep_dt=float(parsed.timestep), seed=int(parsed.seed), 
                        folder_name=parsed.folder, cores=cores)
    elif parsed.single:
        studyDepinnningSingle_mp(tau_min=float(parsed.tau_min), tau_max=float(parsed.tau_max), points=int(parsed.points),
                    time=float(parsed.time), timestep_dt=float(parsed.timestep), seed=int(parsed.seed), 
                    folder_name=parsed.folder, cores=cores)
    else:
        raise Exception("Not specified which type of dislocation must be simulated.")

    # time_elapsed = time.time() - k
    # print(f"It took {time_elapsed} for the script to run.")
    pass

if __name__ == "__main__":
    # studyDepinnningSingle_mp(tau_min=2.7, tau_max=3.25, points=50, timestep_dt=0.05, time=10000, seed=1, folder_name="results/single-dislocation", cores=5)
    triton()
    pass