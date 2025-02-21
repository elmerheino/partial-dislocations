import numpy as np
from partialDislocation import PartialDislocationsSimulation
from singleDislocation import *
import multiprocessing as mp
from functools import partial
from argparse import ArgumentParser
from plots import *
from processData import *
from depinning import *

# import time

save_plots = False      # Don't save any images of figures. Still saves all data as dumps.
    
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

        depinning = DepinningPartial(tau_min=float(parsed.tau_min), tau_max=float(parsed.tau_max), points=int(parsed.points),
                        time=float(parsed.time), dt=float(parsed.timestep), seed=int(parsed.seed), 
                        folder_name=parsed.folder, cores=cores)
        v1, v2, vcm = depinning.run()
        dumpDepinning(depinning.stresses, vcm, depinning.time, depinning.seed, depinning.dt, folder_name=parsed.folder, extra=[v1, v2])

    elif parsed.single:        

        depinning = DepinningSingle(tau_min=float(parsed.tau_min), tau_max=float(parsed.tau_max), points=int(parsed.points),
                        time=float(parsed.time), dt=float(parsed.timestep), seed=int(parsed.seed), 
                        folder_name=parsed.folder, cores=cores)
        vcm = depinning.run()
        dumpDepinning(depinning.stresses, vcm, depinning.time, depinning.seed, depinning.dt, folder_name=parsed.folder)

    else:
        raise Exception("Not specified which type of dislocation must be simulated.")

    # time_elapsed = time.time() - k
    # print(f"It took {time_elapsed} for the script to run.")
    pass

if __name__ == "__main__":
    triton()
    pass