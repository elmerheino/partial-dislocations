import argparse
from pathlib import Path
import shutil
import numpy as np
from singleDislocation import DislocationSimulation
import multiprocessing as mp
from functools import partial
import json
import fcntl
import time
from partialDislocation import PartialDislocationsSimulation

def parse_args():
    parser = argparse.ArgumentParser(description='Find relaxed configurations for some noise levels.')
    parser.add_argument('--rmin', type=float, required=True,
                      help='Minimum delta R value')
    parser.add_argument('--rmax', type=float, required=True,
                      help='Maximum delta R value')
    parser.add_argument('--rpoints', type=int, required=True,
                      help='Number of points between rmin and rmax')
    parser.add_argument('--seeds', type=int, required=True, help='how many realizations of each noise')
    parser.add_argument('--length', type=int, required=True, help='length of the system L=N')

    parser.add_argument('--time', type=int, required=True, help='relaxation time')
    parser.add_argument('--n', type=int, required=True, help='system size')
    parser.add_argument('--dt', type=float, required=True, help='time step')
    parser.add_argument('--folder', type=str, required=True, help='output folder path')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--partial', action='store_true', help='Enable partial dislocations simulation')
    group.add_argument('--perfect', action='store_true', help='Enable perfect dislocations simulation')

    parser.add_argument('-c', '--cores', type=int, required=True, help='output folder path')

    return parser.parse_args()

def relax_one_dislocations(deltaRseed, time, dt, length, bigN, folder, y0=None, t0=None):
    """
    Simulates the relaxation of a dislocation under given parameters and saves the results.
    Args:
        deltaRseed (tuple): A tuple containing the noise amplitude (deltaR) and the random seed.
        time (float): The total simulation time.
        dt (float): The time step for the simulation.
        length (int): The length of the simulation domain.
        bigN (int): The number of points used to discretize the dislocation line.
        folder (str): The folder path to save the simulation results and backup files.
    Returns:
        None
    Raises:
        Exception: If any error occurs during the simulation or saving process.
    Saves:
        - Backup files during the simulation to `folder/failsafes/`.
        - Final relaxed dislocation configurations, VCM history, line profiles, and parameters to `folder/relaxed-configurations/`.
        - `run_params.json` is updated with successful noise amplitudes.
    # selected_ys matrix shape:
    # [[time, y1, y2, y3, ..., yN],
    #  [time, y1, y2, y3, ..., yN],
    #  [time, y1, y2, y3, ..., yN],
    #  ...,
    #  [time, y1, y2, y3, ..., yN]]
    # where time is the simulation time and y1 to yN represent the dislocation shape at that time, N being the bigN used
    in the simulation
    """
    deltaR, seed = deltaRseed
    sim = DislocationSimulation(bigN=bigN, length=length, time=time, dt=dt, deltaR=deltaR, bigB=1, smallB=1, mu=1, tauExt=0, 
                                cLT1=1, seed=seed)

    if (type(y0) != type(None)) and (type(t0) != type(None)):
        sim.setInitialY0Config(y0=y0, t0=t0)
        sim.setTauCutoff(0)

    backup_file = Path(folder).joinpath(f"failsafes/backup-{sim.getUniqueHashString()}.npz")
    backup_file.parent.mkdir(parents=True, exist_ok=True)

    # Find a minima using FIRE, and set it as y0
    y0_fire = sim.relax_w_FIRE()
    if type(y0_fire) != type(None):
        # If successsull, set it as the y0 of the integrator
        sim.setInitialY0Config(y0_fire, sim.t0)

    # Save three dislocation shapes from each chunk
    sim.run_until_relaxed(backup_file, chunk_size=sim.time/10, shape_save_freq=3, method='RK45')

    results_save_path = Path(folder).joinpath(f"relaxed-configurations/dislocation-noise-{deltaR}-seed-{seed}.npz")
    results_save_path.parent.mkdir(exist_ok=True, parents=True)

    sim.saveResults(results_save_path)

    max_retries = 5
    retry_delay = 1

    params_file = Path(folder).joinpath("run_params.json")

    for attempt in range(max_retries):
        try:
            with open(params_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    params = json.load(f)

                    deltaR = float(deltaR)
                    seed = int(seed)

                    noise_seed_pair = [deltaR, seed]

                    params["successful noise-seeds"].append(noise_seed_pair)
                    f.seek(0)
                    json.dump(params, f, indent=4)
                    f.truncate()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            break
        except (IOError, BlockingIOError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

def relax_one_partial_dislocation(deltaRseed, time, dt, length, bigN, folder, y0_1=None, y0_2=None, t0=None):
    # Create the partial dislocation object
    deltaR, seed = deltaRseed
    sim = PartialDislocationsSimulation(bigN=bigN, length=length, time=time, dt=dt, deltaR=deltaR, bigB=1, smallB=1, b_p=1, mu=1, tauExt=0, 
                                cLT1=1, cLT2=1, seed=seed)

    if (type(y0_1) != type(None)) and (type(t0) != type(None)) and (type(y0_2) != type(None)):
        sim.setInitialY0Config(y0_1, y0_2)
        sim.setTauCutoff(0)

    backup_file = Path(folder).joinpath(f"failsafes/backup-{sim.getUniqueHashString()}.npz")
    backup_file.parent.mkdir(parents=True, exist_ok=True)

    # Find a minima using FIRE, and set it as y0
    y1_0_fire, y2_0_fire = sim.relax_w_FIRE()
    if type(y1_0_fire) != type(None):
        # If successsull, set it as the y0 of the integrator
        sim.setInitialY0Config(y1_0_fire, y2_0_fire)

    # Run the simulation for a while using linear interpolation, save three dislocation shapes from each chunk
    sim.run_until_relaxed(backup_file, chunk_size=sim.time/10, shape_save_freq=1, method='RK45')

    results_save_path = Path(folder).joinpath(f"relaxed-configurations/dislocation-noise-{deltaR}-seed-{seed}.npz")
    results_save_path.parent.mkdir(exist_ok=True, parents=True)
    sim.saveResults(results_save_path)

    # Update run_params.json in a way compatible with multiprocessing
    max_retries = 5
    retry_delay = 1

    params_file = Path(folder).joinpath("run_params.json")

    for attempt in range(max_retries):
        try:
            with open(params_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    params = json.load(f)

                    deltaR = float(deltaR)
                    seed = int(seed)

                    noise_seed_pair = [deltaR, seed]

                    params["successful noise-seeds"].append(noise_seed_pair)
                    f.seek(0)
                    json.dump(params, f, indent=4)
                    f.truncate()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            break
        except (IOError, BlockingIOError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)

def fn(x, folder):
    t0  = x[2]
    t_n = x[1]

    print(f"Relaxing dislcoation w/ noise = {x[3]['deltaR'].astype(float)} seed = {x[3]['seed'].astype(int)} from t_0 = {t0} to t_n = {t_n} ")
    relax_one_dislocations( ( x[3]['deltaR'].astype(float),  x[3]['seed'].astype(int) ), t_n, x[3]['dt'].astype(float), x[3]['length'].astype(int), 
                                        x[3]['bigN'].astype(int), Path(folder), x[0], t0 )
    
def pickup_where_left(folder, cores=8):
    # Load parameters of the previous run to dict
    with open(Path(folder).joinpath("run_params.json"), "r") as fp:
        params = json.load(fp)
    
    # Find out which noises have already been relaxed
    succesfull_noises = [(i,j) for i,j in params['successful noise-seeds']]
    total_noises = [(i,j) for i,j in params['noise-seeds']]

    seeds = params["args used"]["seeds"]
    bigN = params["args used"]["n"]
    bigL = params["args used"]["length"]
    dt = params["args used"]["dt"]
    bigTime = params["args used"]["time"]

    unsuccesfull_noises = list(set(total_noises) - set(succesfull_noises))

    # Next find out if any good failsafe files exist for these noises, and gather the relevant parameters.
    failsafe_path = Path(folder).joinpath("failsafes")

    unsuccessful_failsafes = list() # The shape profiles of unsuccessful failsafes
    fail_times = list()     # Time left to integrate until max
    og_times = list()
    unsuccesfull_params = list()    # List of dicts

    for failsafe_path in failsafe_path.iterdir():
        failsafe = np.load(failsafe_path)
        params_i = DislocationSimulation.paramListToDict(failsafe['params'])
        y0_i = failsafe['y_last']               # The shape of dislocation line where if ended
        t_f = failsafe['last_success_time']     # The time where if failed, if it did so
        t_og = failsafe['og_time']              # The time it was meant to run

        deltaR_i = params_i['deltaR']
        seed_i = params_i['seed']

        time_to_integrate = t_og - t_f

        if (deltaR_i, seed_i) in unsuccesfull_noises:
            unsuccessful_failsafes.append(y0_i)
            unsuccesfull_params.append(params_i)
            fail_times.append(t_f)
            og_times.append(t_og)
        
        # Move the failsafe file to archive to prevent failsafe duplication.
        archive_path = Path(folder).joinpath(f"archive-failsafes/{failsafe_path.name}")
        archive_path.parent.mkdir(exist_ok=True, parents=True)
        shutil.move(failsafe_path, archive_path)
            
    
    # Next check if there are some noise levels, which don't have any failsafes
    failsafe_noises = map(lambda x : x['deltaR'], unsuccesfull_params)
    no_failsafe_noises = list(set(unsuccesfull_noises) - set(failsafe_noises))

    print(f"Found dislocation w/ no failsafe at noises {no_failsafe_noises}")
    noise_seed_pairs_no_failsafe = [(float(noise), int(seed)) for noise, seed in no_failsafe_noises]

    # Then relax these dislocations, with and without failsafes until end, 
    # and save the results, first the ones with failsafe, then rest
    with mp.Pool(cores) as pool:
        pool.map(partial(fn, folder=Path(folder)), zip(unsuccessful_failsafes, og_times, fail_times, unsuccesfull_params))
        pool.map(partial(relax_one_dislocations, time=bigTime, dt=dt, length=bigL, bigN=bigN, folder=Path(folder)), 
                noise_seed_pairs_no_failsafe)

def perfect_logic(args):
    noises = np.logspace(args.rmin,args.rmax, args.rpoints)
    noise_seed_pairs = [(float(noise), seed) for noise in noises for seed in range(args.seeds)]

    params_dict = {
        "noise-seeds" : noise_seed_pairs,
        "rmin" : args.rmin,
        "rmax" : args.rmax,
        "rpoints" : args.rpoints,
        "successful noise-seeds" : list(),
        "noise spacing":"log",
        "noise gen command":f"np.logspace({args.rmin}, {args.rmax}, {args.rpoints})",
        "args used":vars(args)
    }

    params_file = Path(args.folder).joinpath("run_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)

    if params_file.exists():
        pickup_where_left(Path(args.folder), args.cores)
    else:
        with open(params_file, 'w') as f:
            json.dump(params_dict, f, indent=4)

        with mp.Pool(args.cores) as pool:
            pool.map(partial(relax_one_dislocations, time=args.time, dt=args.dt, length=args.length, folder=args.folder,
                             bigN=args.n), noise_seed_pairs)

def partial_logic(args):
    noises = np.logspace(args.rmin,args.rmax, args.rpoints)
    noise_seed_pairs = [(float(noise), seed) for noise in noises for seed in range(args.seeds)]

    params_dict = {
        "noise-seeds" : noise_seed_pairs,
        "rmin" : args.rmin,
        "rmax" : args.rmax,
        "rpoints" : args.rpoints,
        "successful noise-seeds" : list(),
        "noise spacing":"log",
        "noise gen command":f"np.logspace({args.rmin}, {args.rmax}, {args.rpoints})",
        "args used":vars(args)
    }

    params_file = Path(args.folder).joinpath("run_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)

    # if params_file.exists():
    #     # pickup_where_left(Path(args.folder), args.cores)
    #     pass
    # else:
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)

    with mp.Pool(args.cores) as pool:
        pool.map(partial(relax_one_partial_dislocation, time=args.time, dt=args.dt, length=args.length, folder=args.folder,
                            bigN=args.n), noise_seed_pairs)
    
    # for i in noise_seed_pairs:
    #     fn = partial(relax_one_partial_dislocation, time=args.time, dt=args.dt, length=args.length, folder=args.folder,
    #                         bigN=args.n)
    #     fn(i)

    pass

def main_w_args():
    args = parse_args()
    if args.perfect:
        perfect_logic(args)
    elif args.partial:
        partial_logic(args)
    else:
        print("Must pass either --partial or --perfect, now neither is passed.")

if __name__ == '__main__':
    main_w_args()