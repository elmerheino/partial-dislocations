import argparse
from pathlib import Path
import shutil
import numpy as np
from src.core.singleDislocation import DislocationSimulation
import multiprocessing as mp
from functools import partial
import json
import fcntl
import time
from src.core.partialDislocation import PartialDislocationsSimulation

def parse_args():
    parser = argparse.ArgumentParser(description='Find relaxed configurations for some noise levels.')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='sub-command help')

    # Create the parser for the "new" command
    parser_new = subparsers.add_parser('new', help='Start a new simulation run')
    parser_new.add_argument('--seeds', type=int, required=True, help='how many realizations of each noise')
    parser_new.add_argument('--length', type=int, required=True, help='length of the system such that L=N')
    parser_new.add_argument('--n', type=int, required=True, help='system size')
    parser_new.add_argument('--rmin', type=float, required=True, help='Minimum delta R value')
    parser_new.add_argument('--rmax', type=float, required=True, help='Maximum delta R value')
    parser_new.add_argument('--rpoints', type=int, required=True, help='Number of points between rmin and rmax')

    parser_new.add_argument('--folder', type=str, required=True, help='output folder path')

    parser_new.add_argument('--d0', type=float, help='Initial separation of partials. Required for --partial.')
    group = parser_new.add_mutually_exclusive_group(required=True)
    group.add_argument('--partial', action='store_true', help='Enable partial dislocations simulation')
    group.add_argument('--perfect', action='store_true', help='Enable perfect dislocations simulation')

    parser_new.add_argument('-c', '--cores', type=int, required=True, help='number of cores to use')

    # Create the parser for the "continue" command
    parser_continue = subparsers.add_parser('continue', help='Continue a previous simulation run')
    parser_continue.add_argument('--run-params', type=str, required=True, help='Path to the run_params.json file of the run to continue.')
    parser_continue.add_argument('-c', '--cores', type=int, required=True, help='number of cores to use')

    return parser.parse_args()

def update_noise_list(run_params_path, deltaR, seed):
    # Updated the run_params.json file in a multiprocessing compatible way.
    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            with open(run_params_path, 'r+') as f:
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

    pass

def relax_one_dislocations(deltaRseed, length, bigN, folder, y0=None, t0=None):
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
    sim = DislocationSimulation(bigN=bigN, length=length, time=1, dt=1, deltaR=deltaR, bigB=1, smallB=1, mu=1, tauExt=0, 
                                cLT1=1, seed=seed)

    # Find a minima using FIRE, and then save it
    y0_fire, success = sim.relax_w_FIRE()
    if success:
        sim.setInitialY0Config(y0_fire, 0)
    
    failsafe_path = Path(folder).joinpath(f"relaxation-failsafes/dislocation-{sim.getUniqueHashString()}.npz")
    failsafe_path.parent.mkdir(exist_ok=True, parents=True)
    sim.run_in_chunks(failsafe_path, sim.time/10, shape_save_freq=1)
    y = sim.getLineProfiles()

    results_save_path = Path(folder).joinpath(f"relaxed-configurations/dislocation-noise-{deltaR}-seed-{seed}.npz")
    results_save_path.parent.mkdir(exist_ok=True, parents=True)

    np.savez( results_save_path, y_fire=y0_fire, y=y, params=sim.getParameteters(), success=success )

    # If relaxation was successfull, update the run_params file to keep track
    if success:
        params_file = Path(folder).joinpath("run_params.json")
        update_noise_list(params_file, deltaR, seed)

def relax_one_partial_dislocation(deltaRseed, length, bigN, folder, d0, y0_1=None, y0_2=None, t0=None):
    # Unpack seed and noise then construct the results path
    deltaR, seed = deltaRseed
    results_save_path = Path(folder).joinpath(f"relaxed-configurations/dislocation-noise-{deltaR}-seed-{seed}.npz")
    results_save_path.parent.mkdir(exist_ok=True, parents=True)

    # Create the partial dislocation object
    sim = PartialDislocationsSimulation(bigN=bigN, length=length, time=200000, dt=100, deltaR=deltaR, bigB=1, smallB=1, b_p=1, mu=1, tauExt=0, 
                                cLT1=1, cLT2=1, seed=seed, d0=d0)
    
    # Check if the results file already exists, if it does, use it as initial config for the FIRE relaxation
    if results_save_path.exists():
        data = np.load(results_save_path)
        if not data['success']:
            data['y1_fire']
            data['y2_fire']
            sim.setInitialY0Config(data['y1_fire'], data['y2_fire'])

    # Find a minima using FIRE and then save it to a file
    y1_0_fire, y2_0_fire, success = sim.relax_w_FIRE()
    if success:
        sim.setInitialY0Config(y1_0_fire, y2_0_fire, 0)     # Integrate for a while after all, but just for a short while
        np.savez(results_save_path, y1_fire=y1_0_fire, y2_fire=y2_0_fire, params=sim.getParameters(), 
            success=success)

    # else: just use the default initial value from the constructor
    
    failsafe_path = Path(folder).joinpath(f"relaxation-failsafes/dislocation-{sim.getUniqueHashString()}.npz")
    failsafe_path.parent.mkdir(exist_ok=True, parents=True)
    sim.run_in_chunks(failsafe_path, chunk_size=sim.time/10, shape_save_freq=1)

    y1, y2 = sim.getLineProfiles()

    np.savez(results_save_path, y1_fire=y1_0_fire, y2_fire=y2_0_fire, y1=y1, y2=y2, params=sim.getParameters(), 
             success=success)

    # If the simulation was successfull, update the list to keep track
    if success:
        params_file = Path(folder).joinpath("run_params.json")
        update_noise_list(params_file, deltaR, seed)

def fn(x, folder, only_fire):
    t0  = x[2]
    t_n = x[1]

    print(f"Relaxing dislcoation w/ noise = {x[3]['deltaR'].astype(float)} seed = {x[3]['seed'].astype(int)} from t_0 = {t0} to t_n = {t_n} ")
    relax_one_dislocations( ( x[3]['deltaR'].astype(float),  x[3]['seed'].astype(int) ), t_n, x[3]['dt'].astype(float), x[3]['length'].astype(int), 
                                        x[3]['bigN'].astype(int), Path(folder), only_fire, y0=x[0], t0=t0 )
    
def pickup_where_left(run_params: Path, cores=8):
    # Load parameters of the previous run to dict
    run_params = Path(run_params)
    with open(run_params, "r") as fp:
        params = json.load(fp)

    seeds = params["args used"]["seeds"]
    bigN = params["args used"]["n"]
    bigL = params["args used"]["length"]
    is_partial_dislocation = params["args used"]["partial"]
    d0 = params["args used"]["d0"]
    
    folder = run_params.parent

    # Find out which noises have already been relaxed and which not
    succesfull_noises = [(i,j) for i,j in params['successful noise-seeds']]
    total_noises = [(i,j) for i,j in params['noise-seeds']]

    unsuccesfull_noises = list(set(total_noises) - set(succesfull_noises))

    # Next relax with FIRE all these unsuccesfull_noises
    # for noise_seed in list(unsuccesfull_noises):

    if is_partial_dislocation:
        with mp.Pool(cores) as pool:
            pool.map(partial(relax_one_partial_dislocation, length=bigL, bigN=bigN, folder=folder, d0=d0), list(unsuccesfull_noises))
            # relax_one_partial_dislocation(noise_seed, bigL, bigN, folder, d0)
    else:
        with mp.Pool(cores) as pool:
            pool.map(partial(relax_one_dislocations, length=bigL, bigN=bigN, folder=folder), list(unsuccesfull_noises))
    # Equivalent sequential code for debugging
    # for i in zip(unsuccessful_failsafes, og_times, fail_times, unsuccesfull_params):
    #     partial(fn, folder=Path(folder), only_fire=only_fire)(i)
    
    # for i in noise_seed_pairs_no_failsafe:
    #     partial(relax_one_dislocations, time=bigTime, dt=dt, length=bigL, bigN=bigN, folder=Path(folder))(i)
                


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

    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)

    with mp.Pool(args.cores) as pool:
        pool.map(partial(relax_one_dislocations, length=args.length, folder=args.folder, bigN=args.n), noise_seed_pairs)

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

    # TODO: implement option to pick up from where we left or where the simulation timed out
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)

    with mp.Pool(args.cores) as pool:
        pool.map(partial(relax_one_partial_dislocation, length=args.length, folder=args.folder,
                            bigN=args.n, d0=args.d0), noise_seed_pairs)

    pass

def main_w_args():
    args = parse_args()
    if args.mode == "new":
        if args.perfect:
            perfect_logic(args)
        elif args.partial:
            partial_logic(args)
    elif args.mode == "continue":
        pickup_where_left(args.run_params, args.cores)

if __name__ == '__main__':
    main_w_args()