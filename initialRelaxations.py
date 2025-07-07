import argparse
from pathlib import Path
import numpy as np
from singleDislocation import DislocationSimulation
import multiprocessing as mp
from functools import partial
import json
import fcntl
import time

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

    parser.add_argument('-c', '--cores', type=int, required=True, help='output folder path')

    return parser.parse_args()

def relax_one_dislocations(deltaRseed, time, dt, length, bigN, folder):
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
    try:
        deltaR, seed = deltaRseed
        sim = DislocationSimulation(bigN=bigN, length=length, time=time, dt=dt, deltaR=deltaR, bigB=1, smallB=1, mu=1, tauExt=0, 
                                    cLT1=1, seed=seed)
        
        backup_file = Path(folder).joinpath(f"failsafes/backup-{sim.getUniqueHashString()}.npz")
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        # Save three dislocation shapes from each chunk
        sim.run_until_relaxed(backup_file, chunk_size=sim.time/10, shape_save_freq=3, method='RK45')

        results_save_path = Path(folder).joinpath(f"relaxed-configurations/dislocation-noise-{deltaR}-seed-{seed}.npz")
        results_save_path.parent.mkdir(exist_ok=True, parents=True)

        v_cm_hist = sim.getVCMhist()
        y_t = sim.getLineProfiles()
        parameters = sim.getParameteters()
        selected_ys = sim.getSelectedYshapes()

        np.savez(results_save_path, v_cm_hist=v_cm_hist, y_last=y_t, 
                 selected_ys=selected_ys,
                 params=parameters)

        max_retries = 5
        retry_delay = 1

        params_file = Path(folder).joinpath("run_params.json")
    except:
        raise

    for attempt in range(max_retries):
        try:
            with open(params_file, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    params = json.load(f)
                    params["successful noises"].append(deltaR.astype(float))
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

def main_w_args():
    args = parse_args()
    noises = np.logspace(args.rmin,args.rmax, args.rpoints)

    params_dict = {
        "noises" : noises.tolist(),
        "rmin" : args.rmin,
        "rmax" : args.rmax,
        "rpoints" : args.rpoints,
        "successful noises" : list(),
        "noise spacing":"log",
        "noise gen command":f"np.logspace({args.rmin}, {args.rmax}, {args.rpoints})"
    }

    params_file = Path(args.folder).joinpath("run_params.json")
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with open(params_file, 'w') as f:
        json.dump(params_dict, f, indent=4)

    noise_seed_pairs = [(noise, seed) for noise in noises for seed in range(args.seeds)]

    # with mp.Pool(args.cores) as pool:
    #     pool.map(partial(relax_one_dislocations, time=args.time, dt=args.dt, length=args.length, folder=args.folder, bigN=args.n), noise_seed_pairs)
    for pair in noise_seed_pairs:
        fn = partial(relax_one_dislocations, time=args.time, dt=args.dt, length=args.length, folder=args.folder, bigN=args.n)
        fn(pair)

if __name__ == '__main__':
    main_w_args()