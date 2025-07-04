import argparse
from pathlib import Path
import numpy as np
from singleDislocation import DislocationSimulation
import multiprocessing as mp
from functools import partial

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
    deltaR, seed = deltaRseed
    sim = DislocationSimulation(bigN=bigN, length=length, time=time, dt=dt, deltaR=deltaR, bigB=1, smallB=1, mu=1, tauExt=0, 
                                cLT1=1, seed=seed)
    
    backup_file = Path(folder).joinpath(f"failsafes/backup-{sim.getUniqueHashString()}.npz")
    backup_file.parent.mkdir(parents=True, exist_ok=True)

    sim.run_until_relaxed(backup_file, chunk_size=sim.time/10, shape_save_freq=10000, method='RK45')

    results_save_path = Path(folder).joinpath(f"relaxed-configurations/dislocation-noise-{deltaR}-seed-{seed}.npz")
    results_save_path.parent.mkdir(exist_ok=True, parents=True)

    v_cm_hist = sim.getVCMhist()
    y_t = sim.getLineProfiles()
    parameters = sim.getParameteters()
    selected_ys = sim.getSelectedYshapes()

    np.savez(results_save_path, v_cm_hist=v_cm_hist, y_last=y_t, selected_y=selected_ys, params=parameters)

if __name__ == '__main__':
    args = parse_args()
    noises = np.logspace(args.rmin,args.rmax, args.rpoints)

    noise_seed_pairs = [(noise, seed) for noise in noises for seed in range(args.seeds)]

    with mp.Pool(args.cores) as pool:
        pool.map(partial(relax_one_dislocations, time=args.time, dt=args.dt, length=args.length, folder=args.folder, bigN=args.n), noise_seed_pairs)