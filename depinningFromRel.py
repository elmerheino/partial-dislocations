from depinning import *
import numpy as np
import multiprocessing as mp
import argparse
from pathlib import Path
from singleDislocation import DislocationSimulation

def getTauLimits(noise):
    return (0, noise/10)

def perfect_depinning_worker(y0, params, points, limits, folder):
    try:
        tau_min, tau_max = limits
        depinning = DepinningSingle(tau_min, tau_max, points, 1000, 1, cores=0, folder_name=folder, 
                                    deltaR=params['deltaR'], seed=int(params['seed']), bigB=params['bigN'], 
                                    length=params['length'], sequential=True)
        depinning.run(y0_rel=y0)
        depinning.dump_res_to_pickle(folder.parent.joinpath(f"depinning-{tau_min}-{tau_max}-{points}"))
    except Exception as e:
        print(e)
    # depinning.save_results(folder.parent.joinpath(f"depinning-{tau_min}-{tau_max}-{points}"))
    pass

def perfect_depinning(folder : Path, cores, points):
    folder_relaxed = folder.joinpath("relaxed-configurations")
    with mp.Pool(cores) as pool:
        async_results = []

        for intial_conf in folder_relaxed.iterdir():
            data_i = np.load(intial_conf)
            y0 = data_i['y_last']
            params = DislocationSimulation.paramListToDict(data_i['params'])

            limits = getTauLimits(params['deltaR'])

            res_async = pool.apply_async(perfect_depinning_worker, args=(y0, params, points, limits, folder))
            async_results.append(res_async)
        
        for i, res in enumerate(async_results):
            try:
                result = res.get()
            except:
                print(f"Error getting result from task {i+1}")

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A script to run simulations.")
    parser.add_argument('--folder', type=str, required=True, help='Folder where the initially relaxed configurations are.')
    parser.add_argument('--partial', action='store_true', help='Partial dislocation.')
    parser.add_argument('--perfect', action='store_true', help='Perfect dislocation.')
    parser.add_argument('--cores', type=int, required=True, help='Perfect dislocation.')

    args = parser.parse_args()

    rel_path = Path(args.folder)

    if args.partial:
        print(f"Depinning for partial dislcoation")
    
    if args.perfect:
        print(f"Depinnign for perfect dislocation")
        perfect_depinning(rel_path, args.cores, 10)

    print(f"Relaxed intial configurations are in folder {args.folder}")