#!/usr/bin/env python3
from depinning import *
import numpy as np
import multiprocessing as mp
import argparse
from pathlib import Path
from singleDislocation import DislocationSimulation

def getTauLimits(noise):
    if noise < 0.01:
        return (0, noise/10)
    else:
        return (0, noise)

def compute_depinnings_from_dir(input_folder : Path, task_id : int, cores : int, points, time : int, dt : int, output_folder, perfect : bool):
    # Read metadata that is left to dir from last run to figure out range of allowed params and print helpful info
    with open(input_folder.joinpath("run_params.json"), "r") as fp:
        metadata = json.load(fp)
        noise_count = metadata['args used']['rpoints']
        seed_count = metadata['args used']['seeds']
    
    max_task_id = noise_count*seed_count - 1
    if not ( (0 <= task_id) and (task_id <= max_task_id) ):     # Ensure that 0 < task_id < max_task_id
        print(f"Task id should be within range {0} <= task-id <= {max_task_id} now it is {task_id}")
        return

    # Find all the relaxed configurations preesent in the passed dir
    rel_folder = input_folder.joinpath("relaxed-configurations")
    paths_list = [str(i) for i in rel_folder.iterdir()]

    # Load a file keeping track of all the relaxed configurations in the file, create it if it doesn't exist
    paths_file = input_folder.joinpath("paths_index.json")
    if paths_file.exists():
        with open(paths_file, "r") as fp:
            paths_data = json.load(fp)
        paths = paths_data["paths"]
    else:
        with open(paths_file, "w+") as fp:
            json.dump({"paths" : paths_list, "len" : len(paths_list)}, fp) # Len should be = seeds * (no. of noises)
        paths = paths_list
    
    # Load the relaxed configuration at hand
    initial_config_path = paths[task_id]
    initial_config = np.load(initial_config_path)

    if perfect:
        print(initial_config.files)
        y0 = initial_config['y_last']
        params = DislocationSimulation.paramListToDict(initial_config['params'])

        # Create the approproate depinning object
        tau_min, tau_max = getTauLimits(params['deltaR'])

        depinning_perfect = DepinningSingle(tau_min=tau_min, tau_max=tau_max, points=points, time=time, dt=dt, cores=cores,
                                            folder_name=input_folder, deltaR=params['deltaR'], seed=params['seed'].astype(int), 
                                            bigN=params['bigN'].astype(int), length=params['length'].astype(int) )
        depinning_perfect.run(y0_rel=y0)
        # depinning_perfect.dump_res_to_pickle(perfect_folder.joinpath(f"depinning-pickle-dumps"))
        depinning_perfect.save_results(output_folder)
    else:
        params = PartialDislocationsSimulation.paramListToDict(initial_config['params'])
        y_last = initial_config['y_last']
        print(y_last.shape)
        tau_min, tau_max = getTauLimits(params['deltaR'])
        depinnin_partial = DepinningPartial(tau_min, tau_max, points, time, dt, cores, output_folder, float(params['deltaR']),
                                            int(params['seed']), int(params['bigN']),  int(params['length']), 10)
        depinnin_partial.run(y1_0=y_last[0], y2_0=y_last[1])
        depinnin_partial.save_results(output_folder)
        pass
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        A script to run simulations. The array requested in triton should be from 0-(seed*noises - 1) since it will be passed
        as index to a list containing all the file paths in the directory
        """
        )
    parser.add_argument('--folder', type=str, required=True, 
                        help="""
                        Folder where the initially relaxed configurations are. should be be the one which containts folder 
                        intitial-relaxations
                        """
                        )
    parser.add_argument('--out-folder', type=str, required=True, help="Output folder of depinning.")
    parser.add_argument('--partial', action='store_true', help='Partial dislocation.')
    parser.add_argument('--perfect', action='store_true', help='Perfect dislocation.')
    parser.add_argument('--cores', type=int, required=True, help='Perfect dislocation.')
    parser.add_argument('--task-id', type=int, required=True, help='SLURM_ARRAY_TASK_ID from triton. ')
    parser.add_argument('--points', type=int, required=True, help='Number of tau_exts to be integrated.')
    parser.add_argument('--time', type=int, required=True, help='Time to integrate each simulation.')
    parser.add_argument('--dt', type=int, required=True, help='Timestep for sampling the solution.')



    args = parser.parse_args()

    rel_path = Path(args.folder)

    if args.partial:
        print(f"Depinning for partial dislcoation")
        compute_depinnings_from_dir(input_folder=Path(args.folder), task_id=args.task_id, cores=args.cores, points=args.points, time=args.time, dt=args.dt, output_folder=Path(args.out_folder), perfect=False)
    
    if args.perfect:
        print(f"Depinnign for perfect dislocation")
        compute_depinnings_from_dir(input_folder=Path(args.folder), task_id=args.task_id, cores=args.cores, points=args.points, time=args.time, dt=args.dt, output_folder=Path(args.out_folder), perfect=True)
