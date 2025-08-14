#!/usr/bin/env python3
from src.core.depinning import *
import numpy as np
import multiprocessing as mp
import argparse
from pathlib import Path
from src.core.singleDislocation import DislocationSimulation

def getInitialConfig(input_folder, task_id):
    # Load a file keeping track of all the relaxed configurations available, create it if it doesn't exist
    input_folder = Path(input_folder)
    paths_file = input_folder.joinpath("paths_index.json")
    if paths_file.exists():
        with open(paths_file, "r") as fp:
            paths_data = json.load(fp)
        paths = paths_data["paths"]    
    else:
        return None
    # Load the relaxed configuration that is associated with the current task_id. The task_id corresponds uniquely to a
    # (noise, seed) pair
    initial_config_path = paths[task_id]
    initial_config = np.load(initial_config_path)
    return initial_config

def getIntegrationTime(noise_delta_r : float) -> int:
    """
    Determines the required integration time based on the noise level.

    Args:
        noise_delta_r: The noise magnitude (ΔR).

    Returns:
        The suggested integration time in seconds.
    """
    if noise_delta_r > 91e-4:
        return 10000
    elif noise_delta_r >= 9e-4:
        return 20000
    elif noise_delta_r > 1e-4:
        return 40000
    else:  # Handles the case for approximately 1e-4 and smaller
        return 60000

def getTauLimits(noise):
    # For a system of size 64, approximately np.log(tau_c) = 1.24*np.log(noise) - 10 or tau_c = 10^(-0.1) noise**1.24
    # So a good guess for the limits of tau would be tau_c - tau_c, 2*tau_c most likely
    tau_c_guess = 10**(-0.1)*noise**1.24
    return (0, tau_c_guess*4)

def compute_depinnings_from_dir(input_folder : Path, task_id : int, cores : int, points, time : int, dt : int, output_folder : Path, perfect : bool):
    input_folder = Path(input_folder)

    # Read metadata that is left to dir from last run to figure out range of allowed params and print helpful info
    with open(input_folder.joinpath("run_params.json"), "r") as fp:
        metadata = json.load(fp)
        noise_count = metadata['args used']['rpoints']
        seed_count = metadata['args used']['seeds']
    
    max_task_id = noise_count*seed_count - 1
    if not ( (0 <= task_id) and (task_id <= max_task_id) ):     # Ensure that 0 < task_id < max_task_id
        print(f"Task id should be within range {0} <= task-id <= {max_task_id} now it is {task_id}")
        return

    # Find all the relaxed configurations in the passed dir
    rel_folder = input_folder.joinpath("relaxed-configurations")
    paths_list = [str(i) for i in rel_folder.iterdir()]

    # Load a file keeping track of all the relaxed configurations available, create it if it doesn't exist
    paths_file = input_folder.joinpath("paths_index.json")
    if paths_file.exists():
        with open(paths_file, "r") as fp:
            paths_data = json.load(fp)
        paths = paths_data["paths"]
    else:
        with open(paths_file, "w+") as fp:
            json.dump({"paths" : paths_list, "len" : len(paths_list)}, fp) # Len should be = seeds * (no. of noises)
        paths = paths_list
    
    # Load the relaxed configuration that is associated with the current task_id. The task_id corresponds uniquely to a
    # (noise, seed) pair
    initial_config_path = paths[task_id]
    initial_config = np.load(initial_config_path)

    if perfect:
        print(initial_config.files)
        if 'y_last' in initial_config.files:
            y0 = initial_config['y_last']
        elif 'y_fire' in initial_config.files:
            y0 = initial_config['y_fire']
        else:
            raise Exception('No key for the initial config found in the intial config file, it must be corrupted.')
        
        params = DislocationSimulation.paramListToDict(initial_config['params'])

        # Create the approproate depinning object
        tau_min, tau_max = getTauLimits(params['deltaR'])
        integration_time = getIntegrationTime(params['deltaR'])

        depinning_output_folder = Path(output_folder).joinpath(f"deltaR_{params['deltaR']}-seed-{params['seed']}")

        depinning_perfect = DepinningSingle(tau_min=tau_min, tau_max=tau_max, points=points, time=integration_time, dt=dt, cores=cores,
                                            folder_name=depinning_output_folder, deltaR=params['deltaR'], seed=params['seed'].astype(int), 
                                            bigN=params['bigN'].astype(int), length=params['length'].astype(int) )
        depinning_perfect.run(y0_rel=y0)
        depinning_perfect.dump_res_to_pickle(output_folder.joinpath(f"depinning-pickle-dumps"))
        # depinning_perfect.save_results(output_folder)
    else:
        params = PartialDislocationsSimulation.paramListToDict(initial_config['params'])

        if 'y_last' in initial_config.files:
            y_last = initial_config['y_last']
        else:
            y1_last = initial_config['y1_fire']
            y2_last = initial_config['y2_fire']
        
        depinning_output_folder = Path(output_folder).joinpath(f"deltaR_{params['deltaR']}-seed-{params['seed']}")

        tau_min, tau_max = getTauLimits(params['deltaR'])
        integration_time = getIntegrationTime(params['deltaR'])
        depinnin_partial = DepinningPartial(tau_min=tau_min, tau_max=tau_max, points=points, time=integration_time, dt=dt, 
                                            cores=cores, folder_name=depinning_output_folder, deltaR=float(params['deltaR']),
                                            seed=int(params['seed']), bigN=int(params['bigN']),  length=int(params['length']), d0=params['d0'])
        depinnin_partial.run(y1_0=y1_last, y2_0=y2_last)
        depinnin_partial.dump_res_to_pickle(output_folder.joinpath(f"depinning-pickle-dumps"))
        pass
    pass

def continue_depinning(path_to_params, integrate_further=False, further_time=None):
    """
    path_to_params : 
    """
    path_to_params = Path(path_to_params)
    depining_folder = path_to_params.parent

    depinning = DepinningPartial.from_json_config(path_to_params)
    print(depinning.deltaR)
    depinning.run_recovered_parallel(integrate_further=integrate_further, further_time=further_time)
    depinning.dump_res_to_pickle(depining_folder.parent.joinpath("depinning-pickle-dumps"))
    pass

def argsmain():
    parser = argparse.ArgumentParser(
        description="""
        A script to run or continue depinning simulations.
        """
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # Command to start a new simulation
    parser_new = subparsers.add_parser('new', help='Start a new depinning simulation from relaxed configurations.')
    parser_new.add_argument('--folder', type=str, required=True,
                            help="""
                            Folder where the initially relaxed configurations are. Should be the one which contains the folder
                            'relaxed-configurations'.
                            """
                            )
    parser_new.add_argument('--out-folder', type=str, required=True, help="Output folder for the new depinning simulation.")
    parser_new.add_argument('--partial', action='store_true', help='Flag for partial dislocation.')
    parser_new.add_argument('--perfect', action='store_true', help='Flag for perfect dislocation.')
    parser_new.add_argument('--cores', type=int, required=True, help='Number of cores to use.')
    parser_new.add_argument('--task-id', type=int, required=True, help='SLURM_ARRAY_TASK_ID from Triton, used as an index.')
    parser_new.add_argument('--points', type=int, required=True, help='Number of tau_ext points to be integrated.')
    parser_new.add_argument('--time', type=int, required=True, help='Time to integrate each simulation.')
    parser_new.add_argument('--dt', type=float, required=True, help='Timestep for sampling the solution.')

    # Command to continue a simulation
    parser_continue = subparsers.add_parser('continue', help='Continue an interrupted depinning simulation.')
    parser_continue.add_argument('--depinning-params', type=str, required=True,
                                 help='Path to the depinning_params.json file from the simulation to continue.')

    parser_run_further = subparsers.add_parser('further', help='Integrate finished depinning simulation further in time.')
    parser_run_further.add_argument('--depinning-params', type=str, required=True,
                                 help='Path to the depinning_params.json file from the simulation to continue.')
    parser_run_further.add_argument('--time', type=int, required=True, help='How much longer should the simulation be run.')

    args = parser.parse_args()

    if args.command == 'new':
        if not args.partial and not args.perfect:
            parser.error("Either --partial or --perfect must be specified for the 'new' command.")
        if args.partial:
            print(f"Starting new depinning for partial dislocation")
            compute_depinnings_from_dir(input_folder=Path(args.folder), task_id=args.task_id, cores=args.cores, points=args.points, time=args.time, dt=args.dt, output_folder=Path(args.out_folder), perfect=False)
        if args.perfect:
            print(f"Starting new depinning for perfect dislocation")
            compute_depinnings_from_dir(input_folder=Path(args.folder), task_id=args.task_id, cores=args.cores, points=args.points, time=args.time, dt=args.dt, output_folder=Path(args.out_folder), perfect=True)
    elif args.command == 'continue':
        print(f"Continuing depinning simulation.")
        continue_depinning(path_to_params=args.depinning_params, integrate_further=False)
    elif args.command == 'further':
        print(f"Integrating depinning simulations even further.")
        continue_depinning(path_to_params=args.depinning_params, integrate_further=True, further_time=args.time)

    return # The code below this point in the original function is now handled within the command logic.

if __name__ == "__main__":
    argsmain()