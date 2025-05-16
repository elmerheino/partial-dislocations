import multiprocessing as mp
from argparse import ArgumentParser
from plots import *
from depinning import *
import json
from scipy import optimize
from processData import velocity_fit

save_plots = False      # Don't save any images of figures. Still saves all data as dumps.

def grid_search(rmin, rmax, array_task_id : int, seeds : int, array_length : int,
                time : int, timestep : float, points, cores : int, partial : bool,
                folder, sequential = False):
    # k = time.time()

    # estimate = (time/timestep)*1024*2*4*1e-6
    # input(f"One simulation will take up {estimate:.1f} MB disk space totalling {estimate*int(parsed.points)*1e-3:.1f} GB")

    # Map the array task id to a 2d grid

    task_id = array_task_id
    cols = seeds
    arr_max = array_length

    no_of_rows = arr_max // cols

    row = task_id // cols   # Let this be noise
    col = task_id % cols    # Let this be the seed

    seed = col

    print(f"seed : {seed} row : {row} col : {col} no of rows : {no_of_rows}")

    interval = np.logspace(rmin,rmax, no_of_rows) # Number of points is determined from seed count and array len
    deltaR = interval[row - 1] # Map the passed slurm index to a value

    tau_min = 0
    tau_max = deltaR*1.7
    
    print(f"tau_min : {tau_min}  tau_max : {tau_max} deltaR : {deltaR}")

    if cores == None:
        cores = mp.cpu_count()
    else:
        cores = cores

    if partial:
        partial_dislocation_depinning(tau_min, tau_max, cores, seed, deltaR, points, time, timestep, folder, sequential)
    elif not partial:
        perfect_dislocation_depinning(tau_min, tau_max, cores, seed, deltaR, points, time, timestep, folder, sequential)
    else:
        raise Exception("Not specified which type of dislocation must be simulated.")

    pass

def search_tau_c(tau_min_0, tau_max_0, deltaR, time, timestep, seed,folder, cores, partial=False):
    found = False
    attemtpts = 0

    tau_min = tau_min_0
    tau_max = tau_max_0

    while not found and attemtpts < 20:
        depinning = DepinningSingle(tau_min=tau_min, tau_max=tau_max, points=5,
                    time=float(time), dt=float(timestep), seed=seed,
                    folder_name=folder, cores=cores, sequential=False, deltaR=deltaR)
        
        if partial:
            depinning = DepinningPartial(tau_min=tau_min, tau_max=tau_max, points=5,
                    time=float(time), dt=float(timestep), seed=seed,
                    folder_name=folder, cores=cores, sequential=False, deltaR=deltaR)
            v1, v2, vcm, l_range, avg_w12s, y1_last, y2_last, parameters = depinning.run()
        else:
            vcm, l_range, roughnesses, y_last, parameters = depinning.run()
        
        t_c_arvio = ( max(depinning.stresses) - min(depinning.stresses) ) / 2
        try:
            fit_params, pcov = optimize.curve_fit(velocity_fit, depinning.stresses, vcm,
                p0 = [
                    t_c_arvio,
                    0.8,
                    0.05
                ], 
                bounds=(0, [max(depinning.stresses), 5, 100] )
            )
            t_c, beta, a = fit_params
            print(f"Fit results tau_c = {t_c}  beta = {beta}  A = {a}")
        except:
            return tau_min, tau_max


        tolerance = 0.085

        if t_c < (1 - tolerance)*max(depinning.stresses) and t_c > tolerance*min(depinning.stresses):
            found = True
            print(f"Found good parameters")
            if deltaR < 0.5:
                tau_min = 0
                tau_max = t_c*1.5
            else:
                tau_min = t_c*0.2
                tau_max = t_c*1.5
        
        delta = t_c*0.5
        if t_c > max(depinning.stresses)*(1 - tolerance):
            # Shift limits to the right increasing them by delta
            tau_min = tau_min + delta
            tau_max = tau_max + delta
            attemtpts += 1
            print(f"Adjusting limits to the right")
        
        if t_c < min(depinning.stresses)*tolerance:
            # Shift limits to the left decreasing them by delta
            print(f"Adjusting limits to the left")
            tau_min = tau_min - delta
            tau_max = tau_max - delta
            attemtpts += 1

    return tau_min, tau_max

def partial_dislocation_depinning(tau_min, tau_max, cores, seed, deltaR, points, time, timestep, folder, sequential=False):
        # tau_min_opt, tau_max_opt = search_tau_c(tau_min, tau_max, deltaR, time, timestep, seed, folder, cores, partial=True)

        if 0 <= deltaR < 0.1: 
            tau_min_opt = 0
            tau_c = 2.620 * deltaR**1.885
            tau_max_opt = tau_c*1.4
        elif 0.1 <= deltaR < 1.0:
            tau_min_opt = 0
            tau_c = 0.954 * deltaR**1.398
            tau_max_opt = tau_c*1.4
        elif 1.0 <= deltaR <= 10:
            tau_c = 1.026 * deltaR**1.255
            tau_min_opt = tau_c*0.7
            tau_max_opt = tau_c*1.3
        else:
            tau_c = 1.026 * deltaR**1.255
            tau_min_opt = tau_c*0.7
            tau_max_opt = tau_c*1.3

        depinning = DepinningPartial(tau_min=tau_min_opt, tau_max=tau_max_opt, points=points,
                    time=time, dt=timestep, seed=seed,
                    folder_name=folder, cores=cores, sequential=sequential, deltaR=deltaR)
    
        v1, v2, vcm, l_range, avg_w12s, y1_last, y2_last, parameters = depinning.run()

        tau_min_ = min(depinning.stresses.tolist())
        tau_max_ = max(depinning.stresses.tolist())
        points = len(depinning.stresses.tolist())

        # Save the depinning to a .json file
        depining_path = Path(folder)
        depining_path = depining_path.joinpath("depinning-dumps").joinpath(f"noise-{deltaR}")
        depining_path.mkdir(exist_ok=True, parents=True)
        depining_path = depining_path.joinpath(f"depinning-tau-{tau_min_}-{tau_max_}-p-{points}-t-{time}-s-{seed}-R-{deltaR}.json")

        with open(str(depining_path), 'w') as fp:
            json.dump({
                "stresses": depinning.stresses.tolist(),
                "v_rel": vcm,
                "seed":depinning.seed,
                "time":depinning.time,
                "dt":depinning.dt,
                "v_1" : v1,
                "v_2" : v2
            },fp)
        
        # Save the roughnesses in an organized way
        for tau, avg_w12, params in zip(depinning.stresses, avg_w12s, parameters):
            tauExt_i = params[11]
            deltaR_i = params[4]
            p = Path(folder).joinpath(f"averaged-roughnesses").joinpath(f"noise-{deltaR}").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau}-R-{deltaR}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w12, parameters=params)
        
        # Save the dislocation at the end of simulation in an organized way
        for y1_i, y2_i, params in zip(y1_last, y2_last, parameters):
            tauExt_i = params[11]
            deltaR_i = params[4]
            p = Path(folder).joinpath(f"dislocations-last").joinpath(f"noise-{deltaR}").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt_i}-R-{deltaR_i}.npz")
            np.savez(p0, y1=y1_i, y2=y2_i, parameters=params)
            pass


def perfect_dislocation_depinning(tau_min, tau_max, cores, seed, deltaR, points, time, timestep, folder, sequential=False):
    # Searching for better limits to find critical force
    #tau_min_opt, tau_max_opt = search_tau_c(tau_min, tau_max, deltaR, time, timestep, seed, folder, cores, partial=False)
    if 0 < deltaR < 0.1:
        tau_c = 1.392 * deltaR**1.736
        tau_min_opt = 0
        tau_max_opt = tau_c*1.5
    elif 0.1 <= deltaR < 1.0:
        tau_c =  0.677 * deltaR ** 1.374
        tau_min_opt = 0
        tau_max_opt = tau_c*1.5
    elif 1.0 <= deltaR <= 10:
        tau_c = 0.736 * deltaR ** 1.280
        tau_min_opt = tau_c*0.7
        tau_max_opt = tau_c*1.5
    else:
        tau_c = 0.736 * deltaR ** 1.280
        tau_min_opt = tau_c*0.7
        tau_max_opt = tau_c*1.5

    depinning = DepinningSingle(tau_min=tau_min_opt, tau_max=tau_max_opt, points=int(points),
                time=float(time), dt=float(timestep), seed=seed, 
                folder_name=folder, cores=cores, sequential=sequential, deltaR=deltaR)
    
    vcm, l_range, roughnesses, y_last, parameters = depinning.run() # Velocity of center of mass, the l_range for roughness, all roughnesses and parameters for each simulation

    # Save the results to a .json file
    depining_path = Path(folder)
    depining_path = depining_path.joinpath("depinning-dumps").joinpath(f"noise-{deltaR}")
    depining_path.mkdir(exist_ok=True, parents=True)

    tau_min_ = min(depinning.stresses.tolist())
    tau_max_ = max(depinning.stresses.tolist())
    points = len(depinning.stresses.tolist())
    depining_path = depining_path.joinpath(f"depinning-tau-{tau_min_}-{tau_max_}-p-{points}-t-{depinning.time}-s-{depinning.seed}-R-{deltaR}.json")
    with open(str(depining_path), 'w') as fp:
        json.dump({
            "stresses":depinning.stresses.tolist(),
            "v_rel":vcm,
            "seed":depinning.seed,
            "time":depinning.time,
            "dt":depinning.dt
        },fp)

    # Save all the roughnesses
    for tau, avg_w, params in zip(depinning.stresses, roughnesses, parameters): # Loop through tau as well to save it along data
        deltaR_i = params[4]
        p = Path(folder).joinpath(f"averaged-roughnesses").joinpath(f"noise-{deltaR}").joinpath(f"seed-{depinning.seed}")
        p.mkdir(exist_ok=True, parents=True)
        p = p.joinpath(f"roughness-tau-{tau}-R-{deltaR_i}.npz")
        
        np.savez(p, l_range=l_range, avg_w=avg_w, parameters=params)

        pass

    # Save all the relaxed dislocaiton profiles at the end of simulation
    for y_i, params in zip(y_last, parameters):
        tauExt = params[10]
        deltaR_i = params[4]
        p = Path(folder).joinpath(f"dislocations-last").joinpath(f"noise-{deltaR}").joinpath(f"seed-{depinning.seed}")
        p.mkdir(exist_ok=True, parents=True)
        p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt}-R-{deltaR}.npz")
        np.savez(p0, y=y_i, parameters=params)

if __name__ == "__main__":
    parser = ArgumentParser(prog="Dislocation simulation")
    subparsers = parser.add_subparsers(help="Do a grid search on noise or just a single depinning.", dest="command")
    grid_parser = subparsers.add_parser("grid")
    # parser.add_argument('-s', '--seed', help='Specify seed for the individual depinning study. If not specified, seed will be randomized between stresses.', required=True)

    parser.add_argument('-f', '--folder', help='Specify the output folder for all the dumps and results.', required=True)
    parser.add_argument('-dt', '--timestep', help='Timestep size in (s).', required=True, type=float)
    parser.add_argument('-t', '--time', help='Total simulation time in (s).', required=True, type=int)
    parser.add_argument('-p', '--points', help='How many points to consider between tau_min and tau_max', required=True, type=int)

    parser.add_argument('-c', '--cores', help='Cores to use in multiprocessing pool. Is not specified, use all available.', type=int)
    parser.add_argument('--partial', help='Simulate a partial dislocation.', action="store_true")
    parser.add_argument('--single', help='Simulate a single dislocation.', action="store_true")
    parser.add_argument('--seq', help='Sequential.', action="store_true", default=False)

    grid_parser.add_argument('-id', '--array-task-id', help="The array task id.", required=True, type=int)
    grid_parser.add_argument('-sl', '--seeds', help="Number of seeds in the grid (or columns)", required=True, type=int)
    grid_parser.add_argument('-arr', '--array-length', help="Length of the array job. (determines the n. of rows)", required=True, type=int)

    # Calculate noise magnitude and seed based on this parameter

    grid_parser.add_argument("--rmin", help="Minimun value of noise", default=0.0, type=float)
    grid_parser.add_argument("--rmax", help="Maximum value of noise", default=2.0, type=float)

    pinning_parser = subparsers.add_parser("pinning")

    pinning_parser.add_argument('-tmin', '--tau-min', help='Start value for stress.', required=True, type=float)
    pinning_parser.add_argument('-tmax', '--tau-max', help='End value for stress.', required=True, type=float)
    pinning_parser.add_argument('-s', '--seed', help='Specify seed for the individual depinning study. If not specified, seed will be randomized between stresses.', required=True, type=int)
    pinning_parser.add_argument("-R", "--delta-r", help='Index of random noise from triton.', default=1.0, type=float)

    parsed = parser.parse_args()
    
    if parsed.command == "grid":
        partial_ = None

        if parsed.partial:
            partial_ = True
        elif parsed.single:
            partial_ = False

        grid_search(parsed.rmin, parsed.rmax, parsed.array_task_id, parsed.seeds, parsed.array_length, parsed.time, parsed.timestep, parsed.points, parsed.cores, partial_, parsed.folder, 
                    parsed.seq)
    elif parsed.command == "pinning":
        if parsed.partial:
            partial_dislocation_depinning(parsed.tau_min, parsed.tau_max, parsed.cores, parsed.seed, parsed.delta_r, parsed.points, parsed.time, parsed.timestep,
                                          parsed.folder, parsed.seq)
        if parsed.single:
            perfect_dislocation_depinning(parsed.tau_min, parsed.tau_max, parsed.cores, parsed.seed, parsed.delta_r, parsed.points, parsed.time, parsed.timestep,
                                          parsed.folder, parsed.seq)
        pass

    pass