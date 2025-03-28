import multiprocessing as mp
from argparse import ArgumentParser
from plots import *
from depinning import *
import json

# import time

save_plots = False      # Don't save any images of figures. Still saves all data as dumps.
    
def triton():
    # k = time.time()
    parser = ArgumentParser(prog="Dislocation simulation")
    # parser.add_argument('-s', '--seed', help='Specify seed for the individual depinning study. If not specified, seed will be randomized between stresses.', required=True)

    parser.add_argument('-f', '--folder', help='Specify the output folder for all the dumps and results.', required=True)
    parser.add_argument('-dt', '--timestep', help='Timestep size in (s).', required=True)
    parser.add_argument('-t', '--time', help='Total simulation time in (s).', required=True)

    parser.add_argument('-id', '--array-task-id', help="The array task id.", required=True)
    parser.add_argument('-sl', '--seeds', help="Number of seeds in the grid (or columns)", required=True)
    parser.add_argument('-arr', '--array-length', help="Length of the array job. (determines the n. of rows)", required=True)

    # Calculate noise magnitude and seed based on this parameter

    # parser.add_argument("-R", "--delta-r", help='Index of random noise from triton.', default=1.0)
    parser.add_argument("--rmin", help="Minimun value of noise", default=0.0)
    parser.add_argument("--rmax", help="Maximum value of noise", default=2.0)
    # parser.add_argument("--rpoints", help="Number of points of noise in triton.", default=10)

    # parser.add_argument('-tmin', '--tau-min', help='Start value for stress.', required=True)
    # parser.add_argument('-tmax', '--tau-max', help='End value for stress.', required=True)
    parser.add_argument('-p', '--points', help='How many points to consider between tau_min and tau_max', required=True)

    parser.add_argument('-c', '--cores', help='Cores to use in multiprocessing pool. Is not specified, use all available.')
    parser.add_argument('--partial', help='Simulate a partial dislocation.', action="store_true")
    parser.add_argument('--single', help='Simulate a single dislocation.', action="store_true")
    parser.add_argument('--seq', help='Sequential.', action="store_true", default=False)

    parsed = parser.parse_args()

    estimate = (int(parsed.time)/float(parsed.timestep))*1024*2*4*1e-6
    # input(f"One simulation will take up {estimate:.1f} MB disk space totalling {estimate*int(parsed.points)*1e-3:.1f} GB")

    # Map the array task id to a 2d grid

    task_id = int(parsed.array_task_id)
    cols = int(parsed.seeds)
    arr_max = int(parsed.array_length)

    no_of_rows = arr_max // cols

    row = task_id // cols   # Let this be noise
    col = task_id % cols    # Let this be the seed

    seed = col

    print(f"seed : {seed} row : {row} col : {col} no of rows : {no_of_rows}")

    rmin = float(parsed.rmin)
    rmax = float(parsed.rmax)

    interval = np.logspace(rmin,rmax, no_of_rows) # Number of points is determined from seed count and array len
    deltaR = interval[row - 1] # Map the passed slurm index to a value

    # Determine here initial guesses tau search limits

    if rmin <= deltaR <= 0.1:
        tau_min = 0
        tau_max = deltaR*(1 + 0.05)
    elif 0.1 < deltaR < 10:
        tau_min = deltaR*(1 - 0.2)
        tau_max = deltaR*(1 + 2)
    elif 10 <= deltaR <= rmax:
        tau_min = deltaR*(1.02)
        tau_max = deltaR*(1 + 1)
    else:
        tau_min = deltaR*(1.02)
        tau_max = deltaR*(1 + 1)
    
    print(f"tau_min : {tau_min}  tau_max : {tau_max} deltaR : {deltaR}")

    if parsed.cores == None:
        cores = mp.cpu_count()
    else:
        cores = int(parsed.cores)

    if parsed.partial:

        # Run here mulptiple small depinning simulations to get an idea which tau interval contains the critical force starting from the intial guesses.

        delta = tau_min*0.1
        tau_c_tolerance = 0.1

        for i in range(0,10):
            test = DepinningPartial(tau_min=tau_min, tau_max=tau_max, points=5,
                            time=float(parsed.time), dt=float(parsed.timestep), seed=seed,
                            folder_name=parsed.folder, cores=cores, sequential=parsed.seq, deltaR=deltaR)
            
            v1, v2, vcm, l_range, avg_w12s, y1_last, y2_last, parameters = test.run()
            v_range = max(vcm) - min(vcm)

            tau_crit_i = np.argmax(np.array(vcm) > tau_c_tolerance)

            middle = len(vcm) / 2

            # Handle spcial cases first

            if tau_crit_i == 0 and v_range > 0.1: # All are too big probably so tau_c resides on the left -> Decrease limits
                print(f"Limits were: {tau_min} < tau < {tau_max}")
                tau_min -= delta
                tau_max -= delta
                print(f"Limits are lowered to: {tau_min} < tau < {tau_max}")
                continue
            elif tau_crit_i == 0 and v_range < 0.1: # All are too small so tau_c resides on the right -> Increase them
                print(f"Limits were: {tau_min} < tau < {tau_max}")
                tau_min += delta
                tau_max += delta
                print(f"Limits are increased to: {tau_min} < tau < {tau_max}")
                continue


            if tau_crit_i < middle:
                print(f"Limits were: {tau_min} < tau < {tau_max}")
                tau_min += delta
                tau_max += delta
                print(f"Limits are increased to: {tau_min} < tau < {tau_max}")
                pass # increase limits
            elif tau_crit_i > middle:
                print(f"Limits were: {tau_min} < tau < {tau_max}")
                tau_min -= delta
                tau_max -= delta
                print(f"Limits are lowered to: {tau_min} < tau < {tau_max}")
                pass # decrease limits
        
        print(f"Proceeding with limits {tau_min:.3f} < tau < {tau_max:.3f} (noise: {deltaR})")

        depinning = DepinningPartial(tau_min=tau_min, tau_max=tau_max, points=int(parsed.points),
                        time=float(parsed.time), dt=float(parsed.timestep), seed=seed,
                        folder_name=parsed.folder, cores=cores, sequential=parsed.seq, deltaR=deltaR)
        
        v1, v2, vcm, l_range, avg_w12s, y1_last, y2_last, parameters = depinning.run()

        # Save the depinning to a .json file
        depining_path = Path(parsed.folder)
        depining_path = depining_path.joinpath("depinning-dumps").joinpath(f"noise-{deltaR:.4f}")
        depining_path.mkdir(exist_ok=True, parents=True)
        depining_path = depining_path.joinpath(f"depinning-tau-{tau_min}-{tau_max}-p-{int(parsed.points)}-t-{parsed.time}-s-{seed}-R-{deltaR:.4f}.json")

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
            p = Path(parsed.folder).joinpath(f"averaged-roughnesses").joinpath(f"noise-{deltaR:.4f}").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau:.3f}-R-{deltaR:.4f}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w12, parameters=params)
        
        # Save the dislocation at the end of simulation in an organized way
        for y1_i, y2_i, params in zip(y1_last, y2_last, parameters):
            tauExt_i = params[11]
            deltaR_i = params[4]
            p = Path(parsed.folder).joinpath(f"dislocations-last").joinpath(f"noise-{deltaR:.4f}").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt_i:.3f}-R-{deltaR_i:.4f}.npz")
            np.savez(p0, y1=y1_i, y2=y2_i, parameters=params)

            # with open(p.joinpath(f"dislocation-shapes-tau-{tauExt:.3f}.json"), "w") as fp:
            #     json.dump({"y1" : y1_i.tolist(), "y2" : y2_i.tolist(), "parameters" : params.tolist()}, fp)
            pass

    elif parsed.single:

        depinning = DepinningSingle(tau_min=tau_min, tau_max=tau_max, points=int(parsed.points),
                        time=float(parsed.time), dt=float(parsed.timestep), seed=seed, 
                        folder_name=parsed.folder, cores=cores, sequential=parsed.seq, deltaR=deltaR)
        
        vcm, l_range, roughnesses, y_last, parameters = depinning.run() # Velocity of center of mass, the l_range for roughness, all roughnesses and parameters for each simulation

        # Save the results to a .json file

        depining_path = Path(parsed.folder)
        depining_path = depining_path.joinpath("depinning-dumps").joinpath(f"noise-{deltaR:.4f}")
        depining_path.mkdir(exist_ok=True, parents=True)

        tau_min_ = min(depinning.stresses.tolist())
        tau_max_ = max(depinning.stresses.tolist())
        points = len(depinning.stresses.tolist())
        depining_path = depining_path.joinpath(f"depinning-tau-{tau_min_}-{tau_max_}-p-{points}-t-{depinning.time}-s-{depinning.seed}-R-{deltaR:.4f}.json")
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
            p = Path(parsed.folder).joinpath(f"averaged-roughnesses").joinpath(f"noise-{deltaR:.4f}").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau:.3f}-R-{deltaR_i:.4f}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w, parameters=params)

            pass

        for y_i, params in zip(y_last, parameters):
            tauExt = params[10]
            deltaR_i = params[4]
            p = Path(parsed.folder).joinpath(f"dislocations-last").joinpath(f"noise-{deltaR:.4f}").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt:.3f}-R-{deltaR:.4f}.npz")
            np.savez(p0, y=y_i, parameters=params)

    else:
        raise Exception("Not specified which type of dislocation must be simulated.")

    pass

if __name__ == "__main__":
    triton()
    pass