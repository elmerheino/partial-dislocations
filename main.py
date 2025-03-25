import multiprocessing as mp
from argparse import ArgumentParser
from plots import *
from processData import dumpDepinning
from depinning import *
import json

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
    parser.add_argument("-R", "--delta-r", help='Magnitude of random noise.', default=1.0)

    parser.add_argument('-c', '--cores', help='Cores to use in multiprocessing pool. Is not specified, use all available.')
    parser.add_argument('--partial', help='Simulate a partial dislocation.', action="store_true")
    parser.add_argument('--single', help='Simulate a single dislocation.', action="store_true")
    parser.add_argument('--seq', help='Sequential.', action="store_true", default=False)

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
                        folder_name=parsed.folder, cores=cores, sequential=parsed.seq, deltaR=float(parsed.delta_r))
        
        v1, v2, vcm, l_range, avg_w12s, y1_last, y2_last, parameters = depinning.run()

        # Save the depinning to a .json file
        depining_path = Path(parsed.folder)
        depining_path = depining_path.joinpath("depinning-dumps")
        depining_path.mkdir(exist_ok=True, parents=True)
        depining_path = depining_path.joinpath(f"depinning-tau-{parsed.tau_min}-{parsed.tau_max}-p-{int(parsed.points)}-t-{parsed.time}-s-{parsed.seed}-R-{parsed.delta_r}.json")

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
            tauExt = params[11]
            p = Path(parsed.folder).joinpath(f"averaged-roughnesses").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau:.3f}-R-{parsed.delta_r}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w12, parameters=params)
        
        # Save the dislocation at the end of simulation in an organized way
        for y1_i, y2_i, params in zip(y1_last, y2_last, parameters):
            tauExt = params[11]
            p = Path(parsed.folder).joinpath(f"dislocations-last").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt:.3f}-R-{parsed.delta_r}.npz")
            np.savez(p0, y1=y1_i, y2=y2_i, parameters=params)

            # with open(p.joinpath(f"dislocation-shapes-tau-{tauExt:.3f}.json"), "w") as fp:
            #     json.dump({"y1" : y1_i.tolist(), "y2" : y2_i.tolist(), "parameters" : params.tolist()}, fp)
            pass

    elif parsed.single:

        depinning = DepinningSingle(tau_min=float(parsed.tau_min), tau_max=float(parsed.tau_max), points=int(parsed.points),
                        time=float(parsed.time), dt=float(parsed.timestep), seed=int(parsed.seed), 
                        folder_name=parsed.folder, cores=cores, sequential=parsed.seq, deltaR=float(parsed.delta_r))
        
        vcm, l_range, roughnesses, y_last, parameters = depinning.run() # Velocity of center of mass, the l_range for roughness, all roughnesses and parameters for each simulation

        # Save the results to a .json file

        depining_path = Path(parsed.folder)
        depining_path = depining_path.joinpath("depinning-dumps")
        depining_path.mkdir(exist_ok=True, parents=True)

        tau_min_ = min(depinning.stresses.tolist())
        tau_max_ = max(depinning.stresses.tolist())
        points = len(depinning.stresses.tolist())
        depining_path = depining_path.joinpath(f"depinning-tau-{tau_min_}-{tau_max_}-p-{points}-t-{depinning.time}-s-{depinning.seed}-R-{parsed.delta_r}.json")
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
            p = Path(parsed.folder).joinpath(f"averaged-roughnesses").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau:.3f}-R-{parsed.delta_r}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w, parameters=params)

            pass

        for y_i, params in zip(y_last, parameters):
            tauExt = params[10]
            p = Path(parsed.folder).joinpath(f"dislocations-last").joinpath(f"seed-{depinning.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt:.3f}-R-{parsed.delta_r}.npz")
            np.savez(p0, y=y_i, parameters=params)

    else:
        raise Exception("Not specified which type of dislocation must be simulated.")

    # time_elapsed = time.time() - k
    # print(f"It took {time_elapsed} for the script to run.")
    pass

if __name__ == "__main__":
    triton()
    pass