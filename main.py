import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from simulation import PartialDislocationsSimulation
from pathlib import Path
import pickle
from tqdm import tqdm
import json
import multiprocessing as mp
from functools import partial

def studyAvgDistance():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()

    # Run 4 simulations right away
    for i in range(0,4):
        np.random.seed(i)

        sim_i = PartialDislocationsSimulation(timestep_dt=0.5, time=100, c_gamma=60, cLT1=4, cLT2=4)

        t = sim_i.getTvalues()
        sim_i.run_simulation()
        avgI = sim_i.getAverageDistances()

        axes_flat[i].plot(t,avgI)
        axes_flat[i].set_xlabel("Time (s)")
        axes_flat[i].set_ylabel("Average distance")
        print(f"Simulation {i+1}, min: {min(avgI)}, max: {max(avgI)}, delta: {max(avgI) - min(avgI)}")

    plt.tight_layout()
    plt.show()

    
def dumpResults(sim: PartialDislocationsSimulation, folder_name: str):
    # Dumps the results of a simulation to a json file
    if not sim.has_simulation_been_run:
        raise Exception("Simulation has not been run.")
    
    parameters = [
        sim.bigN, sim.length, sim.time, sim.dt,
        sim.deltaR, sim.bigB, sim.smallB, sim.b_p,
        sim.cLT1, sim.cLT2, sim.mu, sim.tauExt, sim.c_gamma,
        sim.d0, sim.seed, sim.tau_cutoff
        ] # From these parameters you should be able to replicate the simulation
    
    y1_as_list = [list(row) for row in sim.y1]  # Convert the ndarray to a list of lists
    y2_as_list = [list(row) for row in sim.y2]

    result = {
        "parameters":parameters,
        "y1":sim.y1,
        "y2":sim.y2
    }

    dump_path = Path(folder_name).joinpath("pickle-dumps")
    dump_path = dump_path.joinpath(f"seed-{sim.seed}")
    dump_path.mkdir(exist_ok=True, parents=True)

    dump_path = dump_path.joinpath(f"sim-{sim.tauExt}.json")
    with open(str(dump_path), "wb") as fp:
        pickle.dump(result,fp)
    pass

def loadResults(file_path):
    # Load the results from a file at file_path to a new simulation object for further processing

    with open(file_path, "rb") as fp:
        res_dict = pickle.load(fp)

    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff = res_dict["parameters"]
    res = PartialDislocationsSimulation(bigN=bigN, bigB=bigB, length=length,
                                        time=time,timestep_dt=dt, deltaR=deltaR, 
                                        smallB=smallB, b_p=b_p, cLT1=cLT1, cLT2=cLT2,
                                        mu=mu, tauExt=tauExt, c_gamma=c_gamma, d0=d0, seed=seed)
    
    # Modify the internal state of the object according to preferences
    res.tau_cutoff = tau_cutoff
    res.has_simulation_been_run = True

    res.y1 = res_dict["y1"]
    res.y2 = res_dict["y2"]

    return res

def studyConstantStress(tauExt,
                        timestep_dt,
                        time, seed=None, folder_name="results",):
    # Run a simulation with a specified constant stress

    simulation = PartialDislocationsSimulation(tauExt=tauExt, bigN=1024, length=1024, 
                                              timestep_dt=timestep_dt, time=time, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1, seed=seed)
    # print(" ".join(simulation.getParamsInLatex()))

    simulation.run_simulation()

    dumpResults(simulation, folder_name)

    rV1, rV2, totV2 = simulation.getRelaxedVelocity(time_to_consider=1000) # The velocities after relaxation

    # makeStressPlot(simulation, folder_name)

    return (rV1, rV2, totV2)

def studyDepinning_mp(tau_min, tau_max, points,
                   timestep_dt, time, seed=123, folder_name="results"):
    # Multiprocessing compatible version of a single depinning study, here the studies
    # are distributed between threads by stress letting python mp library determine the best way
    
    stresses = np.linspace(tau_min, tau_max, points)
    
    with mp.Pool(8) as pool:
        results = pool.map(partial(studyConstantStress, folder_name=folder_name, timestep_dt=timestep_dt, time=time, seed=seed), stresses)
    
    
    v1_rel = [i[0] for i in results]
    v2_rel = [i[1] for i in results]
    v_cm = [i[2] for i in results]

    makeDepinningPlot(stresses, v_cm, time, seed, folder_name=folder_name)

    results_json = {
        "stresses":stresses.tolist(),
        "v1_rel": v1_rel,
        "v2_rel": v2_rel,
        "relaxed_velocities_total":v_cm,
        "seed":seed,
        "time":time,
        "dt":timestep_dt
    }

    depining_path = Path(folder_name)
    depining_path = depining_path.joinpath("depinning-dumps")
    depining_path.mkdir(exist_ok=True, parents=True)
    depining_path = depining_path.joinpath(f"depinning-{tau_min}-{tau_max}-{points}-{time}-{seed}.json")
    with open(str(depining_path), 'w') as fp:
        json.dump(results_json,fp)

def studyDepinning(tau_min, tau_max, points,
                   timestep_dt, time,folder_name="results", seed=None): # Fix the stress field for different tau_ext
    
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    stresses = np.linspace(tau_min, tau_max, points)
    r_velocities = list()

    for s,p in zip(stresses, tqdm(range(points), desc="Running simulations", unit="simulation")):
        rel_v1, rel_v2, rel_tot = studyConstantStress(tauExt=s, folder_name=folder_name, timestep_dt=timestep_dt, time=time, seed=seed)
        # print(f"Simulation finished with : v1_cm = {rel_v1}  v2_cm = {rel_v2}  v2_tot = {rel_tot}")
        r_velocities.append(rel_tot)
    
    makeDepinningPlot(stresses, r_velocities, time, seed, folder_name=folder_name)

    results = {
        "stresses":stresses.tolist(),
        "relaxed_velocities":r_velocities,
        "seed":seed
    }

    with open(f"{folder_name}/depinning-{tau_min}-{tau_max}-{points}-{time}-{seed}.json", 'w') as fp:
        json.dump(results,fp)
    

def makeDepinningPlot(stresses, relVelocities, time, seed, folder_name="results"):
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    plt.clf()
    plt.scatter(stresses, relVelocities, marker='x')
    plt.title("Depinning")
    plt.xlabel("$\\tau_{ext}$")
    plt.ylabel("$v_{CM}$")
    plt.savefig(f"{folder_name}/depinning-{min(stresses)}-{max(stresses)}-{len(stresses)}-{time}-{seed}.png")
    # plt.show()

def makeStressPlot(sim: PartialDislocationsSimulation, folder_name):
    # Make a plot with velocities and average distance from the given simulation sim
    # The sim object must have methods: getTValues, getAverageDistance, getRelaxedVeclovity
    # and getCM

    Path(folder_name).mkdir(exist_ok=True, parents=True)

    t = sim.getTvalues()
    avgD = sim.getAverageDistances()
    start = sim.time - 1000
    rV1, rV2, totV2 = sim.getRelaxedVelocity(time_to_consider=1000)

    cm_y1, cm_y2, cm_tot = sim.getCM()

    tauExt = sim.tauExt

    fig, axes = plt.subplots(1,2, figsize=(12, 8))

    axes_flat = axes.ravel()

    axes[0].plot(t, np.gradient(cm_y1), color="black", label="$y_1$") # Plot the velocity of the cm of y_1
    axes[0].plot(t, np.gradient(cm_y2), color="red", label="$y_2$") # Plot the velocity of the average of y_2
    axes[0].plot([start,sim.time], rV1*np.ones(2), '-', color="red")
    axes[0].plot([start,sim.time], rV2*np.ones(2), '-' ,color="blue")

    axes[0].set_title(sim.getTitleForPlot())
    axes[0].set_xlabel("Time t (s)")
    axes[0].set_ylabel("$ v_{CM} $")
    axes[0].legend()

    
    axes[1].plot(t, avgD, label="Average distance")

    axes[1].set_title(sim.getTitleForPlot())
    axes[1].set_xlabel("Time t (s)")
    axes[1].set_ylabel("$ d $")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{folder_name}/constant-stress-tau-{tauExt:.2f}.png") # It's nice to have a plot from each individual simulation
    plt.clf()
    plt.close()

    pass

def makeGif(sim : PartialDislocationsSimulation, file_path : str):
    # Run the simulation
    total_time = 400

    # Get revelant info from simulation
    y1,y2 = sim.getLineProfiles()

    x = sim.getXValues()
    t_axis = sim.getTvalues()

    gifTitle = sim.getTitleForPlot()

    # Make the gif
    frames = list()
    save_path = Path(file_path)
    save_path.mkdir(exist_ok=True, parents=True)

    frames_path = save_path.joinpath("frames")
    frames_path.mkdir(exist_ok=True, parents=True)

    for h1,h2,t in zip(y1,y2,t_axis):

        plt.plot(x, h1, color="blue")
        plt.plot(x, h2, color="red")

        plt.legend(["line 1", "line 2"])
        plt.title(gifTitle)
        plt.xlabel(f"t={t:.2f}")

        
        frame_path = frames_path.joinpath(f"tmp-{t}.png")
        plt.savefig(str(frame_path))
        plt.clf()
        
        frames.append(Image.open(str(frame_path)))

    gif_path = save_path.joinpath(f"lines-moving-aroung-C-{sim.cLT1:.4f}-C_gamma-{sim.c_gamma:.4f}.gif")
    frames[0].save(str(gif_path), save_all=True, append_images=frames[1:], duration=20, loop=0)

    avgDists = sim.getAverageDistances()

    plt.clf()
    plt.plot(t_axis, avgDists)
    plt.title("Average distance")
    plt.xlabel("Time t (s)")
    avg_dist_path = save_path.joinpath(f"avg_dist-C-{sim.cLT1:.4f}-G-{sim.c_gamma:.4f}.png")
    plt.savefig(str(avg_dist_path))
    pass

def multiple_depinnings_mp(seed):
    # Helper function to run depinning studies one study per thread
    studyDepinning(folder_name="results/11-feb-n2", tau_min=2, tau_max=4, timestep_dt=0.05, time=10000, seed=seed)

if __name__ == "__main__":
    # Run multiple such depinning studies with varying seeds
    #time = 10000
    dt = 0.5
    #studyDepinning_mp(tau_min=2.25, tau_max=2.75, points=50, time=time, timestep_dt=dt, seed=2, folder_name="/Volumes/Tiedostoja/dislocationData/14-feb-n1")
    # seeds = range(11,21)
    # for seed,_ in zip(seeds, tqdm(range(len(seeds)), desc="Running depinning studies", unit="study")):
    #     studyDepinning_mp(tau_min=2.25, tau_max=2.75, points=50, time=time, timestep_dt=dt, seed=seed, folder_name="results/13-feb-n2")

    loadedSim = loadResults("joku-simulaatio.pickle")
    makeGif(loadedSim, "results/gifs")