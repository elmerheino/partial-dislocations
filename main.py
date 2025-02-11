import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from simulation import PartialDislocationsSimulation
from pathlib import Path
# import pickle
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

def studyConstantStress(tauExt=1, 
                        folder_name="results", # Leave out the final / when defining value
                        timestep_dt=0.05,
                        time=20000, seed=None):
    
    # Run simulations with varying external stress

    simulation = PartialDislocationsSimulation(tauExt=tauExt, bigN=200, length=200, 
                                              timestep_dt=timestep_dt, time=time, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1, seed=seed)
    # print(" ".join(simulation.getParamsInLatex()))

    simulation.run_simulation()

    # Dump the simulation object to a pickle
    # Path(f"{folder_name}/pickles").mkdir(exist_ok=True, parents=True)
    # with open(f"{folder_name}/pickles/simulation-state-tau-{tauExt}.pickle", "wb") as fp:
    #     pickle.dump(simulation, fp)

    rV1, rV2, totV2 = simulation.getRelaxedVelocity(time_to_consider=1000) # The velocities after relaxation

    # makeStressPlot(simulation, folder_name)

    return (rV1, rV2, totV2)

def makeStressPlot(sim: PartialDislocationsSimulation, folder_name):
    # Make a plot with velocities and average distance from the given simulation sim

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


def studyDepinning_mp(tau_min=2.85, tau_max=3.05, points=50, 
                   folder_name="results",            # Leave out the final / when defining value
                   timestep_dt=0.05, time=10000, seed=123):
    # Multiprocessing compatible version of a single depinning study
    
    stresses = np.linspace(tau_min, tau_max, points)
    
    with mp.Pool(5) as pool:
        results = pool.map(partial(studyConstantStress, folder_name=folder_name, timestep_dt=timestep_dt, time=time, seed=seed), stresses)
    
    v_cm = [i[2] for i in results]
    makeDepinningPlot(stresses, v_cm, time, seed, folder_name=folder_name)

def studyDepinning(tau_min=0, tau_max=2, points=50, 
                   folder_name="results",            # Leave out the final / when defining value
                   timestep_dt=0.05, time=30000, seed=123): # Fix the stress field for different tau_ext
    
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
    plt.clf()
    plt.scatter(stresses, relVelocities, marker='x')
    plt.title("Depinning")
    plt.xlabel("$\\tau_{ext}$")
    plt.ylabel("$v_{CM}$")
    plt.savefig(f"{folder_name}/depinning-{min(stresses)}-{max(stresses)}-{len(stresses)}-{time}-{seed}.png")
    # plt.show()

def makePotentialPlot():
    sim = PartialDislocationsSimulation()
    sim.jotain_saatoa_potentiaaleilla()

def makeGif(gradient_term=0.5, potential_term=60, total_dt=0.25, tau_ext=1):
    # Run the simulation
    total_time = 400

    simulation = PartialDislocationsSimulation(time=total_time, timestep_dt=total_dt, bigN=200, length=200, 
                                               c_gamma=potential_term,
                                               cLT1=gradient_term, cLT2=gradient_term, 
                                               deltaR=1, tauExt=tau_ext, d0=39)
    
    simulation.run_simulation()
    simulation.run_further(50,new_dt=total_dt)

    # Get revelant info from simulation
    y1,y2 = simulation.getLineProfiles()

    x = simulation.getXValues()
    t_axis = simulation.getTvalues()

    gifTitle = simulation.getTitleForPlot()

    # Make the gif
    frames = list()
    counter = 0
    for h1,h2,t in zip(y1,y2,t_axis):
        # counter = counter + 1
        # time_at_t = counter*total_dt

        plt.plot(x, h1, color="blue")
        plt.plot(x, h2, color="red")

        plt.legend(["line 1", "line 2"])
        plt.title(gifTitle)
        plt.xlabel(f"t={t:.2f}")

        plt.savefig(f"frames/tmp-{t}.png")
        plt.clf()
        frames.append(Image.open(f"frames/tmp-{t}.png"))

    frames[0].save(f"lines-moving-aroung-C-{gradient_term}-C_gamma-{potential_term}.gif", save_all=True, append_images=frames[1:], duration=20, loop=0)

    avgDists = simulation.getAverageDistances()

    plt.clf()
    plt.plot(t_axis, avgDists)
    plt.title("Average distance")
    plt.xlabel("Time t (s)")
    plt.savefig(f"avg_dist-C-{gradient_term}-G-{potential_term}.png")
    pass

def multiple_depinnings_mp(seed):
    studyDepinning(folder_name="results/11-feb-n2", tau_min=2, tau_max=4, timestep_dt=0.05, time=10000, seed=seed)

if __name__ == "__main__":
    # with mp.Pool(mp.cpu_count() - 5) as pool:
    #     r = pool.map(multiple_depinnings_mp, range(2000,2000+1))
    # studyDepinning(folder_name="results/11-feb-n3", tau_min=2, tau_max=4, timestep_dt=0.05, time=10000, seed=3424)
    studyDepinning_mp()