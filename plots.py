from simulation import PartialDislocationsSimulation
from singleDislocation import *
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

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
    plt.savefig(str(avg_dist_path), dpi=300)
    pass

def makeVelocityPlot(sim: PartialDislocationsSimulation, folder_name):
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
    plt.savefig(f"{folder_name}/constant-stress-tau-{tauExt:.2f}.png", dpi=300) # It's nice to have a plot from each individual simulation
    plt.clf()
    plt.close()

    pass

def makeVelocityPlot(sim: DislocationSimulation, folder_name:str = "results"):
        # Make a plot with velocities and average distance from the given simulation sim
    # The sim object must have methods: getTValues, getAverageDistance, getRelaxedVeclovity
    # and getCM

    Path(folder_name).mkdir(exist_ok=True, parents=True)

    t = sim.getTvalues()
    avgD = sim.getAverageDistances()
    start = sim.time - 1000
    vR = sim.getRelaxedVelocity(time_to_consider=1000)

    cm = sim.getCM()

    tauExt = sim.tauExt

    fig, axes = plt.subplots(1,2, figsize=(12, 8))

    axes_flat = axes.ravel()

    axes[0].plot(t, np.gradient(cm), color="black", label="$y_1$") # Plot the velocity of the cm of y_1
    axes[0].plot([start,sim.time], vR*np.ones(2), '-', color="red")

    axes[0].set_title(sim.getTitleForPlot())
    axes[0].set_xlabel("Time t (s)")
    axes[0].set_ylabel("$ v_{CM} $")
    axes[0].legend()

    
    axes[1].plot(t, avgD, label="Distance from y=0.")

    axes[1].set_title(sim.getTitleForPlot())
    axes[1].set_xlabel("Time t (s)")
    axes[1].set_ylabel("$ d $")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{folder_name}/single-dislocation-tau-{tauExt:.2f}.png", dpi=300) # It's nice to have a plot from each individual simulation
    plt.clf()
    plt.close()

    pass

    pass

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

def makeDepinningPlot(stresses, relVelocities, time, seed, folder_name="results"):
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    plt.clf()
    plt.scatter(stresses, relVelocities, marker='x')
    plt.title(f"Depinning, seed={seed}")
    plt.xlabel("$\\tau_{ext}$")
    plt.ylabel("$v_{CM}$")
    plt.savefig(f"{folder_name}/depinning-{min(stresses)}-{max(stresses)}-{len(stresses)}-{time}-{seed}.png", dpi=300)
    # plt.show()

def makeDepinningPlotAvg(stresses, vCm, time, count, folder_name="results"):
    # This function is designed for data obtained by averaging.

    Path(folder_name).mkdir(exist_ok=True, parents=True)
    plt.clf()
    averages = np.mean(vCm, axis=0)

    plt.scatter(stresses, averages, 10, marker='x', linewidths=1, label="mean")
    plt.plot(stresses, averages + np.std(vCm, axis=0), '--', color="red", linewidth=1, label="$\\sigma$")
    plt.plot(stresses, averages - np.std(vCm, axis=0), '--', color="red", linewidth=1)

    plt.title(f"Depinning    N={count}")
    plt.xlabel("$\\tau_{ext}$")
    plt.ylabel("$v_{CM}$")
    plt.legend()

    plt.savefig(f"{folder_name}/depinning-tau-{min(stresses)}-{max(stresses)}-p-{len(stresses)}-t-{time}-N-{count}.png", dpi=300)