from partialDislocation import PartialDislocationsSimulation
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
    start = sim.time - sim.time/10 # Consider the last 10% of time
    rV1, rV2, totV2 = sim.getRelaxedVelocity(time_to_consider=sim.time/10)

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

def makeVelocityPlotSingle(sim: DislocationSimulation, folder_name:str = "results"):
        # Make a plot with velocities and average distance from the given simulation sim
    # The sim object must have methods: getTValues, getAverageDistance, getRelaxedVeclovity
    # and getCM

    Path(folder_name).mkdir(exist_ok=True, parents=True)

    t = sim.getTvalues()
    avgD = sim.getAverageDistances()
    start = sim.time - sim.time/10
    vR = sim.getRelaxedVelocity(time_to_consider=sim.time/10)

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

def makeDepinningPlotAvg(time, count, stresses:list, vCms:list, names:list, folder_name="results", colors=["red"]):
    # This function is designed for data obtained by averaging.

    Path(folder_name).mkdir(exist_ok=True, parents=True)
    plt.clf()
    plt.figure(figsize=(8,8))
    for stress, vCm, name, color in zip(stresses, vCms, names, colors):
        averages = np.mean(vCm, axis=0)

        plt.scatter(stress, averages, 10, marker='x', linewidths=1, label=f"$\\bar{{x}}$ {name} N={len(vCm)}", color=color)
        sd_plot, = plt.plot(stress, averages + np.std(vCm, axis=0), '--', linewidth=0.5, label=f"$\\sigma$ {name}")
        plt.plot(stress, averages - np.std(vCm, axis=0), '--', color=sd_plot.get_color(), linewidth=0.5)

        plt.title(f"Depinning    N={count}")
        plt.xlabel("$\\tau_{ext}$")
        plt.ylabel("$v_{CM}$")
        plt.legend()

    plt.savefig(f"{folder_name}/depinning-tau-{min(stresses[0])}-{max(stresses[0])}-p-{len(stresses[0])}-t-{time}-N-{count}.png", dpi=300)

def plotRandTau():
    noise = 1000
    sim = PartialDislocationsSimulation(1024, 1024,100,0.05,noise,1,1,1,1,0)

    x = 100
    z = list()
    y = np.linspace(1024-10,1024+10,100)
    times = list()
    for y_i in y:
        y_all = y_i*np.ones(sim.bigN)
        t0 = time.time()
        z_i = sim.tau_no_interpolation(y_all)[x]
        t1 = time.time()
        times.append(t1 - t0)
        z.append(z_i)

    avg_time = sum(times)/len(times)
    print(f"No interp call takes on avg {avg_time*1e6} mu s per call")

    x = 100
    z_1 = list()
    y_1 = np.linspace(1024-10,1024+10,100)
    times_1 = list()
    for y_i in y_1:
        y_all = y_i*np.ones(sim.bigN)
        t0 = time.time()
        z_i = sim.tau_interpolated(y_all)[x]
        t1 = time.time()
        times_1.append(t1 - t0)
        z_1.append(z_i)

    scipy_avg_time = sum(times_1)/len(times_1)
    print(f"Scipy call takes on avg {scipy_avg_time*1e6} mu s per call")
    
    x = 100
    z_2 = list()
    y_2 = np.linspace(1024-10,1024+10,100)
    times_2 = list()
    for y_i in y_2:
        y_all = y_i*np.ones(sim.bigN)
        t0 = time.time()
        z_i = sim.tau_interpolated_static(y_all, sim.bigN, sim.stressField, sim.x_indices)[x]
        t1 = time.time()

        times_2.append(t1 - t0)

        z_2.append(z_i)
    numba_avg_time = sum(times_2)/len(times_2)
    print(f"Numba call takes on avg {numba_avg_time*1e6} mu s per call")


    fig, axes = plt.subplots(3, 1, figsize=(12, 4))  # 1 row, 3 columns
    axes[0].plot(y, z)
    axes[0].set_title("Cross section of $z = \\tau(100,y)$ when x=100")
    axes[0].set_xlabel("$y$")
    axes[0].set_ylabel("$z$")

    axes[1].plot(y_1, z_1)
    axes[1].set_title("Interpolated (scipy) Cross section of $z = \\tau(100,y)$ when x=100")
    axes[1].set_xlabel("$y$")
    axes[1].set_ylabel("$z$")

    axes[2].plot(y_2, z_2)
    axes[2].set_title("Interpolated (static numba) Cross section of $z = \\tau(100,y)$ when x=100")
    axes[2].set_xlabel("$y$")
    axes[2].set_ylabel("$z$")

    fig.tight_layout()

    plt.show()

    print(f"Scipy is {numba_avg_time/scipy_avg_time} faster than numba")

if __name__ == "__main__":
    plotRandTau()