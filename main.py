import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from simulation import PartialDislocationsSimulation

# run 4 simulations right away
def studyAvgDistance():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()

    for i in range(0,4):
        np.random.seed(i)

        sim_i = PartialDislocationsSimulation(timestep_dt=0.5, time=100, c_gamma=60, cLT1=4, cLT2=4)

        t = sim_i.getTvalues()
        avgI = sim_i.run_simulation()

        axes_flat[i].plot(t,avgI)
        axes_flat[i].set_xlabel("Time (s)")
        axes_flat[i].set_ylabel("Average distance")
        print(f"Simulation {i+1}, min: {min(avgI)}, max: {max(avgI)}, delta: {max(avgI) - min(avgI)}")

    plt.tight_layout()
    plt.show()

def studyStress():
    # Run simulations with varying external stress
    n_simulations = 200
    min_stress = 0
    max_stress = 10

    stresses = np.linspace(min_stress,max_stress,n_simulations) # Vary stress from 0 to 5
    avgH_y1 = list()
    avgH_y2 = list()

    for stress in stresses:
        sim_i = PartialDislocationsSimulation(tauExt=stress, bigN=200, length=200, 
                                              timestep_dt=0.1, time=10000, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1)
        print(sim_i.getParamsInLatex())
        avgD = sim_i.run_simulation()

        y1, y2 = sim_i.getLineProfiles()

        # avg_y1 = [np.average(i) for i in y1] # Average places of y1, which for uniform density is the centre of mass?
        # avg_y2 = [np.average(i) for i in y2] # Average places of y2

        cm_y1 = np.average(y1, axis=1)
        cm_y2 = np.average(y2, axis=1)

        # v_y1 = np.gradient(avg_y1) # Velocities of the average places
        # v_y2 = np.gradient(avg_y2)
        v_y1 = np.gradient(cm_y1[900:])
        v_y2 = np.gradient(cm_y2[900:])

        avgH_y1.append(np.average(v_y1))
        avgH_y2.append(np.average(v_y2))
    
    plt.plot(stresses, np.gradient(avgH_y1), color="blue", label="$y_1$") # Plot the velocity of the average place of y_1
    plt.plot(stresses, np.gradient(avgH_y2), color="red", label="$y_2$") # Plot the velocity of the average of y_2

    plt.title("Average velocity")
    plt.xlabel("Stress $\\tau_{ext}$")
    plt.ylabel("$ v_{avg} $")

    plt.legend()

    plt.show()

def studyConstantStress(tauExt=1):
    # Run simulations with varying external stress
    n_simulations = 200
    min_stress = 0
    max_stress = 10

    simulation = PartialDislocationsSimulation(tauExt=tauExt, bigN=200, length=200, 
                                              timestep_dt=0.05, time=10000, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1)
    print(simulation.getParamsInLatex())

    avgD = simulation.run_simulation()

    y1, y2 = simulation.getLineProfiles()

    cm_y1 = np.average(y1, axis=1)
    cm_y2 = np.average(y2, axis=1)

    t = simulation.getTvalues()

    start = 10000 - 1000        # Consider only the last 1000 s
    rV1, rV2 = simulation.getRelaxedVelocity(time_to_consider=1000) # The velocities after relaxation

    fig, axes = plt.subplots(1,2, figsize=(12, 8))

    axes_flat = axes.ravel()

    axes[0].plot(t, np.gradient(cm_y1), color="black", label="$y_1$") # Plot the velocity of the cm of y_1
    axes[0].plot(t, np.gradient(cm_y2), color="red", label="$y_2$") # Plot the velocity of the average of y_2
    axes[0].plot([start,simulation.time], rV1*np.ones(2), '-', color="red")
    axes[0].plot([start,simulation.time], rV2*np.ones(2), '-' ,color="blue")

    axes[0].set_title(simulation.getTitleForPlot())
    axes[0].set_xlabel("Time t (s)")
    axes[0].set_ylabel("$ v_{CM} $")
    axes[0].legend()

    
    axes[1].plot(t, avgD, label="Average distance")

    axes[1].set_title(simulation.getTitleForPlot())
    axes[1].set_xlabel("Time t (s)")
    axes[1].set_ylabel("$ d $")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"constant-stress-tau-{tauExt}.png")

    return (rV1, rV2)


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
    
    avgDists = simulation.run_simulation()

    # Get revelant info from simulation
    y1,y2 = simulation.getLineProfiles()
    x = simulation.getXValues()
    t_axis = simulation.getTvalues()

    gifTitle = simulation.getTitleForPlot()

    # Make the gif
    frames = list()
    counter = 0
    for h1,h2,t in zip(y1,y2,np.arange(total_time/total_dt)):
        counter = counter + 1
        time_at_t = counter*total_dt

        plt.plot(x, h1, color="blue")
        plt.plot(x, h2, color="red")

        plt.legend(["line 1", "line 2"])
        plt.title(gifTitle)
        plt.xlabel(f"t={time_at_t}")

        plt.savefig(f"frames/tmp-{t}.png")
        plt.clf()
        frames.append(Image.open(f"frames/tmp-{t}.png"))

    frames[0].save(f"lines-moving-aroung-C-{gradient_term}-C_gamma-{potential_term}.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)

    plt.clf()
    plt.plot(t_axis, avgDists)
    plt.title("Average distance")
    plt.xlabel("Time t (s)")
    plt.savefig(f"avg_dist-C-{gradient_term}-G-{potential_term}.png")
    pass


if __name__ == "__main__":
    a,b = studyConstantStress(tauExt=1)
    print(f" avg v1: {a} avg v2: {b}")
