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

        x = sim_i.getTvalues()
        avgI, v_avg = sim_i.run_simulation()

        axes_flat[i].plot(x,avgI)
        axes_flat[i].set_xlabel("Time (s)")
        axes_flat[i].set_ylabel("Average distance")
        print(f"Simulation {i+1}, min: {min(avgI)}, max: {max(avgI)}, delta: {max(avgI) - min(avgI)}, v_avg: {v_avg}")

    plt.tight_layout()
    plt.show()

def studyStress():
    # Run simulations with varying external stress
    n_simulations = 50
    min_stress = 1
    max_stress = 200

    velocities = list()
    stresses = np.linspace(min_stress,max_stress,n_simulations) # Vary stress from 0 to 5

    for stress in stresses:
        sim_i = PartialDislocationsSimulation(tauExt=stress, timestep_dt=0.5, time=100, c_gamma=60, cLT1=4, cLT2=4)
        avgD, v_avg = sim_i.run_simulation()
        velocities.append(v_avg)
    
    plt.plot(stresses, velocities)
    plt.show()

def makeGif(gradient_term=0.5, potential_term=60):
    # Run the simulation
    total_time = 100
    total_dt = 0.25

    simulation = PartialDislocationsSimulation(time=total_time, timestep_dt=total_dt, bigN=50, length=50, 
                                               c_gamma=potential_term,
                                               cLT1=gradient_term, cLT2=gradient_term, 
                                               deltaR=1)
    
    avgDists, avgVelocities = simulation.run_simulation()

    # Get revelant info from simulation
    y1,y2 = simulation.getLineProfiles()
    x = simulation.getXValues()
    t_axis = simulation.getTvalues()

    gifTitle = "$ " + ", ".join(simulation.getParamsInLatex()) + " $"
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

    frames[0].save(f"gifs/lines-moving-aroung-C-{gradient_term}-C_gamma-{potential_term}.gif", save_all=True, append_images=frames[1:], duration=50, loop=0)

    plt.clf()
    plt.plot(t_axis, avgDists)
    plt.title("Average distance")
    plt.xlabel("Time t (s)")
    plt.savefig(f"avg_dist-C-{gradient_term}-G-{potential_term}.png")
    pass

def makePotentialPlot():
    sim = PartialDislocationsSimulation()
    sim.jotain_saatoa_potentiaaleilla()

if __name__ == "__main__":
    for g in [0.1, 0.5, 0.8, 1, 1.5]:
        for gamma in [40,60,80]:
            print(f"making gif with C_LT1=C_LT2={g} and C_gamma={gamma}")
            makeGif(g, gamma)