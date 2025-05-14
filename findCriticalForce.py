from argparse import ArgumentParser
from pathlib import Path
from singleDislocation import DislocationSimulation
from partialDislocation import PartialDislocationsSimulation
import numpy as np
import multiprocessing as mp

def find_critical_force(noise_level, seed, delta_tau,tolerance=0.001):
    """
    Find the critical force for a given noise level by incremeng the external stress
    by delta_tau until the velocity exceeds the tolerance. Cutoff velocity is defined as
    v_cutoff = v0 * (1 + tolerance), where v0 is the relaxed velocity at tau=0.

    Parameters:
    noise_level (float): The noise level to use in the simulation.
    seed (int): The random seed for the simulation.
    delta_tau (float): The increment for the external stress.
    tolerance (float): The tolerance for the velocity as a fractional increment of the relaxed velocity at tau=0.
    """

    tau_ext_i = 0

    simulation_0 = DislocationSimulation(deltaR=noise_level, bigB=1, smallB=1,
                                            mu=1, bigN=1024, length=1024, 
                                            dt=0.05, time=10000, d0=39,
                                            cLT1=1, seed=seed,
                                            tauExt=tau_ext_i)
    simulation_0.run_simulation()
    velocity_0 = simulation_0.getRelaxedVelocity()
    v_cutoff = velocity_0 * (1 + tolerance)


    while tau_ext_i <= noise_level*1.1:
        # Create a new simulation object
        simulation = DislocationSimulation(deltaR=noise_level, bigB=1, smallB=1,
                                            mu=1, bigN=1024, length=1024, 
                                            dt=0.05, time=10000, d0=39,
                                            cLT1=1, seed=seed,
                                            tauExt=tau_ext_i)
        
        # Run the simulation
        simulation.run_simulation()
        
        # Get the results
        velocity = simulation.getRelaxedVelocity()
        
        # Check if the results are valid
        if velocity > v_cutoff:
            tau_crit = tau_ext_i
            return tau_crit
        
        tau_ext_i += delta_tau
        # print(f"tau_ext_i: {tau_ext_i}, velocity: {velocity}, v_cutoff: {v_cutoff} valid: {velocity > v_cutoff}")


if __name__ == "__main__":
    # Example usage
    parser = ArgumentParser(prog="Dislocation simulation")
    parser.add_argument('-id', '--array-task-id', help='slurm array task id', required=True, type=int)
    parser.add_argument('-len', '--array-length', help='slurm array length', required=True, type=int)
    parser.add_argument('-f', '--folder', help='data output folder', required=False, type=str)

    args = parser.parse_args()

    all_noises = np.logspace(-2, 1, args.array_length)

    noise = all_noises[args.array_task_id - 1]

    tau_crit = find_critical_force(noise, seed=42, delta_tau=noise/1000)

    path = Path(args.folder)
    path.mkdir(parents=True, exist_ok=True)
    path = path.joinpath("noise_data.txt")

    if path.exists():
        with open(path, "a") as f:
            f.write(f"{noise}\t{tau_crit}\n")
    else:
        with open(path, "w") as f:
            f.write(f"{noise}\t{tau_crit}\n")