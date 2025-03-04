from pathlib import Path
import json
from plots import *
import numpy as np
from partialDislocation import PartialDislocationsSimulation
from scipy import optimize

def dumpResults(sim: PartialDislocationsSimulation, folder_name: str):
    # TODO: Säilö mielummin useampi tollanen musitiin ja kirjoita harvemmin
    # Dumps the results of a simulation to a npz file
    if not sim.has_simulation_been_run:
        raise Exception("Simulation has not been run.")
    
    parameters = np.array([
        sim.bigN, sim.length, sim.time, sim.dt,
        sim.deltaR, sim.bigB, sim.smallB, sim.b_p,
        sim.cLT1, sim.cLT2, sim.mu, sim.tauExt, sim.c_gamma,
        sim.d0, sim.seed, sim.tau_cutoff
        ]) # From these parameters you should be able to replicate the simulation
    

    dump_path = Path(folder_name).joinpath("simulation-dumps")
    dump_path = dump_path.joinpath(f"seed-{sim.seed}")
    dump_path.mkdir(exist_ok=True, parents=True)

    dump_path = dump_path.joinpath(f"sim-{sim.tauExt:.4f}.npz")
    np.savez(str(dump_path), params=parameters, y1=sim.y1, y2=sim.y2)

    pass

def saveLastState_partial(sim: PartialDislocationsSimulation, folder_name):
    parameters = np.array([
        sim.bigN, sim.length, sim.time, sim.dt,
        sim.deltaR, sim.bigB, sim.smallB, sim.b_p,
        sim.cLT1, sim.cLT2, sim.mu, sim.tauExt, sim.c_gamma,
        sim.d0, sim.seed, sim.tau_cutoff
        ]) # From these parameters you should be able to replicate the simulation

    dump_path = Path(folder_name).joinpath("simulation-dumps")
    dump_path = dump_path.joinpath(f"seed-{sim.seed}")
    dump_path.mkdir(exist_ok=True, parents=True)

    dump_path = dump_path.joinpath(f"sim-partial-tauExt-{sim.tauExt:.4f}-at-t-{sim.time}.npz")
    np.savez(str(dump_path), params=parameters, y1=sim.y1[sim.timesteps-1], y2=sim.y2[sim.timesteps-1])

def saveLastState_single(sim: DislocationSimulation, folder_name):
    parameters = np.array([
        sim.bigN, sim.length, sim.time, sim.dt,
        sim.deltaR, sim.bigB, sim.smallB, sim.b_p,
        sim.cLT1, sim.mu, sim.tauExt,
        sim.d0, sim.seed, sim.tau_cutoff
        ]) # From these parameters you should be able to replicate the simulation

    dump_path = Path(folder_name).joinpath("simulation-dumps")
    dump_path = dump_path.joinpath(f"seed-{sim.seed}")
    dump_path.mkdir(exist_ok=True, parents=True)

    dump_path = dump_path.joinpath(f"sim-single-tauExt-{sim.tauExt:.4f}-at-t-{sim.time}.npz")
    np.savez(str(dump_path), params=parameters, y1=sim.y1[sim.timesteps-1])

def plotDislocation(type:str,path_to_file,save_path):
    # Loads and plots a dislocation a a certain point in time
    loaded = np.load(path_to_file)

    plt.clf()
    if type == "partial":
        bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff = loaded["params"]
        y1, y2 = loaded["y1"], loaded["y2"]
        x_axis = np.arange(bigN)*(length/bigN)

        plt.figure(figsize=(8,8))
        plt.plot(x_axis, y1, label="$y_1 (x)$", color="orange")
        plt.plot(x_axis, y2, label="$y_2 (x)$", color="blue")
        plt.title(f"Partial dislocation at time t={time} w/ $\\tau_{{ext}}$ = {tauExt:.3f}")
        plt.legend()

        p = Path(save_path).joinpath(f"partial-dislocation-s-{seed:.2f}-t-{tauExt:.3f}.png")
        plt.savefig(p, dpi=300)
        

    elif type == "single":
        bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, mu, tauExt, d0, seed, tau_cutoff = loaded["params"]
        y1 = loaded["y1"]
        x_axis = np.arange(bigN)*(length/bigN)

        plt.figure(figsize=(8,8))
        plt.plot(x_axis, y1, label="$y_1 (x)$", color="blue")
        plt.title(f"Single dislocation at time t={time} w/ $\\tau_{{ext}}$ = {tauExt:.3f}")
        plt.legend()

        p = Path(save_path).joinpath(f"single-dislocation-s-{seed}-t-{tauExt}.png")
        plt.savefig(p, dpi=300)
    
def loadResults(file_path):
    # Load the results from a file at file_path to a new simulation object for further processing

    loaded  = np.load(file_path)

    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff = loaded["params"]

    res = PartialDislocationsSimulation(bigN=bigN.astype(int), bigB=bigB, length=length,
                                        time=time,timestep_dt=dt, deltaR=deltaR, 
                                        smallB=smallB, b_p=b_p, cLT1=cLT1, cLT2=cLT2,
                                        mu=mu, tauExt=tauExt, c_gamma=c_gamma, d0=d0, seed=seed.astype(int))
    
    # Modify the internal state of the object according to loaded parameters
    res.tau_cutoff = tau_cutoff
    res.has_simulation_been_run = True

    res.y1 = loaded["y1"]
    res.y2 = loaded["y2"]

    return res

def dumpDepinning(tauExts:np.ndarray, v_rels:list, time, seed, dt, folder_name:str, extra:list = None):
    results_json = {
        "stresses":tauExts.tolist(),
        "v_rel":v_rels,
        "seed":seed,
        "time":time,
        "dt":dt
    }
    if extra != None:
        results_json["extra"] = extra

    depining_path = Path(folder_name)
    depining_path = depining_path.joinpath("depinning-dumps")
    depining_path.mkdir(exist_ok=True, parents=True)
    depining_path = depining_path.joinpath(f"depinning-{min(tauExts)}-{max(tauExts)}-{len(tauExts)}-{time}-{seed}.json")
    with open(str(depining_path), 'w') as fp:
        json.dump(results_json,fp)

    pass

def loadDepinningDumps(folder, partial:bool):
    vCm = list()
    stresses = None
    for file in Path(folder).iterdir():
        with open(file, "r") as fp:
            try:
                depinning = json.load(fp)
            except:
                print(f"File {file} is corrupted.")

            if stresses == None:
                stresses = depinning["stresses"]
            
            vCm_i = depinning["v_rel"]
            if len(vCm_i) == 0: # Check for empty lists
                seed = depinning["seed"]
                print(f"There was problem loading data with seed {seed}")
                continue

            vCm.append(vCm_i)

    return (stresses, vCm) # Retuns a single list of stresses and a list of lists of velocities vCm

def calculateCoarseness(path_to_dislocation, l):
    # Given a path to a single dislocation at some time t calculate the coarseness.
    loaded = np.load(path_to_dislocation)
    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, mu, tauExt, d0, seed, tau_cutoff = loaded["params"]
    y = np.array(loaded["y1"])

    deltaL = length/bigN
    avgY = np.average(y)

    res = [ ((y[i] - avgY)*(y[ (i+l) % y.size ] - avgY))**2 for i in np.arange(y.size) ] # TODO: check fomula here
    res = sum(res)/len(res)
    res = np.sqrt(res)

    return res

def makeRoughnessPlot(path_to_dislocation:str, save_path:str, l_range:tuple):
    # Loads a dislocation and computes the coarseness in a given range.
    loaded = np.load(path_to_dislocation)

    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, mu, tauExt, d0, seed, tau_cutoff = loaded["params"]
    y = np.array(loaded["y1"])

    avgY = np.average(y)

    l_range = range(0,int(bigN))    # TODO: Use range parameter instead
    coarsnesses = np.empty(int(bigN))
    
    for l in l_range:
        res = [ ( (y[i] - avgY)*(y[ (i+l) % y.size ] - avgY) )**2 for i in np.arange(y.size) ] # TODO: check fomula here
        res = sum(res)/len(res)
        c = np.sqrt(res)
        coarsnesses[l] = c
    
    plt.clf()
    plt.figure(figsize=(8,8))
    plt.plot(l_range, coarsnesses, color="blue",label="Coarseness")
    plt.title(f"Roughness single dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt}")
    plt.xlabel("L")
    plt.ylabel("W(L)")

    p = Path(save_path).joinpath(f"coarseness-s-{seed}-{tauExt}.png")
    plt.savefig(p, dpi=300)
    pass

def v_fit(tau_ext, tau_crit, beta, a):
    v_res = np.empty(len(tau_ext))
    for n,tau in enumerate(tau_ext):
        if tau > tau_crit:
            v_res[n] = a*(tau - tau_crit)**beta
        else:
            v_res[n] = 0
    return v_res

def getCriticalForce(file_path):

    with open(file_path, 'r') as fp:
        depinning = json.load(fp)
    
    stresses = depinning["stresses"]
    vcm = depinning["v_rel"]

    fit_params, pcov = optimize.curve_fit(v_fit, stresses, vcm, p0=[2.5, 1.5, 1], maxfev=800)

    critical_force, beta, a = fit_params

    xnew = np.linspace(min(stresses), max(stresses), 200)
    ynew = v_fit(xnew, *fit_params)

    # plt.clf()
    # plt.cla()
    # plt.close('all')
    # plt.figure(figsize=(8,8))

    # plt.scatter(stresses, vcm, marker='x', color="red", label="Depinning")
    # plt.plot(xnew, ynew, color="blue", label="fit")

    # plt.legend()

    # plt.show()

    return critical_force

def getCriticalForces(dir_path):
    # Get all the critical forces from the depinning dir path.
    crit_forces = list()
    for file_path in Path(dir_path).iterdir():
        c = getCriticalForce(file_path)
        crit_forces.append(c)
    return crit_forces

if __name__ == "__main__":
    results_root = Path("results/25-feb-small-interval")

    stresses, vCm = loadDepinningDumps(results_root.joinpath('single-dislocation/depinning-dumps'), partial=False)
    stresses1, vCm_partial = loadDepinningDumps(results_root.joinpath('partial-dislocation/depinning-dumps'), partial=True)

    makeDepinningPlotAvg(10000, 100, [stresses, stresses1], [vCm[0:100], vCm_partial[0:100]], ["single", "partial"], 
                         folder_name=results_root, colors=["red", "blue"])
    
    makeRoughnessPlot(results_root.joinpath("single-dislocation/simulation-dumps/seed-1/sim-single-tauExt-2.6081-at-t-10000.0.npz"), 
                            results_root,
                            (None,None))
    
    
    # Makes a gif from a complete saved simualation
    # sim = loadResults("results/15-feb-1/pickle-dumps/seed-100/sim-3.0000.npz")
    # makeVelocityPlot(sim, "results/15-feb-1/")
    # makeGif(sim, "results/15-feb-1/")

    # Plots dislocations at the end of simulatino w/ tau_ext=0.0000
    plotDislocation("single",
                    results_root.joinpath("single-dislocation/simulation-dumps/seed-1/sim-single-tauExt-2.6081-at-t-10000.0.npz"),
                    results_root)
    plotDislocation("partial", 
                    results_root.joinpath("partial-dislocation/simulation-dumps/seed-0/sim-partial-tauExt-2.6000-at-t-10000.0.npz"),
                    results_root)
    
    c = getCriticalForce(results_root.joinpath("single-dislocation/depinning-dumps/depinning-2.6-3.0-100-10000.0-5.json"))
    
    tau_c_single = getCriticalForces(results_root.joinpath("single-dislocation/depinning-dumps/"))
    tau_c_partial = getCriticalForces(results_root.joinpath("partial-dislocation/depinning-dumps/"))

    print(f"Single dislocation: tau_c={sum(tau_c_single)/len(tau_c_single)}")
    print(f"Partial dislocation: tau_c={sum(tau_c_partial)/len(tau_c_partial)}")