#!/usr/bin/env python3

from pathlib import Path
import json
from plots import *
import numpy as np
from partialDislocation import PartialDislocationsSimulation
from scipy import optimize
from scipy import stats
from argparse import ArgumentParser
import multiprocessing as mp
from functools import partial
from numba import jit
import shutil

def plotDislocation_partial(path_to_file, save_path, point=None):
    loaded = np.load(path_to_file)

    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff = loaded["params"]

    y1 = loaded["y1"]
    y2 = loaded["y2"]

    if point == None:
        point = len(y1) - 1

    y1, y2 = y1[point], y2[point]
    x_axis = np.arange(bigN)*(length/bigN)

    plt.figure(figsize=(8,8))
    plt.plot(x_axis, y1, label="$y_1 (x)$", color="orange")
    plt.plot(x_axis, y2, label="$y_2 (x)$", color="blue")
    plt.title(f"Partial dislocation at time t={time} w/ $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.legend()

    p = Path(save_path).joinpath("partial-imgs")
    p.mkdir(parents=True, exist_ok=True)
    p = p.joinpath(f"partial-dislocation-s-{seed:.2f}-t-{tauExt:.3f}.png")
    plt.savefig(p, dpi=300)
    pass

def plotDislocation_nonp(path_to_file, save_path, point=None):
    loaded = np.load(path_to_file)
    bigN, length, time, dt, deltaR, bigB, smallB, b_p, cLT1, mu, tauExt, d0, seed, tau_cutoff = loaded["params"]
    y1 = loaded["y1"]

    if point == None:
        point = len(y1) - 1
    
    y1 = y1[point]

    x_axis = np.arange(bigN)*(length/bigN)

    plt.figure(figsize=(8,8))
    plt.plot(x_axis, y1, label="$y_1 (x)$", color="blue")
    plt.title(f"Single dislocation at time t={time} w/ $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.legend()

    p = Path(save_path).joinpath("non-partial-imgs")
    p.mkdir(parents=True, exist_ok=True)
    p = p.joinpath(f"single-dislocation-s-{seed}-t-{tauExt}.png")
    plt.savefig(p, dpi=300)
    pass
    
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

def dumpDepinning(tauExts:np.ndarray, v_rels:list, time, seed, dt, folder_name:str, extra:list = None): # TODO: get rid of this function
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

def getCriticalForceFromFile(file_path):

    with open(file_path, 'r') as fp:
        depinning = json.load(fp)
    
    stresses = depinning["stresses"]
    vcm = depinning["v_rel"]

    critical_force = getCriticalForce(stresses, vcm)

    return critical_force

def getCriticalForce(stresses, vcm):
    fit_params, pcov = optimize.curve_fit(v_fit, stresses, vcm, p0=[2.5, 1.5, 1], maxfev=1600)

    critical_force, beta, a = fit_params

    return critical_force, beta, a

# Roughness plots, averaging, rearranging

def exp_beheavior(l, c, zeta):
    return c*(l**zeta)

def roughness_fit(l, c, zeta, cutoff):
    # if l < cutoff:
    #     return exp_heavor(l, c, zeta)
    # else:
    #     return k
    # For continuity we need exp_beheavior(cutoff) = k making the parameter irrelevant
    return np.array(list(map(lambda i : exp_beheavior(i, c, zeta) if i < cutoff else c*(cutoff**zeta), l)))

def roughness_partial(l, c, zeta, k, cutoff):
    return np.array(list(map(lambda i : c*(i**zeta) + k if i < cutoff else c*(cutoff**zeta) + k, l)))

def makeRoughnessPlot_partial(l_range, avg_w12, params, save_path):
    # TODO: incorporate similar
    bigN, length,   time,   dt, selfdeltaR, selfbigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    fit_params, pcov = optimize.curve_fit(roughness_fit, l_range, avg_w12, p0=[1.1, # C
                                                                                1.1, # zeta
                                                                                4.1 ]) # cutoff
    c, zeta, cutoff = fit_params
    k = c*(cutoff**zeta)
    ynew = roughness_fit(l_range, *fit_params)

    plt.clf()
    plt.figure(figsize=(8,8))
    plt.scatter(np.log(l_range), np.log(avg_w12), label="$W_{{12}}$", marker="x")
    plt.plot(np.log(l_range), np.log(ynew), label=f"fit, c={c}, $\\zeta = $ {zeta}")

    plt.title(f"Roughness of a partial dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.xlabel("log(L)")
    plt.ylabel("$\\log(W_{{12}}(L))$")
    plt.legend()

    plt.savefig(save_path, dpi=300)

    return (tauExt, seed, c, zeta, cutoff, k)


def loadRoughnessData_partial(path_to_file, root_dir):
    # Helper function to enable multiprocessing
    loaded = np.load(path_to_file)
    params = loaded["parameters"]
    avg_w12 = loaded["avg_w"]
    l_range = loaded["l_range"]
    bigN, length,   time,   dt, selfdeltaR, selfbigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    p = Path(root_dir)
    p = p.joinpath("roughness-partial").joinpath(f"seed-{seed}")
    p.mkdir(parents=True, exist_ok=True)
    p = p.joinpath(f"avg-roughness-tau-{tauExt:.3f}.png")

    tauExt, seed, c, zeta, cutoff, k = makeRoughnessPlot_partial(l_range, avg_w12, params, p)

    return (tauExt, seed, c, zeta, cutoff, k)


def makeRoughnessPlot_np(l_range, avg_w, params, save_path : Path): # Non-partial -> perfect dislocation
    bigN, length,   time,   dt, selfdeltaR, selfbigB, smallB,  b_p, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    # Regular piecewise fit on all data
    fit_params, pcov = optimize.curve_fit(roughness_fit, l_range, avg_w, p0=[1.1, # C
                                                                            1.1, # zeta
                                                                            4.1 ]) # cutoff
    c, zeta_piecewise, cutoff_piecewise = fit_params
    k = c*(cutoff_piecewise**zeta_piecewise)

    ynew = roughness_fit(l_range, *fit_params)

    # Make an exponential plot to only part of data.
    l_0_range = np.linspace(np.exp(1),np.exp(4), 50) # not logarithmic here! from log(1) to log(4)
    zetas = np.empty(50)
    c_values = np.empty(50)

    for n, l_i in enumerate(l_0_range):
        last = np.argmax(l_range > l_i)
        # print(f"n : {n} l_i : {l_i} last : {last}")

        exp_l_i = l_range[:last]
        exp_w_i = avg_w[:last]

        exp_fit_p, pcov = optimize.curve_fit(exp_beheavior, exp_l_i, exp_w_i, p0 = [1.1, 1.1], maxfev=1600)
        c, zeta = exp_fit_p
        ynew_exp = exp_beheavior(exp_l_i, *exp_fit_p)

        zetas[n] = zeta
        c_values[n] = c

    # Make a plot of zeta as function of L_0
    plt.clf()
    plt.figure(figsize=(8,8))

    plt.plot(np.log(l_0_range), zetas, label="zeta")
    plt.axhline(y=zetas[0]*(1-0.05), label="$5 \\% $", linestyle='--', color="blue")
    plt.axhline(y=zetas[0]*(1-0.10), label="$10 \\% $", linestyle='--', color="red")

    plt.title(f"Fit parameter for perfect dislocation roughness $\\tau_{{ext}} = {tauExt:.3f}$")
    plt.xlabel("$ \\log(l_0) $")
    plt.ylabel("$ \\zeta $")
    plt.legend()
    zeta_save_path = save_path.parent.parent.joinpath("zeta-plots")
    zeta_save_path.mkdir(parents=True, exist_ok=True)
    zeta_save_path = zeta_save_path.joinpath(f"l0-zeta-{tauExt:.4f}.png")
    plt.savefig(zeta_save_path)

    # Now make the actual plot with suitable fit
    plt.clf()
    plt.figure(figsize=(8,8))

    plt.scatter(np.log(l_range), np.log(avg_w), label="$W$", marker="x")
    plt.plot(np.log(l_range), np.log(ynew), label="piecewise fit", color="blue")

    # Select zeta that is 10% at most smaller than initial zeta
    target_zeta = zetas[0]*(1-0.10)  # TODO: check this more closely, there might be something wrong.
    n_selected = np.argmax(zetas < target_zeta)

    last = np.argmax(l_range > l_0_range[n_selected])
    exp_l = l_range[:last]

    c, zeta = c_values[n_selected], zetas[n_selected]
    ynew_exp = exp_beheavior(exp_l, c, zeta)

    plt.plot(np.log(exp_l), np.log(ynew_exp), label=f"$ \\log (L) \\leq {np.log(l_0_range[n_selected]):.2f} $  fit", color="red")

    # Now find out the constant behavior

    start = int(round(len(l_range)/4))
    end = int(round(3*len(l_range)/4 ))

    const_l = l_range[start:end]
    const_w = avg_w[start:end]

    fit_c, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])
    
    new_c = np.ones(len(const_l))*fit_c

    plt.plot(np.log(const_l), np.log(new_c), label=f"N/4 < L < 3N/4", color="orange")

    # Take the constant from the piecewise cutoff
    start = int(round(cutoff_piecewise))

    const_l = l_range[start:]
    const_w = avg_w[start:]

    fit_c, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])
    
    new_c = np.ones(len(const_l))*fit_c

    plt.plot(np.log(const_l), np.log(new_c), label=f"log(L) > {np.log(start):.3f}", color="magenta")


    plt.title(f"Roughness of a perfect dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.xlabel("log(L)")
    plt.ylabel("$\\log(W_(L))$")
    plt.legend()

    plt.savefig(save_path, dpi=300)

    return (tauExt, seed, c, zeta, cutoff_piecewise, k)

def loadRoughnessData_np(f1, root_dir):
    # Helper function to enable use of multiprocessing when making plots
    loaded = np.load(f1)
    avg_w = loaded["avg_w"]
    l_range = loaded["l_range"]
    params = loaded["paramerters"]
    bigN, length,   time,   dt, selfdeltaR, selfbigB, smallB,  b_p, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    p = Path(root_dir)
    p = p.joinpath("roughness-non-partial").joinpath(f"seed-{seed}")
    p.mkdir(parents=True, exist_ok=True)
    p = p.joinpath(f"avg-roughness-tau-{tauExt:.3f}.png")

    tauExt, seed, c, zeta, cutoff, k = makeRoughnessPlot_np(l_range, avg_w, params, p)

    return (tauExt, seed, c, zeta, cutoff, k)

def makeAvgRoughnessPlots(root_dir):
    # Makes roughness plots that have been averaged only at simulation (that is velocity) level
    p = Path(root_dir).joinpath("partial-dislocation").joinpath("averaged-roughnesses")
    roughnesses_partial = {
        "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
        "cutoff" : list(), "k":list()
    }
    for seed_folder in [s for s in p.iterdir() if s.is_dir()]:

        # For now only make plots for seed-0 to save time
        if (str(seed_folder) != "results/14-mar-roughness/partial-dislocation/averaged-roughnesses/seed-0"):
            continue

        print(f"Making plots for seed {seed_folder}")

        with mp.Pool(7) as pool:
            results = pool.map(partial(loadRoughnessData_partial, root_dir=root_dir), seed_folder.iterdir())
            tauExt, seed_r, c, zeta, cutoff, k = zip(*results)
            roughnesses_partial["tauExt"] += tauExt
            roughnesses_partial["seed"] += seed_r
            roughnesses_partial["c"] += c
            roughnesses_partial["zeta"] += zeta
            roughnesses_partial["cutoff"] += cutoff
            roughnesses_partial["k"] += k

    roughnesses_np = {
        "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
        "cutoff" : list(), "k":list()
    }
    
    with open(Path(root_dir).joinpath("roughness_partial.json"), "w") as fp:
        json.dump(roughnesses_partial, fp)
    
    # analyzeRoughnessFitParamteters(root_dir)

    p = Path(root_dir).joinpath("single-dislocation").joinpath("averaged-roughnesses")
    for seed_folder in [s for s in p.iterdir() if s.is_dir()]:

        # For now only make plots for seed-0 to save time
        if (str(seed_folder) != "results/14-mar-roughness/single-dislocation/averaged-roughnesses/seed-0"):
            continue

        print(f"Making plots for seed {seed_folder}")
        with mp.Pool(7) as pool:
            results = pool.map(partial(loadRoughnessData_np, root_dir=root_dir), seed_folder.iterdir())
            tauExt, seed_r, c, zeta, cutoff, k = zip(*results)
            roughnesses_np["tauExt"] += tauExt
            roughnesses_np["seed"] += seed_r
            roughnesses_np["c"] += c
            roughnesses_np["zeta"] += zeta
            roughnesses_np["cutoff"] += cutoff
            roughnesses_np["k"] += k
        
    with open(Path(root_dir).joinpath("roughness_np.json"), "w") as fp:
        json.dump(roughnesses_np, fp)
    
    analyzeRoughnessFitParamteters(root_dir)
    
    pass

def analyzeRoughnessFitParamteters(root_dir):
    with open(Path(root_dir).joinpath("roughness_np.json"), "r") as fp:
        roughnesses_np = json.load(fp)
    with open(Path(root_dir).joinpath("roughness_partial.json"), "r") as fp:
        roughnesses_partial = json.load(fp)
    
    plt.clf()
    plt.figure(figsize=(8,8))
    data = np.column_stack([
        roughnesses_np["tauExt"],
        roughnesses_np["zeta"],
        roughnesses_np["seed"]
    ])
    # x = data[data[:,2] == 0][:,0] # Only take seed 0
    # y = data[data[:,2] == 0][:,1]
    x = data[:,0] # Take all fit data
    y = data[:,1]
    plt.scatter(x, y, label="paramteri", marker="x", color="blue")
    plt.title("Roughness fit exponent for seed 0")
    plt.xlabel("$\\tau_{{ext}}$")
    plt.ylabel("$\\zeta$")
    plt.savefig(Path(root_dir).joinpath("tau_ext-zeta-all-perfect.png"), dpi=300)

def rearrangeRoughnessDataByTau(root_dir):
    for dislocation_dir in ["single-dislocation", "partial-dislocation"]: # Do the rearranging for both dirs
        p = Path(root_dir).joinpath(dislocation_dir)
        dest = p.joinpath("avg-rough-arranged")

        if dest.exists(): # Don't rearrange data twice
            print(f"{dislocation_dir} already rearranged.")
            continue

        for seed_folder in [i for i in p.joinpath("averaged-roughnesses").iterdir() if i.is_dir()]:
            for file_path in seed_folder.iterdir():
                fname = file_path.stem
                tauExt = fname.split("-")[2]
                seed = seed_folder.stem.split("-")[1]

                new_dir = dest.joinpath(f"tau-{tauExt}")
                new_dir.mkdir(parents=True, exist_ok=True)

                new_path = new_dir.joinpath(f"roughness-tau-{tauExt}-seed-{seed}.npz")

                shutil.copy(file_path, new_path)
                pass
    pass

def averageRoughnessBySeed(root_dir):
    rearrangeRoughnessDataByTau(parsed.folder)

    dest = Path(root_dir).joinpath("roughness-avg-tau")
    for dislocation_dir in ["single-dislocation", "partial-dislocation"]: # Do the rearranging for both dirs
        p = Path(root_dir).joinpath(dislocation_dir)
        for tauExt in p.joinpath("avg-rough-arranged").iterdir():
            l_range = list()
            roughnesses = list()
            params = list()
            tauExtValue = tauExt.name.split("-")[1]
            for seed_file in tauExt.iterdir():
                loaded = np.load(seed_file)
                if dislocation_dir == "single-dislocation": # This code is here bc before 25-3 there was a typo in the simulation code
                    params = loaded["paramerters"]
                else:
                    params = loaded["parameters"]
                # params = loaded["parameters"]
                params = params
                w = loaded["avg_w"]
                l_range = loaded["l_range"]

                roughnesses.append(w)
                l_range = l_range
                pass

            roughnesses = np.array(roughnesses)
            w_avg = np.mean(roughnesses, axis=0)
            
            dest_plot = None
            
            if dislocation_dir == "single-dislocation":
                dest_plot = dest.joinpath(dislocation_dir).joinpath("plots")
                dest_plot.mkdir(parents=True, exist_ok=True)
                dest_plot = dest_plot.joinpath(f"roughness-tau-{tauExtValue}.png")

                makeRoughnessPlot_np(l_range, w_avg, params, dest_plot)
            elif dislocation_dir == "partial-dislocation":
                dest_plot = dest.joinpath(dislocation_dir).joinpath("plots")
                dest_plot.mkdir(parents=True, exist_ok=True)
                dest_plot = dest_plot.joinpath(f"roughness-tau-{tauExtValue}.png")

                makeRoughnessPlot_partial(l_range, w_avg, params, dest_plot)
    pass

# Depinning, critical force, etc.

def v_fit(tau_ext, tau_crit, beta, a):
    v_res = np.empty(len(tau_ext))
    for n,tau in enumerate(tau_ext):
        if tau > tau_crit:
            v_res[n] = a*(tau - tau_crit)**beta
        else:
            v_res[n] = 0
    return v_res

def normalizedDepinnings(depinning_path : Path, save_folder : Path, optimized=False):
    # Make such plots for a single dislocation first
    # Collect all values of tau_c

    tau_c = dict()

    # Collect all datapoints for binning

    data_perfect = dict()

    for noise_path in depinning_path.iterdir():
        noise = noise_path.name.split("-")[1]

        tau_c[noise] = list()

        data_perfect[noise] = list()

        for fpath in noise_path.iterdir():
            with open(fpath, "r") as fp:
                depinning = json.load(fp)
            
            tauExt = depinning["stresses"]
            vCm = depinning["v_rel"]
            seed = depinning["seed"]

            t_c_arvaus = (max(tauExt) - min(tauExt))/2

            try:
                fit_params, pcov = optimize.curve_fit(v_fit, tauExt, vCm, p0=[t_c_arvaus,   # tau_c
                                                                            0.9,          # beta
                                                                            0.9           # a
                                                                            ], bounds=(0, [ max(tauExt), 2, 2 ]), maxfev=1600)
            except:
                print(f"Could not find fit w/ perfect dislocation noise : {noise} seed : {seed}")
                continue

            tauCrit, beta, a = fit_params

            tau_c[noise].append(tauCrit)

            xnew = np.linspace(min(tauExt), max(tauExt), 100)
            ynew = v_fit(xnew, *fit_params)

            # Scale the original data
            x = (tauExt - tauCrit)/tauCrit
            y = vCm

            # Scale the fit x-axis as well
            xnew = (xnew - tauCrit)/tauCrit

            data_perfect[noise] += zip(x,y)

            plt.clf()
            plt.figure(figsize=(8,8))
            plt.scatter(x,y, marker='x', color="red", label="Depinning")
            plt.plot(xnew, ynew, color="blue", label="fit")
            plt.title(f"Dislocation $\\tau_{{c}} = $ {tauCrit:.3f}, A={a:.3f}, $\\beta$ = {beta:.3f}, seed = {seed}")
            plt.xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{ext}}$")
            plt.ylabel("$v_{{cm}}$")
            plt.legend()

            p = save_folder.joinpath(f"noise-{noise}")
            p.mkdir(exist_ok=True, parents=True)
            plt.savefig(p.joinpath(f"normalized-depinning-{seed}.png"))
            plt.close()
    
    # TODO: save tau_c in a reasonable dir

    with open(save_folder.joinpath("tau_c.json"), "w") as fp:
        json.dump(tau_c, fp)

    return data_perfect

def confidence_interval_lower(l, c_level):
    # Does not handle empy lists at all, assumes normal distribution
    n = len(l)
    m,s = np.mean(l), np.std(l)/np.sqrt(n)
    c = stats.norm.interval(c_level, loc=m, scale=s)
    return c[0]

def confidence_interval_upper(l, c_level):
    # Does not handle empy lists at all, assumes normal distribution
    n = len(l)
    m,s = np.mean(l), np.std(l)/np.sqrt(n)
    c = stats.norm.interval(c_level, loc=m, scale=s)
    return c[1]

def binning(data : dict, res_dir, conf_level, bins=100): # non-partial and partial dislocation global data, respectively
    # TODO: update this function to handle the new file structure
    tau_c_perfect, tau_c_partial = analyze_tau(results_root)
    for perfect_partial in data.keys():
        for noise in data[perfect_partial].keys():
            d = data[perfect_partial]
            x,y = zip(*d[noise])

            bin_means, bin_edges, _ = stats.binned_statistic(x,y,statistic="mean", bins=100)

            lower_confidence, _, _ = stats.binned_statistic(x,y,statistic=partial(confidence_interval_lower, c_level=conf_level), bins=bins)
            upper_confidence, _, _ = stats.binned_statistic(x,y,statistic=partial(confidence_interval_upper, c_level=conf_level), bins=bins)

            bin_counts, _, _ = stats.binned_statistic(x,y,statistic="count", bins=100)

            print(f'Total of {sum(bin_counts)} datapoints. The bins have {" ".join(bin_counts.astype(str))} respectively.')

            plt.clf()
            plt.close('all')
            plt.figure(figsize=(8,8))

            if perfect_partial == "perfect_data":
                plt.title(f"Perfect dislocation binned depinning $ \\langle \\tau_c \\rangle = {tau_c_perfect[noise]:.4f} $ ")
            elif perfect_partial == "partial_data":
                plt.title(f"Perfect dislocation binned depinning $ \\langle \\tau_c \\rangle = {tau_c_partial[noise]:.4f} $ ")
            
            plt.xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{ext}}$")
            plt.ylabel("$v_{{cm}}$")

            bin_width = (bin_edges[1] - bin_edges[0])
            bin_centers = bin_edges[1:] - bin_width/2

            plt.scatter(x,y, marker='x', linewidths=0.2, label="data", color="grey")
            plt.plot(bin_centers, lower_confidence, color="blue", label=f"${conf_level*100} \\%$ confidence")
            plt.plot(bin_centers, upper_confidence, color="blue")

            plt.scatter(bin_centers, bin_means, color="red", marker="x",
                label='Binned depinning data')
            plt.legend()
            
            # Save to a different directory depending on whether its a partial or perfect dislocation
            p = Path(res_dir)
            if perfect_partial == "perfect_data":
                p = p.joinpath("binned-depinnings-perfect")
            elif perfect_partial == "partial_data":
                p = p.joinpath("binned-depinnings-partial")
            else:
                raise Exception("Data is saved wrong.")
            
            p.mkdir(parents=True, exist_ok=True)
            p = p.joinpath(f"binned-depinning-noise-{noise}-conf-{conf_level}.png")
            plt.savefig(p, dpi=600)
    pass

def analyze_tau(dir):
    p_root = Path(dir)
    p = p_root.joinpath("tau_c.json")

    with open(p, "r") as fp:
        data = json.load(fp)
    
    tau_np = data["non-partial dislocation"]["tau_c"]
    tau_p = data["partial dislocation"]["tau_c"]

    perfect_tau_means = dict()
    partial_tau_means = dict()

    for noise in tau_np.keys():
        mean_tau_perfect = sum(tau_np[noise])/len(tau_np[noise])
        mean_tau_partial = sum(tau_p[noise])/len(tau_p[noise])

        perfect_tau_means[noise] = mean_tau_perfect
        partial_tau_means[noise] = mean_tau_partial

        plt.clf()
        plt.figure(figsize=(8,8))
        plt.title(f"Non partial dislocation R={noise} $\\langle \\tau_c \\rangle = $ {mean_tau_perfect:.3f}")
        plt.hist(tau_np[noise], 15, edgecolor="black")
        plt.xlabel("$ \\tau_c $")
        plt.ylabel("Frequency")
        plt.savefig(p_root.joinpath(f"perfect_tau_c_histogram-R-{noise}.png"), dpi=300)

        plt.clf()
        plt.figure(figsize=(8,8))
        plt.title(f"Partial dislocation R={noise} $\\langle \\tau_c \\rangle = $ {mean_tau_partial:.3f}")
        plt.hist(tau_p[noise], 15, edgecolor="black")
        plt.xlabel("$ \\tau_c $")
        plt.ylabel("Frequency")
        plt.savefig(p_root.joinpath(f"partial_tau_c_histogram-R-{noise}.png"), dpi=300)
    
    return perfect_tau_means, partial_tau_means

def getCriticalForces(dir_path):
    # Get all the critical forces from the depinning dir path.
    crit_forces = list()
    for file_path in Path(dir_path).iterdir():
        c = getCriticalForceFromFile(file_path)
        crit_forces.append(c)
    return crit_forces

def globalFit(dir_path):
    global_data = list() # Global data for partial dislocation
    # TODO: Also gather global data for single dislocation
    # TODO: Handle new directtory structure here as well
    for sd, pd in zip(Path(dir_path).joinpath("single-dislocation/depinning-dumps").iterdir(), Path(dir_path).joinpath("partial-dislocation/depinning-dumps").iterdir()):
        # sd # path to single dislocation depinning
        # pd # path to partial dislocation depinning
        with open(pd, 'r') as fp2:
            partial_depinning = json.load(fp2)

            partial_stresses = partial_depinning["stresses"]
            partial_vcm = partial_depinning["v_rel"]

            global_data += zip(partial_stresses, partial_vcm)
    
    x,y = zip(*global_data) # Here x is external stress and y is velocity

    fit_params, pcov = optimize.curve_fit(v_fit, x, y, p0=[2.5, 1.5, 1], maxfev=1600)
    tauC, beta, a = fit_params
    xnew = np.linspace(min(x), max(x), 100)
    ynew = v_fit(xnew, *fit_params)

    plt.clf()
    plt.figure(figsize=(8,8))
    plt.scatter(x,y, marker='x', linewidths=0.1, label="data")
    plt.plot(xnew, ynew, label=f"Fit $\\tau_{{c}} = $ {tauC:.2f}", color="red")
    plt.title("Depinning of a partial dislocation with 100 runs and fot done globally.")
    plt.legend()
    plt.savefig(Path(dir_path).joinpath("global-depinning.png"), dpi=300)
    print(f"Fit done with parameters tau_c = {tauC:.4f} beta = {beta:.4f} and A = {a}")

def makeAveragedDepnningPlots(dir_path, opt=False):
    partial_depinning_path = Path(dir_path).joinpath("partial-dislocation/depinning-dumps")
    perfect_depinning_path = Path(dir_path).joinpath("single-dislocation/depinning-dumps")

    if opt:
        partial_depinning_path = Path(dir_path).joinpath("partial-dislocation/optimal-depinning-dumps")
        perfect_depinning_path = Path(dir_path).joinpath("single-dislocation/optimal-depinning-dumps")

    for noise_dir in partial_depinning_path.iterdir():
        noise = noise_dir.name.split("-")[1]
        velocities = list()
        stresses = None
        seed = None
        for depining_file in noise_dir.iterdir():
            with open(depining_file, "r") as fp:
                loaded = json.load(fp)
                stresses = loaded["stresses"]
                velocities.append(loaded["v_rel"])
            pass

        x = np.array(stresses)
        y = np.average(np.array(velocities), axis=0)

        plt.clf()
        plt.figure(figsize=(8,8))

        plt.scatter(x,y, marker="x")

        plt.title(f"Depinning noise = {noise}")
        plt.xlabel("$\\tau_{ext}$")
        plt.ylabel("$v_{CM}$")

        dest = Path(dir_path).joinpath(f"averaged-depinnings/partial/noise-{noise}")
        dest.mkdir(parents=True, exist_ok=True)
        if opt:
            plt.savefig(dest.joinpath(f"depinning-noise-opt.png"))
        else:
            plt.savefig(dest.joinpath(f"depinning-noise.png"))
        pass

    for noise_dir in perfect_depinning_path.iterdir():
        noise = noise_dir.name.split("-")[1]
        velocities = list()
        stresses = None
        seed = None
        for depining_file in noise_dir.iterdir():
            with open(depining_file, "r") as fp:
                loaded = json.load(fp)
                stresses = loaded["stresses"]
                velocities.append(loaded["v_rel"])
            pass

        x = np.array(stresses)
        y = np.average(np.array(velocities), axis=0)

        plt.clf()
        plt.figure(figsize=(8,8))

        plt.scatter(x,y, marker="x")

        plt.title(f"Depinning noise = {noise}")
        plt.xlabel("$\\tau_{ext}$")
        plt.ylabel("$v_{CM}$")

        dest = Path(dir_path).joinpath(f"averaged-depinnings/perfect/noise-{noise}")
        dest.mkdir(parents=True, exist_ok=True)
        plt.savefig(dest.joinpath(f"depinning.png"))
        pass

def makeNoisePlot(tau_c_path, save_path, title):
    with open(tau_c_path, "r") as fp:
        loaded = json.load(fp)
    
    # partial dislocation
    partial_data = loaded

    noises = list()
    tau_cs = list()

    for noise in partial_data.keys():
        tau_c = partial_data[noise]
        noises.append(float(noise))
        tau_cs.append(tau_c)
    
    plt.figure(figsize=(8,8))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.scatter(noises, tau_cs, marker='x')
    plt.title(title)
    plt.xlabel("R")
    plt.ylabel("$ \\tau_c $")
    plt.savefig(save_path, dpi=300)
    plt.close()
    pass

if __name__ == "__main__":
    parser = ArgumentParser(prog="Dislocation data processing")
    parser.add_argument('-f', '--folder', help='Specify the output folder of the simulation.', required=True)
    parser.add_argument('--all', help='Make all the plots.', action="store_true")
    parser.add_argument('--np', help='Make normalized depinning plots.', action="store_true")
    parser.add_argument('--avg', help='Make an averaged plot from all depinning simulations.', action="store_true")
    parser.add_argument('--roughness', help='Make a log-log rougness plot.', action="store_true")
    parser.add_argument('-ar','--avg-roughness', help='Make a log-log rougness plot that has been averaged by seed for each tau_ext.', action="store_true")
    parser.add_argument('--dislocations', help='Plot dislocations at the end of simulation.', action="store_true")
    parser.add_argument('--binning', help='Make a binned plot. --np must have been called before', action="store_true")
    parser.add_argument('--confidence', help='Confidence level for depinning, must be called with --binning.', type=float, default=0.95)
    parser.add_argument('--noise', help="Analyse critical force as function of noise amplitude", action="store_true")

    parsed = parser.parse_args()

    results_root = Path(parsed.folder)

    if parsed.all or parsed.avg:
        makeAveragedDepnningPlots(parsed.folder)
        makeAveragedDepnningPlots(parsed.folder, opt=True)
        pass
        
    if parsed.all or parsed.roughness:
        print("Making roughness plots. (w/o averaging by default)")
        makeAvgRoughnessPlots(parsed.folder)

    if parsed.all or parsed.avg_roughness:
        averageRoughnessBySeed(parsed.folder)

    # Plots dislocations at the end of simulation
    if parsed.all or parsed.dislocations:
        print("Making plots of some singe dislocations at time t.")
        np_dir = results_root.joinpath("single-dislocation/simulation-dumps")
        if np_dir.exists():
            for seed in np_dir.iterdir():
                with mp.Pool(7) as pool:
                    pool.map(partial(plotDislocation_nonp, save_path=results_root), seed.iterdir())
        else:
            print("Raw dislocation data not saved.")
        
        p_dir = results_root.joinpath("partial-dislocation/simulation-dumps")
        if p_dir.exists():
            for seed in p_dir.iterdir():
                with mp.Pool(7) as pool:
                    pool.map(partial(plotDislocation_partial, save_path=results_root), seed.iterdir())
        else:
            print("Raw dislocation data not saved.")
        
    # Make normalized depinning plots
    if parsed.all or parsed.np:
        print("Making normalized depinning plots.")

        partial_data = normalizedDepinnings(
            results_root.joinpath("partial-dislocation").joinpath("depinning-dumps"),
            save_folder=results_root.joinpath("partial-dislocation/normalized-plots")
        )
        
        non_partial_data = normalizedDepinnings(
            results_root.joinpath("single-dislocation").joinpath("depinning-dumps"),
            save_folder=results_root.joinpath("single-dislocation/normalized-plots")
        )

        with open(Path(results_root).joinpath("global_data_dump.json"), "w") as fp:
            json.dump({"perfect_data":non_partial_data, "partial_data":partial_data}, fp, indent=2)
        
        # TODO: deal with the optimal depinnings
        partial_data_opt = normalizedDepinnings(
            results_root.joinpath("partial-dislocation").joinpath("optimal-depinning-dumps"),
            save_folder=results_root.joinpath("partial-dislocation/normalized-plots-opt")
        )
        
        non_partial_data_opt = normalizedDepinnings(
            results_root.joinpath("single-dislocation").joinpath("optimal-depinning-dumps"),
            save_folder=results_root.joinpath("single-dislocation/normalized-plots-opt")
        )

        with open(Path(results_root).joinpath("global_data_dump_opt.json"), "w") as fp:
            json.dump({"perfect_data":non_partial_data_opt, "partial_data":partial_data_opt}, fp, indent=2)

    if parsed.all or parsed.binning:
        p = Path(results_root).joinpath("global_data_dump.json")

        if not p.exists():
            raise Exception("--np must be called before this one.")
        
        with open(Path(results_root).joinpath("global_data_dump.json"), "r") as fp:
            data = json.load(fp)

        binning(data, results_root, conf_level=parsed.confidence)

    if parsed.all:
        print("Making a global fit with a global plot.")
        globalFit(results_root)

    if parsed.all or parsed.noise:
        Path(parsed.folder).joinpath("noise-plots").mkdir(parents=True, exist_ok=True)
        makeNoisePlot(
            Path(parsed.folder).joinpath("single-dislocation/normalized-plots/tau_c.json"),
            save_path=Path(parsed.folder).joinpath("noise-plots/noise-tau_c-perfect.png"),
            title="Noise magnitude and external force for perfect dislocation"
            )
        makeNoisePlot(
            Path(parsed.folder).joinpath("partial-dislocation/normalized-plots/tau_c.json"),
            save_path=Path(parsed.folder).joinpath("noise-plots/noise-tau_c-partial.png"),
            title="Noise magnitude and external force for partial dislocation"
            )
        makeNoisePlot(
            Path(parsed.folder).joinpath("partial-dislocation/normalized-plots-opt/tau_c.json"),
            save_path=Path(parsed.folder).joinpath("noise-plots/noise-tau_c-partial-opt.png"),
            title="Noise magnitude and external force for partial dislocation from closeup data"
            )
        makeNoisePlot(
            Path(parsed.folder).joinpath("single-dislocation/normalized-plots-opt/tau_c.json"),
            save_path=Path(parsed.folder).joinpath("noise-plots/noise-tau_c-perfect-opt.png"),
            title="Noise magnitude and external force for perfect dislocation from closeup data"
            )