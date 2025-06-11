from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import multiprocessing as mp
from scipy import optimize, stats
from functools import partial
from numba import jit
import math
from sklearn.linear_model import LinearRegression
import csv

linewidth = 5.59164

def makeAvgRoughnessPlots(root_dir):
    # Makes roughness plots that have been averaged only at simulation (that is velocity) level first
    # for partial dislocations

    try:
        makePartialRoughnessPlots(root_dir)
    except FileNotFoundError:
        print("No partial roughness data skipping.")

    try:
        makePerfectRoughnessPlots(root_dir)
    except FileNotFoundError:
        print("No perfect roughness data skipping.")

    try:
        analyzeRoughnessFitParamteters(root_dir)
    except FileNotFoundError:
        print("No roughness fit parameters data skipping.")
    
    pass

def rearrangeRoughnessDataByTau(root_dir):
    for dislocation_dir in ["single-dislocation", "partial-dislocation"]: # Do the rearranging for both dirs
        p = Path(root_dir).joinpath(dislocation_dir)
        dest = p.joinpath("avg-rough-arranged")

        if dest.exists(): # Don't rearrange data twice
            print(f"{dislocation_dir} already rearranged.")
            continue

        for noise_folder in [i for i in p.joinpath("averaged-roughnesses").iterdir() if i.is_dir()]:
            for seed_folder in noise_folder.iterdir():
                for file_path in seed_folder.iterdir():
                    fname = file_path.stem
                    tauExt = fname.split("-")[2]
                    seed = seed_folder.stem.split("-")[1]
                    noise = noise_folder.stem.split("-")[1]

                    new_dir = dest.joinpath(f"noise-{noise}/tau-{tauExt}")
                    new_dir.mkdir(parents=True, exist_ok=True)

                    new_path = new_dir.joinpath(f"roughness-tau-{tauExt}-seed-{seed}.npz")

                    shutil.copy(file_path, new_path)
                pass
    pass

def analyzeRoughnessFitParamteters(root_dir):
    # TODO: also analyze the partial dislocation fit parameters
    with open(Path(root_dir).joinpath("roughness_perfect.json"), "r") as fp:
        roughnesses_np = json.load(fp)
    
    try:
        with open(Path(root_dir).joinpath("roughness_partial.json"), "r") as fp:
            roughnesses_partial = json.load(fp)
    except:
        print(f"No partial data")
    
    for noise in roughnesses_np.keys():
        plt.clf()
        plt.figure(figsize=(8,8))
        data = np.column_stack([
            roughnesses_np[noise]["tauExt"],
            roughnesses_np[noise]["zeta"],
            roughnesses_np[noise]["seed"],
            roughnesses_np[noise]["cutoff"]
        ])
        # x = data[data[:,2] == 0][:,0] # Only take seed 0
        # y = data[data[:,2] == 0][:,1]
        x = data[:,0] # Take all fit data
        y = data[:,1]
        data_count = len(x) # = len(y)
        plt.scatter(x, y, label="$\\zeta$", marker="x", color="blue")
        plt.title(f"Roughness fit exponent for noise R={noise} and N={data_count}")
        plt.xlabel("$\\tau_{{ext}}$")
        plt.ylabel("$\\zeta$")

        save_path = Path(root_dir).joinpath("roughness-fit-exponents-perfect")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path.joinpath(f"roughness-fit-exponent-{noise}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        plt.clf()
        plt.figure(figsize=(8,8))

        x = data[:,0] # Take all fit data
        y = data[:,3]

        plt.scatter(x, y, label="Transition", marker="x", color="blue")
        plt.title(f"Transition from power to constant behavior for seed 0 R={noise}")
        plt.xlabel("$\\tau_{{ext}}$")
        plt.ylabel("$\\k$")

        save_path = Path(root_dir).joinpath("tau-ext-transitions-perfect")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path.joinpath(f"tau_ext-transition-{noise}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

def makePartialRoughnessPlots(root_dir):
    p = Path(root_dir).joinpath("partial-dislocation").joinpath("averaged-roughnesses")

    roughnesses_partial = dict()

    for noise_folder in [s for s in p.iterdir() if s.is_dir()]:

        noise_val = noise_folder.name.split("-")[1]

        roughnesses_partial_noise = {
            "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
            "cutoff" : list(), "k":list()
        }

        for seed_folder in filter(lambda x : int(x.name.split("-")[1]) == 1, noise_folder.iterdir()):
            seed = int(seed_folder.stem.split("-")[1])
            print(f"Making roughness plots for noise {noise_val} seed {seed}")

            with mp.Pool(8) as pool:
                results = pool.map(partial(loadRoughnessData_partial, root_dir=root_dir), seed_folder.iterdir())
                tauExt, seed_r, c, zeta, cutoff, k = zip(*results)
                roughnesses_partial_noise["tauExt"] += tauExt
                roughnesses_partial_noise["seed"] += seed_r
                roughnesses_partial_noise["c"] += c
                roughnesses_partial_noise["zeta"] += zeta
                roughnesses_partial_noise["cutoff"] += cutoff
                roughnesses_partial_noise["k"] += k
        
        roughnesses_partial[noise_val] = roughnesses_partial_noise

    
    with open(Path(root_dir).joinpath("roughness_partial_from_plots.json"), "w") as fp:
        json.dump(roughnesses_partial, fp)

def makeRoughnessPlot_partial(l_range, avg_w12, params, save_path : Path):
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
    plt.scatter(l_range, avg_w12, label="$W_{{12}}$", marker="x")
    # plt.plot(l_range, ynew, label=f"fit, c={c}, $\\zeta = $ {zeta}")

    plt.xscale("log")
    plt.yscale("log")

    plt.title(f"Roughness of a partial dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.xlabel("log(L)")
    plt.ylabel("$\\log(W_{{12}}(L))$")
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()

    return (tauExt, seed, c, zeta, cutoff, k)

def loadRoughnessData_partial(path_to_file, root_dir):
    # Helper function to enable multiprocessing
    loaded = np.load(path_to_file)
    params = loaded["parameters"]
    avg_w12 = loaded["avg_w"]
    l_range = loaded["l_range"]
    bigN, length,   time,   dt, deltaR, bigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    # Loop through every noise level here

    save_path_partial_roughness = Path(root_dir)
    save_path_partial_roughness = save_path_partial_roughness.joinpath("roughness-partial-pdf").joinpath(f"noise-{deltaR:.5f}/seed-{seed}/avg-roughness-tau-{tauExt:.3f}-partial.pdf")
    save_path_partial_roughness.parent.mkdir(parents=True, exist_ok=True)

    tauExt, seed, c, zeta, cutoff, k = makeRoughnessPlotPerfect(l_range, avg_w12, params, save_path_partial_roughness)

    return (tauExt, seed, c, zeta, cutoff, k)


def find_fit_interval(l_range, avg_w, l_0_range, save_path : Path, tauExt):
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

    # Select zeta that is 10% at most smaller than initial zeta
    target_zeta = zetas[0]*(1-0.10)
    n_selected = np.argmax(zetas < target_zeta)

    last_exp = np.argmax(l_range > l_0_range[n_selected]) # Last point where the exponent fit extends

    c, zeta = c_values[n_selected], zetas[n_selected]

    # Make a plot of zeta as function of L_0
    plt.clf()
    plt.figure(figsize=(8,8))

    plt.plot(l_0_range, zetas, label="zeta")
    plt.xscale("log")
    plt.grid(True)
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
    plt.close()

    return n_selected, last_exp, zeta, c

def makeRoughnessPlotPerfect(l_range, avg_w, params, save_path : Path):
    if len(params) == 13:
        bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params
    elif len(params) == 16:
        bigN, length,   time,   dt, deltaR, bigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params
    else:
        print(f"Lenght of params list len(params) = {len(params)}")


    if np.isnan(avg_w).any() or np.isinf(avg_w).any():
        print("avg_w contains NaN or Inf values. Skipping this plot.")
        return (tauExt, seed, np.nan, np.nan, np.nan, np.nan)

    if np.isnan(l_range).any() or np.isinf(l_range).any():
        print("l_range contains NaN or Inf values. Skipping this plot.")
        return (tauExt, seed, np.nan, np.nan, np.nan, np.nan)

    # Make an exponential plot to only the first decade of data.

    plt.clf()
    plt.figure(figsize=(linewidth/2,linewidth/2))

    plt.scatter(l_range, avg_w, marker="o", s=1) # Plot the data as is

    # Make exponential plot to first 10% of data.

    start = 1
    end = 10

    dekadi_l = l_range[start:end]
    dekadi_w = avg_w[start:end]

    x = np.log(dekadi_l)
    y = np.log(dekadi_w)

    # Use scipy.stats.linregress for linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    c = np.exp(intercept)
    zeta = slope

    y_pred = x*slope + intercept

    plt.plot(dekadi_l, np.exp(y_pred), color="green")

    # Now find out the constant behavior from N/4 < L < 3N/4

    start = int(round(len(l_range)/4))
    end = int(round(3*len(l_range)/4 ))

    const_l = l_range[start:end]
    const_w = avg_w[start:end]

    fit_constant_params, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])
    
    new_c = np.ones(len(const_l))*fit_constant_params

    plt.plot(const_l, new_c, color="orange")

    # Next find out the transition point between power law and constant behavior

    fit_constant_params = fit_constant_params[0]    # This is the constant of constant behavior y = c
    zeta = zeta                     # This is the exponent of power law y = c*l^zeta
    c = c                           # This is the factor in power law y=c*l^zeta

    change_p = (fit_constant_params / c)**(1/zeta)  # This is the points l=change_p where is changes from power to constant.

    plt.scatter([change_p], [fit_constant_params], label="$\\xi$")

    # Plot dashed lines to illustrate the intersection
    dashed_x = np.linspace(change_p, max(l_range)/4, 10)
    dahsed_y = np.ones(10)*fit_constant_params
    plt.plot(dashed_x, dahsed_y, linestyle="dashed", color="grey")

    dashed_x = np.linspace(10, change_p, 10)
    dahsed_y = exp_beheavior(dashed_x, c, zeta)
    plt.plot(dashed_x, dahsed_y, linestyle="dashed", color="grey")

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    # plt.title(f"Roughness of a perfect dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.xlabel("L")
    plt.ylabel("$W(L)$")

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

    return (tauExt, seed, c, zeta, change_p, fit_constant_params)

def loadRoughnessDataPerfect(f1, root_dir):
    # Helper function to enable use of multiprocessing when making plots
    loaded = np.load(f1)
    avg_w = loaded["avg_w"]
    l_range = loaded["l_range"]
    params = loaded["parameters"]
    bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    save_path = Path(root_dir)
    save_path = save_path.joinpath("roughness-perfect-pdf").joinpath(f"noise-{deltaR:.5f}/seed-{seed}")
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path.joinpath(f"avg-roughness-tau-{tauExt}-perfect.pdf")

    try:
        tauExt, seed, c, zeta, correlation_len, constant_val = makeRoughnessPlotPerfect(l_range, avg_w, params, save_path)
    except Exception as e:
        print(f1)
        print(e)

    return (tauExt, seed, c, zeta, correlation_len, constant_val)

def makePerfectRoughnessPlots(root_dir, test=False):
    p = Path(root_dir).joinpath("single-dislocation").joinpath("averaged-roughnesses")

    roughnesses_perfect = dict()

    for n,noise_folder in enumerate([s for s in p.iterdir() if s.is_dir()]):
        noise_val = noise_folder.name.split("-")[1]

        roughnesses_np_noise = {
            "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
            "cutoff" : list(), "k":list()
        }

        if not n % 10 == 0 and test:
            continue

        for seed_folder in filter(lambda x : int(x.name.split("-")[1]) == 1, noise_folder.iterdir()):
            seed = int(seed_folder.stem.split("-")[1])
            if seed != 0 and test:
                continue
            print(f"Making roughness plots for noise {noise_val} seed {seed}")

            with mp.Pool(8) as pool:
                results = pool.map(partial(loadRoughnessDataPerfect, root_dir=root_dir), seed_folder.iterdir())

                tauExt, seed_r, c, zeta, cutoff, k = zip(*results)
                roughnesses_np_noise["tauExt"] += tauExt
                roughnesses_np_noise["seed"] += seed_r
                roughnesses_np_noise["c"] += c
                roughnesses_np_noise["zeta"] += zeta
                roughnesses_np_noise["cutoff"] += cutoff
                roughnesses_np_noise["k"] += k
        
        roughnesses_perfect[noise_val] = roughnesses_np_noise
        
    with open(Path(root_dir).joinpath("roughness_perfect_from_plots.json"), "w") as fp:
        json.dump(roughnesses_perfect, fp)

def extractRoughnessExponent(l_range, avg_w, params):
    bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    # Make a piecewise fit on all data
    # Make exponential plot to first 10% of data.

    dekadi_l = l_range[1:10]
    dekadi_w = avg_w[1:10]

    x = np.log(dekadi_l)
    y = np.log(dekadi_w)

    # Use scipy.stats.linregress for linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    c = np.exp(intercept)
    zeta = slope

    # Find out constant behavior and the transition point

    start = int(round(len(l_range)/4))
    end = int(round(3*len(l_range)/4 ))

    const_l = l_range[start:end]
    const_w = avg_w[start:end]

    fit_c_range, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])


    change_p = (fit_c_range / c)**(1/zeta)

    return c, zeta, change_p

def multiprocessing_helper(f1, root_dir):
    loaded = np.load(f1)
    params = loaded["parameters"]
    avg_w = loaded["avg_w"]
    l_range = loaded["l_range"]
    bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    c, zeta, transition = extractRoughnessExponent(l_range, avg_w, params)

    return  (np.float64(deltaR), np.float64(tauExt), np.float64(seed), np.float64(c), np.float64(zeta), np.float64(transition))


def makeRoughnessExponentDataset(root_dir, seq=False):
    # Make a dataset of roughness exponents for all noise levels
    p = Path(root_dir).joinpath("single-dislocation").joinpath("averaged-roughnesses")
    data = list() # List of tuples (noise, tau_ext, seed, c, zeta)

    progess = 0

    for noise_folder in [s for s in p.iterdir() if s.is_dir()]:
        noise = noise_folder.name.split("-")[1]

        for seed_folder in noise_folder.iterdir():
            seed = int(seed_folder.stem.split("-")[1])
            print(f"Extracting params from data with noise {noise} and seed {seed}")

            if seq: # Make each plot sequentially
                for file_path in seed_folder.iterdir():
                    loaded = np.load(file_path)
                    params = loaded["parameters"]
                    avg_w = loaded["avg_w"]
                    l_range = loaded["l_range"]
                    bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params

                    c, zeta = extractRoughnessExponent(l_range, avg_w, params)
                    data.append((noise, tauExt, seed, c, zeta))
                    progess += 1
                    print(f"Progress: {progess}/{1000*100}")
            else:
                with mp.Pool(7) as pool:
                    results = pool.map(partial(multiprocessing_helper, root_dir=root_dir), seed_folder.iterdir())
                    data += results
                    progess += len(results)
                    print(f"Progress: {progess}/{100000} = {progess/1000:.2f}%")
    
    # Save the data to a file
    data = np.array(data)
    np.savez(Path(root_dir).joinpath("roughness_parameters_perfect.npz"), data=data, columns=["noise", "tauExt", "seed", "c", "zeta", "transitions", "correlation"])

    with open(Path(root_dir).joinpath("roughness_exponents.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["noise", "tauExt", "seed", "c", "zeta", "transitions"])
        writer.writerows(data)
    np.savetxt(Path(root_dir).joinpath("roughness_parameters_perfect.csv"), data, delimiter=",", header="noise,tauExt,seed,c,zeta,correlation")

def makeZetaPlot(data, chosen_noise, root_dir):
    # Make a plot of the roughness exponent zeta as function of tauExt
    chosen_noise = np.round(chosen_noise, 4)
    noise = data[:,0]
    tauExt = data[:,1]
    seed = data[:,2]
    c = data[:,3]
    zeta = data[:,4]

    # Get the data for the chosen noise and tauExt
    noise_rounded = np.round(noise, 4)
    chosen_data = data[ noise_rounded == chosen_noise]
    print(f"Chosen data: {chosen_data.shape}")

    taus = chosen_data[:,1]
    zetas = chosen_data[:,4]

    # Create a binned plot of the data

    bin_means, bin_edges, bin_counts = stats.binned_statistic(taus, zetas, statistic="mean", bins=100)
    stds, bin_edges, bin_counts = stats.binned_statistic(taus, zetas, statistic="std", bins=100)

    plt.figure(figsize=(linewidth/2,linewidth/2))

    plt.scatter(taus, zetas, label="zeta", marker="x", color="lightgrey", alpha=0.5)

    plt.scatter(bin_edges[:-1], bin_means, label="binned zeta", marker="o", color="blue")

    plt.scatter(bin_edges[:-1], bin_means+stds, label="binned zeta", marker="_", color="blue")
    plt.scatter(bin_edges[:-1], bin_means-stds, label="binned zeta", marker="_", color="blue")

    plt.xlabel("$\\tau_{{ext}}$")
    plt.ylabel("$\\zeta$")
    plt.title(f"Roughness exponent for noise {chosen_noise}, N={len(taus)}")
    plt.legend()
    plt.grid(True)

    save_path = Path(root_dir)
    save_path = save_path.joinpath(f"roughness-hurst-exponent-plots/roughness-hurst-exponent-R-{chosen_noise}.eps")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def makeCorrelationPlot(data, chosen_noise, root_dir):
    noise = data[:,0]
    tau_ext = data[:,1]
    correlation = data[:,5]
    seed = data[:,2]

    mask = np.round(noise, 5) == chosen_noise

    noise = noise[mask]
    cor = correlation[mask]

    # print(noise, cor)
    # print(noise.shape, cor.shape)

    bin_means, bin_edges, bin_counts = stats.binned_statistic(tau_ext[mask], cor, statistic="mean", bins=100)
    stds, bin_edges, bin_counts = stats.binned_statistic(tau_ext[mask], cor, statistic="std", bins=100)

    plt.figure(figsize=(linewidth/2,linewidth/2))

    plt.scatter(tau_ext[mask], cor, label="correlation", marker="x", color="lightgrey", alpha=0.5)

    plt.scatter(bin_edges[:-1], bin_means, label="mean", marker="o", color="blue")

    plt.scatter(bin_edges[:-1], bin_means+stds, label="std", marker="_", color="blue")
    plt.scatter(bin_edges[:-1], bin_means-stds, marker="_", color="blue")

    plt.xlabel("$\\tau_{{ext}}$")
    plt.ylabel("Correlation")
    plt.title(f"Correlation for noise {chosen_noise}, N={len(noise)}")
    plt.legend()
    plt.grid(True)

    save_path = Path(root_dir)
    save_path = save_path.joinpath("correlation-plots")
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path.joinpath(f"correlation-{chosen_noise}.eps")
    plt.savefig(save_path)
    plt.close()
    pass

def processExponentData(root_dir):
    path = Path(root_dir).joinpath("roughness_exponents.npz")
    loaded = np.load(path)
    data = loaded["data"]
    columns = loaded["columns"]

    noise = data[:,0]
    tauExt = data[:,1]
    seed = data[:,2]
    c = data[:,3]
    zeta = data[:,4]
    correlation = data[:,5]

    print(f"Noise: {min(noise)} - {max(noise)} count {len(set(noise))}")
    print(f"tauExt: {min(tauExt)} - {max(tauExt)}")
    print(f"Seed: {min(seed)} - {max(seed)}")
    print(f"c: {min(c)} - {max(c)}")
    print(f"zeta: {min(zeta)} - {max(zeta)}")

    unique_noises = set(noise)

    for unique_noise in unique_noises:
        makeZetaPlot(data, np.round(unique_noise, 4), root_dir)
        makeCorrelationPlot(data, np.round(unique_noise, 5), root_dir)

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

def averageRoughnessBySeed(root_dir):
    rearrangeRoughnessDataByTau(root_dir)

    dest = Path(root_dir).joinpath("roughness-avg-tau")
    for dislocation_dir in ["single-dislocation", "partial-dislocation"]: # Do the rearranging for both dirs
        p = Path(root_dir).joinpath(dislocation_dir)
        for noise_folder in p.joinpath("avg-rough-arranged").iterdir():

            noise_val = noise_folder.name.split("-")[1]

            for tauExt in noise_folder.iterdir():
                l_range = list()
                roughnesses = list()
                params = list()
                tauExtValue = tauExt.name.split("-")[1]
                for seed_file in tauExt.iterdir():
                    loaded = np.load(seed_file)
                    params = loaded["parameters"]

                    w = loaded["avg_w"]
                    l_range = loaded["l_range"]

                    roughnesses.append(w)
                    l_range = l_range
                    pass

                roughnesses = np.array(roughnesses)
                w_avg = np.mean(roughnesses, axis=0)
                
                dest_plot = None
                
                if dislocation_dir == "single-dislocation":
                    dest_plot = dest.joinpath(dislocation_dir).joinpath(f"plots/noise-{noise_val}")
                    dest_plot.mkdir(parents=True, exist_ok=True)
                    dest_plot = dest_plot.joinpath(f"roughness-tau-{tauExtValue}.png")

                    makeRoughnessPlotPerfect(l_range, w_avg, params, dest_plot)
                elif dislocation_dir == "partial-dislocation":
                    dest_plot = dest.joinpath(dislocation_dir).joinpath(f"plots/noise-{noise_val}")
                    dest_plot.mkdir(parents=True, exist_ok=True)
                    dest_plot = dest_plot.joinpath(f"roughness-tau-{tauExtValue}.png")

                    makeRoughnessPlot_partial(l_range, w_avg, params, dest_plot)
    pass

@jit(nopython=True)
def roughnessW(y, bigN): # Calculates the cross correlation W(L) of a single dislocation
    l_range = np.arange(0,int(bigN))    # TODO: Use range parameter instead
    roughness = np.empty(int(bigN))

    y_size = int(bigN) # TODO: check if len(y) = bigN ?
    
    for l in l_range:
        res = 0
        for i in range(0,1024):
            res = res + ( y[i] - y[ (i+l) % y_size ] )**2
        
        res = res/y_size
        c = np.sqrt(res)
        roughness[l] = c

    return l_range, roughness

def extractRoughnessFromLast(root_dir):
    path = Path(root_dir).joinpath("partial-dislocation").joinpath("dislocations-last")
    for noise_folder in [s for s in path.iterdir() if s.is_dir()]:
        noise_val = noise_folder.name.split("-")[1]

        for seed_folder in noise_folder.iterdir():
            seed = int(seed_folder.stem.split("-")[1])
            print(f"Extracting params from data with noise {noise_val} and seed {seed}")

            for file_path in seed_folder.iterdir():
                loaded = np.load(file_path)
                parameters = loaded["parameters"]
                bigN, length,   time,   dt, deltaR, bigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = parameters
                new_params = np.array([bigN, length, time, dt, deltaR, bigB, smallB, cLT1, mu, tauExt, d0, seed, tau_cutoff])
                y1 = loaded["y1"]
                y2 = loaded["y2"]

                save_path = Path(root_dir).joinpath("partial-dislocation").joinpath("roughness-from-last").joinpath(f"noise-{noise_val}/seed-{seed}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path.joinpath(f"roughness-tau-{tauExt}.png")

                if save_path.exists():
                    print("skip")
                    continue


                y_avg = (y1 + y2)/2

                l_range, roughness = roughnessW(y_avg.flatten(), 1024)

                makeRoughnessPlotPerfect(l_range, roughness, new_params, save_path)
    pass

def makeTranitionPointPlots():
    pass

if __name__ == "__main__":
    pass