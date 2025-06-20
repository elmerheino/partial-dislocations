from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import multiprocessing as mp
from scipy import optimize, stats
from functools import partial
from numba import jit
from sklearn.linear_model import LinearRegression
import csv
import matplotlib as mpl

linewidth = 5.59164

mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern Roman",  # Or 'sans-serif', 'Computer Modern Roman', etc.
    "font.size": 12,         # Match LaTeX document font size
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 12
})

def rearrangeRoughnessDataByTau(root_dir):
    for dislocation_dir in ["single-dislocation", "partial-dislocation"]: # Do the rearranging for both dirs
        p = Path(root_dir).joinpath(dislocation_dir)
        dest = p.joinpath("avg-rough-arranged")

        if dest.exists(): # Don't rearrange data twice
            print(f"{dislocation_dir} already rearranged.")
            continue

        averaged_roughnesses_path = p.joinpath("averaged-roughnesses")
        if not averaged_roughnesses_path.exists():
            print(f"Directory not found, skipping: {averaged_roughnesses_path}")
            continue

        for noise_folder in [i for i in averaged_roughnesses_path.iterdir() if i.is_dir()]:
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
    perfect_path = Path(root_dir).joinpath("roughness_perfect.json")
    if perfect_path.exists():
        with open(perfect_path, "r") as fp:
            roughnesses_np = json.load(fp)
    else:
        print(f"File not found: {perfect_path}")
        return

    partial_path = Path(root_dir).joinpath("roughness_partial.json")
    if partial_path.exists():
        try:
            with open(partial_path, "r") as fp:
                roughnesses_partial = json.load(fp)
        except Exception as e:
            print(f"Could not load partial data from {partial_path}: {e}")
    else:
        print(f"No partial data file found: {partial_path}")

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
    if not p.exists():
        print(f"Directory not found, skipping: {p}")
        return

    roughnesses_partial = dict()

    for noise_folder in [s for s in p.iterdir() if s.is_dir()]:

        noise_val = noise_folder.name.split("-")[1]

        roughnesses_partial_noise = {
            "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
            "cutoff" : list(), "k":list()
        }

        for seed_folder in filter(lambda x : int(x.name.split("-")[1]) == 1, noise_folder.iterdir()):
            seed_in_name = int(seed_folder.stem.split("-")[1])
            print(f"Making roughness plots for noise {noise_val} seed {seed_in_name}")

            with mp.Pool(8) as pool:
                results = pool.map(partial(loadRoughnessData_partial, root_dir=root_dir), seed_folder.iterdir())
                tauExt, seed, c, zeta, correlation, fit_constant_params = zip(*results)
                roughnesses_partial_noise["tauExt"] += tauExt
                roughnesses_partial_noise["seed"] += seed
                roughnesses_partial_noise["c"] += c
                roughnesses_partial_noise["zeta"] += zeta
                roughnesses_partial_noise["cutoff"] += correlation
                roughnesses_partial_noise["k"] += fit_constant_params
        
        roughnesses_partial[noise_val] = roughnesses_partial_noise

    
    with open(Path(root_dir).joinpath("roughness_partial_from_plots.json"), "w") as fp:
        json.dump(roughnesses_partial, fp)

def loadRoughnessData_partial(path_to_file, root_dir):
    # Helper function to enable multiprocessing
    if not Path(path_to_file).exists():
        print(f"File not found: {path_to_file}")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    loaded = np.load(path_to_file)
    params = loaded["parameters"]
    avg_w12 = loaded["avg_w"]
    l_range = loaded["l_range"]
    bigN, length,   time,   dt, deltaR, bigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    # Loop through every noise level here

    save_path_partial_roughness = Path(root_dir)
    save_path_partial_roughness = save_path_partial_roughness.joinpath("roughness-partial").joinpath(f"noise-{deltaR:.5f}/seed-{seed}/avg-roughness-tau-{tauExt:.3f}-partial.pdf")
    save_path_partial_roughness.parent.mkdir(parents=True, exist_ok=True)

    fig, ax, fit_params = makeRoughnessPlotPerfect(l_range, avg_w12, params, save_path_partial_roughness)
    tauExt, seed, c, zeta, correlation, fit_constant_params = fit_params
    fig.savefig(save_path_partial_roughness)

    return (tauExt, seed, c, zeta, correlation, fit_constant_params)


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
    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))

    ax.scatter(l_range, avg_w, marker="o", s=1) # Plot the data as is

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

    ax.plot(dekadi_l, np.exp(y_pred), color="green")

    # Now find out the constant behavior from N/4 < L < 3N/4

    start = int(round(len(l_range)/4))
    end = int(round(3*len(l_range)/4 ))

    const_l = l_range[start:end]
    const_w = avg_w[start:end]

    fit_constant_params, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])
    
    new_c = np.ones(len(const_l))*fit_constant_params

    ax.plot(const_l, new_c, color="orange")

    # Next find out the transition point between power law and constant behavior

    fit_constant_params = fit_constant_params[0]    # This is the constant of constant behavior y = c
    zeta = zeta                     # This is the exponent of power law y = c*l^zeta
    c = c                           # This is the factor in power law y=c*l^zeta

    change_p = (fit_constant_params / c)**(1/zeta)  # This is the points l=change_p where is changes from power to constant.

    fit_parameters = (tauExt, seed, c, zeta, change_p, fit_constant_params)

    ax.scatter([change_p], [fit_constant_params], label="$\\xi$")

    # Plot dashed lines to illustrate the intersection
    dashed_x = np.linspace(change_p, max(l_range)/4, 10)
    dahsed_y = np.ones(10)*fit_constant_params
    ax.plot(dashed_x, dahsed_y, linestyle="dashed", color="grey")

    dashed_x = np.linspace(10, change_p, 10)
    dahsed_y = exp_beheavior(dashed_x, c, zeta)
    ax.plot(dashed_x, dahsed_y, linestyle="dashed", color="grey")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    # plt.title(f"Roughness of a perfect dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt:.3f}")
    ax.set_xlabel("L")
    ax.set_ylabel("$W(L)$")

    fig.tight_layout()

    plt.close()

    return fig, ax, fit_parameters

def loadRoughnessDataPerfect(f1, root_dir):
    # Helper function to enable use of multiprocessing when making plots
    if not Path(f1).exists():
        print(f"File not found: {f1}")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
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
        fig, ax, fit_parameters = makeRoughnessPlotPerfect(l_range, avg_w, params, save_path)
        tauExt, seed, c, zeta, correlation_len, constant_val = fit_parameters
        fig.savefig(save_path)
    except Exception as e:
        print(f1)
        print(e)

    return (tauExt, seed, c, zeta, correlation_len, constant_val)

def makePerfectRoughnessPlots(root_dir):
    p = Path(root_dir).joinpath("single-dislocation").joinpath("averaged-roughnesses")
    if not p.exists():
        print(f"Directory not found, skipping: {p}")
        return

    roughnesses_perfect = dict()

    for n,noise_folder in enumerate([s for s in p.iterdir() if s.is_dir()]):
        noise_val = noise_folder.name.split("-")[1]

        roughnesses_np_noise = {
            "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
            "cutoff" : list(), "k":list()
        }

        for seed_folder in filter(lambda x : int(x.name.split("-")[1]) == 1, noise_folder.iterdir()): # Only make roughness plots for seed = 1
            seed = int(seed_folder.stem.split("-")[1])

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
    if len(params) == 13:
        bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params
    elif len(params) == 16:
        bigN, length,   time,   dt, deltaR, bigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    # Make a piecewise fit on all data
    # Make exponential plot to first 10% of data.

    dekadi_l = l_range[1:10]
    dekadi_w = avg_w[1:10]

    x = np.log(dekadi_l)
    y = np.log(dekadi_w)

    # Use scipy.stats.linregress for linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    prefactor = np.exp(intercept)
    zeta = slope

    # Find out constant behavior and the transition point

    start = int(round(len(l_range)/4))
    end = int(round(3*len(l_range)/4 ))

    const_l = l_range[start:end]
    const_w = avg_w[start:end]

    fit_c_range, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])


    change_p = (fit_c_range / prefactor)**(1/zeta)

    return prefactor, zeta, change_p, fit_c_range

def find_tau_c(noise, root_dir):

    noise = np.round(noise, 6)

    noise_partial_path = Path(root_dir).joinpath(f"noise-data/partial-noises.csv")
    noise_perfect_path = Path(root_dir).joinpath(f"noise-data/perfect-noises.csv")

    tau_c_partial = np.empty((0, 11))
    if noise_partial_path.exists():
        tau_c_partial = np.genfromtxt(noise_partial_path, delimiter=",", skip_header=1)
    else:
        print(f"No partial noise data file found: {noise_partial_path}")

    tau_c_perfect = np.empty((0, 11))
    if noise_perfect_path.exists():
        tau_c_perfect = np.genfromtxt(noise_perfect_path, delimiter=",", skip_header=1)
    else:
        print(f"No perfect noise data file found: {noise_perfect_path}")

    column1_perfect = np.round(tau_c_perfect[:, 0], 6)
    mask_perfect = np.abs(column1_perfect - noise) < 1e-5

    column1_partial = np.round(tau_c_partial[:, 0], 6)
    mask_partial = np.abs(column1_partial - noise) < 1e-5

    row_perfect = tau_c_perfect[mask_perfect]
    row_partial = tau_c_partial[mask_partial]

    if row_perfect.shape[0] == 0 or row_perfect.shape[1] < 11:
        print(f"No critical force for perfect dislocation with noise={noise}")
        tau_c_mean_perfect = np.nan
    else:
        tau_c_mean_perfect = np.nanmean(row_perfect[:, 1:11])

    if row_partial.shape[0] == 0 or row_partial.shape[1] < 11:
        print(f"No critical force for partial dislocation with noise={noise}")
        tau_c_mean_partial = np.nan
    else:
        tau_c_mean_partial = np.nanmean(row_partial[:, 1:11])

    return tau_c_mean_perfect, tau_c_mean_partial

def multiprocessing_helper(f1, root_dir):
    f1_path = Path(f1)
    if not f1_path.exists():
        print(f"File with path {f1} does not exist")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    try:
        loaded = np.load(f1)
    except Exception as e:
        print(f"File with path {f1} is corrupted: {e}")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    
    try:
        params = loaded["parameters"]
        avg_w = loaded["avg_w"]
        l_range = loaded["l_range"]
    except Exception as e:
        print(f"Exception {e} and loadled.files : {loaded.files}")
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        

    if len(params) == 13:
        bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params
    elif len(params) == 16:
        bigN, length,   time,   dt, deltaR, bigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    prefactor, zeta, transition, constant = extractRoughnessExponent(l_range, avg_w, params)

    return  (np.float64(deltaR), np.float64(tauExt), np.float64(seed), np.float64(prefactor), np.float64(zeta), np.float64(transition), np.float64(constant))

def extractAllFitParams(path, root_dir, seq=False):
    if not path.exists():
        print(f"Directory not found, cannot extract fit parameters: {path}")
        return np.array([])
    progess = 0
    data = list()
    for noise_folder in [s for s in path.iterdir() if s.is_dir()]:
        noise = noise_folder.name.split("-")[1]

        for seed_folder in noise_folder.iterdir():
            seed = int(seed_folder.stem.split("-")[1])
            print(f"Extracting params from data with noise {noise} and seed {seed}")

            with mp.Pool(10) as pool:
                results = pool.map(partial(multiprocessing_helper, root_dir=root_dir), seed_folder.iterdir())
                data += results
                progess += len(results)
                print(f"Progress: {progess}/{100000} = {progess/1000:.2f}%")
    return np.array(data)

def makeRoughnessExponentDataset_perfect(root_dir, seq=False):
    # Make a dataset of roughness exponents for all noise levels
    p = Path(root_dir).joinpath("single-dislocation").joinpath("averaged-roughnesses")
    data_perfect = extractAllFitParams(p, root_dir)
    
    # Save the data to a file
    np.savez(Path(root_dir).joinpath("roughness_parameters_perfect.npz"), data=data_perfect, columns=["noise", "tauExt", "seed", "c", "zeta", "transitions", "correlation", "constant"])

    with open(Path(root_dir).joinpath("roughness_parameters_perfect.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["noise", "tauExt", "seed", "prefactor", "zeta", "transition", "constant"])
        writer.writerows(data_perfect)
    np.savetxt(Path(root_dir).joinpath("roughness_parameters_perfect.csv"), data_perfect, delimiter=",", header="noise,tauExt,seed,c,zeta,correlation,constant")

def makeRoughnessExponentDataset_partial(root_dir):
    p = Path(root_dir).joinpath("partial-dislocation").joinpath("averaged-roughnesses")
    data_partial = extractAllFitParams(p, root_dir)
    
    # Save the data to a file
    np.savez(Path(root_dir).joinpath("roughness_parameters_partial.npz"), data=data_partial, columns=["noise", "tauExt", "seed", "c", "zeta", "transitions", "correlation", "constant"])

    with open(Path(root_dir).joinpath("roughness_parameters_partial.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(["noise", "tauExt", "seed", "prefactor", "zeta", "transition", "constant"])
        writer.writerows(data_partial)
    np.savetxt(Path(root_dir).joinpath("roughness_parameters_partial.csv"), data_partial, delimiter=",", header="noise,tauExt,seed,c,zeta,correlation,constant")


def makeZetaPlot(taus, zetas):
    # Make a plot of the roughness exponent zeta as function of tauExt
    # Create a binned plot of the data

    fig,ax = plt.subplots()
    if len(taus) == 0:
        print("len(taus) = 0, so there is no data for this noise")
        return None, None

    bin_means, bin_edges, bin_counts = stats.binned_statistic(taus, zetas, statistic="mean", bins=100)
    stds, bin_edges, bin_counts = stats.binned_statistic(taus, zetas, statistic="std", bins=100)

    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))

    ax.scatter(taus, zetas, label="zeta", marker="x", color="lightgrey", alpha=0.5)

    ax.scatter(bin_edges[:-1], bin_means, label="binned zeta", marker="o", color="blue")

    ax.scatter(bin_edges[:-1], bin_means+stds, label="$\\sigma$", marker="_", color="blue")
    ax.scatter(bin_edges[:-1], bin_means-stds, marker="_", color="blue")

    ax.set_xlabel("$\\tau_{{ext}}$")
    ax.set_ylabel("$\\zeta$")
    ax.legend()
    ax.grid(True)

    return fig,ax    

def makeCorrelationPlot(tau_ext, cor, tau_c, noise, color):

    bin_means, bin_edges, bin_counts = stats.binned_statistic(tau_ext, cor, statistic="mean", bins=100)
    stds, bin_edges, bin_counts = stats.binned_statistic(tau_ext, cor, statistic="std", bins=100)

    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))


    ax.set_ylim(0, 300)

    x = tau_ext
    y = cor

    if not np.isnan(tau_c): # Normalize data
        x = (x - tau_c)/tau_c

    ax.scatter(x, y, marker="x", color="lightgrey", alpha=0.5, linewidth=0.2)

    x = bin_edges[:-1]
    y = bin_means

    if not np.isnan(tau_c): # Normalize data
        x = (x - tau_c)/tau_c
        ax.set_xlabel("$(\\tau_{{ext}} - \\tau_c)/\\tau_c$")
        normalized_binned_data = { "x":x, "y":y, "noise": noise}
    else:
        ax.set_xlabel("$\\tau_{{ext}}$")
        normalized_binned_data = None

    ax.scatter(x, y, label="$\\vec{\\xi}$", linewidth=0.2, color=color, marker='o', s=2)

    ax.scatter(x, y+stds, label="$\\sigma$", marker="_",linewidth=0.5, color=color)
    ax.scatter(x, y-stds, marker="_", linewidth=0.5, color=color)

    ax.set_ylabel("$\\xi$")
    # ax.set_title(f"Correlation for noise {chosen_noise}, N={len(noise)}")
    fig.tight_layout()
    ax.legend()
    ax.grid(True)

    return fig, ax, normalized_binned_data

def makeConstPlot(tau_ext, const, tau_c, color):
    fig, ax = plt.subplots()
    if tau_c != np.nan:
        tau_ext = (tau_ext - tau_c)/tau_c

    bin_means, bin_edges, bin_counts = stats.binned_statistic(tau_ext, const, statistic="mean", bins=100)
    stds, bin_edges, bin_counts = stats.binned_statistic(tau_ext, const, statistic="std", bins=100)

    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))

    ax.scatter(tau_ext, const, label="constant", marker="x", color="lightgrey", alpha=0.5)

    ax.scatter(bin_edges[:-1], bin_means, label="binned constant", marker="o", color=color)

    ax.scatter(bin_edges[:-1], bin_means+stds, label="$\\sigma$", marker="_", color=color)
    ax.scatter(bin_edges[:-1], bin_means-stds, marker="_", color="blue")

    if tau_c != np.nan:
        ax.set_xlabel("$( \\tau_{{ext}} - \\tau_c ) / \\tau_c$")
    else:
        ax.set_xlabel("$ \\tau_{{ext}} $")

    ax.set_ylabel("Constant")

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    return fig,ax

def processRoughnessFitParamData(path, root_dir, save_folder, perfect):
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"File not found, cannot process data: {path_obj}")
        return
    loaded = np.load(path)
    data = loaded["data"]
    columns = loaded["columns"]

    noise = data[:,0] 
    tauExt = data[:,1]
    seed = data[:,2]
    c = data[:,3]
    zeta = data[:,4]
    correlation = data[:,5]
    const = data[:,6]

    print(f"Noise: {min(noise)} - {max(noise)} count {len(set(noise))}")
    print(f"tauExt: {min(tauExt)} - {max(tauExt)}")
    print(f"Seed: {min(seed)} - {max(seed)}")
    print(f"c: {min(c)} - {max(c)}")
    print(f"zeta: {min(zeta)} - {max(zeta)}")

    unique_noises = set(noise)

    slices_for_3d_plot = []

    for unique_noise in [i for i in unique_noises if not np.isnan(i)]:
        truncated_noise = np.round(unique_noise, 5)
        noise = data[:,0]
        tau_ext = data[:,1]
        seed = data[:,2]
        zeta = data[:,4]
        correlation = data[:,5]

        # Get the data for the chosen noise and tauExt
        mask = np.round(noise, 5) == truncated_noise
        chosen_data = data[ mask]

        taus = chosen_data[:,1]
        zetas = chosen_data[:,4]

        tau_c_perfect, tau_c_partial = find_tau_c(truncated_noise, root_dir)
        
        if not len(taus) == 0:
            fig, ax = makeZetaPlot(taus, zetas)

            save_path = Path(save_folder)
            save_path = save_path.joinpath(f"roughness-hurst-exponent-plots/roughness-hurst-exponent-R-{truncated_noise}.pdf")
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(save_path)
            plt.close()
        else:
            print(f"No data for noise = {truncated_noise}")

        tau_ext_at_noise = tau_ext[mask]
        noise = noise[mask]
        cor = correlation[mask]
        const_current = const[mask]

        if len(tau_ext_at_noise) == 0:
            print(f"No data for noise = {truncated_noise}")
            continue
        
        if perfect:
            fig, ax, normalized_data = makeCorrelationPlot(tau_ext_at_noise, cor, tau_c_perfect, truncated_noise, color="blue")
        else:
            fig, ax, normalized_data = makeCorrelationPlot(tau_ext_at_noise, cor, tau_c_partial, truncated_noise, color="red")

        save_path = Path(save_folder)
        save_path = save_path.joinpath("correlation-plots")
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path.joinpath(f"correlation-{truncated_noise}.pdf")
        fig.savefig(save_path)
        plt.close()

        if perfect:
            fig_const, ax = makeConstPlot(tau_ext_at_noise, const_current, tau_c_perfect, color="blue")
        else:
            fig_const, ax = makeConstPlot(tau_ext_at_noise, const_current, tau_c_partial, color="red")
        
        save_path = Path(save_folder)
        save_path = save_path.joinpath(f"const-plots/constant-{truncated_noise}.pdf")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig_const.savefig(save_path)

        slices_for_3d_plot.append(normalized_data)
        
    np.savez(Path(save_folder).joinpath("correaltion-3dplot-data.npz"), slices_for_3d_plot)

def selectedCorrelationPlots(path, root_dir, save_path, perfect, unique_noises):
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"File not found, cannot create plots: {path_obj}")
        return
    loaded = np.load(path)
    data = loaded["data"]
    columns = loaded["columns"]

    noise = data[:,0] 
    tauExt = data[:,1]
    seed = data[:,2]
    c = data[:,3]
    zeta = data[:,4]
    correlation = data[:,5]
    const = data[:,6]

    colors = ["red", "green", "blue", "orange", "purple"]

    fig, ax, normalized_binned_data = None, None, None

    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))
    ax.set_ylim(0, 300)

    fig_std, ax_std = plt.subplots(figsize=(linewidth/2,linewidth/2))
    ax_std.set_ylim(0, 300)

    scatter_objs = list()

    for i, unique_noise in enumerate(unique_noises):
        truncated_noise = np.round(unique_noise, 5)
        noise = data[:,0]
        tau_ext = data[:,1]
        seed = data[:,2]
        zeta = data[:,4]
        correlation = data[:,5]
        const = data[:,6]

        # Get the data for the chosen noise and tauExt
        mask = np.round(noise, 5) == truncated_noise
        chosen_data = data[ mask]

        taus = chosen_data[:,1]
        zetas = chosen_data[:,4]
        correlations = chosen_data[:,5]
        tau_exts = chosen_data[:,1]

        tau_c_perfect, tau_c_partial = find_tau_c(truncated_noise, root_dir)

        tau_ext_at_noise = tau_ext[mask]
        noise = noise[mask]
        cor = correlation[mask]
        const_current = const[mask]

        if len(tau_ext_at_noise) == 0:
            print(f"No data for noise = {truncated_noise}")
            continue

        # Normalize data
        x = tau_exts
        y = correlations

        tau_c = tau_c_perfect if perfect else tau_c_partial

        if not np.isnan(tau_c):  # Normalize data
            x = (x - tau_c) / tau_c

        bin_means, bin_edges, bin_counts = stats.binned_statistic(x, y, statistic="mean", bins=100)
        stds, bin_edges, bin_counts = stats.binned_statistic(x, y, statistic="std", bins=100)

        x = bin_edges[:-1]
        y = bin_means

        sc = ax.scatter(x, y, label=f"R={truncated_noise:.2f}", linewidth=0.5, color=colors[i], marker='o', s=2)
        scatter_objs.append(sc)
        # ax.scatter(x, y + stds, marker="_", linewidth=0.5, color=colors[i])
        # ax.scatter(x, y - stds, marker="_", linewidth=0.5, color=colors[i])

        ax_std.scatter(x, y, label=f"R={truncated_noise:.2f}", linewidth=0.5, color=colors[i], marker='o', s=2)
        ax_std.scatter(x, y + stds, marker="_", linewidth=0.5, color=colors[i])
        ax_std.scatter(x, y - stds, marker="_", linewidth=0.5, color=colors[i])


    ax.set_xlabel("$(\\tau_{{ext}} - \\tau_c)/\\tau_c$")
    ax.set_ylabel("$\\xi$")
    if perfect:
        legend1 = ax.legend(handles=scatter_objs[0:3], loc='upper right', fontsize='small', handletextpad=0.1, borderpad=0.1 )
        legend2 = ax.legend(handles=scatter_objs[3:], loc='lower right', fontsize='small', handletextpad=0.1, borderpad=0.1 )
        ax.add_artist(legend1)
        ax.add_artist(legend2)
    else:
        ax.legend(fontsize='small')

    ax.grid(True)
    fig.tight_layout()

    ax_std.set_xlabel("$(\\tau_{{ext}} - \\tau_c)/\\tau_c$")
    ax_std.set_ylabel("$\\xi$")
    ax_std.legend(fontsize='small')
    ax_std.grid(True)
    fig_std.tight_layout()

    return fig, ax, fig_std, ax_std


def selectedZetaPlots(path, root_dir, save_path, perfect, unique_noises):
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"File not found, cannot create plots: {path_obj}")
        return None, None, None, None
    loaded = np.load(path)
    data = loaded["data"]

    colors = ["red", "green", "blue", "orange", "purple"]

    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))
    fig_std, ax_std = plt.subplots(figsize=(linewidth/2,linewidth/2))

    for i, unique_noise in enumerate(unique_noises):
        truncated_noise = np.round(unique_noise, 5)
        
        # Get the data for the chosen noise
        mask = np.round(data[:, 0], 5) == truncated_noise
        chosen_data = data[mask]

        if chosen_data.shape[0] == 0:
            print(f"No data for noise = {truncated_noise}")
            continue

        tau_exts = chosen_data[:, 1]
        zetas = chosen_data[:, 4]

        tau_c_perfect, tau_c_partial = find_tau_c(truncated_noise, root_dir)
        tau_c = tau_c_perfect if perfect else tau_c_partial

        # Normalize data
        x = tau_exts
        y = zetas

        if not np.isnan(tau_c):  # Normalize data
            x = (x - tau_c) / tau_c
        
        # Remove nans from x and y before binning
        valid_indices = ~np.isnan(x) & ~np.isnan(y)
        x = x[valid_indices]
        y = y[valid_indices]

        if len(x) == 0:
            continue

        bin_means, bin_edges, _ = stats.binned_statistic(x, y, statistic="mean", bins=100)
        stds, _, _ = stats.binned_statistic(x, y, statistic="std", bins=100)
        
        plot_x = bin_edges[:-1]
        plot_y = bin_means

        ax.scatter(plot_x, plot_y, label=f"R={truncated_noise:.2f}", linewidth=0.5, color=colors[i], marker='o', s=2)

        ax_std.scatter(plot_x, plot_y, label=f"R={truncated_noise:.2f}", linewidth=0.5, color=colors[i], marker='o', s=2)
        ax_std.scatter(plot_x, plot_y + stds, marker="_", linewidth=0.5, color=colors[i])
        ax_std.scatter(plot_x, plot_y - stds, marker="_", linewidth=0.5, color=colors[i])

    ax.set_xlabel("$(\\tau_{{ext}} - \\tau_c)/\\tau_c$")
    ax.set_ylabel("$\\zeta$")
    ax.legend(fontsize='small')
    ax.grid(True)
    fig.tight_layout()

    ax_std.set_xlabel("$(\\tau_{{ext}} - \\tau_c)/\\tau_c$")
    ax_std.set_ylabel("$\\zeta$")
    ax_std.legend(fontsize='small')
    ax_std.grid(True)
    fig_std.tight_layout()

    return fig, ax, fig_std, ax_std

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
        avg_rough_path = p.joinpath("avg-rough-arranged")
        if not avg_rough_path.exists():
            print(f"Directory not found, skipping: {avg_rough_path}")
            continue
        for noise_folder in avg_rough_path.iterdir():

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

                    fig,ax, fit_params = makeRoughnessPlotPerfect(l_range, w_avg, params, dest_plot)
                    fig.savefig(dest_plot)

                elif dislocation_dir == "partial-dislocation":
                    dest_plot = dest.joinpath(dislocation_dir).joinpath(f"plots/noise-{noise_val}")
                    dest_plot.mkdir(parents=True, exist_ok=True)
                    dest_plot = dest_plot.joinpath(f"roughness-tau-{tauExtValue}.png")

                    fig,ax, fit_params = makeRoughnessPlotPerfect(l_range, w_avg, params, dest_plot)
                    fig.savefig(dest_plot)
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
    if not path.exists():
        print(f"Directory not found, cannot extract roughness: {path}")
        return
    for noise_folder in [s for s in path.iterdir() if s.is_dir()]:
        noise_val = noise_folder.name.split("-")[1]

        for seed_folder in noise_folder.iterdir():
            seed = int(seed_folder.stem.split("-")[1])
            print(f"Extracting params from data with noise {noise_val} and seed {seed}")

            for file_path in seed_folder.iterdir():
                if not file_path.exists():
                    print(f"File not found, skipping: {file_path}")
                    continue
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

                fig,ax, fit_params = makeRoughnessPlotPerfect(l_range, roughness, new_params, save_path)
                fig.savefig(save_path)
    pass

if __name__ == "__main__":
    root = "results/2025-06-08-merged-final"
    path_perfect = Path(root).joinpath("roughness_parameters_perfect.npz")
    path_partial = Path(root).joinpath("roughness_parameters_partial.npz")
    
    save_path = Path(root).joinpath("correlation-plots/correlation-selected-perfect.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, fig_std, ax_std = selectedCorrelationPlots(path_perfect, Path(root),save_path, perfect=True, unique_noises = [0.01072, 1.07227, 10.0, 107.22672, 1000.0])
    fig.savefig(save_path)
    fig_std.savefig(save_path.parent.joinpath("correlation-w-std-selected-perfect.pdf"))

    save_path = Path(root).joinpath("correlation-plots/correlation-selected-partial.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, fig_std, ax_std = selectedCorrelationPlots(path_partial, Path(root), save_path, perfect=False, unique_noises = [0.01072, 1.07227, 10.0, 107.22672, 869.74900])
    fig.savefig(save_path)
    fig_std.savefig(save_path.parent.joinpath("correlation-w-std-selected-partial.pdf"))

    save_path = Path(root).joinpath("selected-zeta-plots/zeta-selected-perfect.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, fig_std, ax_std = selectedZetaPlots(path_perfect, Path(root), save_path, perfect=True, unique_noises = [0.01072, 1.07227, 10.0, 107.22672, 1000.0])
    fig.savefig(save_path)
    fig_std.savefig(save_path.parent.joinpath("zeta-selected-w-std-perfect.pdf"))

    save_path = Path(root).joinpath("selected-zeta-plots/zeta-selected-partial.pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax, fig_std, ax_std = selectedZetaPlots(path_partial, Path(root), save_path, perfect=True, unique_noises = [0.01072, 1.07227, 10.0, 107.22672, 869.74900])
    fig.savefig(save_path)
    fig_std.savefig(save_path.parent.joinpath("zeta-selected-w-std-partial.pdf"))