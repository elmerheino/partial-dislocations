from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
import multiprocessing as mp
from scipy import optimize
from functools import partial

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
    with open(Path(root_dir).joinpath("roughness_partial.json"), "r") as fp:
        roughnesses_partial = json.load(fp)
    
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

        for seed_folder in noise_folder.iterdir():
            seed = int(seed_folder.stem.split("-")[1])
            # if seed != 0:
            #     continue
            print(f"Making roughness plots for noise {noise_val} seed {seed}")

            with mp.Pool(7) as pool:
                results = pool.map(partial(loadRoughnessData_partial, root_dir=root_dir), seed_folder.iterdir())
                tauExt, seed_r, c, zeta, cutoff, k = zip(*results)
                roughnesses_partial_noise["tauExt"] += tauExt
                roughnesses_partial_noise["seed"] += seed_r
                roughnesses_partial_noise["c"] += c
                roughnesses_partial_noise["zeta"] += zeta
                roughnesses_partial_noise["cutoff"] += cutoff
                roughnesses_partial_noise["k"] += k
        
        roughnesses_partial[noise_val] = roughnesses_partial_noise

    
    with open(Path(root_dir).joinpath("roughness_partial.json"), "w") as fp:
        json.dump(roughnesses_partial, fp)


def makePerfectRoughnessPlots(root_dir):
    p = Path(root_dir).joinpath("single-dislocation").joinpath("averaged-roughnesses")

    roughnesses_perfect = dict()

    for noise_folder in [s for s in p.iterdir() if s.is_dir()]:
        noise_val = noise_folder.name.split("-")[1]

        roughnesses_np_noise = {
            "tauExt" : list(), "seed" : list(), "c" : list(), "zeta" : list(),
            "cutoff" : list(), "k":list()
        }

        for seed_folder in noise_folder.iterdir():
            seed = int(seed_folder.stem.split("-")[1])
            # if seed != 0:
            #     continue
            print(f"Making roughness plots for noise {noise_val} seed {seed}")

            with mp.Pool(7) as pool:
                results = pool.map(partial(loadRoughnessDataPerfect, root_dir=root_dir), seed_folder.iterdir())
                tauExt, seed_r, c, zeta, cutoff, k = zip(*results)
                roughnesses_np_noise["tauExt"] += tauExt
                roughnesses_np_noise["seed"] += seed_r
                roughnesses_np_noise["c"] += c
                roughnesses_np_noise["zeta"] += zeta
                roughnesses_np_noise["cutoff"] += cutoff
                roughnesses_np_noise["k"] += k
        
        roughnesses_perfect[noise_val] = roughnesses_np_noise
        
    with open(Path(root_dir).joinpath("roughness_perfect.json"), "w") as fp:
        json.dump(roughnesses_perfect, fp)


def makeAvgRoughnessPlots(root_dir):
    # Makes roughness plots that have been averaged only at simulation (that is velocity) level first
    # for partial dislocations

    makePerfectRoughnessPlots(root_dir)
    makePartialRoughnessPlots(root_dir)
    analyzeRoughnessFitParamteters(root_dir)
    
    pass

def loadRoughnessDataPerfect(f1, root_dir):
    # Helper function to enable use of multiprocessing when making plots
    loaded = np.load(f1)
    avg_w = loaded["avg_w"]
    l_range = loaded["l_range"]
    params = loaded["parameters"]
    bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    # TODO: loop over each noise level here

    save_path = Path(root_dir)
    save_path = save_path.joinpath("roughness-non-partial").joinpath(f"noise-{deltaR:.4f}/seed-{seed}")
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path.joinpath(f"avg-roughness-tau-{tauExt:.3f}.png")

    tauExt, seed, c, zeta, cutoff, k = makeRoughnessPlotPerfect(l_range, avg_w, params, save_path)

    return (tauExt, seed, c, zeta, cutoff, k)

def makeRoughnessPlotPerfect(l_range, avg_w, params, save_path : Path):
    bigN, length, time, dt, deltaR, bigB, smallB, cLT, mu, tauExt, d0, seed, tau_cutoff = params

    # Make a piecewise fit on all data
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

    # Now make the actual plot with suitable fit
    plt.clf()
    plt.figure(figsize=(8,8))

    plt.scatter(l_range, avg_w, label="$W$", marker="x") # Plot the data as is
    # plt.plot(l_range, ynew, label="piecewise fit", color="blue") # Plot the piecewise fit

    # Select zeta that is 10% at most smaller than initial zeta
    target_zeta = zetas[0]*(1-0.10)  # TODO: check this more closely, there might be something wrong.
    n_selected = np.argmax(zetas < target_zeta)

    last_exp = np.argmax(l_range > l_0_range[n_selected])
    exp_l = l_range[:last_exp]

    c, zeta = c_values[n_selected], zetas[n_selected]
    ynew_exp = exp_beheavior(exp_l, c, zeta)

    plt.plot(exp_l, ynew_exp, label=f"$ \\log (L) \\leq {np.log(l_0_range[n_selected]):.2f} $  fit with $\\zeta = $ {zeta:.3f}", color="red")

    # Now find out the constant behavior from N/4 < L < 3N/4

    start = int(round(len(l_range)/4))
    end = int(round(3*len(l_range)/4 ))

    const_l = l_range[start:end]
    const_w = avg_w[start:end]

    fit_c_range, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])
    
    new_c = np.ones(len(const_l))*fit_c_range

    plt.plot(const_l, new_c, label=f"N/4 < L < 3N/4", color="orange")

    # Take the constant from the piecewise cutoff
    start = int(round(cutoff_piecewise))

    const_l = l_range[start:]
    const_w = avg_w[start:]

    fit_c_piecewise, pcov = optimize.curve_fit(lambda x,c : c, const_l, const_w, p0=[4.5])
    
    new_c = np.ones(len(const_l))*fit_c_piecewise

    # plt.plot(const_l, new_c, label=f"log(L) > {np.log(start):.3f}", color="magenta")

    # Next find out the transition point between power law and constant behavior

    fit_c_range = fit_c_range[0]   # This is the constant of constant behavior y = c
    zeta = zeta                     # This is the exponent of power law y = c*l^zeta
    c = c                           # This is the factor in power law y=c*l^zeta

    change_p = (fit_c_range / c)**(1/zeta)  # This is the points l=change_p where is changes from power to constant.

    plt.scatter([change_p], [fit_c_range], label="Tansition point")

    # Plot dashed lines to illustrate the intersection
    dashed_x = np.linspace(change_p, max(l_range)/4, 10)
    dahsed_y = np.ones(10)*fit_c_range
    plt.plot(dashed_x, dahsed_y, linestyle="dashed", color="grey")

    dashed_x = np.linspace(last_exp, change_p, 10)
    dahsed_y = exp_beheavior(dashed_x, c, zeta)
    plt.plot(dashed_x, dahsed_y, linestyle="dashed", color="grey")

    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    plt.title(f"Roughness of a perfect dislocation s = {seed} $\\tau_{{ext}}$ = {tauExt:.3f}")
    plt.xlabel("log(L)")
    plt.ylabel("$\\log(W_(L))$")
    plt.legend()

    plt.savefig(save_path, dpi=300)
    plt.close()

    return (tauExt, seed, c, zeta, change_p, fit_c_range)

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
    bigN, length,   time,   dt, selfdeltaR, selfbigB, smallB,  b_p, cLT1,   cLT2,   mu,   tauExt,   c_gamma, d0,   seed,   tau_cutoff = params

    # Loop through every noise level here

    p = Path(root_dir)
    p = p.joinpath("roughness-partial").joinpath(f"seed-{seed}")
    p.mkdir(parents=True, exist_ok=True)
    p = p.joinpath(f"avg-roughness-tau-{tauExt:.3f}.png")

    tauExt, seed, c, zeta, cutoff, k = makeRoughnessPlot_partial(l_range, avg_w12, params, p)

    return (tauExt, seed, c, zeta, cutoff, k)

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
