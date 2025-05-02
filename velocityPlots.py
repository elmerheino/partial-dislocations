import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats, optimize
from functools import partial

def velocity_fit(tau_ext, tau_crit, beta, a):
    v_res = np.empty(len(tau_ext))
    for n,tau in enumerate(tau_ext):
        if tau > tau_crit:
            v_res[n] = a*(tau - tau_crit)**beta
        else:
            v_res[n] = 0
    return v_res

def normalizedDepinnings(depinning_path : Path, save_folder : Path, optimized=False):
    """
    Make normalized velocity-tau_ext plots for partial and perfect dislocations. Also 
    save the tau_c values for each noise in a json file. The veclocities are normalized
    with v = (tau_ext - tau_c)/tau_ext. The tau_c values are calculated by fitting the
    function v = A*(tau_ext - tau_c)^beta to the data.
    """

    # Make such plots for a single dislocation first
    # Collect all values of tau_c

    tau_c = dict()

    # Collect all datapoints for binning

    data_perfect = dict()

    for noise_path in depinning_path.iterdir():
        if not noise_path.is_dir():
            continue
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
                fit_params, pcov = optimize.curve_fit(velocity_fit, tauExt, vCm, p0=[t_c_arvaus,   # tau_c
                                                                            0.9,          # beta
                                                                            0.9           # a
                                                                            ], bounds=(0, [ max(tauExt), 2, 2 ]), maxfev=1600)
            except:
                print(f"Could not find fit w/ perfect dislocation noise : {noise} seed : {seed}")
                continue

            tauCrit, beta, a = fit_params

            tau_c[noise].append(tauCrit)

            xnew = np.linspace(min(tauExt), max(tauExt), 100)
            ynew = velocity_fit(xnew, *fit_params)

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
            plt.title(f"Depinning $\\tau_{{c}} = $ {tauCrit:.3f}, A={a:.3f}, $\\beta$ = {beta:.3f}, seed = {seed}")
            plt.xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{ext}}$")
            plt.ylabel("$v_{{cm}}$")
            plt.legend()

            p = save_folder.joinpath(f"noise-{noise}")
            p.mkdir(exist_ok=True, parents=True)
            plt.savefig(p.joinpath(f"normalized-depinning-{seed}.png"))
            plt.close()
    
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
    # TODO: do this for each noise

    """
    Make binned depinning plots from the data. The data is binned and the mean and confidence intervals are calculated.

    Here dict is a dictionary containing all the normalized data for each noise level.
    """

    with open(res_dir.joinpath("single-dislocation/normalized-plots/tau_c.json"), "r") as fp:
        data_tau_perfect = json.load(fp)
    
    first_key = next(iter(data_tau_perfect.keys()))
    

    with open(res_dir.joinpath("partial-dislocation/normalized-plots/tau_c.json"), "r") as fp:
        data_tau_partial = json.load(fp)
    

    
    for perfect_partial in data.keys():
        for noise in data[perfect_partial].keys():
            d = data[perfect_partial]
            x,y = zip(*d[noise])

            tau_c_perfect = sum(data_tau_perfect[str(noise)])/len(data_tau_perfect[str(noise)])
            tau_c_partial = sum(data_tau_partial[str(noise)])/len(data_tau_partial[str(noise)])

            bin_means, bin_edges, _ = stats.binned_statistic(x,y,statistic="mean", bins=bins)

            lower_confidence, _, _ = stats.binned_statistic(x,y,statistic=partial(confidence_interval_lower, c_level=conf_level), bins=bins)
            upper_confidence, _, _ = stats.binned_statistic(x,y,statistic=partial(confidence_interval_upper, c_level=conf_level), bins=bins)

            bin_counts, _, _ = stats.binned_statistic(x,y,statistic="count", bins=bins)

            # print(f'Total of {sum(bin_counts)} datapoints. The bins have {" ".join(bin_counts.astype(str))} points respectively.')

            plt.clf()
            plt.close('all')
            plt.figure(figsize=(8,8))

            if perfect_partial == "perfect_data":
                plt.title(f"$ \\langle \\tau_c \\rangle = {tau_c_perfect:.4f} $")
            elif perfect_partial == "partial_data":
                plt.title(f"$ \\langle \\tau_c \\rangle = {tau_c_partial:.4f} $")
            
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

def makeAveragedDepnningPlots(dir_path, opt=False):
    print("Making averaged depinning plots.")
    partial_depinning_path = Path(dir_path).joinpath("partial-dislocation/depinning-dumps")
    perfect_depinning_path = Path(dir_path).joinpath("single-dislocation/depinning-dumps")

    if opt:
        partial_depinning_path = Path(dir_path).joinpath("partial-dislocation/optimal-depinning-dumps")
        perfect_depinning_path = Path(dir_path).joinpath("single-dislocation/optimal-depinning-dumps")

    for noise_dir in partial_depinning_path.iterdir():
        if not noise_dir.is_dir():
            continue
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
        plt.close()

    for noise_dir in perfect_depinning_path.iterdir():
        if not noise_dir.is_dir():
            continue

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
        if opt:
            plt.savefig(dest.joinpath(f"depinning-noise-opt.png"))
        else:
            plt.savefig(dest.joinpath(f"depinning-noise.png"))
        pass
        plt.close()
