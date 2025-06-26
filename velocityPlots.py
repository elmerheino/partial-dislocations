import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats, optimize
from functools import partial
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

def velocity_fit(tau_ext, tau_crit, beta, a):
    v_res = np.empty(len(tau_ext))
    for n,tau in enumerate(tau_ext):
        if tau > tau_crit:
            v_res[n] = a*(tau - tau_crit)**beta
        elif tau < tau_crit:
            v_res[n] = 0
    return v_res


def getWindow(tauExt_np, vCm_np, initial_t_c_arvaus, window_width=10):
    """
    Performs a window search on tauExt and vCm to find the critical force.
    """
    refined_t_c = initial_t_c_arvaus
    window_bounds = None  # Will store (tau_window[0], tau_window[-1]) if a window is found
    array_bounds = None

    if len(tauExt_np) >= window_width and np.any(vCm_np > 1e-9): # 1e-9 is a small number to check for non-zero velocity
        max_v = np.max(vCm_np)
        if max_v <= 1e-9:
            transition_threshold = 1e-6 # Default absolute minimum
        else:
            transition_threshold = max(1e-6, 0.001 * max_v)

        found_suitable_window = False
        for i in range(len(tauExt_np) - window_width + 1):
            tau_window = tauExt_np[i : i + window_width]
            v_window = vCm_np[i : i + window_width]

            if v_window[0] < transition_threshold and v_window[-1] > transition_threshold:
                for k_in_window in range(window_width):
                    if v_window[k_in_window] > transition_threshold:
                        refined_t_c = tau_window[k_in_window]
                        window_bounds = (tau_window[0], tau_window[-1])
                        array_bounds = (i, i + window_width)
                        found_suitable_window = True
                        break 
                if found_suitable_window:
                    break
            
    return refined_t_c, window_bounds, array_bounds

def make_a_closeup_plot(x_closeup, y_closeup, save_path : Path, truncated_key, refined_t_c, seed):

    # if truncated_key == "4.328761":
    #     print("saatana vittusaatana")
    
    fit_params, pcov = optimize.curve_fit(velocity_fit, x_closeup, y_closeup, p0=[refined_t_c*1.02,   # tau_c
                                                            0.5,          # beta
                                                            0.9           # A
                                                            ], bounds=(
                                                                [0, 0.3, 0],
                                                                [max(x_closeup), 0.6, np.inf]
                                                            ), maxfev=3000)
    t_c, beta, a = fit_params

    deltaTau = (max(x_closeup) - min(x_closeup))/len(x_closeup)
    
    # Calculate mean squared error for the fit
    v_fit = velocity_fit(x_closeup, *fit_params)
    mse = np.mean((y_closeup - v_fit)**2)

    xnew_closeup = np.linspace(min(x_closeup), max(x_closeup), 100)
    ynew_closeup = velocity_fit(xnew_closeup, *fit_params)

    xnew_closeup = (xnew_closeup - t_c)/t_c
    x_closeup = (x_closeup - t_c)/t_c

    plt.clf()
    plt.figure(figsize=(linewidth/2,linewidth/2))

    plt.scatter(x_closeup,y_closeup, marker='o', s=10, facecolors='none', edgecolors='red', label="Data")
    plt.plot(xnew_closeup, ynew_closeup)

    plt.title(f"Depinning $\\tau_{{c}} = $ {t_c:.3f}, A={a:.3f}, $\\beta$ = {beta:.3f}, seed = {seed}")
    plt.xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{ext}}$")
    plt.ylabel("$v_{{cm}}$")
    plt.legend()

    closeup_save_path_ = save_path.parent.joinpath(f"closeups/noise-{truncated_key}/normalized-depinning-closeup-{seed}.png")
    closeup_save_path_.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(closeup_save_path_)
    plt.tight_layout()
    plt.close()

    return fit_params

def normalizedDepinnings(depinning_path : Path, plot_save_folder : Path, data_save_path : Path, json_save_path : Path, optimized=False, seed_count = 10):
    """
    Make normalized velocity-tau_ext plots for partial and perfect dislocations. Also 
    save the tau_c values for each noise in a json file. The veclocities are normalized
    with v = (tau_ext - tau_c)/tau_ext. The tau_c values are calculated by fitting the
    function v = A*(tau_ext - tau_c)^beta to the data.
    """

    # Make such plots for a single dislocation first
    # Collect all values of tau_c

    tau_c = dict()
    fit_errors = dict()
    delta_tau_values = dict()
    beta_values = dict()
    prefactor_values = dict()

    # Collect all datapoints for binning
    data_perfect = dict()

    # Prepare to save all fit data to a csv file
    tau_c_csv = list()

    for n,noise_path in enumerate(depinning_path.iterdir()):
        if not noise_path.is_dir():
            continue

        # if not n % 10 == 0:
        #     continue
        
        noise = noise_path.name.split("-")[1]

        truncated_key = str(f"{float(noise):.6f}") # Using 6 decimal places, minimum to distinguish values from np.logspace(-3,3,100) would be 4

        if truncated_key not in tau_c:
            tau_c[truncated_key] = list()
        if truncated_key not in fit_errors:
            fit_errors[truncated_key] = list()
        if truncated_key not in data_perfect:
            data_perfect[truncated_key] = list()
        if truncated_key not in delta_tau_values:
            delta_tau_values[truncated_key] = list()
        if truncated_key not in beta_values:
            beta_values[truncated_key] = list()
        if truncated_key not in prefactor_values:
            prefactor_values[truncated_key] = list()

        for fpath in noise_path.iterdir():
            try:
                with open(fpath, "r") as fp:
                    depinning = json.load(fp)
            except Exception as e:
                print(f"Failed to load file {fpath} with exception {e}")
            tauExt = depinning["stresses"]
            vCm = np.array(depinning["v_rel"])*10 # Multiply by ten to account for 0.1s dt between dislocation 
            seed = depinning["seed"]

            t_c_arvaus = (max(tauExt) - min(tauExt))/2
            deltaTau = (max(tauExt) - min(tauExt)) / len(tauExt)

            bounds = None
            tauCrit, beta, a = None, None, None

            xnew_closeup = np.array([])
            ynew_closeup = np.array([])
            
            xnew_all = np.array([])
            ynew_all = np.array([])

            if float(noise) > 0.1:
                bounds = None
                tauCrit, beta, a = None, None, None

                refined_t_c, _, bounds = getWindow(np.array(tauExt), np.array(vCm), t_c_arvaus)
                print(f"bounds {bounds} and refined_t_c = {refined_t_c}")

                start_ = bounds[0] + 4
                end_ = bounds[1] + 6

                x_closeup = tauExt[start_:end_]
                y_closeup = vCm[start_:end_]

                tauCrit, beta, a = make_a_closeup_plot(x_closeup, y_closeup, plot_save_folder, truncated_key, refined_t_c, seed)

                xnew_closeup = np.linspace(min(x_closeup), max(x_closeup), 100)
                ynew_closeup = velocity_fit(xnew_closeup, tauCrit, beta, a)

                xnew_closeup = (xnew_closeup - tauCrit)/tauCrit

                # Calculate mean squared error for the fit
                v_fit =velocity_fit(x_closeup, tauCrit, beta, a)
                mse = np.mean((v_fit - y_closeup)**2)
            else:
                bounds = None
                tauCrit, beta, a = None, None, None

                fit_params, pcov = optimize.curve_fit(velocity_fit, tauExt, vCm, p0=[t_c_arvaus,   # tau_c
                                                                            0.7,          # beta
                                                                            0.9           # A
                                                                            ], bounds=(0, [ max(tauExt), 2, 2 ]), maxfev=10000)
                # Calculate mean squared error for the fit
                v_fit = velocity_fit(tauExt, *fit_params)
                mse = np.mean((vCm - v_fit)**2)

                xnew_all = np.linspace(min(tauExt), max(tauExt), 100)
                ynew_all = velocity_fit(xnew_all, *fit_params)

                tauCrit, beta, a = fit_params
                xnew_all = (xnew_all - tauCrit)/tauCrit

            x_all = (tauExt - tauCrit)/tauCrit
            y_all = vCm

            # Save the critical force, mse and delta tau to dicts for later use
            tau_c[truncated_key].append(tauCrit)
            beta_values[truncated_key].append(beta)
            prefactor_values[truncated_key].append(a)
            fit_errors[truncated_key].append(mse)
            delta_tau_values[truncated_key].append(deltaTau)

            # Scale the original data
            filtered_raw = [(t, v) for t, v in zip(x_all, y_all) if not (np.isnan(t) or np.isinf(t) or np.isnan(v) or np.isinf(v))]
            data_perfect[truncated_key] += filtered_raw

            plt.clf()
            plt.figure(figsize=(linewidth/2,linewidth/2))

            plt.scatter(x_all,y_all, marker='o', s=10, facecolors='none', edgecolors='red', label="Data")

            if xnew_closeup.size != 0:
                plt.plot(xnew_closeup, ynew_closeup)
            
            if xnew_all.size != 0:
                plt.plot(xnew_all, ynew_all)

            # plt.title(f"Depinning $\\tau_{{c}} = $ {tauCrit:.3f}, A={a:.3f}, $\\beta$ = {beta:.3f}, seed = {seed}")
            plt.xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{ext}}$")
            plt.ylabel("$v_{{cm}}$")
            # plt.legend()
            plt.tight_layout()

            p = plot_save_folder.joinpath(f"noise-{truncated_key}")
            p.mkdir(exist_ok=True, parents=True)
            plt.savefig(p.joinpath(f"normalized-depinning-{seed}.pdf"))
            plt.close()
    
    k = 10  # This should equal the no of realizations of noise in the data

    tau_c_csv.clear()

    for truncated_key in tau_c.keys():
        current_tau_crits = list(tau_c.get(truncated_key, []))
        current_fit_errors = list(fit_errors.get(truncated_key, []))
        current_delta_taus = list(delta_tau_values.get(truncated_key, []))
        current_beta_vals = list(beta_values.get(truncated_key, []))
        current_prefcator_vals = list(prefactor_values.get(truncated_key, []))

        if len(current_tau_crits) < k:
            current_tau_crits.extend([None] * (k - len(current_tau_crits)))
        elif len(current_tau_crits) > k:
            current_tau_crits = current_tau_crits[:k]

        if len(current_fit_errors) < k:
            current_fit_errors.extend([None] * (k - len(current_fit_errors)))
        elif len(current_fit_errors) > k:
            current_fit_errors = current_fit_errors[:k]
        
        if len(current_delta_taus) < k:
            current_delta_taus.extend([None] * (k - len(current_delta_taus)))
        elif len(current_delta_taus) > k:
            current_delta_taus = current_delta_taus[:k]

        if len(current_beta_vals) < k:
            current_beta_vals.extend([None] * (k - len(current_beta_vals)))
        elif len(current_beta_vals) > k:
            current_beta_vals = current_beta_vals[:k]

        if len(current_prefcator_vals) < k:
            current_prefcator_vals.extend([None] * (k - len(current_prefcator_vals)))
        elif len(current_prefcator_vals) > k:
            current_prefcator_vals = current_prefcator_vals[:k]

        compiled_row = [truncated_key] + current_tau_crits + current_fit_errors + current_delta_taus + current_beta_vals + current_prefcator_vals
        tau_c_csv.append(compiled_row)

    tau_c_csv.sort(key=lambda row: float(row[0]))

    data_save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(data_save_path, "w") as fp:
        writer = csv.writer(fp)
        # Write header
        header = ["noise"] + [f"tau_c_{i+1}" for i in range(k)] + [f"mse_{i+1}" for i in range(k)]  + [f"d_tau_{i+1}" for i in range(k)] + [f"beta_{i+1}" for i in range(k)] + [f"A_{i+1}" for i in range(k)]
        writer.writerow(header)
        # Write data rows
        writer.writerows(tau_c_csv)
    
    json_save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_save_path, "w") as fp:
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

def makeOneBinnedPlot(x,y, ax, tau_c, color, label, bins=100, conf_level=0.9):
    bin_means, bin_edges, _ = stats.binned_statistic(x,y,statistic="mean", bins=bins)

    lower_confidence, _, _ = stats.binned_statistic(x,y,statistic=partial(confidence_interval_lower, c_level=conf_level), bins=bins)
    upper_confidence, _, _ = stats.binned_statistic(x,y,statistic=partial(confidence_interval_upper, c_level=conf_level), bins=bins)

    bin_counts, _, _ = stats.binned_statistic(x,y,statistic="count", bins=bins)

    # print(f'Total of {sum(bin_counts)} datapoints. The bins have {" ".join(bin_counts.astype(str))} points respectively.')

    ax.set_xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{ext}}$")
    ax.set_ylabel("$v_{{CM}}$")

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    # ax.scatter(x,y, marker='x', linewidths=0.2, color="grey")
    # plt.plot(bin_centers, lower_confidence, color="blue", label=f"${conf_level*100} \\%$ confidence")
    # plt.plot(bin_centers, upper_confidence, color="blue")

    print(f"Mean condfidence : {np.nanmean(upper_confidence - bin_means)}")


    if np.nanmean(upper_confidence - bin_means) < 0.5:
        ax.scatter(bin_centers, bin_means, marker="o", s=2, linewidth=0.5,
            label=label, color=color)
    else:
        ax.errorbar(bin_centers, bin_means, yerr=[bin_means - lower_confidence, upper_confidence - bin_means], color=color, fmt="s", ms=2, capsize=2,  linewidth=0.5, label=label)

def binning(data : dict, res_dir : Path, conf_level, bins=100): # non-partial and partial dislocation global data, respectively
    """
    Make binned depinning plots from the data. The data is binned and the mean and confidence intervals are calculated.

    Here dict is a dictionary containing all the normalized data for each noise level.
    """

    res_dir = Path(res_dir)

    with open(res_dir.joinpath("binning-data/tau_c_perfect.json"), "r") as fp:
        data_tau_perfect = json.load(fp)
    
    first_key = next(iter(data_tau_perfect.keys()))

    with open(res_dir.joinpath("binning-data/tau_c_partial.json"), "r") as fp:
        data_tau_partial = json.load(fp)
    
    figures = dict()

    for perfect_partial in data.keys():
        for noise in data[perfect_partial].keys():
            d = data[perfect_partial]
            x,y = zip(*d[noise])

            if not noise in figures.keys():
                figures[noise] = plt.subplots(figsize=(linewidth/2,linewidth/2))
            
                        
            if perfect_partial == "perfect_data":
                tau_c_perfect = sum(data_tau_perfect[noise])/len(data_tau_perfect[noise])

                fig, ax = figures[noise]
                makeOneBinnedPlot(x,y,ax, tau_c_perfect, color="blue", label="Perfect")
                figures[noise] = (fig, ax)
                
            elif perfect_partial == "partial_data":
                tau_c_partial = sum(data_tau_partial[noise])/len(data_tau_partial[noise])

                fig, ax = figures[noise]
                makeOneBinnedPlot(x,y,ax, tau_c_partial, color="red", label="Partial")
                figures[noise] = (fig, ax)

    for key in figures.keys():
        path_binning_combined = Path(res_dir).joinpath(f"binned-depinnings-combined/binned-depinning-noise-{key}-conf-{conf_level}.pdf")
        path_binning_combined.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = figures[key]
        ax.legend(loc='lower right', handletextpad=0.1, borderpad=0.1)
        ax.grid(True)

        fig.tight_layout()
        fig.savefig(path_binning_combined)

    pass

def makeBetaPlot(ax, csv_path : Path, color, label):
    with open(csv_path, "r") as fp:
        header = fp.readline().strip().split(',')
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        pass

    # partial dislocation

    noises = loaded[:,0]
    tau_c_means = np.nanmean(loaded[:,1:11], axis=1)
    tau_c_stds = np.nanstd(loaded[:,1:11], axis=1)
    tau_c_mses = np.nanmean(loaded[:, 11:21], axis=1)
    deltaTaus = np.nanmean(loaded[:, 21:31], axis=1)
    betas = np.nanmean(loaded[:, 31:41], axis=1)

    ax.scatter(noises, betas, marker="o", s=0.5, color=color, label=label)

    ax.set_xlabel("$\\Delta R$")
    ax.set_ylabel("$\\beta$")

    # ax.set_title("$\\beta$ vs Noise")
    ax.set_xscale('log')
    ax.grid(True)

    pass

def makeAveragedDepnningPlots(dir_path, opt=False):
    print("Making averaged depinning plots.")
    partial_depinning_path = Path(dir_path).joinpath("partial-dislocation/depinning-dumps")
    perfect_depinning_path = Path(dir_path).joinpath("single-dislocation/depinning-dumps")

    if opt:
        partial_depinning_path = Path(dir_path).joinpath("partial-dislocation/optimal-depinning-dumps")
        perfect_depinning_path = Path(dir_path).joinpath("single-dislocation/optimal-depinning-dumps")
    
    if partial_depinning_path.exists():

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
            plt.figure(figsize=(linewidth/2,linewidth/2))

            plt.scatter(x,y, marker="x")

            plt.title(f"Depinning noise = {noise}")
            plt.xlabel("$\\tau_{ext}$")
            plt.ylabel("$v_{CM}$")

            dest = Path(dir_path).joinpath(f"averaged-depinnings/partial/{float(noise)*1e3}-noise")
            dest.mkdir(parents=True, exist_ok=True)
            if opt:
                plt.savefig(dest.joinpath(f"depinning-noise-opt.png"))
            else:
                plt.savefig(dest.joinpath(f"depinning-noise.png"))
            pass
            plt.close()
    else:
        print("No partial depinning dumps")

    if perfect_depinning_path.exists():
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
            plt.figure(figsize=(linewidth/2,linewidth/2))

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
    else:
        print("No perfect depinning dumps")

def copy_all_files(from_path, to_path):
    # Get a list of all files and directories in the source directory
    items = os.listdir(from_path)

    # Iterate over the items
    for item in items:
        # Create the full path to the item in the source directory
        s = os.path.join(from_path, item)
        # Create the full path to the item in the destination directory
        d = os.path.join(to_path, item)
        
        # If the item is a directory, copy it recursively
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        # Otherwise, copy the file
        else:
            shutil.copy2(s, d)
    pass

if __name__ == "__main__":
    root = Path("/Volumes/contenttii/2025-06-08-merged-final")

    fig, ax = plt.subplots(figsize=(linewidth/2,linewidth/2))

    makeBetaPlot(ax, root.joinpath("noise-data/perfect-noises.csv"), color="blue", label="Perfect")
    makeBetaPlot(ax, root.joinpath("noise-data/partial-noises.csv"), color="red", label="Partial")

    save_path = root.joinpath("beta-vs-noise.pdf")

    shutil.copy2(save_path, "/Users/elmerheino/Documents/kandi-repo/figures")
    
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
