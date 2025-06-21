#!/usr/bin/env python3

import math
from pathlib import Path
import json
import time
from plots import *
import numpy as np
from scipy import optimize
from scipy import stats
from argparse import ArgumentParser
import multiprocessing as mp
from functools import partial
import matplotlib as mpl
import matplotlib.pyplot as plt

from roughnessPlots import *
from velocityPlots import *

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

linewidth = 5.59164

def makeNoisePlot(noises, tau_c_means, point_0, point_1, point_2, y_error, color='blue', fit_color='red'):
    fig, ax = plt.subplots(figsize=(linewidth/2, linewidth/2))

    ax.errorbar(noises, tau_c_means, yerr=y_error, fmt='s', markersize=2, capsize=2, color=color, linewidth=0.5, zorder=0)

    data = np.array(list(zip(noises, tau_c_means)))
    sorted_args = np.argsort(data[:,0])
    data = data[sorted_args]

    # Fit power law to first region of data

    region0_index = np.where(data[:,0] > point_0)[0][0]

    try:
        if np.where(data[:,0] > point_1)[0].size == 0:
            x = data[:,0]
            y = data[:,1]
            region1_index = len(x) - 1
        else:
            region1_index = np.where(data[:,0] > point_1)[0][0]
            x = data[region0_index:region1_index,0]
            y = data[region0_index:region1_index,1]
        print(region1_index, x.shape, y.shape)
        fit_params, pcov = optimize.curve_fit(
            lambda x, a, b: a*x**b,
            x, y)
        fit_x = np.linspace(data[0,0], data[region1_index,0], 100)
        fit_y = fit_params[0]*fit_x**fit_params[1]
        ax.plot(fit_x, fit_y, color=fit_color, linewidth=2)
        ax.text(fit_x[1], fit_y[1], f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", ha='left', va='top')
        print(f"Noise fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except Exception as e:
        print(f"No data in first region of data. {e}")

    # Fit power law to second region of data
    try:
        region1_end = np.where(data[:,0] > point_2)[0][0]
        x = data[region1_index:region1_end,0]
        y = data[region1_index:region1_end,1]
        fit_params, pcov = optimize.curve_fit(
            lambda x, a, b: a*x**b,
            x, y)
        fit_x = np.linspace(data[region1_index,0], data[region1_end,0], 100)
        fit_y = fit_params[0]*fit_x**fit_params[1]
        ax.plot(fit_x, fit_y,
                color='black', linestyle='--', linewidth=2
                )
        ax.text(fit_x[len(fit_x)//4], fit_y[len(fit_y)//4], f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", ha='right', va='bottom')
        print(f"Noise fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except:
        print("No data in second region of data.")
        region1_index = 0

    # Fit power law to third region of data
    try:
        x = data[region1_end:,0]
        y = data[region1_end:,1]
        fit_params, pcov = optimize.curve_fit(
            lambda x, a, b: a*x**b,
            x, y)
        fit_x = np.linspace(data[region1_end,0], data[-1,0], 100)
        fit_y = fit_params[0]*fit_x**fit_params[1]
        ax.plot(fit_x, fit_y, color=fit_color, linestyle=':', linewidth=2)
        ax.text(fit_x[len(fit_x)//4], fit_y[len(fit_y)//4], f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", ha='right', va='bottom')
        print(f"Perfect dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except:
        print("No data in third region of data.")
    
    return fig, ax

def makePerfectNoisePlot(results_root : Path, save_path):
    with open(results_root.joinpath("noise-data/perfect-noises.csv"), "r") as fp:
        header = fp.readline().strip().split(',')
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        pass

    # partial dislocation

    noises = loaded[:,0]
    tau_c_means = np.nanmean(loaded[:,1:11], axis=1)
    tau_c_stds = np.nanstd(loaded[:,1:11], axis=1)
    tau_c_mses = np.nanmean(loaded[:, 11:21], axis=1)
    deltaTaus = np.nanmean(loaded[:, 21:31], axis=1)
    
    point_1 = 10**(0)
    point_2 = 10**1.4

    fig, ax = makeNoisePlot(noises, tau_c_means, 0, point_1, point_2, y_error=deltaTaus)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    # ax.set_title(f"Perfect dislocation")
    ax.set_xlabel("$\\Delta R$")
    ax.set_ylabel("$ \\tau_c $")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def makePartialNoisePlot(res_root : Path, save_path):
    with open(results_root.joinpath("noise-data/partial-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        pass

    # partial dislocation

    noises = loaded[:,0]
    tau_c_means = np.nanmean(loaded[:,1:11], axis=1)
    tau_c_stds = np.nanstd(loaded[:,1:11], axis=1)
    tau_c_mses = np.nanmean(loaded[:, 11:21], axis=1)
    deltaTaus = np.nanmean(loaded[:, 21:31], axis=1)

    point_0 = 10**(-1.2)
    point_1 = 10**(0.25)
    point_2 = 10**(1.45)

    fig, ax = makeNoisePlot(noises, tau_c_means, point_0, point_1, point_2, color='red', fit_color='blue', y_error=deltaTaus)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)
    # ax.set_title(f"Partial dislocation")
    ax.set_xlabel("$\\Delta R$")
    ax.set_ylabel("$ \\tau_c $")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close()

def makeCommonNoisePlot(root_dir : Path):
    with open(results_root.joinpath("noise-data/perfect-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)

        noises = loaded[:,0]
        tau_c_means = np.nanmean(loaded[:,1:10], axis=1)
        tau_c_deltaTaus = np.nanmean(loaded[:, 21:31], axis=1)

        data_perfect = np.column_stack([
            noises,
            tau_c_means,
            tau_c_deltaTaus
        ])

    with open(results_root.joinpath("noise-data/partial-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        pass

        data_partial = np.column_stack([
            loaded[:,0],
            np.nanmean(loaded[:,1:11], axis=1),
            np.nanmean(loaded[:, 21:31], axis=1)
        ])

    fig, ax = plt.subplots(figsize=(linewidth/2, linewidth/2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    ax.errorbar(data_partial[:,0], data_partial[:,1], yerr=data_partial[:,2],
                 fmt='o', markersize=2, capsize=2, label="Partial", color='red', linewidth=0.2, zorder=0)
    
    ax.errorbar(data_perfect[:,0], data_perfect[:,1], yerr=data_perfect[:,2],
                 fmt='s', markersize=2, capsize=2, label="Perfect", color="blue", linewidth=0.2, zorder=0)
        
    ax.set_xlabel("R")
    ax.set_ylabel("$ \\tau_c $")
    ax.legend()
    fig.savefig(root_dir.joinpath("noise-plots/noise-tau_c-both-w-error.pdf"), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(linewidth/2, linewidth/2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    ax.plot(data_partial[:,0], data_partial[:,1],
                 'o', markersize=2, label="Partial", color='red', linewidth=0.2, zorder=0)
    
    ax.plot(data_perfect[:,0], data_perfect[:,1],
                 's', markersize=2, label="Perfect", color="blue", linewidth=0.2, zorder=0)
        
    ax.set_xlabel("R")
    ax.set_ylabel("$ \\tau_c $")
    ax.legend()
    fig.savefig(root_dir.joinpath("noise-plots/noise-tau_c-both.pdf"), bbox_inches='tight')
    pass

def makeDislocationPlots(folder):
    # First plot all perfect dislocations
    path_partial = Path(folder).joinpath("single-dislocation/dislocations-last")
    dest_folder = Path(folder).joinpath("single-dislocation/dislocations-last-pictures")
    print("Making perfect dislocation plots.")
    for n_noise, noise_path in enumerate([p for p in path_partial.iterdir() if p.is_dir()]):
        noise = noise_path.name.split("-")[1]
        noise = float(noise)

        if noise != 0.1:
            continue

        for seed_path in noise_path.iterdir():
            seed = seed_path.name.split("-")[1]
            seed = int(seed)

            if seed != 0:
                continue

            for dislocation_file in seed_path.iterdir():
                try:
                    loaded = np.load(dislocation_file)
                except:
                    print(f"File {dislocation_file} is somwhow corrupted.")
                    continue
                parameters = loaded['parameters']

                bigN, length, time, dt, deltaR, bigB, smallB,  cLT1, mu, tauExt, d0, seed, tau_cutoff =  parameters
                print(f"making plot for perfect dislocation seed {seed}, noise {noise}, tau_ext {tauExt:.4f}")
                bigN = int(bigN)
                length = int(length)
                seed = int(seed)

                y = loaded['y']

                x = np.linspace(0, length, bigN)

                np.random.seed(seed)
                stressField = np.random.normal(0,deltaR,[bigN, 2*bigN])

                plt.clf()
                plt.figure(figsize=(linewidth/2, linewidth/2))

                min_y = int(math.floor(min(y)) - 1)
                max_y = int(math.ceil(max(y)) + 1)

                if f"{tauExt*1e3:.2f}" == "22.32": # Select one dislocation to be included in thesis
                    print(f"Found selected tau w/ {tauExt}")
                    ylim_min = 0
                    ylim_max = 50
                    plt.ylim(ylim_min,ylim_max)
                    relevantPart = stressField[:,ylim_min:ylim_max]*10
                    plt.imshow(np.transpose(relevantPart), extent=[0, 1024, ylim_min, ylim_max], origin='lower', aspect='auto', label="stress field")
                    dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.4f}/perfect-{tauExt*1e3}-1e-3-tau-dislocation-R-{deltaR:.4f}-seed-{seed}.pdf")
                else:
                    relevantPart = stressField[:,min_y:max_y]*10
                    plt.imshow(np.transpose(relevantPart), extent=[0, 1024, min_y, max_y], origin='lower', aspect='auto', label="stress field")                
                    dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.4f}/{tauExt*1e3}-1e-3-tau-dislocation.pdf")

                plt.xlabel("$x$")
                plt.ylabel("$y(x)$")

                plt.plot(x, y, color='blue')

                dest_file.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(dest_file, dpi=600, bbox_inches='tight')

                plt.close()
        pass

    # Next plot all partial dislocations
    path_perfect = Path(folder).joinpath("partial-dislocation/dislocations-last")
    dest_folder = Path(folder).joinpath("partial-dislocation/dislocations-last-pictures")
    print("Making partial dislocation plots.")
    for n_noise, noise_path in enumerate([p for p in path_perfect.iterdir() if p.is_dir()]):
        noise = noise_path.name.split("-")[1]

        noise = float(noise)

        if noise != 0.1:
            continue

        for n,seed_folder in enumerate(noise_path.iterdir()):
            seed = seed_folder.name.split("-")[1]
            seed = int(seed)
            print(seed)

            if seed != 0:
                continue

            for dislocation_file in seed_folder.iterdir():
                loaded = np.load(dislocation_file)
                parameters = loaded['parameters']

                bigN, length, time, dt, deltaR, bigB, smallB, b_p,  cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff =  parameters
                print(f"making plot for partial dislocation seed {seed}, noise {noise}, tau_ext {tauExt:.4f}")
                bigN = int(bigN)
                length = int(length)
                seed = int(seed)

                y1 = loaded['y1']
                y2 = loaded['y2']

                x = np.linspace(0, length, bigN)
                y1 = y1[0]
                y2 = y2[0]

                np.random.seed(seed)
                stressField = np.random.normal(0,deltaR,[bigN, 2*bigN])

                plt.clf()
                plt.figure(figsize=(linewidth/2, linewidth/2))

                min_y = min([
                    int(math.floor(min(y1)) - 1), int(math.floor(min(y2)) - 1)
                    ])
                max_y = max([
                    int(math.ceil(max(y1)) + 1), int(math.ceil(max(y2)) + 1)
                ])
                if f"{tauExt*1e3:.2f}" == "22.26":  # Select one dislocation to be included in thesis
                    print(f"Found selected tau w/ {tauExt}")
                    plt.ylim(30,80)

                    relevantPart = stressField[:,30:80]*10
                    plt.imshow(np.transpose(relevantPart), extent=[0, 1024, 30, 80], origin='lower', aspect='auto', label="stress field")
                    dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.4f}/partial-{tauExt*1e3}-1e-3-tau-dislocation-R-{deltaR:.4f}-seed-{seed}.pdf")
                else:
                    relevantPart = stressField[:,min_y:max_y]*10
                    plt.imshow(np.transpose(relevantPart), extent=[0, 1024, min_y, max_y], origin='lower', aspect='auto', label="stress field")
                    dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.4f}/{tauExt*1e3}-1e-3-tau-dislocation.pdf")


                plt.xlabel("$x$")
                plt.ylabel("$y(x)$")

                plt.plot(x, y1, color='blue')
                plt.plot(x, y2, color='red')
                plt.plot(x, (y1 + y2)/2, color='yellow', linestyle='--')


                dest_file.parent.mkdir(exist_ok=True, parents=True)
                plt.savefig(dest_file, dpi=600, bbox_inches='tight')
                plt.close()
            pass
        pass
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
    parser.add_argument('--rearrange', help="Rearrange roughness data by tau instead of seed.", action="store_true")
    parser.add_argument('--analyze-hurst-exponent', help="Compute roughness exponents and store them separately, them plot them (only perfect dislocation)", action="store_true")

    parsed = parser.parse_args()

    results_root = Path(parsed.folder)

    if parsed.all or parsed.avg:
        makeAveragedDepnningPlots(parsed.folder)
        # makeAveragedDepnningPlots(parsed.folder, opt=True)
        pass
        
    if parsed.all or parsed.roughness:
        print("Making roughness plots. (w/o averaging by default)")
            # for partial dislocations
        try:
            print("Making partial dislocation roughness plots.")
            makePartialRoughnessPlots(parsed.folder)
        except FileNotFoundError:
            print("No partial roughness data skipping.")

        try:
            print("Making perfect dislocation roughness plots.")
            makePerfectRoughnessPlots(parsed.folder)
        except FileNotFoundError:
            print("No perfect roughness data skipping.")

        try:
            analyzeRoughnessFitParamteters(parsed.folder)
        except FileNotFoundError:
            print("No roughness fit parameters data skipping.")
        
        pass
    
    if parsed.all or parsed.analyze_hurst_exponent:
        path = Path(parsed.folder).joinpath("roughness_parameters_perfect.npz")
        path2 = Path(parsed.folder).joinpath("roughness_parameters_partial.npz")

        if not ( path.exists() or path2.exists() ):
            makeRoughnessExponentDataset_perfect(parsed.folder)
            makeRoughnessExponentDataset_partial(parsed.folder)
            print("Making roughness dataset.")
        
        processRoughnessFitParamData(path, Path(parsed.folder), Path(parsed.folder).joinpath("perfect-correlations"), perfect=True)
        processRoughnessFitParamData(path2, Path(parsed.folder), Path(parsed.folder).joinpath("partial-correlations"), perfect=False)

    if parsed.all or parsed.avg_roughness:
        averageRoughnessBySeed(parsed.folder)

    # Plots dislocations at the end of simulation
    if parsed.all or parsed.dislocations:
        t0 = time.time()
        makeDislocationPlots(parsed.folder)
        t1 = time.time()
        print(f"Time taken to make dislocation plots: {t1-t0:.2f} seconds")

    # Make normalized depinning plots
    if parsed.all or parsed.np:
        print("Making normalized depinning plots.")
        partial_depinning_dumps = results_root.joinpath("partial-dislocation").joinpath("depinning-dumps")
        if partial_depinning_dumps.exists():
            partial_data = normalizedDepinnings(
                partial_depinning_dumps,
                plot_save_folder=results_root.joinpath("partial-dislocation/normalized-plots"),
                data_save_path=results_root.joinpath("noise-data/partial-noises.csv"),
                json_save_path=results_root.joinpath("binning-data/tau_c_partial.json")
            )
        else:
            print("No partial disloation depinning dumps.")
            partial_data = {}
        
        perfect_dumps = results_root.joinpath("single-dislocation").joinpath("depinning-dumps")
        if perfect_dumps.exists():
            non_partial_data = normalizedDepinnings(
                perfect_dumps,
                plot_save_folder=results_root.joinpath("single-dislocation/normalized-plots"),
                data_save_path=results_root.joinpath("noise-data/perfect-noises.csv"),
                json_save_path=results_root.joinpath("binning-data/tau_c_perfect.json")
            )
        else:
            print("No perfect dislocation depinning dumps found. Skipping perfect depinning normalization.")
            non_partial_data = {}

        with open(Path(results_root).joinpath("global_data_dump.json"), "w") as fp:
            json.dump({"perfect_data":non_partial_data, "partial_data":partial_data}, fp, indent=2)
        
    if parsed.all or parsed.binning:
        p = Path(results_root).joinpath("global_data_dump.json")

        if not p.exists():
            raise Exception("--np must be called before this one.")
        
        with open(Path(results_root).joinpath("global_data_dump.json"), "r") as fp:
            data = json.load(fp)

        binning(data, results_root, conf_level=parsed.confidence)

    if parsed.all or parsed.noise:
        Path(parsed.folder).joinpath("noise-plots").mkdir(parents=True, exist_ok=True)
        try:
            print(f"Make noise plot from partial dislocation data")
            makePartialNoisePlot(Path(parsed.folder),
                Path(parsed.folder).joinpath("noise-plots/noise-tau_c-partial.pdf")
            )
        except Exception as e:
            print("No partial dislocation depinning dumps found. Skipping partial noise plot.")
            print(e)

        try:
            print("Making noise plot from perfect dislocation data")
            makePerfectNoisePlot(Path(parsed.folder), 
                Path(parsed.folder).joinpath("noise-plots/noise-tau_c-perfect.pdf")
            )
        except FileNotFoundError:
            print("No perfect dislocation depinning dumps found. Skipping perfect noise plot.")

        try:    
            makeCommonNoisePlot(Path(parsed.folder))
        except FileNotFoundError:
            print("No depinning dumps found. Skipping common noise plot.")
        
    if parsed.rearrange or parsed.all:
        rearrangeRoughnessDataByTau(parsed.folder)