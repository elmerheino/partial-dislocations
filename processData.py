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

def makePartialNoisePlot(res_root : Path, save_path):
    with open(results_root.joinpath("noise-data/partial-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        pass

    # partial dislocation

    noises = loaded[:,0]
    tau_c_means = np.nanmean(loaded[:,1:10], axis=1)
    tau_c_stds = np.nanstd(loaded[:,1:10], axis=1)

    plt.figure(figsize=(linewidth/2, linewidth/2))
    plt.errorbar(noises, tau_c_means, yerr=tau_c_stds, fmt='o', markersize=2, capsize=2, color='red', linewidth=0.5, zorder=0)

    data_partial = np.array(list(zip(noises, tau_c_means)))
    sorted_args = np.argsort(data_partial[:,0])
    data_partial = data_partial[sorted_args]

    # Mask away nan values
    mask = ~np.isnan(data_partial[:,1])
    data_partial = data_partial[mask]

    point_0 = 10**(-1.2)
    point_1 = 10**(0.25)
    point_2 = 10**(1.45)
    
    # Fit power law to first region of data
    try:
        region1_index = np.where(data_partial[:,0] > point_1)[0][0]
        region0_index = np.where(data_partial[:,0] > point_0)[0][0]

        x = data_partial[region0_index:region1_index,0]
        y = data_partial[region0_index:region1_index,1]

        print(region1_index, x.shape, y.shape)
        fit_params, pcov = optimize.curve_fit(
            lambda x, a, b: a*x**b, 
            x, y)
        
        fit_x = np.linspace(data_partial[region0_index,0], data_partial[region1_index,0], 100)
        fit_y = fit_params[0]*fit_x**fit_params[1]

        plt.plot(fit_x, fit_y, label=f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", color='blue', linewidth=2, zorder=1)
        print(f"Partial dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except:
        print("No data in first region of data.")
        region1_index = 0

    # Fit power law to second region of data
    try:
        region1_end = np.where(data_partial[:,0] > point_2)[0][0]

        x = data_partial[region1_index:region1_end,0]
        y = data_partial[region1_index:region1_end,1]

        fit_params, pcov = optimize.curve_fit(
            lambda x, a, b: a*x**b,
            x, y, maxfev=3000)
        
        fit_x = np.linspace(data_partial[region1_index,0], data_partial[region1_end,0], 100)
        fit_y = fit_params[0]*fit_x**fit_params[1]
        plt.plot(fit_x, fit_y, label=f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", color='black', linestyle='--', linewidth=2, zorder=1)
        print(f"Partial dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except:
        print("No data in second region of data.")

    # Fit power law to third region of data
    x = data_partial[region1_end:,0]
    y = data_partial[region1_end:,1]

    fit_params,_ = optimize.curve_fit(
        lambda x, a, b: a*x**b,
        x, y)
    fit_x = np.linspace(data_partial[region1_end,0], data_partial[-1,0], 100)
    fit_y = fit_params[0]*fit_x**fit_params[1]
    plt.plot(fit_x, fit_y, label=f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", color='blue', linestyle=':', linewidth=2, zorder=1)
    print(f"Partial dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")


    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.title(f"Partial dislocation")
    # plt.xticks([point_0, point_1, point_2], labels=["$10^{-1.2}$", "$10^{0.0}$", "$10^{1.7}$"])
    plt.legend()
    plt.xlabel("$\\Delta R$")
    plt.ylabel("$ \\tau_c $")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def makeNoisePlot(noises, tau_c_means, tau_c_stds, point_1, point_2):
    fig, ax = plt.subplots(figsize=(linewidth/2, linewidth/2))

    ax.errorbar(noises, tau_c_means, yerr=tau_c_stds, fmt='s', markersize=2, capsize=2, color='blue', linewidth=0.5, zorder=0)

    data = np.array(list(zip(noises, tau_c_means)))
    sorted_args = np.argsort(data[:,0])
    data = data[sorted_args]

    # Fit power law to first region of data

    try:
        region1_index = np.where(data[:,0] > point_1)[0][0]
        x = data[0:region1_index,0]
        y = data[0:region1_index,1]
        print(region1_index, x.shape, y.shape)
        fit_params, pcov = optimize.curve_fit(
            lambda x, a, b: a*x**b,
            x, y)
        fit_x = np.linspace(data[0,0], data[region1_index,0], 100)
        fit_y = fit_params[0]*fit_x**fit_params[1]
        ax.plot(fit_x, fit_y, label=f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", color='red', linewidth=2)
        print(f"Perfect dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except:
        print("No data in first region of data.")

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
        ax.plot(fit_x, fit_y, label=f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", color='black', linestyle='--', linewidth=2)
        print(f"Perfect dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
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
        ax.plot(fit_x, fit_y, label=f"$ \\tau_c \\propto R^{{{fit_params[1]:.3f} }}$", color='red', linestyle=':', linewidth=2)
        print(f"Perfect dislocation fit: {fit_params[0]:.3f} * R^{fit_params[1]:.3f} on interval {min(x)} to {max(x)}")
    except:
        print("No data in third region of data.")
    
    return fig, ax

def makePerfectNoisePlot(results_root : Path, save_path):
    with open(results_root.joinpath("noise-data/perfect-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        pass

    # partial dislocation

    noises = loaded[:,0]
    tau_c_means = np.mean(loaded[:,1:10], axis=1)
    tau_c_stds = np.std(loaded[:,1:10], axis=1)

    point_1 = 10**(-0)
    point_2 = 10**1.4

    fig, ax = makeNoisePlot(noises, tau_c_means, tau_c_stds, point_1, point_2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True)

    ax.set_title(f"Perfect dislocation")
    ax.set_xlabel("$\\Delta R$")
    ax.set_ylabel("$ \\tau_c $")

    fig.legend()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def makeCommonNoisePlot(root_dir : Path):
    with open(results_root.joinpath("noise-data/perfect-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)

        noises = loaded[:,0]
        tau_c_means = np.mean(loaded[:,1:10], axis=1)

        data_perfect = np.column_stack([
            noises,
            tau_c_means
        ])

    with open(results_root.joinpath("noise-data/partial-noises.csv"), "r") as fp:
        loaded = np.genfromtxt(fp, delimiter=',', skip_header=1)
        
        pass

        data_partial = np.column_stack([
            loaded[:,0],
            np.mean(loaded[:,1:10], axis=1)
        ])

    plt.figure(figsize=(linewidth/2, linewidth/2))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)

    plt.errorbar(data_partial[:,0], data_partial[:,1], fmt='o', markersize=2, capsize=2, label="Partial", color='red', linewidth=0.2, zorder=0)
    plt.errorbar(data_perfect[:,0], data_perfect[:,1], fmt='s', markersize=2, capsize=2, label="Perfect", color="blue", linewidth=0.2, zorder=0)
        
    plt.title("Noise magnitude and external force")
    plt.xlabel("R")
    plt.ylabel("$ \\tau_c $")
    plt.legend()
    plt.savefig(root_dir.joinpath("noise-plots/noise-tau_c-both.png"), dpi=300, bbox_inches='tight')
    pass

def makeDislocationPlots(folder):
    # First plot all perfect dislocations
    path = Path(folder).joinpath("single-dislocation/dislocations-last")
    dest_folder = Path(folder).joinpath("single-dislocation/dislocations-last-pictures")
    print("Making perfect dislocation plots.")
    for noise_path in [p for p in path.iterdir() if p.is_dir()]:
        noise = noise_path.name.split("-")[1]
        noise = float(noise)

        # if noise != 1:
        #     continue

        for seed_path in noise_path.iterdir():
            seed = seed_path.name.split("-")[1]
            seed = int(seed)

            # if seed != 1:
            #     continue

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

                relevantPart = stressField[:,min_y:max_y]*10
                # print(relevantPart.shape, min_y, max_y)

                # interpolated = np.array([
                #     np.interp(np.linspace(0, relevantPart.shape[1], 1000), np.arange(relevantPart.shape[1]), relevantPart[row, :]) for row in range(relevantPart.shape[0])
                # ])
                plt.imshow(np.transpose(relevantPart), extent=[0, 1024, min_y, max_y], origin='lower', aspect='auto', label="stress field")

                plt.xlabel("$x$")
                plt.ylabel("$y(x)$")

                plt.plot(x, y, color='blue')


                dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.4f}/")
                dest_file.mkdir(exist_ok=True, parents=True)
                dest_file = dest_file.joinpath(f"dislocation-tau-{tauExt}.png")

                plt.savefig(dest_file, dpi=600, bbox_inches='tight')
                plt.close()
        pass

    # Next plot all partial dislocations
    path = Path(folder).joinpath("partial-dislocation/dislocations-last")
    dest_folder = Path(folder).joinpath("partial-dislocation/dislocations-last-pictures")
    print("Making partial dislocation plots.")
    for noise_path in [p for p in path.iterdir() if p.is_dir()]:
        noise = noise_path.name.split("-")[1]

        noise = float(noise)
        # if noise != 1:
        #     continue

        for seed_folder in noise_path.iterdir():
            seed = seed_folder.name.split("-")[1]
            seed = int(seed)

            # if seed != 1:
            #     continue

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

                relevantPart = stressField[:,min_y:max_y]*10

                # interpolated = np.array([
                #     np.interp(np.linspace(0, relevantPart.shape[1], 1000), np.arange(relevantPart.shape[1]), relevantPart[row, :]) for row in range(relevantPart.shape[0])
                # ])
                plt.imshow(np.transpose(relevantPart), extent=[0, 1024, min_y, max_y], origin='lower', aspect='auto', label="stress field")

                plt.xlabel("$x$")
                plt.ylabel("$y(x)$")

                # plt.ylim(0,50)

                plt.plot(x, y1, color='blue')
                plt.plot(x, y2, color='red')
                plt.plot(x, (y1 + y2)/2, color='yellow', linestyle='--')


                dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.3f}/")
                dest_file.mkdir(exist_ok=True, parents=True)
                dest_file = dest_file.joinpath(f"dislocation-tau-{tauExt}.png")

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
    parser.add_argument('--analyse-roughness', help="Analyse roughness fit parameters.", action="store_true")
    parser.add_argument('--analyze-hurst-exponent', help="Compute roughness exponents and store them separately, them plot them (only perfect dislocation)", action="store_true")

    parsed = parser.parse_args()

    results_root = Path(parsed.folder)

    if parsed.all or parsed.avg:
        makeAveragedDepnningPlots(parsed.folder)
        # makeAveragedDepnningPlots(parsed.folder, opt=True)
        pass
        
    if parsed.all or parsed.roughness:
        print("Making roughness plots. (w/o averaging by default)")
        makeAvgRoughnessPlots(parsed.folder)
    
    if parsed.all or parsed.analyze_hurst_exponent:
        path = Path(parsed.folder).joinpath("roughness_exponents_perfect.npz")
        if not path.exists():
            makeRoughnessExponentDataset(parsed.folder)
            print("Making roughness dataset.")
        processExponentData(Path(parsed.folder))
    
    if parsed.all or parsed.analyse_roughness:
        analyzeRoughnessFitParamteters(parsed.folder)

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

        # try:
        partial_data = normalizedDepinnings(
            results_root.joinpath("partial-dislocation").joinpath("depinning-dumps"),
            plot_save_folder=results_root.joinpath("partial-dislocation/normalized-plots"),
            data_save_path=results_root.joinpath("noise-data/partial-noises.csv"),
            json_save_path=results_root.joinpath("binning-data/tau_c_partial.json")
        )
        # except Exception as e:
            # print("No partial dislocation depinning dumps found. Skipping partial depinning normalization.")
            # print(e)
            # partial_data = {}
        
        try:
            non_partial_data = normalizedDepinnings(
                results_root.joinpath("single-dislocation").joinpath("depinning-dumps"),
                plot_save_folder=results_root.joinpath("single-dislocation/normalized-plots"),
                data_save_path=results_root.joinpath("noise-data/perfect-noises.csv"),
                json_save_path=results_root.joinpath("binning-data/tau_c_perfect.json")
            )
        except Exception as e:
            print("No perfect dislocation depinning dumps found. Skipping perfect depinning normalization.")
            print(e)
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
            makePartialNoisePlot(Path(parsed.folder),
                Path(parsed.folder).joinpath("noise-plots/noise-tau_c-partial.png")
            )
        except Exception as e:
            print("No partial dislocation depinning dumps found. Skipping partial noise plot.")
            print(e)

        try:
            makePerfectNoisePlot(Path(parsed.folder), 
                Path(parsed.folder).joinpath("noise-plots/noise-tau_c-perfect.png")
            )
        except FileNotFoundError:
            print("No perfect dislocation depinning dumps found. Skipping perfect noise plot.")
        try:    
            makeCommonNoisePlot(Path(parsed.folder))
        except FileNotFoundError:
            print("No depinning dumps found. Skipping common noise plot.")
        
    if parsed.rearrange or parsed.all:
        rearrangeRoughnessDataByTau(parsed.folder)