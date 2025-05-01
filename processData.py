#!/usr/bin/env python3

import math
from pathlib import Path
import json
from plots import *
import numpy as np
from scipy import optimize
from scipy import stats
from argparse import ArgumentParser
import multiprocessing as mp
from functools import partial

from roughnessPlots import *
from velocityPlots import *


def makeNoisePlot(tau_c_path, save_path, title):
    print("Making noise plots")
    with open(tau_c_path, "r") as fp:
        loaded = json.load(fp)
    
    # partial dislocation
    partial_data = loaded

    noises = list()
    tau_cs = list()

    for noise in partial_data.keys():
        tau_c = partial_data[noise]
    
        tau_c = sum(tau_c)/len(tau_c)

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

def makeCommonPlot(root_dir : Path):
    tau_c_perfect = root_dir.joinpath("single-dislocation/normalized-plots/tau_c.json")
    tau_c_partial = root_dir.joinpath("partial-dislocation/normalized-plots/tau_c.json")

    data_partial = list()
    data_perfect = list()
    with open(tau_c_partial, "r") as fp:
        tau_c_partial = json.load(fp)
        for noise in tau_c_partial.keys():
            noise_value = float(noise)
            tau_c_partial_values = tau_c_partial[noise]

            datapoint_partial = (noise_value, np.mean(tau_c_partial_values))
            data_partial.append(datapoint_partial)
    
    with open(tau_c_perfect, "r") as fp:
        tau_c_perfect = json.load(fp)
        for noise in tau_c_perfect.keys():
            noise_value = float(noise)
            tau_c_perfect_values = tau_c_perfect[noise]

            datapoint_perfect = (noise_value, np.mean(tau_c_perfect_values))
            data_perfect.append(datapoint_perfect)
    
    data_partial = np.array(data_partial)
    data_perfect = np.array(data_perfect)

    plt.figure(figsize=(8,8))
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.scatter(data_partial[:,0], data_partial[:,1], marker='x', label="Partial dislocation", color='red')
    plt.scatter(data_perfect[:,0], data_perfect[:,1], marker='x', label="Perfect dislocation", color="blue")
    plt.title("Noise magnitude and external force")
    plt.xlabel("R")
    plt.ylabel("$ \\tau_c $")
    plt.legend()
    plt.savefig(root_dir.joinpath("noise-plots/noise-tau_c-both.png"), dpi=300)
    pass

def makeDislocationPlots(folder):
    # First plot all perfect dislocations
    path = Path(folder).joinpath("single-dislocation/dislocations-last")
    dest_folder = Path(folder).joinpath("single-dislocation/dislocations-last-pictures")
    for noise_path in path.iterdir():
        for seed_path in noise_path.iterdir():
            for dislocation_file in seed_path.iterdir():
                loaded = np.load(dislocation_file)
                parameters = loaded['parameters']

                bigN, length, time, dt, deltaR, bigB, smallB,  cLT1, mu, tauExt, d0, seed, tau_cutoff =  parameters
                bigN = int(bigN)
                length = int(length)
                seed = int(seed)

                y = loaded['y']

                x = np.linspace(0, length, bigN)
                y = y[0]

                np.random.seed(seed)
                stressField = np.random.normal(0,deltaR,[bigN, 2*bigN])

                plt.clf()
                plt.figure(figsize=(8,8))

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
                dest_file = dest_file.joinpath("dislocation.png")

                plt.savefig(dest_file, dpi=300)
                plt.close()
        pass

    path = Path(folder).joinpath("partial-dislocation/dislocations-last")
    dest_folder = Path(folder).joinpath("partial-dislocation/dislocations-last-pictures")
    # Next plot all partial dislocations
    for noise_path in path.iterdir():
        for seed_folder in noise_path.iterdir():
            for dislocation_file in seed_folder.iterdir():
                loaded = np.load(dislocation_file)
                parameters = loaded['parameters']

                bigN, length, time, dt, deltaR, bigB, smallB, b_p,  cLT1, cLT2, mu, tauExt, c_gamma, d0, seed, tau_cutoff =  parameters
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
                plt.figure(figsize=(8,8))

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

                plt.plot(x, y1, color='blue')
                plt.plot(x, y2, color='red')


                dest_file = dest_folder.joinpath(f"noise-{deltaR:.4f}/seed-{seed:.3f}/")
                dest_file.mkdir(exist_ok=True, parents=True)
                dest_file = dest_file.joinpath("dislocation.png")

                plt.savefig(dest_file, dpi=300)
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

    parsed = parser.parse_args()

    results_root = Path(parsed.folder)

    if parsed.all or parsed.avg:
        makeAveragedDepnningPlots(parsed.folder)
        # makeAveragedDepnningPlots(parsed.folder, opt=True)
        pass
        
    if parsed.all or parsed.roughness:
        print("Making roughness plots. (w/o averaging by default)")
        makeAvgRoughnessPlots(parsed.folder)

    if parsed.all or parsed.avg_roughness:
        averageRoughnessBySeed(parsed.folder)

    # Plots dislocations at the end of simulation
    if parsed.all or parsed.dislocations:
        makeDislocationPlots(parsed.folder)

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
        
        # partial_data_opt = normalizedDepinnings(
        #     results_root.joinpath("partial-dislocation").joinpath("optimal-depinning-dumps"),
        #     save_folder=results_root.joinpath("partial-dislocation/normalized-plots-opt")
        # )
        
        # non_partial_data_opt = normalizedDepinnings(
        #     results_root.joinpath("single-dislocation").joinpath("optimal-depinning-dumps"),
        #     save_folder=results_root.joinpath("single-dislocation/normalized-plots-opt")
        # )

        # with open(Path(results_root).joinpath("global_data_dump_opt.json"), "w") as fp:
        #     json.dump({"perfect_data":non_partial_data_opt, "partial_data":partial_data_opt}, fp, indent=2)

    if parsed.all or parsed.binning:
        p = Path(results_root).joinpath("global_data_dump.json")

        if not p.exists():
            raise Exception("--np must be called before this one.")
        
        with open(Path(results_root).joinpath("global_data_dump.json"), "r") as fp:
            data = json.load(fp)

        binning(data, results_root, conf_level=parsed.confidence)

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
        makeCommonPlot(Path(parsed.folder))
        # makeNoisePlot(
        #     Path(parsed.folder).joinpath("partial-dislocation/normalized-plots-opt/tau_c.json"),
        #     save_path=Path(parsed.folder).joinpath("noise-plots/noise-tau_c-partial-opt.png"),
        #     title="Noise magnitude and external force for partial dislocation from closeup data"
        #     )
        # makeNoisePlot(
        #     Path(parsed.folder).joinpath("single-dislocation/normalized-plots-opt/tau_c.json"),
        #     save_path=Path(parsed.folder).joinpath("noise-plots/noise-tau_c-perfect-opt.png"),
        #     title="Noise magnitude and external force for perfect dislocation from closeup data"
        #     )
        
    if parsed.rearrange or parsed.all:
        rearrangeRoughnessDataByTau(parsed.folder)