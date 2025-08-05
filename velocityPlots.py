#!/usr/bin/env python3

import os
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy import stats, optimize
from functools import partial
import csv
import matplotlib as mpl
from sklearn.cluster import KMeans
import pandas as pd

from src.core.partialDislocation import PartialDislocationsSimulation
from src.core.singleDislocation import DislocationSimulation
import sys
import argparse

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

    ax.set_xlabel("$( \\tau_{{ext}} - \\tau_{{c}} )/\\tau_{{c}}$")
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

def makeVelHistPlot(fig, ax, v, dt, tau_ext):
    time = np.arange(len(v)) * dt
    
    ax.plot(time, v, label=f"$\\tau_{{ext}} = {tau_ext}$")

    ax.axhline(y=0, color='black', linestyle='--')
    ax.grid(True)
    ax.set_xlabel('time')
    ax.set_ylabel('$v_{cm}$')
    fig.tight_layout()
    pass

def generateVelocityHistoryPlots(velocity_data, output_folder):
    """
    velocity_data : the path to dictionary containing velocity datasets
    output : output directory where the velocity plots are generated
    """
    velocity_data, output_folder = Path(velocity_data), Path(output_folder)
    for noise_dir in velocity_data.iterdir():
        for velocity_file in noise_dir.iterdir():
            df = pd.read_csv(velocity_file, sep=";", header=0, index_col=0)
            noise_val = float(velocity_file.name.split("_")[2])
            seed_str = (velocity_file.name.split("_")[4]).split(".")[0]
            
            for index, row in df.iterrows():
                tau_ext = row['tau_ext']
                velocities = row.iloc[1:].values
                time = df.columns[1:].to_numpy(dtype=float)

                fig, ax = plt.subplots(figsize=(linewidth, linewidth/2))
                makeVelHistPlot(fig, ax, velocities, 100, tau_ext)
                ax.set_title(f"$\\Delta R = {noise_val} $")
                # ax.plot(time, velocities, label=f'$\\tau_{{ext}} = {tau_ext}$')
                ax.legend()

                save_path =output_folder.joinpath(f"{np.log10(noise_val)}-_noise/{tau_ext}_tauext_s_{seed_str}.pdf")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                fig.savefig(save_path)
                plt.close()
            # print(df.head())
            pass
    pass

def extractVeclocitiesFromPickle(pickle_path):
    """
    Extract the velocity histories from a depinning to a pandas data frame which is returned.
    """

    with open(pickle_path, "rb") as fp:
        data = pickle.load(fp)
    
    v1_rel = [r['v1'] for r in data]
    v2_rel = [r['v2'] for r in data]
    v_cm_rel = [r['v_cm'] for r in data]
    l_ranges = [r['l_range'] for r in data]
    avg_w12s = [r['avg_w'] for r in data]
    y1_last = [r['y1_last'] for r in data]
    y2_last = [r['y2_last'] for r in data]
    v_cms = [r['v_cm_hist'] for r in data]
    sf_hists = [r['sf_hist'] for r in data]
    parameters = [ PartialDislocationsSimulation.paramListToDict(r['params']) for r in data]

    sim_tim = parameters[0]['time']
    sim_dt = parameters[0]['dt']
    hist_0 = v_cms[0]

    columns = ["tau_ext"]
    columns.extend(np.linspace(0, len(hist_0)*sim_dt, len(hist_0)))

    rows = list()
    for v_hist, params in zip(v_cms, parameters):
        tau_ext = params['tauExt']
        row = [tau_ext]
        row.extend(v_hist)
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=columns)
    df.attrs = {
        'deltaR' : parameters[0]['deltaR'],
        'seed' : parameters[0]['seed']
    }
    
    return df

def generateVelocityDatasets(path_to_pickle_dumps, out_folder):
    path_to_pickle_dumps = Path(path_to_pickle_dumps)
    out_folder = Path(out_folder)

    for pickle_path in path_to_pickle_dumps.iterdir():
        df = extractVeclocitiesFromPickle(pickle_path)
        fname = f"noise-{float(df.attrs['deltaR'])}/velocities_deltaR_{float(df.attrs['deltaR'])}_seed_{int(df.attrs['seed'])}.csv"
        save_path = out_folder.joinpath(fname)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, sep=";")
    pass

def extractDepinningFromVeclocity(path_to_velocity_data):
    """
    path_to_velocity_data should point to the folder where the velocity dataframes are located, usually named by the noise
    of the data in question.
    """
    path_to_velocity_data = Path(path_to_velocity_data)
    depinning_dataset = list()
    noise_key = float(path_to_velocity_data.name.split("-")[1])
    for vel_file in path_to_velocity_data.iterdir():
        noise_key = vel_file.name.split("_")[2]
        df = pd.read_csv(vel_file, sep=";", header=0, index_col=0)

        tau_ext = df["tau_ext"].values
        velocities = df.iloc[:, -int(df.shape[1] / 20):].mean(axis=1).values  # Average velocities over the last 10th of the time
        depinning_data = [[i,j] for i,j in zip(tau_ext, velocities)]

        # print(f"Processed depinning data from {vel_file}: {depinning_data}")
        depinning_dataset.extend(depinning_data)
    
    df = pd.DataFrame(depinning_dataset, columns=['tau_ext', 'v_rel'])
    df.attrs = {
        'deltaR' : noise_key
    }
    return df

def generateDepinningDatasets(path_to_velocities):
    """
    path to velocities should be the path pointing to velocity-datasets folder.
    """
    path_to_velocities = Path(path_to_velocities)
    for noise_folder in path_to_velocities.iterdir():
        noise_key = noise_folder.name.split("-")[1]
        depinning_df = extractDepinningFromVeclocity(noise_folder)
        
        deltaR = depinning_df.attrs['deltaR']
        save_path = path_to_velocities.parent.joinpath(f"depinning-datasets/depinning-noise-{deltaR}.csv")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        depinning_df.to_csv(save_path, sep=";")

def generateDepinningPlots(path_to_depinning, out_dir):
    """
    Generates depinning plots from depinning data. At the same time collects critical forces using a simple treshold
    method and returns them in a list containing datapoints.

    path_to_depinning:path to folder containing csv files with v_rel against tau_ext

    TODO: handle multiple seeds
    """
    path_to_depinning, out_dir = Path(path_to_depinning), Path(out_dir)
    critical_force_data = list()
    for depinning_csv in path_to_depinning.iterdir():
        df = pd.read_csv(depinning_csv, sep=";", header=0, index_col=0)
        noise = float((depinning_csv.name.split('-')[2]).split('.csv')[0])

        fig, ax = plt.subplots(figsize=(5,5))
        ax.set_title(f'$\\Delta R = {noise}$')
        ax.scatter(df['tau_ext'], df['v_rel'], marker='.')
        fig.tight_layout()

        velocity_threshold = 1e-6
        if len(df[df['v_rel'] < velocity_threshold]) > 0:
            last_pinned_index = df[df['v_rel'] < velocity_threshold].index[-1]
            critical_force = df.loc[last_pinned_index, 'tau_ext']
            ax.axvline(x=critical_force, color='r', linestyle='--', label=f'Critical Force (τc) ≈ {critical_force:.4f}')
        else:
            critical_force = np.nan
        
        critical_force_data.append((noise, critical_force))


        save_path = out_dir.joinpath(f"{np.log(noise):.3f}-10e-depinning.pdf")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_path)
        plt.close()
        pass
    return critical_force_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze dislocation depinning data.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Subparser for vanha_maini
    parser_vanha_maini = subparsers.add_parser('vanha_maini', help='Run vanha_maini function.')

    parser_velocity_dataset = subparsers.add_parser('velocity_dataset', help='Generate a velocity dataset from input pickcle.')
    parser_velocity_dataset.add_argument('pickle_folder', type=str, help='Path to the folder which contains the pickle dumps from depinnign simulations.')

    parser_depinning_dataset = subparsers.add_parser('depinning_dataset', help='Generate a depinning dataset from input velocity datas.')
    parser_depinning_dataset.add_argument('vel_folder', type=str, help='Path to folder containin velocity data.')

    parser_velocity_histories = subparsers.add_parser('hist', help='Generate velocity history plots')
    parser_velocity_histories.add_argument('vel_dataset', type=str,
        default="debug/test-run-further/deltaR_0.00030888435964774815-seed-3.0/velocity-datasets")
    parser_velocity_histories.add_argument('plot_dir', type=str,
        default="debug/test-run-further/plots/vel-history")
    
    parser_depinning_plots = subparsers.add_parser('depinning_plot', help='Generate depinning plots')
    parser_depinning_plots.add_argument('depinning_dataset', type=str)
    parser_depinning_plots.add_argument('plot_dir', type=str)

    args = parser.parse_args()

    if args.command == 'velocity_dataset':
        pickle_folder = Path(args.pickle_folder)
        generateVelocityDatasets(pickle_folder, pickle_folder.parent.joinpath("velocity-datasets"))
    elif args.command == 'depinning_dataset':
        generateDepinningDatasets(args.vel_folder)
    elif args.command == 'hist':
        generateVelocityHistoryPlots(args.vel_dataset, args.plot_dir)
    elif args.command == 'depinning_plot':
        critical_forces = generateDepinningPlots(args.depinning_dataset, args.plot_dir)
        save_path = Path(args.depinning_dataset).parent.joinpath("critical_forces.csv")
        df = pd.DataFrame(critical_forces, columns=['noise', 'critical_force'])
        df.to_csv(save_path, sep=";")
        
    else:
        parser.print_help()