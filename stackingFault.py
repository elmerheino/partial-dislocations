import pickle
import pandas as pd
import numpy as np
from src.core.partialDislocation import PartialDislocationsSimulation
from pathlib import Path
import matplotlib.pyplot as plt

def extractStackingFaultsFromPickle(pickle_path):
    """
    Extract the velocity histories from a depinning to a pandas data frame which is returned.
    """

    with open(pickle_path, "rb") as fp:
        data = pickle.load(fp)
    
    sf_hists = [r['sf_hist'] for r in data]
    parameters = [ PartialDislocationsSimulation.paramListToDict(r['params']) for r in data]

    sim_tim = parameters[0]['time']
    sim_dt = parameters[0]['dt']
    sf_0 = sf_hists[0]

    columns = ["tau_ext"]
    columns.extend(np.linspace(0, len(sf_0)*sim_dt, len(sf_0)))

    rows = list()
    for sf_hist, params in zip(sf_hists, parameters):
        tau_ext = params['tauExt']
        row = [tau_ext]
        row.extend(sf_hist)
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=columns)
    df.attrs = {
        'deltaR' : parameters[0]['deltaR'],
        'seed' : parameters[0]['seed']
    }
    
    return df

def generateAllSFDatasets(pickle_folder_path):
    """    Generates stacking fault datasets from pickle files in a given folder.
        Args:
            pickle_folder_path (str): Path to the folder containing the pickle files of the simulations.

        This function reads pickle files from the specified folder, extracts stacking fault data,
        and saves it into a new directory named "stacking-fault-datasets" located in the parent
        directory of `pickle_folder_path`.
        Each dataset is saved as a CSV file with the following format:

        +--------------+--------------------+--------------------+----------+--------------------+
        |   tau_ext    |       time_0       |       time_1       |   ...    |       time_N       |
        +--------------+--------------------+--------------------+----------+--------------------+
        | <stress_val> | <sf_width_at_t0>   | <sf_width_at_t1>   |   ...    | <sf_width_at_tN>   |
        | <stress_val> | <sf_width_at_t0>   | <sf_width_at_t1>   |   ...    | <sf_width_at_tN>   |
        |      ...     |        ...         |        ...         |   ...    |        ...         |
        +--------------+--------------------+--------------------+----------+--------------------+

        The filename is generated based on the noise level (deltaR) and the seed value
        associated with the simulation.  The noise is log transformed for the filename.
    """ 

    out_folder = Path(pickle_folder_path).parent / "stacking-fault-datasets"
    out_folder.mkdir(exist_ok=True)
    
    for file in Path(pickle_folder_path).iterdir():
        if file.suffix == ".pickle":
            df = extractStackingFaultsFromPickle(file)
            noise = df.attrs['deltaR']
            seed = df.attrs['seed']
                        
            file_name = f"{np.log10(noise)}_noise_stacking_fault_{seed}.csv"
            file_path = out_folder / file_name
            
            df.to_csv(file_path)
    pass

def generateSFvsVRelPlots(stacking_fault_datas, output_folder=None):
    stacking_fault_datas = Path(stacking_fault_datas)

    if type(output_folder) == type(None):
        output_folder = stacking_fault_datas.parent.joinpath("plots/sf_width-at-last10")
        output_folder.mkdir(exist_ok=True, parents=True)

    for csv_path in Path(stacking_fault_datas).iterdir():
        df = pd.read_csv(csv_path)

        deltaR = float(csv_path.name.split("_")[0])     # Log deltaR

        fig, ax = plt.subplots(figsize=(5,5))
        
        last_10 = int(len(df.iloc[:, 0][1:])/10)

        mean_sf = df.iloc[:, -last_10:].mean(axis=1)
        std_sf = df.iloc[:, -last_10:].std(axis=1)

        ax.scatter(df['tau_ext'], mean_sf, marker='.')
        ax.fill_between(df['tau_ext'], mean_sf-std_sf, mean_sf+std_sf, color='blue', alpha=0.2)

        ax.set_xlabel('$\\tau_{ext}$')
        ax.set_ylabel('SF Width when relaxed')
        ax.set_title(f"$\\Delta R = 10^{{{deltaR:.3f}}}$")
        ax.grid(True)

        save_path = output_folder.joinpath(f"{deltaR}_noise-rel-staking-fault.pdf")
        fig.savefig(save_path)
    pass


def generateSFt0Plots(stacking_fault_datas, output_folder=None):
    stacking_fault_datas = Path(stacking_fault_datas)

    if type(output_folder) == type(None):
        output_folder = stacking_fault_datas.parent.joinpath("plots/sf_width-t0_vs_v-rel")
        output_folder.mkdir(exist_ok=True, parents=True)

    for csv_path in Path(stacking_fault_datas).iterdir():
        df = pd.read_csv(csv_path, index_col=0)

        deltaR = float(csv_path.name.split("_")[0])     # Log deltaR

        fig, ax = plt.subplots(figsize=(5,5))
        
        print(df['tau_ext'].to_numpy().shape)
        print(df.iloc[:, -1].to_numpy().shape)

        sf_at_t0 = df.iloc[:, 1]

        ax.scatter(df['tau_ext'], sf_at_t0, marker='.')

        ax.set_xlabel('$\\tau_{ext}$')
        ax.set_ylabel('SF Width at t=0')
        ax.set_title(f"$\\Delta R = 10^{{{deltaR:.3f}}}$")
        ax.grid(True)

        save_path = output_folder.joinpath(f"{deltaR}_noise-rel-staking-fault.pdf")
        fig.savefig(save_path)
    pass

if __name__ == "__main__":
    # for i in ["2.0", "4.0", "8.0", "16.0", "32.0"]:
    #     generateAllSFDatasets(f'results/01-08-2025-depinning/partial/l-64-d0-{i}/depinning-pickle-dumps')
    #     generateSFvsVRelPlots(f'results/01-08-2025-depinning/partial/l-64-d0-{i}/stacking-fault-datasets')
    #     generateSFt0Plots(f'results/01-08-2025-depinning/partial/l-64-d0-{i}/stacking-fault-datasets')

    generateAllSFDatasets('results/01-08-2025-depinning/partial/l-512-d0-8.0/depinning-pickle-dumps')
    generateSFvsVRelPlots('results/01-08-2025-depinning/partial/l-512-d0-8.0/stacking-fault-datasets')
    pass