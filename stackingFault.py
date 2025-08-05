import pickle
import pandas as pd
import numpy as np
from src.core.partialDislocation import PartialDislocationsSimulation
from pathlib import Path

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
                        
            file_name = f"{np.log(noise):.4f}_noise_stacking_fault_{seed}.csv"
            file_path = out_folder / file_name
            
            df.to_csv(file_path)
    pass

if __name__ == "__main__":
    generateAllSFDatasets('results/01-08-2025-depinning/partial/l-64-d0-2.0/depinning-pickle-dumps')
    pass