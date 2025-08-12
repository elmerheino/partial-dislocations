import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.core.partialDislocation import PartialDislocationsSimulation
from pathlib import Path
import argparse

def extractYshapesFromPickle(pickle_path):
    with open(pickle_path, "rb") as fp:
        data = pickle.load(fp)
    
    selected_y1 = [r['selected_y1'] for r in data] # Each elem is of the form [timestamp, y1, y2, ..., yN] 
    selected_y2 = [r['selected_y2'] for r in data]

    parameters = [ PartialDislocationsSimulation.paramListToDict(r['params']) for r in data]

    list_of_dicts = list()
    for i_y1, i_y2, p_i in zip(selected_y1, selected_y2, parameters):
        columns = ['time'] + list(range(1,int(parameters[0]['bigN'])+1))
        df_y1 = pd.DataFrame(i_y1, columns=columns)
        df_y2 = pd.DataFrame(i_y2, columns=columns)
        params_df = pd.DataFrame([
            p_i.values()
        ], columns=p_i.keys())
        excel_sheet = {
            'y1 shapes' : df_y1,
            'y2 shapes' : df_y2,
            'parameters' : params_df
        }
        list_of_dicts.append(excel_sheet)

    return list_of_dicts

def saveDictsToExcel(list_of_dicts, out_folder):
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    for l in list_of_dicts:
        params = l['parameters'].iloc[0].to_numpy()
        params = PartialDislocationsSimulation.paramListToDict(params)
        
        name = f"{params['tauExt']*1e4}e-4_tauExt.xlsx"
        save_path = out_folder.joinpath(name)
        with pd.ExcelWriter(save_path) as writer:
            for key, val in l.items():
                val.to_excel(writer, sheet_name=key)
    pass

def extractAllShapes(pickle_folder, y_shape_folder):
    """
    Extracts dislocation shapes from pickle files in a given folder and saves them as Excel files
    in an organized file structure.

    Args:
        pickle_folder (str): Path to the folder containing the pickle dumps.
        y_shape_folder (str): Path to the folder where the dislocation shapes will be saved as Excel files.
                               The files will be organized in a structured manner within this folder.
    """
    pickle_folder, y_shape_folder = Path(pickle_folder), Path(y_shape_folder)
    for pickle_path in Path(pickle_folder).iterdir():
        list_of_dicts = extractYshapesFromPickle(pickle_path)

        params = list_of_dicts[0]['parameters'].iloc[0].to_numpy()
        params = PartialDislocationsSimulation.paramListToDict(params)

        deltaR = float(params['deltaR'])
        seed = int(params['seed'])

        out_folder = y_shape_folder.joinpath(f"{deltaR*1e4:.3f}e-4_noise/{seed}-seed/")

        saveDictsToExcel(list_of_dicts, out_folder)
    pass


# TODO : compute roughness from the collected y(x) shape data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract dislocation shapes from pickle files and save them as Excel files.")
    parser.add_argument("-p", "--pickles", type=str, help="Path to the folder containing the pickle dumps.")
    parser.add_argument("-o", "--output", type=str, help="Path to the folder where the dislocation shapes will be saved as Excel files.")

    args = parser.parse_args()

    if not args.pickles:
        raise ValueError("Please provide the path to the pickle folder using the --pickles or -p flag.")
    
    if not args.output:
        raise ValueError("Please provide the path to the output folder using the --output or -o flag.")

    extractAllShapes(args.pickles, args.output)