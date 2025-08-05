import pickle
import pandas as pd
import numpy as np
from src.core.partialDislocation import PartialDislocationsSimulation

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
