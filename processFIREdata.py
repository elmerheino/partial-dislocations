from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
import h5py
from src.core.partialDislocation import PartialDislocationsSimulation
from src.core.singleDislocation import DislocationSimulation

# Define roughness
def roughnessW(y, bigN): # Calculates the cross correlation W(L) of a single dislocation
    l_range = np.arange(0,int(bigN))
    roughness = np.empty(int(bigN))

    y_size = int(bigN)
    
    for l in l_range:
        res = 0
        for i in range(0,bigN):
            res = res + ( y[i] - y[ (i+l) % y_size ] )**2
        
        res = res/y_size
        c = np.sqrt(res)
        roughness[l] = c

    return l_range, roughness

def power_spectral_desity(heights):
    k = np.fft.fft(heights) / len(heights)
    fq = np.fft.fftfreq(len(k))

    psd = np.abs(k**2)

    return psd, fq

def process_shape_data(path_to_extra, path_to_force_data, partial=True):
    with open(path_to_extra, "rb") as fp:
        data = pickle.load(fp)
    
    df = pd.read_csv(path_to_force_data)

    force_data = zip(df['deltaR'].to_list(), df['tau_c'].to_list())
    force_data = list(force_data)

    assert len(data) == len(force_data)

    shape_data = list(zip(force_data, data))

    organized_data = list()

    for i in shape_data:
        deltaR, tau_c = i[0]

        tau_exts = i[1]['tau_ext']
        converged = i[1]['converged']
        shapes = i[1]['shapes']

        converged_tau_exts = tau_exts[converged]

        converged_shapes = [ shape for shape, j in zip(shapes, converged) if j]
        converged_shapes

        # From the extracted shapes, compute the roughness of the CM and store it
        roughness_list = list()
        slope_dataseries = list()
        for tau_ext, shapes in zip(converged_tau_exts, converged_shapes):
            if partial:
                cm = (shapes[0] + shapes[1])/2
            else:
                cm = shapes
            l,r = roughnessW(cm, len(cm))
            roughness_list.append((tau_ext, r))

            x = np.log(np.arange(len(r)))[1:10]
            y = np.log(r)[1:10]

            slope, intercept, r_value, p_value, std_err = linregress(x, y)

            slope_dataseries.append((tau_ext, slope))


        res = {
            'deltaR' : deltaR,
            'deltaR_log10' : np.log10(deltaR),
            'tau_c' : tau_c,
            'shape' : list(zip(converged_tau_exts, converged_shapes)),
            'roughness' : roughness_list,
            'roughness exponents' : slope_dataseries
        }
        organized_data.append(res)

    return organized_data

def process_folder(path_p : str, partial=True):
    path_p = Path(path_p)
    save_path = path_p.joinpath("processed_data")
    save_path.mkdir(exist_ok=True, parents=True)

    def extract_sort_key(filepath):
        if 'extra_info_dump-' in filepath.stem:
            return filepath.stem.replace('extra_info_dump-', '')
        elif 'noise_tauc_data_' in filepath.stem:
            return filepath.stem.replace('noise_tauc_data_', '')
        return filepath.stem

    pickle_files = sorted(path_p.joinpath("shape_data").iterdir(), key=extract_sort_key)
    csv_files = sorted(path_p.joinpath("force_data").iterdir(), key=extract_sort_key)


    for pickle_path, csv_path in zip(pickle_files, csv_files):
        processed_data = process_shape_data(pickle_path, csv_path, partial=partial)
        with open(save_path.joinpath(f"processed_data_{extract_sort_key(pickle_path)}.pickle"), "wb") as fp:
            pickle.dump(processed_data, fp)

def collect_shapes(path_to_processed_data, output_file : Path, partial=True):
    path_to_processed_data = Path(path_to_processed_data)
    collected_data = dict()

    h5file = h5py.File(output_file, "w")
    for data_paska in path_to_processed_data.iterdir():
        with open(data_paska, "rb") as fp:
            data = pickle.load(fp)
        
        if partial:
            seed = data_paska.name.split("_")[2].split("-")[3]
            d0 = data_paska.name.split("-")[-1].split(".")[0]
        else:
            seed = data_paska.name.split("_")[2].split("-")[3].split('.')[0]

        for item in data:
            deltaR = item['deltaR']
            heights = item['shape']

            try:
                tau_exts, heights = zip(*heights)
            except:
                print(item)
                continue

            group_deltaR = h5file.require_group(str(deltaR))
            group_seed = group_deltaR.require_group(seed)

            shape_dataset = group_seed.create_dataset("heights", data=heights)

            shape_dataset.attrs['seed'] = int(seed)
            shape_dataset.attrs['deltaR'] = deltaR
            shape_dataset.attrs['tau_exts'] = tau_exts

            y1 = heights[-1][0]
            y2 = heights[-1][1]

            # Integrate the shape further in time to obtain a time average
            bigN = len(heights[-1][0])
            b_p = 0.5773499805
            sim = PartialDislocationsSimulation(bigN, bigN, 5, 0.01, deltaR, 1,1, b_p, 1, tau_exts[-1], int(d0), c_gamma=0.3, 
                                                    seed=int(seed)
                                                )
            sim.setInitialY0Config(y1, y2, 0)
            results = sim.run_in_one_go()

            shapes = np.array(results.y.reshape(2, bigN, -1))
            times = np.array(results.t)

            integration_dataset = group_seed.create_dataset("integrated shapes", data=shapes)
            integration_dataset.attrs['times'] = times

            y1_values_over_time = shapes[0]
            y2_values_over_time = shapes[1]

            psd_y1s, psd_y2s, psd_cms = list(), list(), list()
            for y1_i, y2_i in zip(y1_values_over_time, y2_values_over_time):
                psd_y1_i, fq_y1_i = power_spectral_desity(y1_i)
                psd_y2_i, fq_y2_i = power_spectral_desity(y2_i)
                cm = (y1_i + y2_i)/2
                psd_cm_i,fq_cm_i = power_spectral_desity(cm)

                psd_y1s.append(psd_y1_i)
                psd_y2s.append(psd_y2_i)
                psd_cms.append(psd_cm_i)

            psd_y1s, psd_y2s, psd_cms = np.array(psd_y1s), np.array(psd_y2s), np.array(psd_cms)

            psd_y1_mean = np.mean(psd_y1s, axis=0)
            psd_y2_mean = np.mean(psd_y2s, axis=0)
            psd_cm_mean = np.mean(psd_cms, axis=0)

            group_seed.create_dataset("psd-y1-mean", data=psd_y1_mean)
            group_seed.create_dataset("psd-y2-mean", data=psd_y2_mean)
            group_seed.create_dataset("psd-cm-mean", data=psd_cm_mean)

            cm = (y1 + y2)/2

            psd_critical, fq_critical = power_spectral_desity(cm)
            psd_y1, fq_y1 = power_spectral_desity(y1)
            psd_y2, fq_y2 = power_spectral_desity(y2)

            psd_dataset_critical = group_seed.create_dataset("psd_critical", data=psd_critical)

            psd_dataset_critical.attrs['seed'] = int(seed)
            psd_dataset_critical.attrs['deltaR'] = deltaR
            psd_dataset_critical.attrs['dft-fqns'] = fq_critical

            psd_dataset_y1 = group_seed.create_dataset("psd_y1", data=psd_y1)

            psd_dataset_y1.attrs['seed'] = int(seed)
            psd_dataset_y1.attrs['deltaR'] = deltaR
            psd_dataset_y1.attrs['dft-fqns'] = fq_y1

            psd_dataset_y2 = group_seed.create_dataset("psd_y2", data=psd_y2)

            psd_dataset_y2.attrs['seed'] = int(seed)
            psd_dataset_y2.attrs['deltaR'] = deltaR
            psd_dataset_y2.attrs['dft-fqns'] = fq_y2

    h5file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot dislocation data.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Create the parser for the "process" command
    process_parser = subparsers.add_parser('process', help='Process raw simulation data.')
    process_parser.add_argument("main", type=str, help="Path to the main data folder.")

    args = parser.parse_args()

    if args.command == 'process':
        depinning_folder = args.main

        partial = True

        for folder in Path(f"{depinning_folder}/{'partial' if partial else 'perfect'}/").iterdir():
            print(f"Processing folder {folder}")
            process_folder(folder, partial=partial)

        for folder1 in Path(f"{depinning_folder}/{'partial' if partial else 'perfect'}/").iterdir():

            collect_shapes(
                f"{folder1}/processed_data",
                f"{folder1}/shapes.h5",
                partial=partial
                )
