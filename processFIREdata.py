from scipy.stats import linregress
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import argparse
import h5py

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
            shape_dataset = group_deltaR.create_dataset(seed, data=heights)

            shape_dataset.attrs['seed'] = int(seed)
            shape_dataset.attrs['deltaR'] = deltaR
            shape_dataset.attrs['tau_exts'] = tau_exts

    h5file.close()

def get_psd(h5_file_path='results/5-10-vahemman-rpoints/partial/l-32-d-4/shapes.h5',noise_key=None):
    psds = list()
    
    f = h5py.File(h5_file_path, "r")
    print(f.keys())

    if type(noise_key) == type(None):
        noise_key = list(f.keys())[0]

    for seed in f[noise_key].keys():
        dataset = f[noise_key][seed]

        heights = dataset[-1]
        force = dataset.attrs['tau_exts'][-1]

        # heights = np.mean(heights, axis=0)

        k = np.fft.fft(heights)
        fq = np.fft.fftfreq(len(k))

        psd = np.abs(k**2)
        psds.append(psd)
    
    psds = np.array(psds)
    psd = np.mean(psds, axis=0)
    return psd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and plot dislocation data.")
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    # Create the parser for the "process" command
    process_parser = subparsers.add_parser('process', help='Process raw simulation data.')
    process_parser.add_argument("main", type=str, help="Path to the main data folder.")

    # Create the parser for the "plot" command
    plot_parser = subparsers.add_parser('plot', help='Plot the processed data.')
    # This command currently takes no arguments as the paths are hardcoded in the plotting section

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

    elif args.command == 'plot':
        # The plotting part of the script will be executed.
        # No specific arguments needed for plot command based on current script structure.
        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot(get_psd("results/11-10-pieni-c_gamma/perfect/l-128/shapes.h5", noise_key='1.0'), label="1.0")
        ax.plot(get_psd("results/11-10-pieni-c_gamma/perfect/l-128/shapes.h5", noise_key='0.1389495494373137'), label="0.14")
        ax.plot(get_psd("results/11-10-pieni-c_gamma/perfect/l-128/shapes.h5", noise_key='0.0001'), label="0.0001")
        ax.plot(get_psd("results/11-10-pieni-c_gamma/perfect/l-128/shapes.h5", noise_key='0.01'), label="0.01")
        

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_xlabel('$k$')
        ax.set_ylabel('PSD')

        fig.tight_layout()
        ax.legend()
        ax.grid(True)

        plt.show()
