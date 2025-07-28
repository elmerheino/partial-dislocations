import argparse
from pathlib import Path
import shutil
import json

def update_run_params(path_to_run_params):
    with open(path_to_run_params, "rb") as fp:
        data = json.load(fp)
    
    noise_seeds = data['noise-seeds']
    sc_noise_seeds = data['successful noise-seeds']
    args_used = data['args used']

    new_noise_seeds = list(filter(lambda x : x[1] == 0, noise_seeds))
    new_sc_noise_seeds = list(filter(lambda x : x[1] == 0, sc_noise_seeds))


    new_data = data
    new_data['noise-seeds'] = new_noise_seeds
    new_data['successful noise-seeds'] = new_sc_noise_seeds

    new_args = args_used
    new_args['seeds'] = 1

    new_data['args used'] = new_args

    return new_data

def pick_desired_files(path_to_run_params):
    path_to_run_params = Path(path_to_run_params)
    folder = path_to_run_params.parent
    rel_confs_folder = folder.joinpath(f"relaxed-configurations")
    rel_confs_paths = list(rel_confs_folder.iterdir())

    chosen_paths = filter(lambda x : x.name.split('-')[4] == '0.npz',rel_confs_paths)

    return list(chosen_paths)

def create_new_rels(input_folder, output):
    """
    input path is until this level results/24-7-weak-coupling/partial/l-32-d0-2 and so is output path in the new location.
    """
    input_folder = Path(input_folder)
    output = Path(output)

    files_to_copy = pick_desired_files(input_folder.joinpath('run_params.json'))
    new_run_params = update_run_params(input_folder.joinpath('run_params.json'))

    for file in files_to_copy:
        dest = output.joinpath('relaxed-configurations').joinpath(file.name)
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy(file, dest)
    
    with open(output.joinpath("run_params.json"), "w") as fp:
        json.dump(new_run_params, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that takes an input and output file path."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input file initial configurations, to the same level as results/24-7-weak-coupling"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path for the output where the smaller subset is to be generated, to the same level as input"
    )

    args = parser.parse_args()
    for partial_perfect in Path(args.input).iterdir():
        for oogabooga in partial_perfect.iterdir():
            dest_folder = Path(args.output).joinpath(f"{partial_perfect.name}/{oogabooga.name}")
            create_new_rels(oogabooga, dest_folder)
            pass

    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
