import os
import argparse
import json
from pathlib import Path
from datetime import date


def generate_script_from_template(template_path, output_script_path, replacements):
    """
    Reads a template script, replaces all bracketed {} keys with values from 
    the replacements dictionary, and writes the new script to a new file.

    Args:
        template_path (str): The path to the template file.
        output_script_path (str): The path to write the generated script file.
        replacements (dict): A dictionary where keys correspond to the bracketed
                             placeholders in the template (e.g., 'job-name') and
                             values are what to replace them with.
    """
    with open(template_path, 'r') as f:
        template_content = f.read()

    for key, value in replacements.items():
        template_content = template_content.replace(f'{{{key}}}', str(value))

    with open(output_script_path, 'w') as f:
        f.write(template_content)
    
    # Make the script executable
    os.chmod(output_script_path, 0o755)


def find_unfinished_depinnings(folder):
    """
    folder is the path to a folder containing all the depinnings for a certain configuration of a system,
    for example: results/31-07-2025-depinning/partial/l-64-d0-8.0

    returns all the paths in this folder leading to a depinning_params.json file of an unfinished depinning
    simulation, that is, is finds all unifinished simulations in the directory.

    Then for each of these paths you can create a new job in triton.
    """
    folder = Path(folder)
    run_params_paths = list(folder.rglob("depinning_params.json"))
    unfinished_paths = list()
    for json_path in run_params_paths:
        with open(json_path, "r+") as fp:
            data = json.load(fp)
        
        status_dict = data['status']
        if "ongoing" in status_dict.values():
            unfinished_paths.append(json_path)
    return unfinished_paths

def generate(path_to_run_params, n):
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    template_file = os.path.join(script_dir, 'continue-depinning-template.sh')
    
    with open(Path(path_to_run_params), "r") as fp:
        run_params = json.load(fp)

    args_used = run_params['other parameters']

    length = args_used['length']
    d0 = args_used['d0']
    cores = args_used['cores']

    today_str = date.today().strftime("%d-%m-%Y")
    job_name = f'l-{length}-d0-{d0}-{today_str}_{n}'
    new_script_filename = f"{job_name}.sh"

    repl = {
        'hours':10, 'minutes': '00', 'seconds':'00',
        'job-name':job_name,
        'cores' : cores, 
        'depinning-params' : str(path_to_run_params),
    }

    out_script_path = script_dir.joinpath(f'continue-{job_name}.sh')

    generate_script_from_template(template_file, out_script_path, repl)
    
    return out_script_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate continue scripts for unfinished depinning simulations.")
    parser.add_argument('--path', type=str, required=True, help='Path to the folder containing depinning simulations.')

    args = parser.parse_args()
    folder_path = args.path

    unifnished_depinnings = find_unfinished_depinnings(folder_path)
    print(f"Found {len(unifnished_depinnings)} unifinished depinning simulations.")
    for n,i in enumerate(unifnished_depinnings):
        generate(i, n)
        pass