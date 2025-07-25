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

def generate_scirpt(path_to_run_params):
    path_to_run_params = Path(path_to_run_params)

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    template_file = os.path.join(script_dir, 'depinning-from-rel-template.sh')

    input_folder = path_to_run_params.parent

    with open(Path(path_to_run_params), "r") as fp:
        run_params = json.load(fp)
    args_used = run_params['args used']

    seeds = args_used['seeds']
    rpoints = args_used['rpoints']
    perfect = args_used['perfect']
    length = args_used['length']
    d0 = args_used['d0']

    output_path = f"${{WRKDIR}}/date-depinning/{'perfect' if perfect else 'partial'}/l-{ args_used['length'] }/"
    today_str = date.today().strftime("%d-%m-%Y")
    output_path = f"${{WRKDIR}}/{today_str}-depinning/{'perfect' if perfect else 'partial'}/l-{ args_used['length'] }-d0-{d0}/"

    script_parameters = {
        'hours': 10,
        'minutes': "00",
        'seconds': "00",
        'job-name': f'{today_str}-depinning-l-{length}-d0-{d0}',
        'cores': 10,
        'arr-start': 0,
        'arr-end': rpoints*seeds - 1,
        'input-path': str(input_folder),
        'rel-time': 200000,
        'sample-dt': 100,
        'output-path': output_path,
        'perfect-partial' : f"{'--perfect' if perfect else '--partial'}"
    }

    # Define the name for the new script file
    new_script_filename = f"{script_parameters['job-name']}.sh"
    new_script_path = script_dir.joinpath(new_script_filename)

    # Generate the new script
    generate_script_from_template(template_file, new_script_path, script_parameters)

    print(f"Generated script: {new_script_path}")

    return new_script_path

def find_run_params_files(directory):
    """
    Recursively finds all paths in the given directory that lead to a file named 'run_params.json'.

    Args:
        directory (str): The directory to search.

    Returns:
        list: A list of paths to 'run_params.json' files.
    """
    run_params_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == 'run_params.json':
                run_params_files.append(os.path.join(root, file))
    return run_params_files

# Example usage
directory_to_search = '/path/to/directory'  # Replace with the actual directory path
run_params_files = find_run_params_files(directory_to_search)
print(f"Found run_params.json files: {run_params_files}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a depinning script from a template.")
    parser.add_argument('--path', type=str, required=True, 
                        help='Path to directory called partial or perfect')
    args = parser.parse_args()

    # generate_scirpt(args.initial_config)
    run_params_paths = find_run_params_files("/Users/elmerheino/Documents/partial-dislocations/results/24-7-weak-coupling/partial")
    for path_i in run_params_paths:
        script_path = generate_scirpt(path_i)
        os.system(f"sbatch {script_path}")