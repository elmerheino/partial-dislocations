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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a depinning script from a template.")
    parser.add_argument('--initial-config', type=str, required=True, 
                        help='Path to the initial relaxed configuration run_params.json file')
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    template_file = os.path.join(script_dir, 'depinning-from-rel-template.sh')

    input_folder = Path(args.initial_config).parent

    with open(Path(args.initial_config), "r") as fp:
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
