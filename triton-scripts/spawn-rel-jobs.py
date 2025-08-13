from pathlib import Path
import subprocess
import os

def run_command(command):
    process = subprocess.Popen(command, shell=True, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    stdout, stderr = process.communicate()
    return stdout.decode(), stderr.decode(), process.returncode

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


def spawn_relaxation(rmin, rmax, rpoints, system_size, d0, seeds, save_path, perfect_partial, hours_limit, cores):
    """
    Generates a SLURM script from a template and submits it using sbatch.
    """
    template_path = Path(__file__).parent / 'initial-relaxation-template.sh'
    
    # Create a directory for generated scripts if it doesn't exist
    generated_scripts_dir = Path(__file__).parent / 'generated_scripts'
    generated_scripts_dir.mkdir(exist_ok=True)
    
    job_name = f"relax_l{system_size}_d{d0}"
    output_script_path = generated_scripts_dir / f"{job_name}.sh"

    replacements = {
        'name': job_name,
        'hours': hours_limit,
        'system_size': system_size,
        'rmin': rmin,
        'rmax': rmax,
        'rpoints': rpoints,
        'seeds': seeds,
        'd0': d0,
        'save_path': save_path,
        'perfect_partial': perfect_partial,
        'cores':cores
    }

    generate_script_from_template(template_path, output_script_path, replacements)

    stdout, stderr, returncode = run_command(f"sbatch {output_script_path}")
    
    if returncode == 0:
        print(f"Successfully submitted job: {job_name}")
        print(stdout)
    else:
        print(f"Error submitting job: {job_name}")
        print(f"Return code: {returncode}")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")

kansion_nimi = "13-8-strong-coupling-region-III"

for sys_size in [32, 64, 128, 256, 512, 1024]:
    save_path = f"${{WRKDIR}}/{kansion_nimi}/perfect/l-{sys_size}"
    spawn_relaxation(-4, -1, 50, system_size=sys_size, d0=0, seeds=10, save_path=save_path, 
                     perfect_partial="--perfect", hours_limit=24, cores=10)
    powers_of_two = [2**i for i in range(1, int(sys_size/4).bit_length() + 1)]
    for d0 in powers_of_two:
        save_path = f"${{WRKDIR}}/{kansion_nimi}/partial/l-{sys_size}-d0-{d0}"
        spawn_relaxation(-4, -1, 50, system_size=sys_size, d0=d0, seeds=10, save_path=save_path, 
                         perfect_partial="--partial", hours_limit=24, cores=10)