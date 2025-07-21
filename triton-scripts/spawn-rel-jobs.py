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

def replace_line_in_file(file_path, line_number, new_line):
    """Replaces a specific line in a file with a new line.

    Args:
        file_path (str): The path to the file.
        line_number (int): The line number to replace (1-indexed).
        new_line (str): The new line to write.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if line_number > 0 and line_number <= len(lines):
        lines[line_number - 1] = new_line + '\n'

        with open(file_path, 'w') as f:
            f.writelines(lines)
    else:
        print(f"Line number {line_number} is out of range for file {file_path}")

def spawn_relaxation(time, rmin, rmax, rpoints, system_size, d0, seeds, save_path, perfect_partial):
    # Locate the script
    script_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f"The script is located in: {script_directory}")
    triton_rel_script = script_directory.joinpath("run-one-initial-relaxation.sh")

    # Rename the triton job to be submitted
    replace_line_in_file(
        triton_rel_script, 
                        3, 
        "#SBATCH --job-name=initial-relaxation-4"
        )

    run_command(f"sbatch run-one-initial-relaxation.sh {time} {system_size} {rmin} {rmax} {rpoints} {seeds} {d0} {save_path} {perfect_partial}")
    pass

sys_size=32
save_path = f"${{WRKDIR}}/21-7-testjuttu/sys-{sys_size}/partial"
spawn_relaxation(10000, -4, 4, sys_size, 32, 10, 1, save_path, "--partial")