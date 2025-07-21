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

def spawn_relaxation(time, rmin, rmax, rpoints, system_size, d0, seeds, save_path, perfect_partial, dt, hours_limit):
    # Locate the script
    script_directory = Path(os.path.dirname(os.path.abspath(__file__)))
    print(f"The script is located in: {script_directory}")
    triton_rel_script = script_directory.joinpath("run-one-initial-relaxation.sh")

    # Rename the triton job to be submitted
    replace_line_in_file(
        triton_rel_script, 
                        3, 
        f"#SBATCH --job-name=l-{sys_size}-d0-{d0}-{perfect_partial}"
        )
    # Change time limit accordingly
    replace_line_in_file(
        triton_rel_script, 
                        2, 
        f"#SBATCH --time={hours_limit}:00:00"
        )


    stdout = run_command(f"sbatch run-one-initial-relaxation.sh {time} {system_size} {rmin} {rmax} {rpoints} {seeds} {d0} {save_path} {perfect_partial} {dt}")
    print(stdout)
    pass

def getSimulationTime(noise):
    if noise <= 1e-2:
        return 800000
    else:
        return 100000

for sys_size in [32, 64, 128, 265, 512, 1024]:
    save_path = f"${{WRKDIR}}/21-7-testjuttu/l-{sys_size}/perfect"
    spawn_relaxation(800000, -4, -2, 13, dt=100, system_size=sys_size, d0=d0, seeds=10, save_path=save_path, perfect_partial="--perfect", hours_limit=72)
    spawn_relaxation(100000, -2, 4, 37, dt=10, system_size=sys_size, d0=d0, seeds=10, save_path=save_path, perfect_partial="--perfect", hours_limit=72)

    powers_of_two = [2**i for i in range(1, int(sys_size/4).bit_length() + 1)]
    for d0 in powers_of_two:
        save_path = f"${{WRKDIR}}/21-7-ihan-himona-dataa/l-{sys_size}-d0-{d0}/partial"
        spawn_relaxation(800000, -4, -2, 13, dt=100, system_size=sys_size, d0=d0, seeds=10, save_path=save_path, perfect_partial="--partial", hours_limit=72)
        spawn_relaxation(100000, -2, 4, 37, dt=10, system_size=sys_size, d0=d0, seeds=10, save_path=save_path, perfect_partial="--partial", hours_limit=72)