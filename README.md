# Two interacting partial dislocations: dynamics

Code for the interaction of two partial dislocations modeled as elastic lines. This script allows the simulation of dislocations
using a line tension model following the Edwards-Wilkinson equation with quenched noise.  Multiprocessing is used to parallelize
the code to gain results much faster.

# Outline of the workflow

The workflow for running simulation is composed of three major steps: initial relaxation, depinning and then if necessary running
the depinning simulations further and finally the generation of figures based on the data.


## Running initial relaxations

Before starting a new depinning simulation, you need an initial configuration of the system where a steady state is reached
with zero external force.

This state can be found using two methods, either pure numerical integration with zero external fore or an accelerated algorithm
called FIRE which we are using. Another option is to use both, first relaxing the system using FIRE and then briefly integrating 
from the found configuration to ensure that a steady state is reached. In the current codebase, only FIRE is used.

You can run a new initial relaxation by using the script initialRelaxations.py which has two commands, `new` and `continue`, 
of which new has the following usage:

```
usage: initialRelaxations.py new [-h] --seeds SEEDS --length LENGTH --n N --rmin RMIN --rmax RMAX --rpoints RPOINTS --folder FOLDER [--d0 D0] (--partial | --perfect) -c CORES

options:
  -h, --help            show this help message and exit
  --seeds SEEDS         how many realizations of each noise
  --length LENGTH       length of the system such that L=N
  --n N                 system size
  --rmin RMIN           Minimum delta R value
  --rmax RMAX           Maximum delta R value
  --rpoints RPOINTS     Number of points between rmin and rmax
  --folder FOLDER       output folder path
  --d0 D0               Initial separation of partials. Required for --partial.
  --partial             Enable partial dislocations simulation
  --perfect             Enable perfect dislocations simulation
  -c CORES, --cores CORES
                        number of cores to use
```

The script also has another command `continue` 
which allows you to continue the relaxation from a previous failsafe. This command is relevant only in the case when the
system is integrated in time in addition to FIRE relaxation.

During this step many of the relevant parameters of the coming depinning simulations are already set, the most important 
of which are: the noise magnitude range, number of points within it, the number of seeds and also the separation between 
the partials.

When running these initial relaxations in Triton, there is another script `triton-scripts/spawn-rel-jobs.py`, which generates
the Triton batch scripts automatically and submits the jobs to Triton. In that script many values are hard coded into it, which
must be changes according to what is needed.

## Running a depinning simulation

After some relaxed configurations have been obtained, we are ready to run a depinning simulation from them. The script
`depinningFromRel.py` does just this. The script has three commands: `new`, `continue` and `further`. Command `new`initiates
a new depinning from the given initial configurations, `continue` continues a depinning from the failsafe files in case the
simulation has timed out, and finally `further` integrates a completed simulation further in time.

The `new` command has the following usage:

```
usage: depinningFromRel.py new [-h] --folder FOLDER --out-folder OUT_FOLDER [--partial] [--perfect] --cores CORES --task-id TASK_ID --points POINTS --time TIME --dt DT

options:
  -h, --help            show this help message and exit
  --folder FOLDER       Folder where the initially relaxed configurations are. Should be the one which contains the folder 'relaxed-configurations'.
  --out-folder OUT_FOLDER
                        Output folder for the new depinning simulation.
  --partial             Flag for partial dislocation.
  --perfect             Flag for perfect dislocation.
  --cores CORES         Number of cores to use.
  --task-id TASK_ID     SLURM_ARRAY_TASK_ID from Triton, used as an index.
  --points POINTS       Number of tau_ext points to be integrated.
  --time TIME           Time to integrate each simulation.
  --dt DT               Timestep for sampling the solution.
```

When the simulation is finished, it saves the results in a list of dictionaries which is dumped in a pickle in the folder given
with the flag `--out-folder`.

Usually we are running multiple such depinnings with some array of seeds and noises. The scripts keep track of these automatically
using the task-id from Triton as an index.

To submit such jobs on Triton the script `triton-scripts/generoi-depinning-skripti.py` comes in handy. It generates the needed
batch script for all the array jobs to run the depinnings from the obtained initial relaxations. The script takes only
one argument, the path to the folder containing all the initial relaxations. It can also automatically submit the jobs to Triton
if line no. 111 is uncommented.

If the simulation times out and the integration is left unfinished, the script can recover from the saved failsafe files using
the `continue` command. Its usage is found by using the `-h` flag. In the folder `triton-scripts` there is also a helper script
for submitting such jobs on Triton.

Then finally there is also the `further` command, which allows integrating completed simulation even further in case such a
need arises e.g. if the original simulation was not long enough to reach a steady state.

## Generating figures from the results

There are currently three scripts for processing the simulation results: `velocityPlots.py`, `stackingFault.py`, and 
`selectedYshapes.py`. Each of these scripts has a very straightforward command line interface, and the usage can be found
by simply passing the `-h` flag.

A key principle with these scripts is that they take the pickle dumps, read the data from them and save it in an organized
form to various .xlsx files so that they can be indexed using pandas.

At the moment there is not an automated routine for making critical force plots, but this data is saved to a .csv file when
making depinning plots with `velocityPlots.py` and making such a $\Delta R$ vs. $\tau_c$ plot is very straightforward using
libraries such as pandas and matplotlib.

In principle there exists also the `roughessPlots.py` script, which can generate all the roughness plots. However, since
the way the results are saved was recently changes, it does not work as is. The pickles can be converted to the directory
structure that was previously used by using the `save_results()` function found in classes `DepinningPartial` and `DepinningSingle`