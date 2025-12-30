# Two interacting partial dislocations: dynamics

Code for the interaction of two partial dislocations modeled as elastic lines. This script allows the simulation of dislocations
using a line tension model following the Edwards-Wilkinson equation with quenched noise.  Multiprocessing is used to parallelize
the code to gain results much faster.


# Running simulations using FIRE

In addition to all these other scrips we also have the new option of runnning such simulations to find out the critical
force by just using FIRE relaxation. The script required to do this is called `criticalForceUsingFIRE.py`. It is used as follows:

```
usage: criticalForceUsingFIRE.py [-h] --N N --L L [--partial] --d0 D0 --cores CORES --seed SEED --points POINTS --rmin RMIN --rmax RMAX --save_folder SAVE_FOLDER
                                 --taupoints TAUPOINTS

Run FIRE critical force calculation with NoiseData.

options:
  -h, --help            show this help message and exit
  --N N                 System size N (required) usualy N = L
  --L L                 System length L (required) usually L=N
  --partial             Give this flag to simulate partial dislocations. Otherwise the script simulates just a single dislocation
  --d0 D0               Separation of the partials only relevant if --partial is given
  --cores CORES         Number of cores (required)
  --seed SEED           Random seed (required)
  --points POINTS       Number of points in deltaR (required)
  --rmin RMIN           log10(min deltaR) (required)
  --rmax RMAX           log10(max deltaR) (required)
  --save_folder SAVE_FOLDER
                        Folder where simulations results are saved (required)
  --taupoints TAUPOINTS
                        The number of external forces to test.
```

The script saves all the critical forces that were tested and also the relaxed configurations all as pickle files in the
directory specified using the --save_folder flag.

Example usage

```
python3 ./criticalForceUsingFIRE.py --N 32 --L 32 --d0 1 --cores 10 --seed 0 --points 10 --rmin -3 --rmax 1 --save_folder "here" --partial --taupoints 10
```


## Useful triton commands

First `cd` into the `triton-scripts` folder and then use these to submit triton jobs to simulate partial dislocations with various stacking fault widths:

```
for d in 32 16 8 4 2
do
        sbatch taucWithFIRE.sh 32 $WRKDIR/2025-12-29-depinning-dR--3-1-wider-tauc-limits/partial/l-32-d-$d $d
done


for d in 64 32 16 8 4 2
do
        sbatch taucWithFIRE.sh 64 $WRKDIR/2025-12-29-depinning-dR--3-1-wider-tauc-limits/partial/l-64-d-$d $d
done


for d in 128 64 32 16 8 4 2
do
        sbatch taucWithFIRE.sh 128 $WRKDIR/2025-12-29-depinning-dR--3-1-wider-tauc-limits/partial/l-128-d-$d $d
done

for d in 256 128 64 32 16 8 4 2
do
        sbatch taucWithFIRE.sh 256 $WRKDIR/2025-12-29-depinning-dR--3-1-wider-tauc-limits/partial/l-256-d-$d $d
done

for d in 512 256 128 64 32 16 8 4 2
do
        sbatch taucWithFIRE.sh 512 $WRKDIR/2025-12-29-depinning-dR--3-1-wider-tauc-limits/partial/l-512-d-$d $d
done
```

and this for the perfect dislocation case

```
for l in 32 64 128 256 512
do
        sbatch perfectTaucWithFIRE.sh $l $WRKDIR/2025-12-29-depinning-dR--3-1-wider-tauc-limits/perfect/l-$l 0
done
```

# Running simulations via direct numerical integration

The workflow for running simulation using numerical integration is composed of three major steps: initial relaxation, depinning and then if necessary running
the depinning simulations further and finally the generation of figures based on the data. These steps are separate so that in case the simulation timeouts,
it's always possible to recover.


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

## Output files of the script

The files are organized in the following directory structure:

```
root/
├── partial/
│   ├── l-32-d-2/
│   │   ├── force_data
        └── shape_data
│   └── l-64-d-2/
└── perfect/
│   ├── l-32/
│   └── l-64/
```

As the names suggest the folder partial and perfect contain partial and perfect dislocation data respectively, each 
folder named by the length L and separation d of the partials.

Under each folder is the simulation results of that system. The folder `force_data` contains the critical forces at each
magnitude of disorder for all seeds (realizations of noise) in `.csv` form while `shape_data` on the other hand contains 
the relaxed configurations at each magnitude of noise and critical force.

Each file in `shape_data` is a pickle which is named as `extra_info_dump-l-128-s-1-d0-32.pickle` according to the size L,
separation d, and seed used to generate the pinning field of the system. The pickle contains a simple python list where each
item is a dictionary with keys tau_ext, converged, shapes, deltaR. Under the key tau_ext is a list of the external forces 
attempted. Each item on this list stores the results for some value of deltaR. 

Each key has the following contents: under converged is a list of boolean values indicating whether the simulation 
converged or not under the tau_ext with the same index. Lastly, the shapes key contains a list of the converged or 
nonconverged configurations of the system and deltaR simply indicates the noise magnitude.

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
