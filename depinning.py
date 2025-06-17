import numpy as np
import multiprocessing as mp
from functools import partial
from partialDislocation import PartialDislocationsSimulation
from singleDislocation import DislocationSimulation
from pathlib import Path
import hashlib

class Depinning(object):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR:float, bigB, smallB,
                mu, bigN, length, d0, sequential=False, seed=None):
        # The common constructor for both types of depinning simulations
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.points = points

        self.time = time
        self.dt = dt
        self.seed = seed

        self.sequential = sequential
        self.cores = cores

        self.deltaR = deltaR
        self.bigB = bigB
        self.smallB = smallB
        self.mu = mu

        self.bigN = bigN
        self.length = length
        self.d0 = d0

        self.folder_name = folder_name

        self.stresses = np.linspace(self.tau_min, self.tau_max, self.points)

class DepinningPartial(Depinning):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR : float,
                 seed=None, bigN=1024, length=1024, d0=39, sequential=False,
                        bigB=1,
                        b_p=0.5773499805, # b_p^2 = a^2 / 6 => b_p^2 = 0.3333333333 => b_p = 0.5773499805
                        smallB=1,         # b^2 = a^2 / 2 = 1 => a^2 = 2
                        mu=1,
                        cLT1=1,
                        cLT2=1,
                        c_gamma=1
                        ): # Use realistic values from paper by Zaiser and Wu 2022
        
        super().__init__(tau_min, tau_max, points, time, dt, cores, folder_name, deltaR, bigB, smallB, mu, bigN, length, d0, sequential, seed)
        # The initializations specific to a partial dislocation depinning simulation.
        self.cLT1 = cLT1
        self.cLT2 = cLT2
        self.c_gamma =c_gamma
        self.results = list()
        self.b_p = b_p
    
    def studyConstantStress(self, tauExt):
        simulation = PartialDislocationsSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                                   mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                                   dt=self.dt, time=self.time, d0=self.d0, c_gamma=self.c_gamma,
                                                   cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
        params = simulation.getParameters()
        params_str = np.array2string(params)  # Convert array to string
        hash_object = hashlib.sha256(params_str.encode())
        hex_dig = hash_object.hexdigest()

        backup_file = Path(self.folder_name).joinpath(f"failsafe/dislocaition-{hex_dig}")

        chunk_size = self.time/10

        simulation.run_in_chunks(backup_file=backup_file, chunk_size=chunk_size)

        rV1, rV2, totV2 = simulation.getRelaxedVelocity()   # The velocities after relaxation
        y1_last, y2_last = simulation.getLineProfiles()     # Get the lines at t = time
        l_range, avg_w = simulation.getAveragedRoughness()  # Get averaged roughness from the same time as rel velocity

        return (rV1, rV2, totV2, l_range, avg_w, y1_last, y2_last, simulation.getParameters())
    
    def run(self):
        # Multiprocessing compatible version of a single depinning study, here the studies
        # are distributed between threads by stress letting python mp library determine the best way
        
        if self.sequential: # Sequental does not work
            for s in self.stresses:
                r_i = self.studyConstantStress(tauExt=s)
                self.results.append(r_i)

        else:
            with mp.Pool(self.cores) as pool:
                self.results = pool.map(partial(DepinningPartial.studyConstantStress, self), self.stresses)
        

        v1_rel, v2_rel, v_cm, l_ranges, avg_w12s, y1_last, y2_last, params = zip(*self.results)

        return v1_rel, v2_rel, v_cm, l_ranges[0], avg_w12s, y1_last, y2_last, params
    
    def getStresses(self):
        return self.stresses

class DepinningSingle(Depinning):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR:float, seed=None, bigN=1024, length=1024, d0=39, sequential=False,
                        bigB=1,
                        smallB=1,    # b^2 = a^2 / 2 = 1
                        mu=1,
                        cLT1=1, rtol=1e-8
                ):

        super().__init__(tau_min, tau_max, points, time, dt, cores, folder_name, deltaR, bigB, smallB, mu, bigN, length, d0, sequential, seed)
        self.cLT1 = cLT1
        self.rtol = rtol

    def studyConstantStress(self, tauExt):
        sim = DislocationSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
                                    mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                    dt=self.dt, time=self.time, cLT1=self.cLT1, seed=self.seed, rtol=self.rtol)
        
        # Get simulation parameters and hash them
        params = sim.getParameteters()
        params_str = np.array2string(params)  # Convert array to string
        hash_object = hashlib.sha256(params_str.encode())
        hex_dig = hash_object.hexdigest()

        backup_file = Path(self.folder_name).joinpath(f"failsafe/dislocaition-{hex_dig}")

        chunk_size = self.time/10

        sim.run_in_chucks(backup_file=backup_file, chunk_size=chunk_size)
        v_rel = sim.getRelaxedVelocity() # Consider last 10% of time to get relaxed velocity.
        y_last = sim.getLineProfiles()
        l_range, avg_w = sim.getAveragedRoughness() # Get averaged roughness from the last 10% of time

        if np.isnan(np.sum(avg_w)) or np.isinf(np.sum(avg_w)):
            print(f"Warning: NaN or Inf in average roughness for tau_ext={tauExt}.")
            error_log = Path(self.folder_name).joinpath("nans_or_infs.txt")
            error_log.parent.mkdir(parents=True, exist_ok=True)
            
            if error_log.exists():
                with open(error_log, 'a') as f:
                    f.write(f"{tauExt}\t{self.deltaR} \n")
            else:
                with open(error_log, 'w') as f:
                    f.write(f"{tauExt}\t{self.deltaR} \n")

        return v_rel, l_range, avg_w, y_last, sim.getParameteters()

    def run(self):
        velocities = list()

        if self.sequential: # Sequential does not work
            for s in self.stresses:
                v_i = self.studyConstantStress(tauExt=s)
                velocities.append(v_i)
        else:
            with mp.Pool(self.cores) as pool:
                results = pool.map(partial(DepinningSingle.studyConstantStress, self), self.stresses)
                velocities, l_ranges, w_avgs, y_last, params = zip(*results)
        
        return velocities, l_ranges[0], w_avgs, y_last, params
    
    def getParameteters(self):
        parameters = np.array([
            self.bigN, self.length, self.time, self.dt,
            self.deltaR, self.bigB, self.smallB,
            self.cLT1, self.mu,
            self.d0, self.seed
        ])
        return parameters
    
    def getStresses(self):
        return self.stresses