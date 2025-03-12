import numpy as np
import multiprocessing as mp
from functools import partial
from partialDislocation import PartialDislocationsSimulation
from processData import *

class Depinning(object):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR=1, bigB=1, smallB=1,
                  b_p=1, mu=1, seed=None, bigN=1024, length=1024, d0=39, sequential=False):
        # The common constructor for both types of depinning simulations
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.points = points

        self.time = time
        self.dt = dt
        self.seed = seed

        self.sequential = sequential
        self.cores = cores

        self.deltR = deltaR
        self.bigB = bigB
        self.smallB = smallB
        self.mu = mu
        self.b_p = b_p

        self.bigN = bigN
        self.length = length
        self.d0 = d0

        self.folder_name = folder_name

        self.stresses = np.linspace(self.tau_min, self.tau_max, self.points)

class DepinningPartial(Depinning):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR=1, bigB=1, smallB=1, b_p=1, 
                 mu=1, seed=None, bigN=1024, length=1024, d0=39, c_gamma=20, cLT1=0.1, cLT2=0.1, sequential=False):
        
        super().__init__(tau_min, tau_max, points, time, dt, cores, folder_name, deltaR, bigB, smallB, b_p, mu, 
                         seed, bigN, length, d0, sequential)
        # The initializations specific to a partial dislocation depinning simulation.
        self.cLT1 = cLT1
        self.cLT2 = cLT2
        self.c_gamma =c_gamma
        self.results = list()
    
    def studyConstantStress(self, tauExt):
        simulation = PartialDislocationsSimulation(deltaR=self.deltR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                                   mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                                   dt=self.dt, time=self.time, d0=self.d0, c_gamma=self.c_gamma,
                                                   cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
    
        simulation.run_simulation()
        # dumpResults(simulation, folder_name)
        t_to_consider = self.time/10 # TODO: make time to consider a global parameter
        rV1, rV2, totV2 = simulation.getRelaxedVelocity(time_to_consider=t_to_consider) # The velocities after relaxation
        saveStatesFromTime(simulation, self.folder_name,t_to_consider)

        return (rV1, rV2, totV2)
    
    def run(self):
        # Multiprocessing compatible version of a single depinning study, here the studies
        # are distributed between threads by stress letting python mp library determine the best way
        
        if self.sequential:
            for s in self.stresses:
                r_i = self.studyConstantStress(tauExt=s)
                self.results.append(r_i)

        else:
            with mp.Pool(self.cores) as pool:
                self.results = pool.map(partial(DepinningPartial.studyConstantStress, self), self.stresses)
        
        
        v1_rel = [i[0] for i in self.results]
        v2_rel = [i[1] for i in self.results]
        v_cm = [i[2] for i in self.results]

        return v1_rel, v2_rel, v_cm

class DepinningSingle(Depinning):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR=1, bigB=1, smallB=1, b_p=1, mu=1, seed=None, bigN=1024, length=1024, d0=39, cLT1=0.1, sequential=False):
        super().__init__(tau_min, tau_max, points, time, dt, cores, folder_name, deltaR, bigB, smallB, b_p, mu, seed, bigN, length, d0, sequential)
        self.cLT1 = cLT1

    def studyConstantStress(self, tauExt):
        sim = DislocationSimulation(deltaR=self.deltR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p,
                                    mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                    dt=self.dt, time=self.time, cLT1=self.cLT1, seed=self.seed)

        sim.run_simulation()
        t_to_consider = self.time/10
        v_rel = sim.getRelaxedVelocity(time_to_consider=t_to_consider) # Consider last 10% of time to get relaxed velocity.

        saveStatesFromTime_single(sim, self.folder_name, t_to_consider)

        return v_rel

    def run(self):
        velocities = list()

        if self.sequential:
            for s in self.stresses:
                v_i = self.studyConstantStress(tauExt=s)
                velocities.append(v_i)
        else:
            with mp.Pool(self.cores) as pool:
                velocities = pool.map(partial(DepinningSingle.studyConstantStress, self), self.stresses)
        
        return velocities