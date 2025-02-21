import numpy as np
import multiprocessing as mp
from functools import partial
from partialDislocation import PartialDislocationsSimulation
from processData import *

class DepinningPartial(object):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR=1, bigB=1, smallB=1, b_p=1, mu=1, seed=None, bigN=1024, length=1024, d0=39, c_gamma=20, 
                                              cLT1=0.1, cLT2=0.1, sequential=False):
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
        self.c_gamma = c_gamma
        self.cLT1 = cLT1
        self.cLT2 = cLT2

        self.folder_name = folder_name
    
        pass

    def studyConstantStress(self, tauExt):
        simulation = PartialDislocationsSimulation(deltaR=self.deltR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                                   mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                                   dt=self.dt, time=self.time, d0=self.d0, c_gamma=self.c_gamma,
                                                   cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
    
        simulation.run_simulation()
        # dumpResults(simulation, folder_name)
        rV1, rV2, totV2 = simulation.getRelaxedVelocity(time_to_consider=self.time/10) # The velocities after relaxation
        saveLastState_partial(simulation, self.folder_name)

        return (rV1, rV2, totV2)

    def run(self):
        # Multiprocessing compatible version of a single depinning study, here the studies
        # are distributed between threads by stress letting python mp library determine the best way
        
        stresses = np.linspace(self.tau_min, self.tau_max, self.points)
        results = list()

        if self.sequential:
            for s in stresses:
                r_i = self.studyConstantStress(tauExt=s)
                results.append(r_i)
        else:
            with mp.Pool(self.cores) as pool:
                results = pool.map(partial(DepinningPartial.studyConstantStress, self), stresses)
        
        
        v1_rel = [i[0] for i in results]
        v2_rel = [i[1] for i in results]
        v_cm = [i[2] for i in results]

        return v1_rel, v2_rel, v_cm

    