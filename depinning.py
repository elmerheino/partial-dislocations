import json
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

        self.y1_0 = None
        self.y2_0 = None
    
    def initialRelaxation(self, relaxation_time=1e6):
        """
        This function finds the relaxed configuration of the system with external force being zero.
        """
        sim = PartialDislocationsSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                            mu=self.mu, tauExt=0, bigN=self.bigN, length=self.length, 
                                            dt=self.dt, time=relaxation_time, d0=self.d0, c_gamma=self.c_gamma,
                                            cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
        
        rel_backup_path = Path(self.folder_name).joinpath(f"initial-relaxations/initial-relaxation-{sim.getUniqueHash()}.npz")
        rel_backup_path.parent.mkdir(exist_ok=True, parents=True)
        
        relaxed = sim.run_until_relaxed(rel_backup_path, sim.time/10)

        print(f"Initial relaxation fulfilled criterion: {relaxed}")

        y1_0, y2_0 = sim.getLineProfiles()
        return y1_0, y2_0

    def studyConstantStress(self, tauExt):
        simulation = PartialDislocationsSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                                   mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                                   dt=self.dt, time=self.time, d0=self.d0, c_gamma=self.c_gamma,
                                                   cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
        
        print(f"Shapes {self.y1_0.shape} {self.y2_0.shape}")
        
        simulation.setInitialY0Config(self.y1_0, self.y2_0)
        
        backup_file = Path(self.folder_name).joinpath(f"failsafe/dislocaition-{simulation.getUniqueHash()}")
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        chunk_size = self.time/10

        is_relaxed = simulation.run_in_chunks(backup_file=backup_file, chunk_size=chunk_size)
        # print(f"Dislocaiation was relaxed? {is_relaxed}")

        rV1, rV2, totV2 = simulation.getRelaxedVelocity()   # The velocities after relaxation
        y1_last, y2_last = simulation.getLineProfiles()     # Get the lines at t = time
        l_range, avg_w = simulation.getAveragedRoughness()  # Get averaged roughness from the same time as rel velocity
        v_cm = simulation.getVCMhist()                      # Get the cm velocity history from the time of the whole simulation
        sfHist = simulation.getSFhist()

        return (rV1, rV2, totV2, l_range, avg_w, y1_last, y2_last, v_cm, sfHist, simulation.getParameters())
    
    def run(self):
        # Multiprocessing compatible version of a single depinning study, here the studies
        # are distributed between threads by stress letting python mp library determine the best way

        y1_0, y2_0 = self.initialRelaxation()
        self.y1_0 = y1_0
        self.y2_0 = y2_0
        
        if self.sequential: # Sequental does not work
            for s in self.stresses:
                r_i = self.studyConstantStress(tauExt=s)
                self.results.append(r_i)

        else:
            with mp.Pool(self.cores) as pool:
                self.results = pool.map(partial(DepinningPartial.studyConstantStress, self), self.stresses)
        

        v1_rel, v2_rel, v_cm_rel, l_ranges, avg_w12s, y1_last, y2_last, v_cms, sf_hists, params = zip(*self.results)

        return v1_rel, v2_rel, v_cm_rel, l_ranges[0], avg_w12s, y1_last, y2_last, v_cms, sf_hists, params
    
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

        self.y0_rel = None
        self.results = None

    def initialRelaxation(self, relaxation_time = 50):
        sim = DislocationSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
                            mu=self.mu, tauExt=0, bigN=self.bigN, length=self.length, 
                            dt=self.dt, time=relaxation_time, cLT1=self.cLT1, seed=self.seed, rtol=self.rtol)
        
        rel_backup_path = Path(self.folder_name).joinpath(f"initial-relaxations/initial-relaxation-{sim.getUniqueHashString()}.npz")
        rel_backup_path.parent.mkdir(exist_ok=True, parents=True)

        realaxed = sim.run_until_relaxed(rel_backup_path, chunk_size=sim.time/10)

        print(f"Initial relaxation fulfilled criterion: {realaxed}")

        return sim.getLineProfiles()

    def studyConstantStress(self, tauExt):
        sim = DislocationSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
                                    mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                    dt=self.dt, time=self.time, cLT1=self.cLT1, seed=self.seed, rtol=self.rtol)
        
        sim.setInitialY0Config(self.y0_rel)
        
        # Name a backup file where to save checkpoints
        backup_file = Path(self.folder_name).joinpath(f"failsafe/dislocaition-{sim.getUniqueHashString()}")

        chunk_size = self.time/10

        sim.run_in_chunks(backup_file=backup_file, chunk_size=chunk_size)
        v_rel = sim.getRelaxedVelocity() # Consider last 10% of time to get relaxed velocity.
        y_last = sim.getLineProfiles()
        l_range, avg_w = sim.getAveragedRoughness() # Get averaged roughness from the last 10% of time
        v_cm_over_time = sim.getVCMhist()

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

        return {
            'v_rel': v_rel, 'l_range': l_range, 'avg_w': avg_w, 'y_last': y_last, 'v_cm': v_cm_over_time,
            'params': sim.getParameteters()
        }

    def run(self):
        velocities = list()

        self.y0_rel = self.initialRelaxation()

        if self.sequential: # Sequential does not work
            for s in self.stresses:
                v_i = self.studyConstantStress(tauExt=s)
                velocities.append(v_i)
        else:
            with mp.Pool(self.cores) as pool:
                results = pool.map(partial(DepinningSingle.studyConstantStress, self), self.stresses)
                self.results = results

                velocities, l_ranges, w_avgs, y_last, v_cms, params = zip(*results)
        
        return velocities, l_ranges[0], w_avgs, y_last, v_cms, params
    
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
    
    def save_results(self, folder_path):

        if self.results is None:
            raise ValueError("No results to save. Run the simulation first.")

        parameters = [r['params'] for r in self.results]

        depining_path = Path(folder_path)
        depining_path = depining_path.joinpath("depinning-dumps").joinpath(f"noise-{self.deltaR}")
        depining_path.mkdir(exist_ok=True, parents=True)

        tau_min_ = min(self.stresses.tolist())
        tau_max_ = max(self.stresses.tolist())
        points = len(self.stresses.tolist())
        depining_path = depining_path.joinpath(f"depinning-tau-{tau_min_}-{tau_max_}-p-{points}-t-{self.time}-s-{self.seed}-R-{self.deltaR}.json")

        v_rels = [r['v_rel'] for r in self.results]
        with open(str(depining_path), 'w') as fp:
            json.dump({
                "stresses":self.stresses.tolist(),
                "v_rel":v_rels,
                "seed":self.seed,
                "time":self.time,
                "dt":self.dt
            },fp)

        # Save all the roughnesses
        roughnesses = [r['avg_w'] for r in self.results]
        l_ranges = [r['l_range'] for r in self.results]
        l_range = l_ranges[0]
        for tau, avg_w, params in zip(self.stresses, roughnesses, parameters): # Loop through tau as well to save it along data
            deltaR_i = params[4]
            p = Path(folder_path).joinpath(f"averaged-roughnesses").joinpath(f"noise-{self.deltaR}").joinpath(f"seed-{self.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau}-R-{deltaR_i}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w, parameters=params)
            pass

        # Save all the relaxed dislocation profiles at the end of simulation
        y_last = [r['y_last'] for r in self.results]
        for y_i, params in zip(y_last, parameters):
            tauExt = params[9]
            deltaR_i = params[4]
            p = Path(folder_path).joinpath(f"dislocations-last").joinpath(f"noise-{self.deltaR}").joinpath(f"seed-{self.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt}-R-{self.deltaR}.npz")
            np.savez(p0, y=y_i, parameters=params)

        # Save the velocity of the CM from the last 10% of simulation time
        v_cms = [r['v_cm'] for r in self.results]
        v_cms_over_time = dict()
        for v_cm, params in zip(v_cms, parameters):
            deltaR_i = params[4]
            tauExt_i = params[9]

            v_cms_over_time[f"{tauExt_i}"] = v_cm
        
        vel_save_path = Path(folder_path).joinpath(f"velocties/noise-{self.deltaR}-seed-{self.seed}-v_cm.npz")
        vel_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(vel_save_path, **v_cms_over_time)
