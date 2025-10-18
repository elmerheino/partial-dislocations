import fcntl
import json
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import multiprocessing as mp
from functools import partial
from src.core.partialDislocation import PartialDislocationsSimulation
from src.core.singleDislocation import DislocationSimulation
from pathlib import Path
import hashlib
import csv

class Depinning(object):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR:float, bigB, smallB,
                mu, bigN, length, d0, sequential=False, seed=None):
        # The common constructor for both types of depinning simulations
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.points = points

        self.time = time
        self.dt = dt

        if type(seed) == type(None):
            self.seed = np.random.randint(0, 2**32 - 1)
        else:
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
                        c_gamma=1, failsafe_dict=None, status_dict=None, recovery_config=None
                        ): # Use realistic values from paper by Zaiser and Wu 2022
        
        super().__init__(tau_min, tau_max, points, time, dt, cores, folder_name, deltaR, bigB, smallB, mu, bigN, length, d0, sequential, seed)
        # The initializations specific to a partial dislocation depinning simulation.
        self.cLT1 = cLT1
        self.cLT2 = cLT2
        self.c_gamma = c_gamma
        self.results = None
        self.b_p = b_p

        self.y1_0 = None
        self.y2_0 = None

        self.backup_paths = dict()

        self.failsafe_path = Path(folder_name).joinpath("failsafes/")
        self.json_path = Path(folder_name).joinpath("depinning_params.json")

        self.failsafe_dict = failsafe_dict
        self.status_dict = status_dict
        self.recovery_config = recovery_config
        if not self.json_path.exists():
            self.createInfoJSON(self.json_path)

    @classmethod
    def from_json_config(cls, json_path, cores=None):
        with open(json_path, 'r') as f:
            config = json.load(f)

        if type(cores) == None:
            cores = params['cores']
    
        # Extract parameters from JSON
        params = config['other parameters']
        loaded_failsafes = config['failsafe files']
        loaded_statuses = config['status']
        instance = cls( tau_min=config['tau_min'], tau_max=config['tau_max'], points=params['points'], time=params['time'],
            dt=params['dt'], cores=cores, folder_name=params['folder_name'], deltaR=params['deltaR'], seed=params['seed'],
            bigN=params['bigN'], length=params['length'], d0=params['d0'], sequential=params['sequential'], bigB=params['bigB'],
            smallB=params['smallB'], mu=params['mu'], cLT1=params['cLT1'], cLT2=params['cLT2'], b_p=params['b_p'],
            c_gamma=params['c_gamma'], failsafe_dict=loaded_failsafes, status_dict=loaded_statuses, recovery_config=config
        )
        instance.stresses = config['stresses']
        return instance

    def createInfoJSON(self, path):
        status_dict = {str(float(tau_ext)) : "not_started" for tau_ext in self.stresses}
        data = {
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "stresses": self.stresses.tolist(),
            "failsafe files" : dict(),
            "status" : status_dict,     # Each tau ext has value which is either "not_started", "ongoing", "finished"
            "other parameters": {
                "points": self.points, "time": self.time, "dt": self.dt, "seed": self.seed, "sequential": self.sequential,
                "cores": self.cores, "deltaR": self.deltaR, "bigB": self.bigB, "smallB": self.smallB, "mu": self.mu,
                "bigN": self.bigN, "length": self.length, "d0": self.d0, "folder_name": str(self.folder_name), 
                "cLT1" : self.cLT1, "cLT2" : self.cLT2, "b_p" : self.b_p, "c_gamma" : self.c_gamma
            }
        }
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as fp:
            json.dump(data, fp)

        pass
    
    def appendFailsafeToJSON(self, tauExt, failsafe_path):
        # Appends the failsafe corresponding to tauExt to the list stored in self.json_file
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with open(self.json_path, 'r+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        params = json.load(f)

                        params["failsafe files"][tauExt] = str(failsafe_path)

                        f.seek(0)
                        json.dump(params, f, indent=4)
                        f.truncate()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                break
            except (IOError, BlockingIOError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
    
    def updateStatusDict(self, tauExt, status):
        # Appends the failsafe corresponding to tauExt to the list stored in self.json_file
        max_retries = 5
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                with open(self.json_path, 'r+') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        params = json.load(f)

                        params["status"][str(float(tauExt))] = status

                        f.seek(0)
                        json.dump(params, f, indent=4)
                        f.truncate()
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                break
            except (IOError, BlockingIOError) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)

    def initialRelaxation(self, relaxation_time=10):
        """
        This function finds the relaxed configuration of the system with external force being zero.
        """
        sim = PartialDislocationsSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                            mu=self.mu, tauExt=0, bigN=self.bigN, length=self.length, 
                                            dt=self.dt, time=relaxation_time, d0=self.d0, c_gamma=self.c_gamma,
                                            cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
        
        rel_backup_path = Path(self.folder_name).joinpath(f"initial-relaxations/initial-relaxation-{sim.getUniqueHash()}.npz")
        rel_backup_path.parent.mkdir(exist_ok=True, parents=True)
        
        relaxed = sim.run_in_chunks(rel_backup_path, sim.time/10)
        v_cm_hist = sim.getVCMhist()

        np.savez(rel_backup_path, v_cm_hist=v_cm_hist)

        print(f"Initial relaxation fulfilled criterion: {relaxed}")

        y1_0, y2_0 = sim.getLineProfiles()
        return y1_0, y2_0

    def studyConstantStress(self, tauExt):
        simulation = PartialDislocationsSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB, b_p=self.b_p, 
                                                   mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                                   dt=self.dt, time=self.time, d0=self.d0, c_gamma=self.c_gamma,
                                                   cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
        if (type(self.y1_0) != type(None)) and (type(self.y2_0) != type(None)):
            print(f"Shapes {self.y1_0.shape} {self.y2_0.shape}")
            simulation.setInitialY0Config(self.y1_0, self.y2_0)
        
        # Create a backup file where the data is saved once in a while, and keep track of it in the JSON file
        backup_file = Path(self.folder_name).joinpath(f"failsafe/deltaR-{self.deltaR}-seed-{self.seed}/dislocaition-{simulation.getUniqueHash()}")
        backup_file.parent.mkdir(exist_ok=True, parents=True)
        self.appendFailsafeToJSON(tauExt, backup_file)
        self.updateStatusDict(tauExt, "ongoing")

        chunk_size = self.time/10
        is_relaxed = simulation.run_in_chunks(backup_file=backup_file, chunk_size=chunk_size)
        self.updateStatusDict(simulation.tauExt, "finished")

        final_results = simulation.getResultsAsDict()

        with open(backup_file, "wb") as fp:
            pickle.dump(final_results, fp)

        return final_results
    
    def constantStressFromFailsafe(self, failsafe_path):
        """
        This method has to return the same values as self.studyConstantStress!
        """
        backup_file = Path(failsafe_path)

        simulation = PartialDislocationsSimulation.fromFailsafe(failsafe_path)
        simulation.run_in_chunks(backup_file, simulation.time/10)

        self.updateStatusDict(simulation.tauExt, "finished")

        final_results = simulation.getResultsAsDict()

        with open(backup_file, "wb") as fp:
            pickle.dump(final_results, fp)

        return final_results
    
    def integrateFurther(self,failsafe_path,tau_ext, time_further):
        backup_file = Path(failsafe_path)
        sim = PartialDislocationsSimulation.fromFinishedFailsafe(failsafe_path, time_further)
        # Here 10k is the time how much longer its integrated

        self.updateStatusDict(tau_ext, "ongoing")
        sim.run_in_chunks(backup_file, time_further/10)
        self.updateStatusDict(tau_ext, "finished")

        final_results = sim.getResultsAsDict()

        with open(backup_file, "wb") as fp:
            pickle.dump(final_results, fp)

        return final_results

    def run(self, y1_0=None, y2_0=None):
        # Multiprocessing compatible version of a single depinning study, here the studies
        # are distributed between threads by stress letting python mp library determine the best way
        if type(y1_0) == type(None):
            print("No initial config was passed, running a short relaxation before proceeding to depinning.")
            y1_0, y2_0 = self.initialRelaxation()
            self.y1_0 = y1_0
            self.y2_0 = y2_0
        else:
            print("Initial config was given.")
            self.y1_0 = y1_0
            self.y2_0 = y2_0
        
        if self.sequential: # Sequental does not work
            results = list()
            for s in self.stresses:
                r_i = self.studyConstantStress(tauExt=s)
                results.append(r_i)
            self.results = results
        else:
            with mp.Pool(self.cores) as pool:
                self.results = pool.map(partial(DepinningPartial.studyConstantStress, self), self.stresses)
    
    def run_recovered_sequential(self, y1_0=None, y2_0=None, integrate_further=False, further_time=None):
        res = list()
        self.y1_0 = y1_0
        self.y2_0 = y2_0

        for tau_ext in self.stresses:
            # Check if the simulation is already complete
            # If not, load the failsafe from the dictionary
            # Create the dislocation object based on that failasfe
            self.mp_helper(tau_ext, integrate_further=integrate_further, further_time=further_time)
            pass
        
        self.results = res

    def mp_helper(self, tau_ext, integrate_further=False, further_time=None):
        tau_ext_key = str(float(tau_ext))
        status = self.status_dict[tau_ext_key]

        if status == "ongoing":
            # create the dislocation object from the failsafe
            # TODO : handle case if the failsafe does not simply exist
            if tau_ext_key in self.failsafe_dict.keys():
                failsafe_path = Path(self.failsafe_dict[tau_ext_key])
                if failsafe_path.exists():
                    print(f"Recovering from failsafe with tau_ext = {tau_ext}")
                    results_i = self.constantStressFromFailsafe(failsafe_path)
                else:
                    print(f"Starting new simulation with tau_ext = {tau_ext}")
                    results_i = self.studyConstantStress(tau_ext)
            else:
                # Failsafe does not exist, so go to case "not_started" and start a new simualation from zero
                print(f"Starting new simulation with tau_ext = {tau_ext}")
                results_i = self.studyConstantStress(tau_ext)
            pass
        elif status == "not_started":
            # just run a new simulation from nothing using studyConstantStress
            print(f"Starting new simulation with tau_ext = {tau_ext}")
            results_i = self.studyConstantStress(tau_ext)
            pass
        else:
            # load the final results to memory
            if tau_ext_key in self.failsafe_dict.keys():
                failsafe_path = Path(self.failsafe_dict[tau_ext_key])
                if failsafe_path.exists():
                    # Here the results are loaded to memory, we could also, for example, continue running the simulations here
                    # even further
                    if integrate_further:
                        print(f"Integrating further with tau_ext = {tau_ext}")
                        results_i = self.integrateFurther(failsafe_path, tau_ext, further_time)
                    else:
                        with open(failsafe_path, "rb") as fp:
                            results_i = pickle.load(fp)
                else:
                    # In this case the failsafe does not exist for some reason and a new depinning must be started
                    print(f"Starting new simulation with tau_ext = {tau_ext}")
                    results_i = self.studyConstantStress(tau_ext)
            else:
                results_i = self.studyConstantStress(tau_ext)
            pass

        return results_i

    def run_recovered_parallel(self, y1_0=None, y2_0=None, integrate_further=False, further_time=None):
        self.y1_0 = y1_0
        self.y2_0 = y2_0

        with mp.Pool(self.cores) as pool:
            self.results = pool.map(partial(DepinningPartial.mp_helper, self, integrate_further=integrate_further,
                                            further_time=further_time), self.stresses)

    def getStresses(self):
        return self.stresses
    
    def mp_function(self, tauExt, use_gd=False):
        sim = PartialDislocationsSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
                        mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, b_p=self.b_p, d0=self.d0,
                        dt=self.dt, time=self.time, cLT1=self.cLT1, cLT2=self.cLT2, seed=self.seed)
        print(f"External force is {tauExt}")

        if use_gd:
            h1, h2, success = sim.relax_w_gd()
        else:
            h1, h2, success = sim.relax_w_FIRE()

        shape = (h1,h2)
        return success, shape

    def findCriticalForceWithFIRE(self, use_gd=False):
        res = {
            'tau_ext' : self.stresses,
            'converged' : list(),
            'shapes' : list(),
            'deltaR' : self.deltaR
        }

        with mp.Pool(self.cores) as pool:
            results = pool.map(partial(DepinningPartial.mp_function, self, use_gd=use_gd), self.stresses)
            successes = [i[0] for i in results]
            shapes = [i[1] for i in results]
            res['converged'] = successes
            res['shapes'] = shapes
        
        index = np.argmax( ~(np.array(res['converged'])) )

        tau_c = res['tau_ext'][index]
        shape = res['shapes'][index]

        return tau_c, shape, res
    
    def save_results(self, folder_path):
        """
        Saves the results in a directeory structure of files and folders.
        """
        v1_rel = [r['v1'] for r in self.results]
        v2_rel = [r['v2'] for r in self.results]
        v_cm_rel = [r['v_cm'] for r in self.results]
        l_ranges = [r['l_range'] for r in self.results]
        avg_w12s = [r['avg_w'] for r in self.results]
        y1_last = [r['y1_last'] for r in self.results]
        y2_last = [r['y2_last'] for r in self.results]
        v_cms = [r['v_cm_hist'] for r in self.results]
        sf_hists = [r['sf_hist'] for r in self.results]
        parameters = [r['params'] for r in self.results]

        tau_min_ = min(self.stresses.tolist())
        tau_max_ = max(self.stresses.tolist())
        points = len(self.stresses.tolist())

        # Save the depinning to a .json file
        depining_path = Path(folder_path)
        depining_path = depining_path.joinpath("depinning-dumps").joinpath(f"noise-{self.deltaR}")
        depining_path.mkdir(exist_ok=True, parents=True)
        depining_path = depining_path.joinpath(f"depinning-tau-{tau_min_}-{tau_max_}-p-{points}-t-{self.time}-s-{self.seed}-R-{self.deltaR}.json")

        with open(str(depining_path), 'w') as fp:
            json.dump({
                "stresses": self.stresses.tolist(),
                "v_rel": v_cm_rel,
                "seed":self.seed,
                "time":self.time,
                "dt":self.dt,
                "v_1" : v1_rel,
                "v_2" : v2_rel
            },fp)
        
        # Save the roughnesses in an organized way
        l_range = l_ranges[0]
        for tau, avg_w12, params in zip(self.stresses, avg_w12s, parameters):
            tauExt_i = params[11]
            deltaR_i = params[4]
            p = Path(folder_path).joinpath(f"averaged-roughnesses").joinpath(f"noise-{self.deltaR}").joinpath(f"seed-{self.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p = p.joinpath(f"roughness-tau-{tau}-R-{self.deltaR}.npz")
            
            np.savez(p, l_range=l_range, avg_w=avg_w12, parameters=params)
        
        # Save the dislocation at the end of simulation in an organized way
        for y1_i, y2_i, params in zip(y1_last, y2_last, parameters):
            tauExt_i = params[11]
            deltaR_i = params[4]
            p = Path(folder_path).joinpath(f"dislocations-last").joinpath(f"noise-{self.deltaR}").joinpath(f"seed-{self.seed}")
            p.mkdir(exist_ok=True, parents=True)
            p0 = p.joinpath(f"dislocation-shapes-tau-{tauExt_i}-R-{deltaR_i}.npz")
            np.savez(p0, y1=y1_i, y2=y2_i, parameters=params)
            pass

        # Save the velocity of the CM from the entire duration of simulation
        v_cms_over_time = dict()
        for v_cm, params in zip(v_cms, parameters):
            deltaR_i = params[4]
            tauExt_i = params[11]
            v_cms_over_time[f"{tauExt_i}"] = v_cm
        
        # v_cms_over_time = np.array(v_cms_over_time)
        vel_save_path = Path(folder_path).joinpath(f"velocties/noise-{self.deltaR}-seed-{self.seed}-v_cm.npz")
        vel_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(vel_save_path, **v_cms_over_time )

        # Save stacking faults over time
        sf_stacking_faults_over_time = dict()
        for sf_hist, params in zip(sf_hists, parameters):
            tauExt_i = params[11]
            sf_stacking_faults_over_time[f"{tauExt_i}"] = sf_hist

        sf_save_path = Path(folder_path).joinpath(f"stacking-faults/noise-{self.deltaR}-seed-{self.seed}-sf.npz")
        sf_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sf_save_path, **sf_stacking_faults_over_time)
    
    def dump_res_to_pickle(self, folder):
        dump_path = Path(folder).joinpath(f"result-dump-deltaR-{self.deltaR}-seed-{self.seed}.pickle")
        dump_path.parent.mkdir(exist_ok=True, parents=True)
        with open(dump_path, "wb") as fp:
            pickle.dump(self.results, fp)
        pass

class DepinningSingle(Depinning):

    def __init__(self, tau_min, tau_max, points, time, dt, cores, folder_name, deltaR:float, seed=None, bigN=1024, length=1024, sequential=False,
                        bigB=1,
                        smallB=1,    # b^2 = a^2 / 2 = 1
                        mu=1,
                        cLT1=1, rtol=1e-8
                ):

        super().__init__(tau_min, tau_max, points, time, dt, cores, folder_name, deltaR, bigB, smallB, mu, bigN, length, None, sequential, seed)
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

        realaxed = sim.run_in_chunks(rel_backup_path, chunk_size=sim.time/10, shape_save_freq=1)

        print(f"Initial relaxation fulfilled criterion: {realaxed}")

        return sim.getLineProfiles()

    def studyConstantStress(self, tauExt):
        sim = DislocationSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
                                    mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                                    dt=self.dt, time=self.time, cLT1=self.cLT1, seed=self.seed, rtol=self.rtol)
        
        sim.setInitialY0Config(self.y0_rel, 0)
        
        # Name a backup file where to save checkpoints
        backup_file = Path(self.folder_name).joinpath(f"failsafe/dislocaition-{sim.getUniqueHashString()}")

        chunk_size = self.time/10

        sim.run_in_chunks(backup_file=backup_file, chunk_size=chunk_size, shape_save_freq=2)
        v_rel = sim.getRelaxedVelocity() # Consider last 10% of time to get relaxed velocity.
        y_last = sim.getLineProfiles()
        l_range, avg_w = sim.getAveragedRoughness() # Get averaged roughness from the last 10% of time
        v_cm_over_time = sim.getVCMhist()
        y_selected = sim.getSelectedYshapes()

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
            'params': sim.getParameteters(), 'y_selected' : y_selected
        }

    def run(self, y0_rel=None):
        velocities = list()
        if type(y0_rel) == type(None):
            print(f"No initial config given, running the default initial relaxation with zero ext force")
            self.y0_rel = self.initialRelaxation()
        else:
            print(f"Initial config was given")
            self.y0_rel = y0_rel


        if self.sequential: # Sequential does not work
            for s in self.stresses:
                v_i = self.studyConstantStress(tauExt=s)
                velocities.append(v_i)
        else:
            with mp.Pool(self.cores) as pool:
                results = pool.map(partial(DepinningSingle.studyConstantStress, self), self.stresses)
                self.results = results

                # velocities, l_ranges, w_avgs, y_last, v_cms, params = zip(*results)
        
        # return velocities, l_ranges[0], w_avgs, y_last, v_cms, params
    
    def mp_function(self, tauExt):
        sim = DislocationSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
                        mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
                        dt=self.dt, time=self.time, cLT1=self.cLT1, seed=self.seed, rtol=self.rtol)
        print(f"External force is {tauExt}")
        shape, success = sim.relax_w_FIRE()
        return success, shape

    def findCriticalForceWithFIRE(self):
        res = {
            'tau_ext' : self.stresses,
            'converged' : list(),
            'shapes' : list(),
            'deltaR' : self.deltaR,
            'params' : self.getParameteters()
        }
        ## This is the sequential implementation, keeping it here just in case
        # for tauExt in self.stresses:
        #     sim = DislocationSimulation(deltaR=self.deltaR, bigB=self.bigB, smallB=self.smallB,
        #                             mu=self.mu, tauExt=tauExt, bigN=self.bigN, length=self.length, 
        #                             dt=self.dt, time=self.time, cLT1=self.cLT1, seed=self.seed, rtol=self.rtol)
        #     print(f"External force is {tauExt}")
        #     shape, success = sim.relax_w_FIRE()
        #     res['tau_ext'].append(tauExt)
        #     res['converged'].append(success)
        #     res['shapes'].append(shape)

        with mp.Pool(self.cores) as pool:
            results = pool.map(partial(DepinningSingle.mp_function, self), self.stresses)
            successes = [i[0] for i in results]
            shapes = [i[1] for i in results]
            res['converged'] = successes
            res['shapes'] = shapes
        
        index = np.argmax( ~(np.array(res['converged'])) )

        tau_c = res['tau_ext'][index]
        shape = res['shapes'][index]

        return tau_c, shape, res

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

        v_rels = [r['v_rel'].astype(float) for r in self.results]
        with open(str(depining_path), 'w') as fp:
            json.dump({
            "stresses": [float(s) for s in self.stresses.tolist()],
            "v_rel": [float(v) for v in v_rels],
            "seed": int(self.seed),
            "time": float(self.time),
            "dt": float(self.dt)
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

        # Save the dislocation shapes colleceted at selected times
        y_shape_matrices = [r['y_selected'] for r in self.results]
        selected_ys = dict()
        for y_selected, params in zip(y_shape_matrices, parameters):
            params_dict = DislocationSimulation.paramListToDict(params)
            deltaR_i = params_dict['deltaR']
            tauExt_i = params_dict['tauExt']

            selected_ys[f"{tauExt_i}"] = y_selected

        selected_y_sp = Path(folder_path).joinpath(f"selected-ys/{self.deltaR*1e5:.4f}-1e-5-noise-{self.seed}-tauExt.npz")
        selected_y_sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(selected_y_sp, **selected_ys)

    def dump_res_to_pickle(self, folder):
        dump_path = Path(folder).joinpath(f"result-dump-deltaR-{self.deltaR}-seed-{self.seed}.pickle")
        dump_path.parent.mkdir(exist_ok=True, parents=True)
        with open(dump_path, "wb") as fp:
            pickle.dump(self.results, fp)

class NoiseVsCriticalForce(object):
    def __init__(self, N=32, L=32, cores=10, folder_name="remove_me", seed=0, time=1000, tau_points=20, d0=None):
        self.N = N
        self.L = L
        self.cores = cores
        self.folder_name = folder_name
        self.seed = seed
        self.time = time
        self.d0 = d0
        self.tau_points = tau_points

    def noise_tauc_FIRE(self, rmin, rmax, points):
        data = list()
        extra_infos = list()
        for deltaR in np.logspace(rmin, rmax, points):
            tau_c_guess = deltaR
            depinning = DepinningSingle(0, tau_c_guess*1.3, self.tau_points, 1000, 0.1, cores=self.cores, folder_name="remove_me", 
                                        deltaR=deltaR, seed=self.seed, bigN=self.N, length=self.L)
            tau_c, shape, extra_info = depinning.findCriticalForceWithFIRE()
            data.append((deltaR, tau_c))
            extra_infos.append(extra_info)
        return data, extra_infos
    
    def noise_tauc_FIRE_partial(self, rmin, rmax, points):
        data = list()
        extra_infos = list()

        for deltaR in np.logspace(rmin, rmax, points):
            tau_c_guess = deltaR
            depinning = DepinningPartial(0, tau_c_guess*1.3, self.tau_points, 1000, 0.1, cores=self.cores, folder_name="remove_me", 
                                        deltaR=deltaR, seed=self.seed, bigN=self.N, length=self.L, d0=self.d0, c_gamma=0.3)
            tau_c, shape, extra_info = depinning.findCriticalForceWithFIRE()
            data.append((deltaR, tau_c))
            extra_infos.append(extra_info)
        return data, extra_infos
    
    def save_data(self, data, folder):
        paath = Path(folder).joinpath(f"noise_tauc_data_l-{self.L}-s-{self.seed}-d0-{self.d0}.csv")
        paath.parent.mkdir(exist_ok=True, parents=True)
        with open(paath, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['deltaR', 'tau_c'])  # Write header
            csvwriter.writerows(data)  # Write data rows
    
    def do_all_steps(self, rmin, rmax, rpoints, save_folder):
        data, extra_infos = self.noise_tauc_FIRE(rmin, rmax, rpoints)

        path = Path(save_folder).joinpath("force_data")
        path.mkdir(exist_ok=True, parents=True)
        self.save_data(data, path)

        path1 = Path(save_folder).joinpath(f"shape_data/extra_info_dump-l-{self.L}-s-{self.seed}.pickle")
        path1.parent.mkdir(exist_ok=True, parents=True)

        with open(path1, "wb") as fp:
            pickle.dump(extra_infos, fp)
    
    def do_all_steps_partial(self, rmin, rmax, rpoints, save_folder):
        data, extra_infos = self.noise_tauc_FIRE_partial(rmin, rmax, rpoints)

        path = Path(save_folder).joinpath("force_data")
        path.mkdir(exist_ok=True, parents=True)
        self.save_data(data, path)

        path1 = Path(save_folder).joinpath(f"shape_data/extra_info_dump-l-{self.L}-s-{self.seed}-d0-{self.d0}.pickle")
        path1.parent.mkdir(exist_ok=True, parents=True)

        with open(path1, "wb") as fp:
            pickle.dump(extra_infos, fp)


if __name__ == "__main__":

    tauc_vs_deltaR = NoiseVsCriticalForce(8, 8, 10, seed=0, tau_points=40, d0=1)
    tauc_vs_deltaR.do_all_steps_partial(-4, 0, 20, "debug/6-9-dataa")