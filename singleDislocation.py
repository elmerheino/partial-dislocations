import time
import numpy as np
from simulation import Simulation
from scipy.integrate import solve_ivp
from pathlib import Path
import hashlib

class DislocationSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, cLT1=2, seed=None, d0=10, rtol=1e-6):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1}
        self.d0 = d0
        
        # Pre-allocate memory here
        #self.y1 = np.empty((self.timesteps*10, self.bigN)) # Each timestep is one row, bigN is number of columns
        self.y1 = list() # List of arrays

        # y1 is a matrix that looks like this:
        #
        #                axis=1 (bigN) -->
        #   axis=0       [------------------- dislocation shape 1 -------------------]
        #   (time)       [------------------- dislocation shape 2 -------------------]
        #   |            [------------------- dislocation shape 3 -------------------]
        #   v            [                       ...                                 ]
        #                [------------------- dislocation shape n -------------------]

        self.errors = list()
        self.used_timesteps = 0
        self.rtol = rtol

        self.v_cm_history = list()

        pass
    
    def funktio(self, y1, t):
        dy1 = (
            self.cLT1*self.mu*(self.smallB**2)*self.secondDerivative(y1) # The gradient term # type: ignore
            + self.smallB*(self.tau(y1) + self.tau_ext(t)*np.ones(self.bigN) )
        ) * ( self.bigB/self.smallB )
        return dy1
    
    def rhs(self, t, u_flat : np.ndarray):
        # Reshape the flat array back to its original shape
        u = u_flat.reshape(self.bigN)
                
        # Compute the right-hand side using the function defined above
        dudt = np.array(self.funktio(u, t))
        
        # Flatten the result back to a 1D array
        return dudt.flatten()

    def run_simulation(self, timeit=False):
        if timeit:
            t0 = time.time()
        y0 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d

        sol = solve_ivp(self.rhs, [0, self.time], y0.flatten(), method='RK45', 
                        t_eval=np.arange(self.time*(1 - 0.1), self.time, self.dt),
                        rtol=self.rtol, atol=1e-10) # Use a very small atol to avoid NaNs in the solution

        self.y1 = sol.y.T

        self.used_timesteps = sol.t[1:] - sol.t[:-1] # Get the time steps used

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
    
    def run_in_chucks(self, backup_file, chunk_size = 100, timeit=False):
        if timeit:
            t0 = time.time()

        y0 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d

        total_time_so_far = 0
        last_y0 = y0

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        if self.time % chunk_size != 0:
            print(chunk_size % self.time)
            raise Exception("invalid chunk size")
        
        while total_time_so_far < self.time*(1 - 0.1):
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method='RK45', 
                            t_eval=[end_i],
                            rtol=self.rtol)
            y_i = sol_i.y.T
            last_y0 = y_i[-1]
            np.savez(backup_file, y_last=last_y0, params=self.getParameteters(), time=end_i)
            total_time_so_far += chunk_size
        
        sol = solve_ivp(self.rhs, [self.time*(1 - 0.1), self.time], last_y0.flatten(), method='RK45', 
                            t_eval=np.arange(self.time*(1 - 0.1), self.time, self.dt),
                            rtol=self.rtol)

        self.y1 = sol.y.T

        self.used_timesteps = sol.t[1:] - sol.t[:-1] # Get the time steps used

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
        pass

    def run_until_relaxed(self, backup_file, chunk_size : int, timeit=False, tolerance=1e-6):   
        """
        When using this method to run the simulation, then self.time acts as the maximum simulation time, and chunck_size
        is the timespan from the end that will be saved for for further processing in methods such as getCM, getRelVelocity,
        getRoughness, etc.

        So when using this method, these other varibaled will not be computed from the "last 10%" of simulation time, unless
        chunk_size is one tenth of it and it so happend that exactly ten chunks is used to achieve relaxation.
        """

        if timeit:
            t0 = time.time()
        
        if self.time % chunk_size != 0:
            print(self.time % chunk_size)
            raise Exception("invalid chunk size")
        chunk_size = int(chunk_size)

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        y0 = np.ones((chunk_size, self.bigN), dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d

        total_time_so_far = 0
        last_y = y0
        max_time = self.time

        relaxed = False
        # print("started while loop")
        while (total_time_so_far < max_time) and not relaxed:
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            last_y0 = last_y[-1]
            
            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method='RK45', 
                            t_eval=np.arange(start_i, end_i, self.dt),
                            rtol=self.rtol)
            y_i = sol_i.y.T
            self.used_timesteps = self.used_timesteps + sol_i.t[1:] - sol_i.t[:-1] # Get the time steps used

            last_y = y_i
            np.savez(backup_file, y_last=last_y0, params=self.getParameteters(), time=end_i)
            total_time_so_far = total_time_so_far + chunk_size

            cm_i = np.mean(y_i, axis=1)
            v_cm_i = np.gradient(cm_i, self.dt)
            self.v_cm_history.append(v_cm_i)

            if self.is_relaxed(v_cm_i, tolerance=tolerance):
                relaxed = True
                break # end the simulation here.

            # print(f"{total_time_so_far/self.time:.1f}", "\t")
        
        # print(f"total time so far = {total_time_so_far} max time = {max_time} and total/max_time {total_time_so_far / max_time}")
        # print(f"Relaxed {relaxed}")

        if relaxed:
            self.y1 = last_y
            # print(f"relaxed shape {self.y1.shape}, len/time = {self.y1.shape[0]/self.time*100:2f}")
        else: # If not relaxed run one more chuck
            sol = solve_ivp(self.rhs, [total_time_so_far, total_time_so_far + chunk_size], last_y[-1].flatten(), method='RK45', 
                                t_eval=np.arange(total_time_so_far, total_time_so_far + chunk_size, self.dt),
                                rtol=self.rtol)
            
            # print(sol.y.T)

            self.y1 = np.array(sol.y.T)

            self.used_timesteps = self.used_timesteps + sol.t[1:] - sol.t[:-1] # Get the time steps used
            # print(f"not relaxed shape {self.y1.shape}, len*dt/time = {self.y1.shape[0]*self.dt/self.time*100:2f}")

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
        return relaxed
    
    def getLineProfiles(self):
        start = 0

        start = self.timesteps - 1
        # Otherwise return the last 10% of the simulation

        if self.has_simulation_been_run:
            return self.y1[start:].flatten()
        else:
            raise Exception("Simulation has not been run.")
        
        print("Simulation has not been run yet.")
        return self.y1 # Retuns empty lists
    
    def getAverageDistances(self):
        # Returns the average distance from y=0
        return np.average(self.y1, axis=1)
    
    def getCM(self):
        # Return the centre of mass of the line
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')

        y1_CM = np.mean(self.y1, axis=1)

        return y1_CM
    
    def getRelaxedVelocity(self):
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')
        
        # Returns the velocities of the centres of mass
        y1_CM = self.getCM()
        v1_CM = np.gradient(y1_CM)

        # Condisering only time_to_consider seconds from the end

        start = 0 # Consider only the last 10% of the simulation

        v_relaxed_y1 = np.average(v1_CM[start:])

        return v_relaxed_y1
    
    def getParamsInLatex(self):
        return super().getParamsInLatex() + [f"C_{{LT1}} = {self.cLT1}"]
    
    def getTitleForPlot(self, wrap=6):
        parameters = self.getParamsInLatex()
        plot_title = " ".join([
            "$ "+ i + " $" + "\n"*(1 - ((n+1)%wrap)) for n,i in enumerate(parameters) # Wrap text using modulo
        ])
        return plot_title
    
    def getParameteters(self):
        parameters = np.array([
            self.bigN, self.length, self.time, self.dt,     # Index 0-3
            self.deltaR, self.bigB, self.smallB,            # 4 - 6
            self.cLT1, self.mu, self.tauExt,                # 7 - 9
            self.d0, self.seed, self.tau_cutoff             # 10 - 12
        ])
        return parameters

    def getAveragedRoughness(self):
        start = 0 # Consider only the last 10% of the simulation

        l_range, _ = self.roughnessW(self.y1[0], self.bigN)

        roughnesses = np.empty((self.timesteps, self.bigN))
        for i in range(start,self.timesteps):
            _, rough = self.roughnessW(self.y1[i], self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg
    
    def getErrors(self):
        return self.errors
    
    def retrieveUsedTimesteps(self):
        return self.used_timesteps
    
    def getUniqueHashString(self):
        params = self.getParameteters()
        params_str = np.array2string(params)  # Convert array to string
        hash_object = hashlib.sha256(params_str.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig
    
    def getVCMhist(self):
        return np.array(self.v_cm_history).flatten()


# For debugging
if __name__ == "__main__":
    dislocation = DislocationSimulation( deltaR = 0.01,
                seed=0, bigN=256, length=256, d0=1,
                bigB=1,
                smallB=1,         # b^2 = a^2 / 2 = 1 => a^2 = 2
                mu=1,
                cLT1=1,
                time=1000, dt=2, tauExt=0)
    backup_path = Path("debug").joinpath(f"failsafe/dislocaition-debug.npz")
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    dislocation.run_until_relaxed(chunk_size=1000/10, backup_file=backup_path)
    pass