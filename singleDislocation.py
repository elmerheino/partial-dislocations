import time
import numpy as np
from simulation import Simulation
from scipy.integrate import solve_ivp
from pathlib import Path
import hashlib
from scipy.interpolate import CubicSpline
from scipy import fft
class DislocationSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, cLT1, seed=None, rtol=1e-6):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1}
        
        # Initialize default y0 as flat line at y=0
        self.y0 = np.zeros(self.bigN, dtype=np.float64)
        self.t0 = 0
        
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

        self.shape_save_freq = 10           # The frequency when the whole dislocation line shape is saved in the format
                                            # of every self.shape_save_freq:th self.dt
        self.selected_y_shapes = list()

        # --- System Parameters for FIRE---
        self.N = self.bigN
        self.L = self.length

        # --- FIRE Algorithm Parameters ---
        self.DT_INITIAL = 0.01
        self.DT_MAX = 0.1
        self.N_MIN = 5
        self.F_INC = 1.1
        self.F_DEC = 0.5
        self.ALPHA_START = 0.1
        self.F_ALPHA = 0.99
        self.MAX_STEPS = 500000
        self.CONVERGENCE_FORCE = 1e-10

        # --- Noise Parameters ---
        self.h_max_noise = self.bigN*2
        self.num_noise_points_h = self.bigN
        self.splines = self.setup_splines()

        pass
    
    def setInitialY0Config(self, y0, t0):
        """
        Sets the initial array y0 at time t=0.
        """
        if len(y0) != self.bigN:
            raise Exception(f"Length of input array is invalid len(y0) = {len(y0)} != {self.bigN}")
        self.y0 = np.array(y0, dtype=np.float64)  # Store as numpy array
        self.t0 = t0
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

    def run_simulation(self, y0=None, timeit=False):
        if timeit:
            t0 = time.time()
        
        if type(y0) != type(None):
            self.y0 = y0
            
        sol = solve_ivp(self.rhs, [0, self.time], self.y0.flatten(), method='RK45', 
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
    
    def run_in_chunks(self, backup_file, shape_save_freq, chunk_size = 100, timeit=False):
        """
        runs the simulatio in chunks
        if shape_save_freq == chunck_size, then 
        """
        if timeit:
            t0 = time.time()

        total_time_so_far = 0
        last_y0 = self.y0.copy()
        last_y = self.y0.copy()

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        if self.time % chunk_size != 0:
            print(f"self.time % chunk_size = {self.time % chunk_size} should equal 0")
            raise Exception("invalid chunk size")
        
        while total_time_so_far < self.time:
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            t_evals = np.arange(start_i, end_i, self.dt)

            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method='RK45', 
                            t_eval=t_evals,
                            rtol=self.rtol)
            
            y_i = sol_i.y.T
            current_chunk_y1 = y_i

            # Save selected shapes according to shape_save_freq
            indices = np.arange(0, len(current_chunk_y1), shape_save_freq)
            selected_times = t_evals[indices]
            selected_ys = current_chunk_y1[indices]

            self.selected_y_shapes.extend(zip(selected_times, selected_ys))

            last_y = y_i
            last_y0 = y_i[-1].T

            np.savez(backup_file, y_last=last_y0, params=self.getParameteters(), time=end_i)
            total_time_so_far += chunk_size

            cm_i = np.mean(current_chunk_y1, axis=1)
            try:
                v_cm_i = np.gradient(cm_i, self.dt)
                self.v_cm_history.append(v_cm_i)
            except:
                print(f"Failed to compute the gradient of centre of mass : {cm_i}")
        
        self.y1 = last_y

        self.used_timesteps = sol_i.t[1:] - sol_i.t[:-1] # Get the time steps used

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
        pass

    def run_until_relaxed(self, backup_file, chunk_size : int, shape_save_freq, timeit=False, tolerance=1e-6, method='RK45'):
        """
        When using this method to run the simulation, then self.time acts as the maximum simulation time, and chunk_size
        is the timespan from the end that will be saved for for further processing in methods such as getCM, getRelVelocity,
        getRoughness, etc.

        So when using this method, these other variables will not be computed from the "last 10%" of simulation time, unless
        chunk_size is one tenth of simulation time and it so happened that exactly ten chunks is used to achieve relaxation.

        Shape save freq is the number of dislocations saved from each chunk.
        """

        if timeit:
            t0 = time.time()
        
        if self.time % chunk_size != 0:
            print(self.time % chunk_size)
            raise Exception("invalid chunk size")
        chunk_size = int(chunk_size)

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        total_time_so_far = self.t0
        last_y = [self.y0]
        max_time = self.time

        relaxed = False
        # print("started while loop")
        while (total_time_so_far < max_time) and not relaxed:
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            last_y0 = last_y[-1]
            
            t_evals = np.arange(start_i, end_i, self.dt)
            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method=method, 
                            t_eval=t_evals,
                            rtol=self.rtol)
            y_i = sol_i.y.T
            self.used_timesteps = self.used_timesteps + sol_i.t[1:] - sol_i.t[:-1] # Get the time steps used

            last_y = y_i

            # Save selected shapes according to shape_save_freq
            indices = np.linspace(0, len(y_i) - 1, shape_save_freq, dtype=int)
            selected_times = t_evals[indices]
            selected_ys = y_i[indices]

            self.selected_y_shapes.extend(zip(selected_times, selected_ys))

            # Save a backup

            np.savez(backup_file, y_last=last_y0, params=self.getParameteters(), last_success_time=end_i, og_time=self.time,
                     v_cm_hist=self.getVCMhist())
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

        # Remove the backup file
        backup_file.unlink(missing_ok=True)

        if relaxed:
            self.y1 = last_y
            # print(f"relaxed shape {self.y1.shape}, len/time = {self.y1.shape[0]/self.time*100:2f}")
        else: # If not relaxed run one more chuck
            sol = solve_ivp(self.rhs, [total_time_so_far, total_time_so_far + chunk_size], last_y[-1].flatten(), method=method, 
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
    
    def setup_splines(self):
        """Creates cubic spline interpolators for the generated random noise."""
        # Define grid for the potential
        h_grid = np.linspace(0, 2*self.bigN, 2*self.bigN)
        
        # Generate random force values at grid points
        force_grid = self.stressField
        force_grid[:,-1] = force_grid[:,0]
        
        # Create a list of cubic spline interpolators, one for each x position
        splines = [CubicSpline(h_grid, force_grid[i, :], bc_type='periodic') for i in range(self.bigN)]

        return splines

    def calculate_forces(self, h):
        k = fft.rfftfreq(self.N, d=self.deltaL) * 2 * np.pi  # Wavevectors
        h_k = fft.rfft(h)
        laplacian_k = -(k**2) * h_k         # Second derivative in Fourier space
        line_tension_force = self.cLT1*self.mu*(self.smallB**2) * fft.irfft(laplacian_k, n=self.bigN)

        noise_force = np.array([self.splines[i](h[i]) for i in range(self.N)])

        return line_tension_force + noise_force

    def relax_w_FIRE(self):
        """
        Performs the FIRE relaxation of the dislocation line.
        """
        # Initialize dislocation line (e.g., as a straight line) and velocity
        h = np.zeros(self.bigN)
        v = np.zeros(self.bigN)

        # Initialize FIRE parameters
        dt = self.DT_INITIAL
        alpha = self.ALPHA_START
        steps_since_negative_power = 0

        print("Starting FIRE relaxation...")
        for step in range(self.MAX_STEPS):
            force = self.calculate_forces(h)

            # Check for convergence
            if np.linalg.norm(force) / np.sqrt(self.N) < self.CONVERGENCE_FORCE:
                print(f"✅ Converged after {step} steps.")
                break

            # FIRE dynamics
            if step > 0:
                power = np.dot(force, v)
                if power > 0:
                    steps_since_negative_power += 1
                    if steps_since_negative_power > self.N_MIN:
                        dt = min(dt * self.F_INC, self.DT_MAX)
                        alpha *= self.F_ALPHA
                else:
                    steps_since_negative_power = 0
                    dt *= self.F_DEC
                    v[:] = 0.0  # Reset velocity
                    alpha = self.ALPHA_START

            # Update velocity and position
            v = (1.0 - alpha) * v + alpha * (force / np.linalg.norm(force)) * np.linalg.norm(v)
            v += force * dt
            h += v * dt

            # Simple check to prevent divergence
            if np.any(np.isnan(h)):
                print("🛑 Simulation diverged. Halting.")
                break
        else:
            print("⚠️ Maximum steps reached without convergence.")

        return h
    
    def getLineProfiles(self):
        """
        Always returns the last state of the dislocation.
        """
        if self.has_simulation_been_run:
            return self.y1[-1].flatten()
        else:
            raise Exception("Simulation has not been run.")

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
            self.y0[0], self.seed, self.tau_cutoff          # 10 - 12
        ])
        return parameters
    
    @staticmethod
    def paramListToDict(param_list):
        bigN, length, time, dt, deltaR, bigB, smallB, cLT1, mu, tauExt, y0, seed, tau_cutoff = param_list 

        return {
            'bigN': bigN, 'length': length, 'time': time, 'dt': dt, 'deltaR': deltaR, 'bigB': bigB, 'smallB': smallB,
            'cLT1': cLT1, 'mu': mu, 'tauExt': tauExt, 'y0': y0, 'seed': seed, 'tau_cutoff': tau_cutoff
        }

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
    
    def getSelectedYshapes(self):
        """
        Returns a matrix of selected dislocation shapes over time.

        The `self.selected_y_shapes` list contains tuples of (time, dislocation shape).
        This function transforms this list into a NumPy array where each row represents
        a time point and the corresponding dislocation shape.

        The output matrix has the following structure:

        +------+------+------+------+-----+------+
        | time |  y1  |  y2  |  y3  | ... |  yN  |
        +------+------+------+------+-----+------+
        | time |  y1  |  y2  |  y3  | ... |  yN  |
        +------+------+------+------+-----+------+
        | time |  y1  |  y2  |  y3  | ... |  yN  |
        +------+------+------+------+-----+------+
        | ...  | ...  | ...  | ...  | ... | ...  |
        +------+------+------+------+-----+------+
        | time |  y1  |  y2  |  y3  | ... |  yN  |
        +------+------+------+------+-----+------+

        Where `time` is the time at which the dislocation shape was recorded, and
        `y1`, `y2`, ..., `yN` are the y-coordinates representing the shape of the
        dislocation at that time.

        Returns:
            np.ndarray: A NumPy array representing the selected dislocation shapes over time with the structure:
        """

        times = np.array([i[0] for i in self.selected_y_shapes])
        shapes = np.array([i[1] for i in self.selected_y_shapes])
        times = times.reshape(-1,1)
        selected_ys = np.hstack((times, shapes))

        return selected_ys
    @classmethod
    def from_dict(cls, params):
        """Create a DislocationSimulation from a dictionary of parameters"""
        sim = cls(
            bigN=params['bigN'],
            length=params['length'],
            time=params['time'],
            dt=params['dt'],
            deltaR=params['deltaR'],
            bigB=params['bigB'],
            smallB=params['smallB'],
            mu=params['mu'],
            tauExt=params['tauExt'],
            cLT1=params.get('cLT1', 2),
            seed=params.get('seed', None),
            rtol=params.get('rtol', 1e-6)
        )
        if 'y0' in params:
            sim.setInitialY0Config(params['y0'])
        return sim
    
    @classmethod
    def from_backup(cls, backup_file):
        """Create a DislocationSimulation from a backup file"""
        data = np.load(backup_file)
        
        params = data['params']
        paramsDict = DislocationSimulation.paramListToDict(params)

        og_time = paramsDict['time']
        fail_time = data['last_success_time']
        fail_y = data['y_last']

        # Parameters are stored in the order defined in getParameteters()
        sim = cls(
            bigN=int(params[0]),
            length=params[1],
            time=params[2],
            dt=params[3],
            deltaR=params[4],
            bigB=params[5],
            smallB=params[6],
            cLT1=params[7],
            mu=params[8],
            tauExt=params[9],
            seed=None if params[11] == 0 else int(params[11]),
            rtol=1e-6
        )
        # Set the last state as initial condition if available
        if 'y_last' in data:
            sim.setInitialY0Config(fail_y, fail_time)
        return sim


# For debugging
if __name__ == "__main__":
    # Run simulation with first instance and save backup
    backup_path = Path("results/7-7-relaksaatio/perfect/failsafes/backup-90dc59a8385e7a1e33cda3a0a7be94dc8add9552152da2016c6c49fffeb1d303.npz")
    backup_path.parent.mkdir(parents=True, exist_ok=True)

    # Load from backup file
    dislocation = DislocationSimulation(128, 128, 100, 1, 1000, 1, 1, 1, 1, 1)
    relaxed_h = dislocation.relax_w_FIRE()
    dislocation.setInitialY0Config(relaxed_h, 0)
    dislocation.run_until_relaxed("remove_me", 100/10, 1, True)

    pass