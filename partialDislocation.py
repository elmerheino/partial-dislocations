import hashlib
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import CubicSpline
from simulation import Simulation
from scipy.integrate import solve_ivp
import time
from pathlib import Path
class PartialDislocationsSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, b_p, mu, tauExt, cLT1=2, cLT2=2, d0=40, c_gamma=50, seed=None, rtol=1e-8):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1} and C_{LT2} (tension of the two lines)
        self.cLT2 = cLT2
        # TODO: make sure values of cLT1 and cLT2 align with the line tension tau, and mu

        self.c_gamma = c_gamma                  # Parameter in the interaction force, should be small
        self.d0 = d0                            # Initial distance separating the partials
        self.b_p = b_p

        self.y0 = np.ones((2, self.bigN), dtype=float)*self.d0
        self.y0 = np.vstack((
            np.ones(self.bigN)*self.d0,     # y1
            np.zeros(self.bigN)             # y2 ensure that in the beginning y1 > y2
        ))

        self.y2 = list()
        self.y1 = list()

        self.used_timesteps = [self.dt] # List of the used timesteps
        self.errors = list()         # List of the errors

        self.rtol = rtol

        self.avg_v_cm_history = list()
        self.avg_stacking_fault_history = list()

        self.selected_y1_shapes = list()
        self.selected_y2_shapes = list()

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

    
    def setInitialY0Config(self, y1_0, y2_0):
        """
        Sets the intial arrays y1 and y2 at time t=0.
        """
        if len(y1_0) != self.bigN or len(y2_0) != self.bigN :
            print(f"Length of input array is invalid len(y1_0) = {len(y1_0)} != {self.bigN}")
            print(y1_0)
        self.y0 = np.vstack([y1_0, y2_0])
        pass
        
    def force1(self, y1,y2):
        #return -np.average(y1-y2)*np.ones(bigN)
        #return -(c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

        factor = (1/self.d0)*self.c_gamma*self.mu*(self.b_p**2)
        numerator = ( np.average(y2) - np.average(y1) )*np.ones(self.bigN)
        return factor*(1 + numerator/self.d0)

    def force2(self, y1,y2):
        #return np.average(y1-y2)*np.ones(bigN)
        #return (c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

        factor = -(1/self.d0)*self.c_gamma*self.mu*(self.b_p**2)
        numerator = ( np.average(y2) - np.average(y1) )*np.ones(self.bigN)
        return factor*(1 + numerator/self.d0) # Term from Vaid et Al B.7

    def f1(self, y1,y2, t):
        dy = ( 
            self.cLT1*self.mu*(self.b_p**2)*self.secondDerivative(y1) # The gradient term # type: ignore
            + self.b_p*self.tau(y1) # The random stress term
            + self.force1(y1, y2) # Interaction force
            + (self.smallB/2)*self.tau_ext(t)*np.ones(self.bigN) # The external stress term
            ) * ( self.bigB/self.smallB )

        return dy

    def f2(self, y1,y2,t):
        dy = ( 
            self.cLT2*self.mu*(self.b_p**2)*self.secondDerivative(y2) 
            + self.b_p*self.tau(y2) 
            + self.force2(y1, y2)
            + (self.smallB/2)*self.tau_ext(t)*np.ones(self.bigN) ) * ( self.bigB/self.smallB )

        return dy
    
    def rhs(self, t, u_flat : np.ndarray):
        u = u_flat.reshape(2, self.bigN)
        
        dudt = np.array([
            self.f1(u[0], u[1], t), # y1
            self.f2(u[0], u[1], t)  # y2
        ])
        
        return dudt.flatten()

    def run_simulation(self, evaluate_from=0.1):                                # By default only evaluate from the last 10 % of simulation time

        y0 = self.y0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d

        time_evals = np.arange(self.time*(1 - evaluate_from),self.time, self.dt) # Evaluate the solution only from the last 10% of the time every dt

        sol = solve_ivp(self.rhs, [0, self.time], y0.flatten(), method='RK45', t_eval=time_evals, rtol=self.rtol)

        sol.y = sol.y.reshape(2, self.bigN, -1) # Reshape the solution to have 2 lines
        self.y1 = sol.y[0].T
        self.y2 = sol.y[1].T

        self.used_timesteps = sol.t[1:] - sol.t[:-1] # Get the time steps used

        # print(self.y1.shape, self.y2.shape, sol.y.shape)

        self.y1 = np.array(self.y1)
        self.y2 = np.array(self.y2)
        self.timesteps = len(self.y1) # Update the number of timesteps

        self.has_simulation_been_run = True

    def run_in_chunks(self, backup_file, chunk_size = 100, timeit=False, method='RK45'):

        if timeit:
            t0 = time.time()

        y0 = self.y0

        total_time_so_far = 0
        last_y0 = y0

        last_y = y0

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        if self.time % chunk_size != 0:
            print(f"self.time % chunk_size = {self.time % chunk_size} should equal 0")
            raise Exception("invalid chunk size")
        
        while total_time_so_far < self.time:
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method=method, 
                            t_eval=np.arange(start_i, end_i, self.dt),
                            rtol=self.rtol, atol=1e-14)
            
            if not sol_i.success:
                raise RuntimeError(f"solve_ivp failed: {sol_i.message}")
            
            y_i = sol_i.y.reshape(2, self.bigN, -1)

            current_chunk_y1 = y_i[0].T
            current_chunk_y2 = y_i[1].T

            last_y = y_i

            last_y0 = y_i[:, :, -1]
            np.savez(backup_file, y_last=last_y0, params=self.getParameters(), time=end_i)
            total_time_so_far += chunk_size

            y1_CM_i = np.mean(current_chunk_y1, axis=1)
            y2_CM_i = np.mean(current_chunk_y2, axis=1)
            total_CM_i = (y1_CM_i + y2_CM_i) / 2

            sf_width = np.mean(current_chunk_y1 - current_chunk_y2, axis=1)
            self.avg_stacking_fault_history.append(sf_width)

            if len(total_CM_i) > 2:
                v_cm_i = np.gradient(total_CM_i, self.dt)
                self.avg_v_cm_history.append(v_cm_i)
            else:
                print(f"not true len(total_cm_i) > 2, total_cm_i = {total_CM_i}, start_i = {start_i}, end_i = {end_i}")
        
        self.y1 = last_y[0].T
        self.y2 = last_y[1].T

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.y2 = np.array(self.y2)
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
        pass

    def run_until_relaxed(self, backup_file, chunk_size : int, timeit=False, tolerance=1e-6, shape_save_freq=1, method='RK45'):
        """
        When using this method to run the simulation, then self.time acts as the maximum simulation time, and chunck_size
        is the timespan from the end that will be saved for for further processing in methods such as getCM, getRelVelocity,
        getRoughness, etc.

        So when using this method, these other varibaled will not be computed from the "last 10%" of simulation time, unless
        chunk_size is one tenth of it and it so happend that exactly ten chunks is used to achieve relaxation.
        """

        if timeit:
            t0 = time.time()
        
        chunk_size = int(chunk_size)

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        y0 = self.y0 # Initial condition for two partials
        last_y0 = y0

        total_time_so_far = 0
        max_time = self.time

        relaxed = False
        
        while total_time_so_far < max_time:
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            t_evals = np.arange(start_i, end_i, self.dt)
            
            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method=method, 
                            t_eval=t_evals,
                            rtol=self.rtol)
            
            y_i = sol_i.y.reshape(2, self.bigN, -1)
            
            current_chunk_y1 = y_i[0].T
            current_chunk_y2 = y_i[1].T
            current_chunk_timesteps = sol_i.t[1:] - sol_i.t[:-1]

            last_y0 = y_i[:, :, -1]
            np.savez(backup_file, y_last=last_y0, params=self.getParameters(), time=end_i)
            total_time_so_far += chunk_size

            y1_CM_i = np.mean(current_chunk_y1, axis=1)
            y2_CM_i = np.mean(current_chunk_y2, axis=1)
            total_CM_i = (y1_CM_i + y2_CM_i) / 2

            sf_width = np.mean(current_chunk_y1 - current_chunk_y2, axis=1)

            # Save selected dislocation line shapes
            indices = np.linspace(0, len(y_i) - 1, shape_save_freq, dtype=int)
            selected_times = t_evals[indices]
            selected_y1s = current_chunk_y1[indices]
            selected_y2s = current_chunk_y2[indices]

            self.selected_y1_shapes.extend(zip(selected_times, selected_y1s))
            self.selected_y2_shapes.extend(zip(selected_times, selected_y2s))

            if len(total_CM_i) > 2:
                v_cm_i = np.gradient(total_CM_i, self.dt).flatten()

                self.avg_v_cm_history.append(v_cm_i)        # Save velocity for later use
                self.avg_stacking_fault_history.append(sf_width)

                if self.is_relaxed(v_cm_i, tolerance=tolerance):
                    relaxed = True
                    self.y1 = current_chunk_y1
                    self.y2 = current_chunk_y2
                    self.used_timesteps = current_chunk_timesteps
                    break # end the simulation here.

            # If timeit is enabled print some info on progress
            if timeit:
                t_elapsed = time.time() - t0
                percentage_done = (total_time_so_far / max_time) * 100
                print(f"Progress: {percentage_done:.2f}%, Time elapsed: {t_elapsed:.2f} seconds")

        if not relaxed:
            # If not relaxed after max_time, run one more chunk
            start_t = total_time_so_far
            end_t = total_time_so_far + chunk_size
            sol = solve_ivp(self.rhs, [start_t, end_t], last_y0.flatten(), method=method, 
                                t_eval=np.arange(start_t, end_t, self.dt),
                                rtol=self.rtol)
            
            sol.y = sol.y.reshape(2, self.bigN, -1)
            self.y1 = sol.y[0].T
            self.y2 = sol.y[1].T
            self.used_timesteps = sol.t[1:] - sol.t[:-1]

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.y2 = np.array(self.y2)
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
        
        return relaxed
    
    # From there on define all the methods related to FIRE relaxation.
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
    
    def weak_coupling(self, h1, h2):
        d_avg = np.abs(np.mean(h1 - h2))
        return self.d0 / d_avg - 1

    def fire_force1(self, h1, h2):
        # return self.weak_coupling(h1, h2)
        return self.force1(h1, h2)

    def fire_force2(self, h1, h2):
        # return self.weak_coupling(h1,h2)
        return self.force2(h1, h2)
    
    def calculate_forces_FIRE(self, h1, h2):
        """
        Calculates forces using Fourier method for line tension and spline derivatives for noise.
        """
        # 1. Line Tension Force of first partial (via Fourier Domain)
        k = rfftfreq(self.bigN, d=self.deltaL) * 2 * np.pi  # Wavevectors
        h1_k = rfft(h1)
        laplacian_k1 = -(k**2) * h1_k         # Second derivative in Fourier space
        line_tension_force1 = self.cLT1 * irfft(laplacian_k1, n=self.bigN)

        # 2. Line tension of the second partial
        k = rfftfreq(self.bigN, d=self.deltaL) * 2 * np.pi  # Wavevectors
        h2_k = rfft(h2)
        laplacian_k2 = -(k**2) * h2_k         # Second derivative in Fourier space
        line_tension_force2 = self.cLT2 * irfft(laplacian_k2, n=self.bigN)

        # 2. Quenched Noise Force (from splines)

        noise_force1 = np.array([self.splines[i](h1[i]) for i in range(self.bigN)])
        noise_force2 = np.array([self.splines[i](h2[i]) for i in range(self.bigN)])

        force1_tot = line_tension_force1 + noise_force1 + self.fire_force1(h1, h2)
        force2_tot = line_tension_force2 + noise_force2 + self.fire_force2(h1, h2)

        return force1_tot, force2_tot

    
    def relax_w_FIRE(self):
        """
        Performs the FIRE relaxation of the dislocation line.
        """
        # Initialize dislocation line (e.g., as a straight line) and velocity
        h1 = np.zeros(self.bigN)
        h2 = np.ones(self.bigN)*self.d0

        v1 = np.zeros(self.bigN)
        v2 = np.zeros(self.bigN)

        # Initialize FIRE parameters
        dt = self.DT_INITIAL
        alpha = self.ALPHA_START
        steps_since_negative_power = 0

        print("Starting FIRE relaxation...")
        for step in range(self.MAX_STEPS):
            force1, force2 = self.calculate_forces_FIRE(h1, h2)

            # Check for convergence
            if (np.linalg.norm(force1) / np.sqrt(self.bigN) < self.CONVERGENCE_FORCE) and (np.linalg.norm(force2) / np.sqrt(self.bigN) < self.CONVERGENCE_FORCE):
                print(f"✅ Force 1 and 2 Converged after {step} steps.")
                break

            # FIRE dynamics
            if step > 0:
                power1 = np.dot(force1, v1)
                power2 = np.dot(force2, v2)
                if (power1 > 0) and (power2 > 0):
                    steps_since_negative_power += 1
                    if steps_since_negative_power > self.N_MIN:
                        dt = min(dt * self.F_INC, self.DT_MAX)
                        alpha *= self.F_ALPHA
                else:
                    steps_since_negative_power = 0
                    dt *= self.F_DEC
                    v1[:] = 0.0  # Reset velocity
                    v2[:] = 0.0
                    alpha = self.ALPHA_START

            # Update velocity for the first equation and position
            v1 = (1.0 - alpha) * v1 + alpha * (force1 / np.linalg.norm(force1)) * np.linalg.norm(v1)
            v1 += force1 * dt
            h1 += v1 * dt

            v2 = (1.0 - alpha) * v2 + alpha * (force2 / np.linalg.norm(force2)) * np.linalg.norm(v2)
            v2 += force2 * dt
            h2 += v2 * dt

            # Simple check to prevent divergence
            if np.any(np.isnan(h1)) or np.any(np.isnan(h2)):
                print("🛑 Simulation diverged. Halting.")
                break

        else:
            print("⚠️ Maximum steps reached without convergence.")
            
        return h1, h2

    def getLineProfiles(self):
        """
        Always returns the line shape at the end of the simulation.
        """
        if self.has_simulation_been_run:
            return (self.y1[-1], self.y2[-1])
        
        print("Simulation has not been run yet.")
        return (self.y1, self.y2)
    
    def getAverageDistances(self):
        return np.average(self.y1, axis=1) - np.average(self.y2, axis=1)
    
    def getCM(self):
        # Return the centres of mass of the two lines as functions of time
        if len(self.y1) == 0 or len(self.y2) == 0:
            raise Exception('simulation has probably not been run')

        y1_CM = np.mean(self.y1, axis=1)
        y2_CM = np.mean(self.y2, axis=1)

        total_CM = (y1_CM + y2_CM)/2    # TODO: This is supposed to be the centre of mass of the entire system

        return (y1_CM,y2_CM,total_CM)
    
    def getRelaxedVelocity(self):
        # Returns the relaxed velocity of the two lines and the total velocity from the last 10% of the simulation time
        if len(self.y1) == 0 or len(self.y2) == 0:
            raise Exception('simulation has probably not been run')
        
        # Returns the velocities of the centres of mass
        y1_CM, y2_CM, tot_CM = self.getCM()
        v1_CM = np.gradient(y1_CM)
        v2_CM = np.gradient(y2_CM)
        vTot_CM = np.gradient(tot_CM)

        # Condisering only time_to_consider seconds from the end
        start = 0

        v_relaxed_y1 = np.mean(v1_CM[start:])
        v_relaxed_y2 = np.mean(v2_CM[start:])
        v_relaxed_tot = np.mean(vTot_CM[start:])

        return (v_relaxed_y1, v_relaxed_y2, v_relaxed_tot)
    
    def getAveragedRoughness(self):
        # Returns the averaged roughness of the two lines from their centre of mass (assuming equal mass distribution)x
        # Averages the roughness over the last 10% of the simulation time

        start = 0

        l_range, _ = self.roughnessW((self.y1[0] + self.y2[0])/2, self.bigN)

        roughnesses = np.empty((self.timesteps, self.bigN))
        for i in range(start,self.timesteps):
            centre_of_mass = (self.y1[i] + self.y2[i])/2
            _, rough = self.roughnessW(centre_of_mass, self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg

    def getParameters(self):
        parameters = np.array([
            self.bigN, self.length, self.time, self.dt,                 # Index 0 - 3
            self.deltaR, self.bigB, self.smallB, self.b_p,              # Index 4 - 7
            self.cLT1, self.cLT2, self.mu, self.tauExt, self.c_gamma,   # Index 8 - 12
            self.d0, self.seed, self.tau_cutoff                         # Index 13 - 15
        ])
        return parameters
    
    def getSelectedYshapes(self):
        """
        Return two matriced with the structure, one matric for each partial:

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
        """
        times1 = np.array([i[0] for i in self.selected_y1_shapes])
        shapes1 = np.array([i[1] for i in self.selected_y1_shapes])
        times1 = times1.reshape(-1,1)
        selected_y1s = np.hstack([times1, shapes1])

        times2 = np.array([i[0] for i in self.selected_y2_shapes])
        shapes2 = np.array([i[1] for i in self.selected_y2_shapes])
        times2 = times2.reshape(-1,1)
        selected_y2s = np.hstack([times2, shapes2])

        return selected_y1s, selected_y2s
    
    def getResultsAsDict(self):
        v_cm_hist = self.getVCMhist()
        y_t = self.getLineProfiles()
        parameters = self.getParameters()
        selected_y1, selected_y2 = self.getSelectedYshapes()
        sf_widths = self.getSFhist()

        return { 'v_cm_hist' : v_cm_hist, 'y_last' : y_t, "selected_y1" : selected_y1, "selected_y2" : selected_y2,
                'params' : parameters, 'sf_width' : sf_widths }
    
    def saveResults(self, path : Path):
        if not self.has_simulation_been_run:
            raise Exception('Simulation has not been run yet.')
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(path)
        np.savez(path, **self.getResultsAsDict())
        print("saved")

    
    @staticmethod
    def paramListToDict(params_list):
        bigN, length, time, dt,   deltaR,  bigB,  smallB,  b_p,   cLT1,  cLT2,  mu,  tauExt,  c_gamma,   d0,  seed,  tau_cutoff  = params_list
        return {
            'bigN': bigN,
            'length': length,
            'time': time, 
            'dt': dt,
            'deltaR': deltaR,
            'bigB': bigB,
            'smallB': smallB,
            'b_p': b_p,
            'cLT1': cLT1,
            'cLT2': cLT2,
            'mu': mu,
            'tauExt': tauExt,
            'c_gamma': c_gamma,
            'd0': d0,
            'seed': seed,
            'tau_cutoff': tau_cutoff
        }

    
    def calculateC_gamma(self, v=1, theta=np.pi/2):
        # Calulcates the C_{\gamma} parameter based on dislocation character
        # according to Vaid et al. (12)

        secondTerm = 1 - (2*v*np.cos(2*theta))
        c_gamma = secondTerm*(2 - v)/(8*np.pi*(1-v))

    def equilibriumDistance(self, gamma=60.5, tau_gamma=486):
        # Calculated the supposed equilibrium distance of the two lines
        # according to Vaid et al. (11)

        gamma = 60.5 # Stacking fault energy, gamma is also define through sf stress \tau_{\gamma}*b_p
        d0 = (self.c_gamma*self.mu/gamma)*self.b_p**2

    def getParamsInLatex(self):
        return super().getParamsInLatex()+ [f"C_{{LT1}} = {self.cLT1}", f"C_{{LT2}} = {self.cLT2}"]
    
    def getTitleForPlot(self, wrap=6):
        parameters = self.getParamsInLatex()
        plot_title = " ".join([
            "$ "+ i + " $" + "\n"*(1 - ((n+1)%wrap)) for n,i in enumerate(parameters) # Wrap text using modulo
        ])
        return plot_title
    
    def getErrors(self):
        if len(self.errors) == 0:
            raise Exception('simulation has probably not been run')
        
        return self.errors

    def retrieveUsedTimesteps(self):
        if len(self.used_timesteps) == 0:
            raise Exception('simulation has probably not been run')
        
        return self.used_timesteps
    
    def getSFhist(self):
        return np.array(self.avg_stacking_fault_history)
    
    def getVCMhist(self):
        return np.array(self.avg_v_cm_history).flatten()
    
    def getUniqueHash(self):
        params = self.getParameters()
        params_str = np.array2string(params)  # Convert array to string
        hash_object = hashlib.sha256(params_str.encode())
        hex_dig = hash_object.hexdigest()
        return hex_dig
    
    def getUniqueHashString(self):
        return self.getUniqueHash()


if __name__ == "__main__":
    # Example usage of the PartialDislocationsSimulation class
    sim = PartialDislocationsSimulation(
        bigN=2,           # Number of points
        length=2,        # Length of dislocation
        time=100,          # Total simulation time
        dt=1,            # Time step
        deltaR=1.0,         # Random force correlation length
        bigB=1.0,          # Drag coefficient
        smallB=1.0,         # Burgers vector
        b_p=1.0,           # Partial Burgers vector
        mu=1.0,            # Shear modulus
        tauExt=1.0         # External stress
    )
    # sim.run_simulation()
    # sim.run_until_relaxed("debug/jtn.npz", chunk_size=sim.time/10, shape_save_freq=1, timeit=True, method='RK45')
    # sim.saveResults("debug/jtn/tulokset.npz")
    sim.relax_w_FIRE()
