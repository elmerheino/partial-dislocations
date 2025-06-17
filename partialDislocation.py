import numpy as np
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

        self.y2 = list()
        self.y1 = list()

        self.used_timesteps = [self.dt] # List of the used timesteps
        self.errors = list()         # List of the errors

        self.rtol = rtol
        
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

        y0 = np.ones((2, self.bigN), dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d

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

    def run_in_chunks(self, backup_file, chunk_size = 100, timeit=False):

        if timeit:
            t0 = time.time()

        y0 = np.ones((2, self.bigN), dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d

        total_time_so_far = 0
        last_y0 = y0

        backup_file = Path(backup_file)
        backup_file.parent.mkdir(exist_ok=True, parents=True)

        if self.time % chunk_size != 0:
            print(f"self.time % chunk_size = {self.time % chunk_size} should equal 0")
            raise Exception("invalid chunk size")
        
        while total_time_so_far < self.time*(1 - 0.1):
            start_i = total_time_so_far
            end_i = total_time_so_far + chunk_size

            sol_i = solve_ivp(self.rhs, [start_i, end_i], last_y0.flatten(), method='RK45', 
                            t_eval=[end_i],
                            rtol=self.rtol)
            
            y_i = sol_i.y.reshape(2, self.bigN, -1)

            last_y0 = y_i[:, :, -1]
            np.savez(backup_file, y_last=last_y0, params=self.getParameters(), time=end_i)
            total_time_so_far += chunk_size
        
        sol = solve_ivp(self.rhs, [self.time*(1 - 0.1), self.time], last_y0.flatten(), method='RK45', 
                            t_eval=np.arange(self.time*(1 - 0.1), self.time, self.dt),
                            rtol=self.rtol)

        sol.y = sol.y.reshape(2, self.bigN, -1) # Reshape the solution to have 2 lines
        self.y1 = sol.y[0].T
        self.y2 = sol.y[1].T

        self.used_timesteps = sol.t[1:] - sol.t[:-1] # Get the time steps used

        self.y1 = np.array(self.y1) # Convert to numpy array
        self.y2 = np.array(self.y2)
        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
        pass
    def getLineProfiles(self):
        start = 0

        start = self.timesteps - 1

        if self.has_simulation_been_run:
            return (self.y1[start:], self.y2[start:])
        
        print("Simulation has not been run yet.")
        return (self.y1, self.y2) # Retuns empty lists
    
    def getAverageDistances(self):
        return np.average(self.y1, axis=1) - np.average(self.y2, axis=1)
    
    def getCM(self):
        # Return the centres of mass of the two lines as functions of time (from the last 10% of simulation time)
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