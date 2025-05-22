import time
import numpy as np
from scipy import fft
from simulation import Simulation
from scipy.integrate import solve_ivp

class DislocationSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, cLT1=2, seed=None, d0=10, rtol=1e-6):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1}
        self.d0 = d0
        
        # Pre-allocate memory here
        #self.y1 = np.empty((self.timesteps*10, self.bigN)) # Each timestep is one row, bigN is number of columns
        self.y1 = list() # List of arrays
        self.errors = list()
        self.used_timesteps = 0
        self.rtol = rtol

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

        sol = solve_ivp(self.rhs, [0, self.time], y0.flatten(), method='RK45', t_eval=np.arange(0,self.time, self.dt), rtol=self.rtol)

        self.y1 = sol.y.T

        self.used_timesteps = sol.t[1:] - sol.t[:-1] # Get the time steps used

        self.timesteps = len(self.y1)
        
        self.has_simulation_been_run = True

        if timeit:
            t1 = time.time()
            print(f"Time taken for simulation: {t1 - t0}")
    
    def getLineProfiles(self, time_to_consider=None):
        start = 0

        if time_to_consider != None:
            ratio = time_to_consider/self.time
            start = round(self.timesteps*(1-0.1))
        if time_to_consider == self.time: # In this case only return the last state
            start = self.timesteps - 1
        elif time_to_consider == None: # Also return the last state
            start = self.timesteps - 1

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
    
    def getRelaxedVelocity(self, time_to_consider):
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')
        
        # Returns the velocities of the centres of mass
        y1_CM = self.getCM()
        v1_CM = np.gradient(y1_CM)

        # Condisering only time_to_consider seconds from the end

        start = round(self.timesteps*(1 - 0.1)) # Consider only the last 10% of the simulation

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

    def getAveragedRoughness(self, time_to_consider):
        start = round(self.timesteps*(1 - 0.1)) # Consider only the last 10% of the simulation

        l_range, _ = self.roughnessW(self.y1[0], self.bigN)

        roughnesses = np.empty((start, self.bigN))
        for i in range(start,self.timesteps):
            _, rough = self.roughnessW(self.y1[i], self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg
    
    def getErrors(self):
        return self.errors
    
    def retrieveUsedTimesteps(self):
        return self.used_timesteps