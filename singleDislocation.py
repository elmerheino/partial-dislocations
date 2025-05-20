import numpy as np
from scipy import fft
from simulation import Simulation

class DislocationSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, cLT1=2, seed=None, d0=40):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1}
        self.d0 = d0
        
        # Pre-allocate memory here
        #self.y1 = np.empty((self.timesteps*10, self.bigN)) # Each timestep is one row, bigN is number of columns
        self.y1 = list() # List of arrays
        self.errors = list()
        self.used_timesteps = [self.dt] # it should be that len(y1) = len(used_timesteps)

        pass
    
    def funktio(self, y1):
        dy1 = (
            self.cLT1*self.mu*(self.smallB**2)*self.secondDerivative(y1) # The gradient term # type: ignore
            + self.smallB*(self.tau(y1) + self.tau_ext()*np.ones(self.bigN) )
        ) * ( self.bigB/self.smallB )
        return dy1


    def timestep(self, dt, y):
        k1 = self.funktio(y)
        k2 = self.funktio(y + dt*k1/5)
        k3 = self.funktio( y + dt*(k1*3/40 + k2*9/40) )
        k4 = self.funktio( y + dt*(k1*44/45 - k2*56/15 + k3*32/9) )
        k5 = self.funktio( y + dt*(k1*19372/6561 - k2*25360/2187 + k3*64448/6561 - k4*212/729) )
        k6 = self.funktio( y + dt*(k1*9017/3168 - k2*355/33 + k3*46732/5247 + k4*49/176 - k5*5103/18656) )
        k7 = self.funktio( y + dt*(k1*35/384 + k3*500/1113 + k4*125/192 - k5*2187/6784 + k6*11/84) ) # This has been evaluated at the same point as the k1 of the next step

        
        fifth_order = y + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
        fourth_order = y + dt*(5179/57600*k1 + 7571/16695*k3 + 393/640*k4 - 92097/339200*k5 + 187/2100*k6 + 1/40*k7)

        return fourth_order, fifth_order

    def run_simulation(self):
        y0 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d
        self.y1.append( y0 )

        i = 0
        while self.time_elapsed <= self.time:

            y1_previous = self.y1[i-1]

            fourth_i, fifth_i = self.timestep(self.dt,y1_previous)

            abstol = 1e-6
            reltol = 1e-6

            # sc_i = abstol + reltol*max(np.linalg.norm(fourth_i), np.linalg.norm(y1_previous))
            sc = lambda x : abstol + reltol*max(fourth_i[x], y1_previous[x])

            error = sum([ (    (y5i - y4i)/( sc(n) ) )**2 for n, (y5i, y4i) in enumerate( zip(fifth_i, fourth_i) )])/len(fifth_i)
            error = np.sqrt(error)

            h_opt = 0.25 * self.dt * (1 / error )**(1/5) # Optimal step size

            min_dt = 1e-6
            max_dt = 5

            if error < 0.5: # Accept the step and move to the next timestep and adjust the timestep
                i += 1
                self.time_elapsed += self.dt
                self.y1.append(fourth_i)
                self.errors.append(error)              # Error of the 4th order solution
                self.used_timesteps.append(self.dt)  # Store the optimal timestep for each step
                print(f"Step {i}, \t dt = {self.dt:.5f}, \t error = {error:.5f}, \t h_opt = {h_opt:.5f}")
            else:   # Reject the step and only reduce the timestep
                # Error is too big, so we need to reduce the timestep, thus continuing the loop
                print(f"Adjusting Step {i}, \t dt = {self.dt:.5f}, \t error = {error:.5f}, \t h_opt = {h_opt:.5f}")
                pass

            if h_opt < min_dt:
                self.dt = min_dt
            elif h_opt > max_dt:
                self.dt = max_dt
            else:
                self.dt = h_opt
        
        self.y1 = np.array(self.y1) # Convert to numpy array
        
        self.has_simulation_been_run = True
    
    def getLineProfiles(self, time_to_consider=None):
        start = 0

        avg_dt = np.average(np.array(self.used_timesteps))
        timesteps = len(self.y1)

        if time_to_consider != None:
            steps_to_consider = round(time_to_consider / avg_dt)
            start = timesteps - steps_to_consider
        if time_to_consider == self.time: # In this case only return the last state
            start = timesteps - 1

        if self.has_simulation_been_run:
            return self.y1[start:]
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
        dt_average = np.average(np.array(self.used_timesteps))
        print(dt_average)

        start = round(len(self.used_timesteps) - time_to_consider/dt_average)
        print(start)

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
            self.deltaR, self.bigB, self.smallB,  # 4 - 7
            self.cLT1, self.mu, self.tauExt,                # 8 - 10
            self.d0, self.seed, self.tau_cutoff             # 11 - 13
        ])
        return parameters

    def getAveragedRoughness(self, time_to_consider):
        timesteps = len(self.used_timesteps)
        avg_dt = np.average(np.array(self.used_timesteps))
        steps_to_consider = round(time_to_consider / avg_dt)
        start = len(self.used_timesteps) - steps_to_consider

        l_range, _ = self.roughnessW(self.y1[0], self.bigN)

        roughnesses = np.empty((steps_to_consider, self.bigN))
        for i in range(start,timesteps):
            _, rough = self.roughnessW(self.y1[i], self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg
    
    def getErrors(self):
        return self.errors
    
    def retrieveUsedTimesteps(self):
        return self.used_timesteps