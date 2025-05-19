import numpy as np
from scipy import fft
from simulation import Simulation

class DislocationSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, cLT1=2, seed=None, d0=40):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1}
        self.d0 = d0
        
        # Pre-allocate memory here
        self.y1 = np.empty((self.timesteps, self.bigN)) # Each timestep is one row, bigN is number of columns
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

        error = np.abs(fifth_order - fourth_order) # Error of the 4th order solution

        # TODO: update the time step size dt based on the error

        self.time_elapsed += dt    # Update how much time has elapsed by adding dt

        return fourth_order

    def run_simulation(self):
        y10 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d
        self.y1[0] = y10

        for i in range(1,self.timesteps):
            y1_previous = self.y1[i-1]

            y1_i = self.timestep(self.dt,y1_previous)
            self.y1[i] = y1_i
        
        self.has_simulation_been_run = True
    
    def getLineProfiles(self, time_to_consider=None):
        start = 0

        if time_to_consider != None:
            steps_to_consider = round(time_to_consider / self.dt)
            start = self.timesteps - steps_to_consider
        if time_to_consider == self.time: # In this case only return the last state
            start = self.timesteps - 1

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
        # Return the centres of mass of the two lines as functions of time
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')

        y1_CM = np.mean(self.y1, axis=1)

        return y1_CM
    
    def getRelaxedVelocity(self, time_to_consider=1000):
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')
        
        # Returns the velocities of the centres of mass
        y1_CM = self.getCM()
        v1_CM = np.gradient(y1_CM)

        # Condisering only time_to_consider seconds from the end
        start = round(self.timesteps - time_to_consider/self.dt)

        v_relaxed_y1 = np.average(v1_CM[start:self.timesteps])

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
        steps_to_consider = round(time_to_consider / self.dt)
        start = self.timesteps - steps_to_consider

        l_range, _ = self.roughnessW(self.y1[0], self.bigN)

        roughnesses = np.empty((steps_to_consider, self.bigN))
        for i in range(start,self.timesteps):
            _, rough = self.roughnessW(self.y1[i], self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg