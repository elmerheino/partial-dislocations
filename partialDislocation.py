import numpy as np
from simulation import Simulation

class PartialDislocationsSimulation(Simulation):

    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, b_p, mu, tauExt, cLT1=2, cLT2=2, d0=40, c_gamma=50, seed=None):
        super().__init__(bigN, length, time, dt, deltaR, bigB, smallB, b_p, mu, tauExt, seed)
        
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1} and C_{LT2} (tension of the two lines)
        self.cLT2 = cLT2
        # TODO: make sure values of cLT1 and cLT2 align with the line tension tau, and mu

        self.c_gamma = c_gamma                  # Parameter in the interaction force, should be small

        self.d0 = d0                            # Initial distance separating the partials

        # Pre-allocate memory here
        self.y2 = np.empty((self.timesteps, self.bigN))
        self.y1 = np.empty((self.timesteps, self.bigN))
        
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

    def timestep(self, dt, y1,y2):
        dy1 = ( 
            self.cLT1*self.mu*(self.b_p**2)*self.secondDerivative(y1) # The gradient term # type: ignore
            + self.b_p*self.tau(y1) # The random stress term
            + self.force1(y1, y2) # Interaction force
            + (self.smallB/2)*self.tau_ext()*np.ones(self.bigN) # The external stress term
            ) * ( self.bigB/self.smallB )
        dy2 = ( 
            self.cLT2*self.mu*(self.b_p**2)*self.secondDerivative(y2) 
            + self.b_p*self.tau(y2) 
            + self.force2(y1, y2)
            + (self.smallB/2)*self.tau_ext()*np.ones(self.bigN) ) * ( self.bigB/self.smallB )
        
        newY1 = (y1 + dy1*dt)
        newY2 = (y2 + dy2*dt)

        self.time_elapsed += dt    # Update how much time has elapsed by adding dt

        return (newY1, newY2)


    def run_simulation(self):
        y10 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d
        self.y1[0] = y10

        y20 = np.zeros(self.bigN, dtype=float)
        self.y2[0] = y20

        for i in range(1,self.timesteps):
            y1_previous = self.y1[i-1]
            y2_previous = self.y2[i-1]

            (y1_i, y2_i) = self.timestep(self.dt,y1_previous,y2_previous)
            self.y1[i] = y1_i
            self.y2[i] = y2_i
        
        self.has_simulation_been_run = True
    
    def run_further(self, new_time:int, new_dt:int = 0.05):
        # Runs the simulation further in time with a new timestep if need be
        # self.dt = new_dt
        # TODO: Implement the possibility to change the timestep
        raise Exception("Method does not work yet.")

        self.time = self.time + new_time
        new_timesteps = round(new_time/self.dt)

        for i in range(self.timesteps,self.timesteps+new_timesteps):
            y1_previous = self.y1[i-1]
            y2_previous = self.y2[i-1]

            y1_i = self.timestep(self.dt,y1_previous,y2_previous)
            self.y1[i] = y1_i
            self.y2[i] = y1_i
        
        self.timesteps += new_timesteps # Update the no. of timesteps so that getTValues will function properly

        return 0
        
    def getLineProfiles(self, time_to_consider=None):
        start = 0

        if time_to_consider != None:
            steps_to_consider = round(time_to_consider / self.dt)
            start = self.timesteps - steps_to_consider
        if time_to_consider == self.time: # In this case only return the last state
            start = self.timesteps - 1

        if self.has_simulation_been_run:
            return (self.y1[start:], self.y2[start:])
        
        print("Simulation has not been run yet.")
        return (self.y1, self.y2) # Retuns empty lists
    
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
    
    def getRelaxedVelocity(self, time_to_consider=1000):
        if len(self.y1) == 0 or len(self.y2) == 0:
            raise Exception('simulation has probably not been run')
        
        # Returns the velocities of the centres of mass
        y1_CM, y2_CM, tot_CM = self.getCM()
        v1_CM = np.gradient(y1_CM)
        v2_CM = np.gradient(y2_CM)
        vTot_CM = np.gradient(tot_CM)

        # Condisering only time_to_consider seconds from the end
        start = round(self.timesteps - time_to_consider/self.dt)

        v_relaxed_y1 = np.average(v1_CM[start:self.timesteps])
        v_relaxed_y2 = np.average(v2_CM[start:self.timesteps])
        v_relaxed_tot = np.average(vTot_CM[start:self.timesteps])

        return (v_relaxed_y1, v_relaxed_y2, v_relaxed_tot)
    
    def getAveragedRoughness(self, time_to_consider):
        steps_to_consider = round(time_to_consider / self.dt)
        start = self.timesteps - steps_to_consider

        l_range, _ = self.roughnessW12(self.y1[0], self.y2[0], self.bigN)

        roughnesses = np.empty((steps_to_consider, self.bigN))
        for i in range(start,self.timesteps):
            _, rough = self.roughnessW12(self.y1[i], self.y2[i], self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg

    def getParameters(self):
        parameters = np.array([
            self.bigN, self.length, self.time, self.dt,                 # Index 0 - 3
            self.deltaR, self.bigB, self.smallB, self.b_p,              # Index 4 - 7
            self.cLT1, self.cLT2, self.mu, self.tauExt, self.c_gamma,   # Index 8 - 15
            self.d0, self.seed, self.tau_cutoff                         # Index 16 - 18
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
