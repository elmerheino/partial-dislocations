import numpy as np
from scipy import fft

class PartialDislocationsSimulation:

    def __init__(self, 
                bigN=1024, length=1024, time=500, 
                timestep_dt=0.01, deltaR=1, bigB=1, 
                smallB=1, b_p=1, cLT1=2, cLT2=2, mu=1, 
                tauExt=0, d0=40, c_gamma=50, seed:int =None):
        
        self.bigN = bigN                        # Number of discrete heights in the line so len(y1) = len(y2) = bigN
        self.length = length                    # Length L of the actual line
        self.deltaL = self.length/self.bigN     # The dx value in x direction

        self.time = time                        # Time in seconds
        self.dt = timestep_dt
        self.timesteps = round(self.time/self.dt) # In number of timesteps of dt

        self.deltaR = deltaR                    # Parameters for the random noise

        self.bigB = bigB                        # Bulk modulus
        self.smallB = smallB                    # Size of Burgers vector
        # TODO: make sure the value of b aligns w/ b_p and C_LT1 and C_LT2

        self.b_p = b_p                          # Equal lengths of the partial burgers vectors
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1} and C_{LT2} 
        self.cLT2 = cLT2
        # TODO: make sure values of cLT1 and cLT2 align with the line tension tau
        self.mu = mu

        self.tauExt = tauExt

        self.c_gamma = c_gamma                  # Parameter in the interaction force, should be small
        # self.c_gamma = self.deltaR*50

        self.d0 = d0                            # Initial distance
        self.seed = seed
        if seed != None:
            np.random.seed(seed)
        
        self.stressField = np.random.normal(0,self.deltaR,[self.bigN, 2*self.bigN]) # Generate a random stress field

        # Pre-allocate memory here
        self.y2 = np.empty((self.timesteps, self.bigN))
        self.y1 = np.empty((self.timesteps, self.bigN))

        self.has_simulation_been_run = False

        self.time_elapsed = 0                   # The time elapsed since the beginning of simulation in seconds
        self.tau_cutoff = self.time/3                  # Time when tau is switched on in seconds

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
        d = d0
    
    def getParamsInLatex(self):
        return [
            f"N={self.bigN}", f"L={self.length}", f"t={self.time}",
            f"dt={self.dt}", f"\\Delta R = {self.deltaR}", f"C_{{LT1}} = {self.cLT1}",
            f"C_{{LT2}} = {self.cLT2}", f"\\tau_{{ext}} = {self.tauExt}", f"b_p = {self.b_p}"]
    
    def getTitleForPlot(self, wrap=6):
        parameters = self.getParamsInLatex()
        plot_title = " ".join([
            "$ "+ i + " $" + "\n"*(1 - ((n+1)%wrap)) for n,i in enumerate(parameters) # Wrap text using modulo
        ])
        return plot_title
    
    def getTvalues(self):
        # Returns the times ealaped at each step
        return np.arange(self.timesteps)*self.dt
    
    def getXValues(self):
        # Retuns the correct distance on x axis.
        return np.arange(self.bigN)*self.deltaL
    
    def tau(self, y): # Should be static in time. The index is again the x coordinate here. Takes around 9.6912e-05 s
        yDisc = np.remainder(np.round(y), self.bigN).astype(np.int32) # Round the y coordinate to an integer and wrap around bigN
        return self.stressField[np.arange(self.bigN), yDisc] # x is discrete anyways here
    
    def tau_interpolated(self, y): # Takes around 0.0131 s, which is 1000x slower than above
        x_points = np.arange(self.bigN)
        # tau_res = np.empty(self.bigN)
        # for x in x_points:
        #     yp = self.stressField[x,0:self.bigN]
        #     yx = y[x]
        #     y_res = np.interp(yx, np.arange(self.bigN), yp, period=self.bigN)
        #     tau_res[x] = y_res
        
        tau_res = [ np.interp(y[x], x_points, self.stressField[x,0:self.bigN], period=self.bigN) for x in x_points ] #TODO: Tee intepolaatio ite
        return tau_res
    
    def tau_ext(self):
        if self.time_elapsed >= self.tau_cutoff:
            return self.tauExt
        else:
            return 0
    
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
        numerator = ( np.average(y2 - y1) )*np.ones(self.bigN)
        return factor*(1 + numerator/self.d0) # Term from Vaid et Al B.7

    def secondDerivative(self, x):
        x_hat = fft.fft(x)
        k = fft.fftfreq(n=self.bigN, d=self.deltaL)*2*np.pi

        d_x_hat = -x_hat*(k)**2

        return fft.ifft(d_x_hat).real

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
        
    def getLineProfiles(self):
        if self.has_simulation_been_run:
            return (self.y1, self.y2)
        
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