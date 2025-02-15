import numpy as np
from scipy import fft

class DislocationSimulation:

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
        self.cLT1 = cLT1                        # Parameters of the gradient term C_{LT1}

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
            f"\\tau_{{ext}} = {self.tauExt}", f"b_p = {self.b_p}"]
    
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
    
    def tau(self, y): # Should be static in time. The index is again the x coordinate here
        yDisc = np.remainder(np.round(y), self.bigN).astype(np.int32) # Round the y coordinate to an integer and wrap around bigN
        return self.stressField[np.arange(self.bigN), yDisc] # x is discrete anyways here
    
    def tau_ext(self):
        if self.time_elapsed >= self.tau_cutoff:
            return self.tauExt
        else:
            return 0
    
    def secondDerivative(self, x):
        x_hat = fft.fft(x)
        k = fft.fftfreq(n=self.bigN, d=self.deltaL)*2*np.pi

        d_x_hat = -x_hat*(k)**2

        return fft.ifft(d_x_hat).real

    def timestep(self, dt, y1):
        dy1 = ( 
            self.cLT1*self.mu*(self.b_p**2)*self.secondDerivative(y1) # The gradient term # type: ignore
            + self.b_p*self.tau(y1) # The random stress term
            + (self.smallB/2)*self.tau_ext()*np.ones(self.bigN) # The external stress term
            ) * ( self.bigB/self.smallB )
        
        newY1 = (y1 + dy1*dt)

        self.time_elapsed += dt    # Update how much time has elapsed by adding dt

        return newY1

    def run_simulation(self):
        y10 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d
        self.y1[0] = y10


        for i in range(1,self.timesteps):
            y1_previous = self.y1[i-1]

            y1_i = self.timestep(self.dt,y1_previous)
            self.y1[i] = y1_i
        
        self.has_simulation_been_run = True
    
    def getLineProfiles(self):
        if self.has_simulation_been_run:
            return self.y1
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