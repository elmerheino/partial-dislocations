import numpy as np
import matplotlib.pyplot as plt

class PartialDislocationsSimulation:

    def __init__(self, 
                bigN=1024, length=1024, time=500, 
                timestep_dt=0.01, deltaR=1, bigB=1, 
                smallB=1, b_p=1, cLT1=2, cLT2=2, mu=1, 
                tauExt=0, d0=40, c_gamma=50):
        
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

        self.c_gamma = c_gamma # Parameter in the interaction force, should be small
        # self.c_gamma = self.deltaR*50

        self.d0 = d0 # Initial distance

        self.y2 = list()
        self.y1 = list()

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
            f"dt={self.dt}", f"Delta R = {self.deltaR}", f"C_{{LT1}} = {self.cLT1}",
            f"C_{{LT2}} = {self.cLT2}"]
    
    def getTvalues(self):
        # Returns the times ealaped at each step
        return np.arange(self.timesteps-1)*self.dt
    
    def getXValues(self):
        return np.arange(self.bigN)*self.deltaL
    
    def tau(self, y, stressField): # Should be static in time. The index is again the x coordinate here
        yDisc = (np.round(y).astype(int) & self.bigN ) - 1 # Round the y coordinate to an integer and wrap around bigN
        return stressField[np.arange(self.bigN), yDisc] # x is discrete anyways here

    def force1(self, y1,y2):
        #return -np.average(y1-y2)*np.ones(bigN)
        #return -(c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

        factor = (1/self.d0)*self.c_gamma*self.mu*(self.b_p**2)
        return factor*(1 + (y2-y1)/self.d0)

    def force2(self, y1,y2):
        #return np.average(y1-y2)*np.ones(bigN)
        #return (c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

        factor = -(1/self.d0)*self.c_gamma*self.mu*(self.b_p**2)
        return factor*(1 + (y2-y1)/self.d0) # Term from Vaid et Al B.7

    def derivativePeriodic(self, x, dl):
        res = (np.roll(x, -1) - np.roll(x,1))/(2*dl)
        return res

    def secondDerivative(self, x):
        return self.derivativePeriodic(self.derivativePeriodic(x,self.deltaL),self.deltaL)

    def timestep(self, dt, y1,y2, stressField):
        dy1 = ( 
            self.cLT1*self.mu*(self.b_p**2)*self.secondDerivative(y1) # The gradient term # type: ignore
            + self.b_p*self.tau(y1, stressField) # The random stress term
            + self.force1(y1, y2) # Interaction force
            + (self.smallB/2)*self.tauExt*np.ones(self.bigN) # The external stress term
            ) * ( self.bigB/self.smallB )
        dy2 = ( 
            self.cLT2*self.mu*(self.b_p**2)*self.secondDerivative(y2) 
            + self.b_p*self.tau(y2, stressField) 
            + self.force2(y1, y2)
            + (self.smallB/2)*self.tauExt*np.ones(self.bigN) ) * ( self.bigB/self.smallB )
        
        newY1 = (y1 + dy1*dt)
        newY2 = (y2 + dy2*dt)

        return (newY1, newY2)


    def run_simulation(self):
        y10 = np.ones(self.bigN, dtype=float)*self.d0 # Make sure its bigger than y2 to being with, and also that they have the initial distance d
        self.y1.append(y10)

        y20 = np.zeros(self.bigN, dtype=float)
        self.y2.append(y20)

        averageDist = []

        stressField = np.random.normal(0,self.deltaR,[self.bigN, 2*self.bigN]) # Generate a random sterss field

        for i in range(1,self.timesteps):
            y1_previous = self.y1[i-1]
            y2_previous = self.y2[i-1]

            (y1_i, y2_i) = self.timestep(self.dt,y1_previous,y2_previous, stressField)
            self.y1.append(y1_i)
            self.y2.append(y2_i)

            averageDist.append(np.average(y1_i-y2_i))
        
        velocity = np.average(np.gradient(averageDist)) # Calculate average velocity

        return (averageDist, velocity)
    
    def getLineProfiles(self):
        return (self.y1, self.y2)

    def jotain_saatoa_potentiaaleilla(self):
        forces_f1 = list()
        forces_f2 = list()

        for i in range(0,300):
            y10 = np.ones(1)*i # Generate lines at distance i apart with length 1
            y20 = np.zeros(1)

            f1 = self.force1(y10,y20)
            f2 = self.force2(y10,y20)

            forces_f1.append(f1)
            forces_f2.append(f2)
        
        plt.plot(forces_f1)
        plt.plot(forces_f2)

        plt.legend(["f_1", "f_2"])

        plt.show()
        pass