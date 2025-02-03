import numpy as np
import matplotlib.pyplot as plt

class PartialDislocationsSimulation:

    def __init__(self, 
                bigN=1024, length=1024, time=500, 
                timestep_dt=0.01, deltaR=1, bigB=1, 
                smallB=1, b_p=1, cLT1=2, cLT2=2, mu=1, 
                tauExt=0, d0=40, c_gamma=50):
        
        self.bigN = bigN # Number of discrete heights in the line so len(y1) = len(y2) = bigN
        self.length = length # Length L of the actual line
        self.deltaL = self.length/self.bigN # The dx value in x direction

        self.time = time # Time in seconds
        self.dt = timestep_dt
        self.timesteps = round(self.time/self.dt) # In number of timesteps of dt

        self.deltaR = deltaR # Parameters for the random noise

        self.bigB = bigB    # Bulk modulus
        self.smallB = smallB  # Size of Burgers vector

        self.b_p = b_p
        self.cLT1 = cLT1 # Parameters of the gradient term
        self.cLT2 = cLT2
        self.mu = mu

        self.tauExt = tauExt

        self.c_gamma = c_gamma # Parameter in the interaction force, should be small
        # self.c_gamma = self.deltaR*50

        self.d0 = d0 # Initial distance

    # gamma = 60.5
    # d0 = (c_gamma*mu/gamma)*b_p**2
    # d = d0

    def getXValues(self):
        return np.arange(self.timesteps-1)*self.dt
    
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
        y1 = [y10]

        y20 = np.zeros(self.bigN, dtype=float)
        y2 = [y20]

        averageDist = []

        stressField = np.random.normal(0,self.deltaR,[self.bigN, 2*self.bigN]) # Generate a random sterss field

        for i in range(1,self.timesteps):
            y1_previous = y1[i-1]
            y2_previous = y2[i-1]

            (y1_i, y2_i) = self.timestep(self.dt,y1_previous,y2_previous, stressField)
            y1.append(y1_i)
            y2.append(y2_i)

            averageDist.append(np.average(y1_i-y2_i))
        return averageDist

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

    # run 4 simulations right away
def run4sims():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.ravel()

    for i in range(0,4):
        np.random.seed(i)

        sim_i = PartialDislocationsSimulation()

        x = sim_i.getXValues()
        avgI = sim_i.run_simulation()

        axes_flat[i].plot(x,avgI)
        axes_flat[i].set_xlabel("Time (s)")
        axes_flat[i].set_ylabel("Average distance")
        print(f"Simulation {i+1}, min: {min(avgI)}, max: {max(avgI)}, delta: {max(avgI) - min(avgI)}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run4sims()