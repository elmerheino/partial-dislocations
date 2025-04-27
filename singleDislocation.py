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

        # Compute all the rescaling parameters
        self.lineTension = self.cLT1*self.mu*(self.smallB**2)

        self.a = np.sqrt(self.lineTension)
        self.t0 = self.smallB/self.bigB
        self.epsilon = self.a/self.smallB

        self.scaled_length = self.length/self.a
        self.scaled_time = self.time/self.t0
        self.scaled_deltaL = self.scaled_length / self.bigN
        self.scaled_dt = self.dt/self.t0
        self.stressField = np.random.normal(0,self.deltaR/self.epsilon,[self.bigN, 2*self.bigN])

        self.d0_scaled = self.d0/self.a
        self.tau_cutoff = self.scaled_time/3
        
        # print(f"Scaling parameters a = {self.a}   epsilon = {self.epsilon}  t0 = {self.t0}")
        # print(f"Scaled: dx : {self.scaled_deltaL:.4f}, dt : {self.scaled_dt:.4f}, line tension : {self.lineTension:.4f}, dx^2/2T = {self.scaled_deltaL**2 / 2*self.lineTension :.4f}")
        # using these above computed parameters the original variables are: x = a x'   y = a h'    t = t0 t'
        pass

    def secondDerivative(self, x):
        x_hat = fft.fft(x)
        k = fft.fftfreq(n=self.bigN, d=self.scaled_deltaL)*2*np.pi

        d_x_hat = -x_hat*(k**2)

        return fft.ifft(d_x_hat).real

    def timestep(self, dt, y1):
        # TODO: Update this function to use the rescaled version PDE
        tau = self.tau(y1)
        tau_ext = (self.tau_ext()/self.epsilon)
        second_derivative = self.secondDerivative(y1)
        dy1 = ( second_derivative + tau + tau_ext*np.ones(self.bigN) )
        
        newY1 = (y1 + dy1*self.scaled_dt)

        self.time_elapsed += self.scaled_dt    # Update how much time has elapsed by adding dt

        return newY1

    def run_simulation(self):
        y10 = np.ones(self.bigN, dtype=float)*self.d0_scaled # Make sure its bigger than y2 to being with, and also that they have the initial distance d
        self.y1[0] = y10

        for i in range(1,self.timesteps):
            y1_previous = self.y1[i-1]

            y1_i = self.timestep(self.dt,y1_previous)
            self.y1[i] = y1_i
        
        self.has_simulation_been_run = True
    
    def getResults(self):
        return self.y1*self.a
    
    def getLineProfiles(self, time_to_consider=None):
        # TODO: use getResutls instead
        start = 0

        if time_to_consider != None:
            steps_to_consider = round(time_to_consider / self.dt)
            start = self.timesteps - steps_to_consider
        if time_to_consider == self.time: # In this case only return the last state
            start = self.timesteps - 1

        if self.has_simulation_been_run:
            return self.getResults()[start:]
        else:
            raise Exception("Simulation has not been run.")
        
        print("Simulation has not been run yet.")
        return self.y1 # Retuns empty lists
    
    def getAverageDistances(self):
        # Returns the average distance from y=0
        return np.average(self.getResults(), axis=1)
    
    def getCM(self):
        # Return the centres of mass of the two lines as functions of time
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')

        y1_CM = np.mean(self.getResults(), axis=1)

        return y1_CM
    
    def getRelaxedVelocity(self, time_to_consider=1000):
        if len(self.y1) == 0:
            raise Exception('simulation has probably not been run')
        
        # Returns the velocities of the centres of mass
        y1_CM = self.getCM()
        v1_CM = np.gradient(y1_CM, self.dt)

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
        # TODO: use get results instead
        steps_to_consider = round(time_to_consider / self.dt)
        start = self.timesteps - steps_to_consider

        l_range, _ = self.roughnessW(self.getResults()[0], self.bigN)

        roughnesses = np.empty((steps_to_consider, self.bigN))
        for i in range(start,self.timesteps):
            _, rough = self.roughnessW(self.getResults()[i], self.bigN)
            roughnesses[i-start] = rough
        
        avg = np.average(roughnesses, axis=0)
        return l_range, avg