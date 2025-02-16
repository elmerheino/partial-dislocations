from scipy import fft
import numpy as np

class Simulation(object):
    def __init__(self, bigN, length, time, dt, deltaR, bigB, smallB, b_p, mu, tauExt, seed=None):

        self.bigN = bigN                        # Number of discrete heights in the line so len(y1) = len(y2) = bigN
        self.length = length
        self.deltaL = self.length/self.bigN     # The dx value in x direction

        self.time = time                        # Time in seconds
        self.dt = dt

        self.timesteps = round(self.time/self.dt) # In number of timesteps of dt

        self.deltaR = deltaR                    # Parameter for the random noise

        self.bigB = bigB                        # Bulk modulus
        self.smallB = smallB                    # Size of Burgers vector
        # TODO: make sure the value of b aligns w/ b_p and C_LT1 and C_LT2

        self.b_p = b_p                          # Equal lengths of the partial burgers vectors
        self.mu = mu                            # mu, parameter for tension

        self.tauExt = tauExt

        self.time_elapsed = 0                   # The time elapsed since the beginning of simulation in seconds
        self.tau_cutoff = self.time/3           # Time when tau is switched on in seconds

        self.seed = seed
        if seed != None:
            np.random.seed(seed)

        self.stressField = np.random.normal(0,self.deltaR,[self.bigN, 2*self.bigN]) # Generate a random stress field

        pass

    def secondDerivative(self, x):
        x_hat = fft.fft(x)
        k = fft.fftfreq(n=self.bigN, d=self.deltaL)*2*np.pi

        d_x_hat = -x_hat*(k)**2

        return fft.ifft(d_x_hat).real
    
    def tau_ext(self):
        if self.time_elapsed >= self.tau_cutoff:
            return self.tauExt
        else:
            return 0
    
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
        
        tau_res = [ np.interp(y[x], x_points, self.stressField[x,0:self.bigN], period=self.bigN) for x in x_points ]
        return tau_res

    def getParamsInLatex(self):
        return [
            f"N={self.bigN}", f"L={self.length}", f"t={self.time}",
            f"dt={self.dt}", f"\\Delta R = {self.deltaR}",
            f"\\tau_{{ext}} = {self.tauExt}", f"b_p = {self.b_p}"]
    
    def getTvalues(self):
        # Returns the times ealaped at each step
        return np.arange(self.timesteps)*self.dt
    
    def getXValues(self):
        # Retuns the correct distance on x axis.
        return np.arange(self.bigN)*self.deltaL

    pass