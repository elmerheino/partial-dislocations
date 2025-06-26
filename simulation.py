from scipy import fft
import numpy as np
import math
from numba import jit
from scipy.ndimage import map_coordinates

class Simulation(object):
    def __init__(self, bigN, length, time, dt, deltaR : float, bigB, smallB, mu, tauExt, seed=None):

        self.bigN = bigN                        # Number of discrete heights in the line so len(y1) = len(y2) = bigN
        self.x_indices = np.arange(self.bigN)

        self.length = length
        self.deltaL = self.length/self.bigN     # The dx value in x direction

        self.time = time                        # Time in seconds
        self.dt = dt

        self.timesteps = round(self.time/self.dt) # In number of timesteps of dt

        self.deltaR = deltaR                    # Parameter for the random noise

        self.bigB = bigB                        # Bulk modulus
        self.smallB = smallB                    # Size of Burgers vector
        self.mu = mu                            # mu, parameter for tension

        self.tauExt = tauExt

        self.time_elapsed = 0                   # The time elapsed since the beginning of simulation in seconds
        self.tau_cutoff = self.time/10          # Time when tau is switched on in seconds, (e.g. 10% of the simulation time)

        self.seed = seed
        if seed != None:
            np.random.seed(seed)

        self.stressField = np.random.normal(0,self.deltaR,[self.bigN, 2*self.bigN]) # Generate a random stress field

        self.has_simulation_been_run = False
        
        pass

    def secondDerivative(self, x):
        x_hat = fft.fft(x)
        k = fft.fftfreq(n=self.bigN, d=self.deltaL)*2*np.pi

        d_x_hat = -x_hat*(k**2)

        return fft.ifft(d_x_hat).real
    
    def tau_ext(self, time):
        if time >= self.tau_cutoff:
            return self.tauExt
        else:
            return 0
    
    def tau_no_interpolation(self,y): # Takes around 22.31 mu s
        yDisc = np.remainder(np.round(y), self.bigN).astype(np.int32) # Round the y coordinate to an integer and wrap around bigN
        return self.stressField[np.arange(self.bigN), yDisc] # x is discrete anyways here
    
    def tau(self, y): # This should be the fastest possible way to do this w/ 29.201507568359375 mu s
        # tau_res = [ np.interp(y[x], self.x_points, self.stressField[x,0:self.bigN], period=self.bigN) for x in self.x_points ]
        coords = np.array([self.x_indices, y])
        stress_data = self.stressField[:, :self.bigN]
        return map_coordinates(stress_data, coords, order=1, mode='wrap')
    
    def tau_interpolated(self, y): # This should be the fastest possible way to do this w/ 29.201507568359375 mu s
        return self.tau(y)
    
    def is_relaxed(self, velocities, tolerance=1e-7):
        accel_cm_i = np.gradient(velocities)
        return np.mean(np.abs(accel_cm_i)) < tolerance
    
    @staticmethod
    @jit(nopython=True)
    def tau_interpolated_static(y, bigN, stressField, x_points): # Takes around 1639.9  mu s, which is faster than the two other options
        # tau_res = [ np.interp(y[x], self.x_points, self.stressField[x,0:self.bigN], period=self.bigN) for x in self.x_points ]

        tau_res = np.empty(bigN)
        for x in x_points:
            col = stressField[x,0:bigN]
            y_x = y[x]

            x1 = math.floor(y_x)
            x2 = math.ceil(y_x)

            y1 = col[x1 % bigN]
            y2 = col[x2 % bigN]

            if (x2 - x1) == 0: # handle the case where 
                # print(f"y_x={y_x} y1={y1}=col[x1]  y2={y2}=col[x2] k={k} ")
                tau_res[x] = y1
                continue

            k = (y2 - y1)/(x2 - x1)
            b = y1 - k*x1

            tau_res[x] = k*y_x + b

        return tau_res

    @staticmethod
    @jit(nopython=True)
    def roughnessW(y, bigN): # Calculates the cross correlation W(L) of a single dislocation
        l_range = np.arange(0,int(bigN))
        roughness = np.empty(int(bigN))

        y_size = len(y)
        
        for l in l_range:
            res = 0
            for i in range(0,y_size):
                res += ( y[i] - y[ (i+l) % y_size ] )**2
            
            # res = [ ( y[i] - y[ (i+l) % y.size ] )**2 for i in np.arange(y.size) ]

            res = res/y_size
            c = np.sqrt(res)
            roughness[l] = c

        return l_range, roughness

    @staticmethod
    @jit(nopython=True)
    def roughnessW12(y1, y2, bigN): # Calculated the cross correlation W_12(L) between the dislocations
        avgY1 = sum(y1)/len(y1)
        avgY2 = sum(y2)/len(y2)

        l_range = np.arange(0,int(bigN))    # TODO: Use range parameter instead
        roughness = np.empty(int(bigN))

        y2_size = len(y2)   # TODO: check if len(y1) = bigN ?
        y1_size = len(y1)
        
        for l in l_range:
            res = 0
            for i in range(0,y1_size):
                res += ( y1[i] - avgY1 - (y2[ (i+l) % y2_size ] - avgY2) )**2

            # res = [ ( y1[i] - avgY1 - (y2[ (i+l) % y2_size ] - avgY2) )**2 for i in range(0,y1_size) ] # TODO: check fomula here

            res = res/y1_size # len(range(0,y1_size)) = y1_size = len(res)
            c = np.sqrt(res)
            roughness[l] = c

        return l_range, roughness

    def getParamsInLatex(self):
        return [
            f"N={self.bigN}", f"L={self.length}", f"t={self.time}",
            f"dt={self.dt}", f"\\Delta R = {self.deltaR}",
            f"\\tau_{{ext}} = {self.tauExt}"]
    
    def getTvalues(self):
        # Returns the times ealaped at each step
        return np.arange(self.timesteps)*self.dt
    
    def getXValues(self):
        # Retuns the correct distance on x axis.
        return np.arange(self.bigN)*self.deltaL

    pass