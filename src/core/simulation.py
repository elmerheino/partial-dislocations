from pathlib import Path
from scipy import fft
from scipy.interpolate import CubicSpline
import numpy as np
from numba import jit
from scipy.ndimage import map_coordinates
from src.core.pinned_field import generate_random_field

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

        self.has_simulation_been_run = False

        # --- FIRE Algorithm Parameters ---
        self.DT_INITIAL = 0.01
        self.DT_MAX = 0.1
        self.N_MIN = 5
        self.F_INC = 1.1
        self.F_DEC = 0.5
        self.ALPHA_START = 0.1
        self.F_ALPHA = 0.99
        self.MAX_STEPS = 200000
        self.CONVERGENCE_FORCE = 1e-7

        # --- Setup the random force and its spline interpolation ---
        # self.stressField = np.random.normal(0,self.deltaR,[self.bigN, self.bigN]) # Generate a random stress field

        X, Y, tau = generate_random_field(self.bigN, field_rms=1.5)
        self.stressField = tau
        
        # Set endpoints of the generated random field equal to allow use of periodic boundary conditions
        self.stressField[:,-1] = self.stressField[:,0]
        
        self.splines = self.setup_splines()

        pass

    # From there on define all the methods related to FIRE relaxation.
    def setup_splines(self):
        """Creates cubic spline interpolators for the generated random noise."""
        # Define grid for the potential. It should correspond to the indices of the stress field array.
        h_grid = np.arange(self.bigN)
                
        # Create a list of cubic spline interpolators, one for each x position
        splines = [CubicSpline(h_grid, self.stressField[i, :], bc_type='periodic') for i in range(self.bigN)]

        return splines

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
        
    def tau(self, y):
        noise_force = np.array([self.splines[i](y[i]) for i in range(self.bigN)])
        return noise_force
        
    def is_relaxed(self, velocities, tolerance=1e-9):
        # accel_cm_i = np.gradient(velocities)
        # return np.mean(np.abs(accel_cm_i)) < tolerance

        # t = np.arange(len(velocities))*self.dt
        # slope, intercept, r_value, p_value, std_err = linregress(t, velocities)

        # return slope < tolerance
        return False
        
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
    
    def getTvalues(self):
        # Returns the times ealaped at each step
        return np.arange(self.timesteps)*self.dt
    
    def getXValues(self):
        # Retuns the correct distance on x axis.
        return np.arange(self.bigN)*self.deltaL
    
    def setTauCutoff(self, new_cutoff):
        """
        Sets the cutoff time when self.tau_ext is switched on.
        """
        self.tau_cutoff = new_cutoff
    
    def saveResults(self, path : Path):
        raise NotImplementedError()
    
    def getResultsAsDict(self):
        raise NotImplementedError()
    
    def getUniqueHashString(self):
        raise NotImplementedError()