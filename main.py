import numpy as np
import matplotlib.pyplot as plt

bigN = 1024 # Number of discrete heights in the line so len(y1) = len(y2) = bigN
length = 1024 # Length L of the actual line
deltaL = length/bigN # The dx value in x direction

time = 1000 # Time in seconds
dt = 0.5
timesteps = round(time/dt) # In number of timesteps of dt

deltaR = 1 # Parameters for the random noise

bigB = 1    # Bulk modulus
smallB = 1  # Size of Burgers vector

b_p = 1
cLT1 = 2 # Parameters of the gradient term
cLT2 = 2
mu = 1

tauExt = 0

c_gamma = deltaR/10 # Parameter in the interaction force, should be small

d = 200 # Average distance

# gamma = 60.5
# d0 = (c_gamma*mu/gamma)*b_p**2
# d = d0


def generateRandomTau():
    return np.random.normal(0,deltaR,[bigN, bigN])

def tau(y, stressField): # Should be static in time. The index is again the x coordinate here
    yDisc = (np.round(y).astype(int) & bigN ) - 1 # Round the y coordinate to an integer and wrap around bigN
    return stressField[np.arange(bigN), yDisc] # x is discrete anyways here

def force1(y1,y2):
    #return -np.average(y1-y2)*np.ones(bigN)
    #return -(c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

    factor = (1/d)*c_gamma*mu*(b_p**2)
    return factor*(1 + (y2-y1)/d)

def force2(y1,y2):
    #return np.average(y1-y2)*np.ones(bigN)
    #return (c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

    factor = -(1/d)*c_gamma*mu*(b_p**2)
    return factor*(1 + (y2-y1)/d) # Term from Vaid et Al B.7

def derivativePeriodic(x, dl):

    res = (np.roll(x, -1) - np.roll(x,1))/(2*dl)

    return res

def secondDerivateive(x):
    return derivativePeriodic(derivativePeriodic(x,deltaL),deltaL)

def timestep(dt, y1,y2, stressField):
    dy1 = ( 
        cLT1*mu*(b_p**2)*secondDerivateive(y1) # The gradient term
        + b_p*tau(y1, stressField) # The random stress term
        + force1(y1, y2) # Interaction force
        + (smallB/2)*tauExt*np.ones(bigN) # The external stress term
        ) * ( bigB/smallB )
    dy2 = ( cLT2*mu*(b_p**2)*secondDerivateive(y2) + b_p*tau(y2, stressField) + force2(y1, y2) + (smallB/2)*tauExt*np.ones(bigN) )*( bigB/smallB )
    
    newY1 = (y1 + dy1*dt)
    newY2 = (y2 + dy2*dt)

    return (newY1, newY2)


def run_simulation():
    y10 = np.ones(bigN)*d # Make sure its bigger than y2 to being with, and also that they have the initial distance d
    y1 = [y10]

    y20 = np.zeros(bigN)
    y2 = [y20]

    averageDist = []

    stressField = generateRandomTau()

    for i in range(1,timesteps):
        y1_previous = y1[i-1]
        y2_previous = y2[i-1]

        (y1_i, y2_i) = timestep(dt,y1_previous,y2_previous, stressField)
        y1.append(y1_i)
        y2.append(y2_i)

        averageDist.append(np.average(y1_i-y2_i))
    return averageDist

def jotain_saatoa_potentiaaleilla():
    forces_f1 = list()
    forces_f2 = list()

    for i in range(0,300):
        y10 = np.ones(1)*i # Generate lines at distance i apart with length 1
        y20 = np.zeros(1)

        f1 = force1(y10,y20)
        f2 = force2(y10,y20)

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

    x = np.arange(timesteps-1)*dt

    for i in range(0,4):
        np.random.seed(i)
        avgI = run_simulation()
        axes_flat[i].plot(x,avgI)
        axes_flat[i].set_xlabel("Time (s)")
        axes_flat[i].set_ylabel("Average distance")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run4sims()