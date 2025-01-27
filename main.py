import numpy as np
import matplotlib.pyplot as plt

bigN = 1024 # Number of discrete heights in the line so len(y1) = len(y2) = bigN
length = 1024 # Length L of the actual line
deltaL = length/bigN # The dx value in x direction

time = 5000 # Time in seconds
dt = 0.05
timesteps = round(time/dt) # In number of timesteps of dt

deltaR = 1 # Parameters for the random noise

bigB = 1
smallB = 1

b_p = 1
cLT1 = 2 # Parameters of the gradient term
cLT2 = 2
mu = 1

tauExt = 2
c_gamma = deltaR/1000 # Parameter in the interaction force, should be small
d = 500 # Average distance

# gamma = 60.5
# d0 = (c_gamma*mu/gamma)*b_p**2

stressField = np.zeros([bigN, bigN])

def generateRandomTau():
    stressField = np.random.normal(0,deltaR,[bigN, bigN])

def tau(y): # Should be static in time. The index is again the x coordinate here
    yDisc = (np.round(y).astype(int) & bigN ) - 1 # Round the y coordinate to an integer and wrap around bigN
    return stressField[np.arange(bigN), yDisc] # x is discrete anyways here

def force1(y1,y2):
    #return -np.average(y1-y2)*np.ones(bigN)
    #return -(c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

    factor = (1/d)*c_gamma*mu*(b_p**2)
    return factor*(1 + (y1-y2)/d)

def force2(y1,y2):
    #return np.average(y1-y2)*np.ones(bigN)
    #return (c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

    factor = (1/d)*c_gamma*mu*(b_p**2)
    return factor*(1 + (y2-y1)/d) # Term from Vaid et Al B.7


def secondDerivateive(x):
    return np.gradient(np.gradient(x, deltaL), deltaL)

def timestep(dt, y1,y2):
    dy1 = ( 
        cLT1*mu*(b_p**2)*secondDerivateive(y1) # The gradient term
        + b_p*tau(y1) # The random stress term
        + force1(y1, y2) # Interaction force
        + (smallB/2)*tauExt*np.ones(bigN) # The external stress term
        ) * ( bigB/smallB )
    dy2 = ( cLT2*mu*(b_p**2)*secondDerivateive(y2) + b_p*tau(y2) + force2(y1, y2) + (smallB/2)*tauExt*np.ones(bigN) )*( bigB/smallB )
    
    newY1 = (y1 + dy1*dt)
    newY2 = (y2 + dy2*dt)

    return (newY1, newY2)


def run_simulation():
    y10 = np.random.rand(bigN)+np.ones(bigN)*d # Make sure its bigger than y2 to being with, and also that they have the initial distance d
    y1 = [y10]

    y20 = np.random.rand(bigN)
    y2 = [y20]

    averageDist = []

    generateRandomTau()

    for i in range(1,timesteps):
        y1_previous = y1[i-1]
        y2_previous = y2[i-1]

        (y1_i, y2_i) = timestep(dt,y1_previous,y2_previous)
        y1.append(y1_i)
        y2.append(y2_i)

        averageDist.append(np.average(y1_i-y2_i))
    return averageDist

# run 4 simulations right away

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