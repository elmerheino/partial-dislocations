import numpy as np
import matplotlib.pyplot as plt

bigN = 1024 # Number of discrete heights in the line so len(y1) = len(y2) = bigN
length = 1024
deltaL = length/bigN

time = 2500 # Time in seconds
dt = 0.1
timesteps = round(time/dt) # In number of timesteps of dt

bigB = 1
smallB = 1
mu = 2
b_p = 2
cLT1 = 2
cLT2 = 2
tauExt = 2
c_gamma = 5
d = 200 # Average distance

# gamma = 60.5
# d0 = (c_gamma*mu/gamma)*b_p**2

stressField = np.zeros([bigN, bigN])

def generateRandomTau():
    deltaR = 1
    stressField = np.random.normal(0,deltaR,[bigN, bigN])

def tau(y): # Should be static in time. The index is again the x coordinate here
    yDisc = (np.round(y).astype(int) & bigN ) - 1 # Round the y coordinate to an integer and wrap around bigN
    return stressField[np.arange(bigN), yDisc] # x is discrete anyways here

def force1(y1,y2):
    #return -np.average(y1-y2)*np.ones(bigN)
    #return -(c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

    factor = (1/d)*c_gamma*mu*b_p**2
    return factor*(1 + (y1-y2)/d)

def force2(y1,y2):
    #return np.average(y1-y2)*np.ones(bigN)
    #return (c_gamma*mu*b_p**2/d)*(1-y1/d) # Vaid et Al B.10

    factor = (1/d)*c_gamma*mu*b_p**2
    return factor*(1 + (y2-y1)/d) # Term from Vaid et Al B.7


def secondDerivateive(x):
    return np.gradient(np.gradient(x))

def timestep(dt, y1,y2):
    dy1 = ( cLT1*mu*(b_p**2)*secondDerivateive(y1) + b_p*tau(y1) + force1(y1, y2) + (smallB/2)*tauExt*np.ones(bigN) )*( bigB/smallB )*dt 
    dy2 = ( cLT2*mu*(b_p**2)*secondDerivateive(y2) + b_p*tau(y2) + force2(y1, y2) + (smallB/2)*tauExt*np.ones(bigN) )*( bigB/smallB )*dt
    
    newY1 = (y1 + dy1)
    newY2 = (y2 + dy2)

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

x = [i*dt for i in range(1,timesteps)]

for i in range(0,4):
    avgI = run_simulation()
    axes_flat[i].plot(x,avgI)
    axes_flat[i].set_xlabel("Time (s)")
    axes_flat[i].set_ylabel("Average distance")

plt.tight_layout()
plt.show()