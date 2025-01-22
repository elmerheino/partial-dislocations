import numpy as np
import matplotlib.pyplot as plt

bigN = 1024 # Number of discrete heights in the line
length = 1024
deltaL = length/bigN
xVec = [i*deltaL for i in range(0,bigN)] # Come up with some x values, since the x corodinate is just the indice here.

time = 60 # Time in seconds
dt = 0.005
timesteps = round(time/dt) # In number of timesteps of dt

bigB = 1
smallB = 1
mu = 2
b_p = 2
cLT1 = 3
cLT2 = 1
tauExt = 1
c_gamma = 1
d = 200 # Average distance

def tau(x,y):
    return np.random.rand(bigN) # len(x) = len(y) = bigN

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

    dy1 = (cLT1*mu*(b_p**2)*secondDerivateive(y1) + (smallB/2)*tauExt + b_p*tau(xVec,y1) + force1(y1, y2))*(bigB/smallB)*dt
    dy2 = (cLT2*mu*(b_p**2)*secondDerivateive(y2) + (smallB/2)*tauExt + b_p*tau(xVec,y2) + force2(y1, y2))*(bigB/smallB)*dt
    
    return (y1+dy1, y2+dy2)


y10 = np.random.rand(bigN)+np.ones(bigN)*d # Make sure its bigger than y2 to being with, and also that they have the initial distance d
y1 = [y10]

y20 = np.random.rand(bigN)*0.2
y2 = [y20]

averageDist = []

for i in range(1,timesteps):
    y1_previous = y1[i-1]
    y2_previous = y2[i-1]

    (y1_i, y2_i) = timestep(dt,y1_previous,y2_previous)
    y1.append(y1_i)
    y2.append(y2_i)

    averageDist.append(np.average(y1_i-y2_i))

# plt.plot(y10)
# plt.plot(y1[len(y1)-1])

# plt.plot(y20)
# plt.plot(y2[len(y2)-1])

# plt.legend(
#     ["y1_0", "y1_1000", "y2_0", "y2_1000"]
#     )

# plt.show()

plt.plot([i*dt for i in range(1,timesteps)], averageDist)
plt.xlabel("t (s)")
plt.ylabel("Average distance")
plt.show()