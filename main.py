import numpy as np
import matplotlib.pyplot as plt

bigN = 100 # Number of discrete heights in the line
length = 10
deltaL = length/bigN
xVec = [i*deltaL for i in range(0,bigN)] # Come up with some x values, since the x corodinate is just the indice here.

dt = 0.01
time = 4000 # In number of timesteps of dt, so in this case, simulating 50s


bigB = 0.01
smallB = 1
mu = 0.01
b_p = 0.01
clt1 = 0.01
clt2 = 0.01
tauExt = 0.01

def tau(x,y):
    return np.random.rand(bigN) # len(x) = len(y) = bigN

def force1(y1,y2):
    return -(y1-y2) # Now only adjacent nodes have an effect on each other

def force2(y1,y2):
    return (y1-y2)

def secondDerivateive(x):
    return np.gradient(np.gradient(x))

def timestep(dt, y1,y2):

    dy1 = (clt1*mu*(b_p**2)*secondDerivateive(y1) + (smallB/2)*tauExt + b_p*tau(xVec,y1) + force1(y1, y2))*(bigB/smallB)*dt
    dy2 = (clt1*mu*(b_p**2)*secondDerivateive(y2) + (smallB/2)*tauExt + b_p*tau(xVec,y2) + force2(y1, y2))*(bigB/smallB)*dt
    return (y1+dy1, y2+dy2)


y10 = np.random.rand(bigN)+np.ones(bigN)*2 # Make sure its bigger than y2 to being with
y1 = [y10]

y20 = np.random.rand(bigN)
y2 = [y20]

for i in range(1,time):
    y1_previous = y1[i-1]
    y2_previous = y2[i-1]

    (y1_i, y2_i) = timestep(dt,y1_previous,y2_previous)
    y1.append(y1_i)
    y2.append(y2_i)

plt.plot(y10)
plt.plot(y1[len(y1)-1])

plt.plot(y20)
plt.plot(y2[len(y2)-1])

plt.legend(
    ["y1_0", "y1_1000", "y2_0", "y2_1000"]
    )

plt.show()