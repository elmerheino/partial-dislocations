import numpy as np
import matplotlib.pyplot as plt

h0 = np.random.rand(10)

def timestep(dt, h):
    random_noise = np.random.normal(0,0.1,10)
    tension = 1
    res = np.gradient(np.gradient(h)) + random_noise + tension
    return res

h = [h0]
for i in range(1,100):
    h_previous = h[i-1]
    h_i = timestep(0.2,h_previous)
    h.append(h_i)

plt.plot(h0)
plt.plot(h[len(h)-1])

plt.legend(
    ["h10", "d^2h/dx^2"]
    )

plt.show()