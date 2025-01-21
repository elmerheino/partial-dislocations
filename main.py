import numpy as np
import matplotlib.pyplot as plt

h = np.random.rand(10)

gradientti = np.gradient(np.gradient(h))

plt.plot(h)
plt.plot(gradientti)
plt.legend(
    ["h", "d^2h/dx^2"]
    )
plt.show()