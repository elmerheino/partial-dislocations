from src.core.pinned_field import generate_random_field

import numpy as np
import matplotlib.pyplot as plt

Nx = 512
X, Y, tau = generate_random_field(Nx, field_rms=10.0)

print(tau.shape)   # (1024, 512)

plt.figure()
plt.imshow(tau)
plt.colorbar()
plt.tight_layout()

plt.figure()
plt.hist(tau.flatten())
print(str(np.mean(tau.flatten())) + ' pm ' + str(np.std(tau.flatten())))
plt.tight_layout()

plt.show()