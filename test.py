from struct_opt import *
import matplotlib.pyplot as plt
import numpy as np

b1 = Beam(5e-3, 256)

b1.frac_oam(Ma=2.5, n_modes=21, beta=0, theta_0=0, z=0, pol_index=None)


plt.imshow(b1.int_profile(), cmap='jet')
plt.colorbar()
plt.show()