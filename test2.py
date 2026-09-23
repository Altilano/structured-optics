import matplotlib.pyplot as plt
from structured_optics import *
import numpy as np
from structured_optics.hologram import dmd_hologram
import time



b1 = Beam(nix=5e-3, 
          Dx=1024,
          waist = 1e-3,
          lamb = 633e-9,
          pol_dim=2)


b1.hg(0,0)

b1 = b1*Iris(1e-3)

b1.propagate(1e30)



plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
plt.colorbar()
plt.show()
exit()
