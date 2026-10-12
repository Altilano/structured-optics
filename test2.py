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


b1.Ex = hg(b1, N=0, M=0) - 1j*hg(b1, N=1, M=1)
b1 = b1*Lens(50e-2)
b1.propagate(40e-2, method="blue_fix")




plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
#plt.colorbar()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()
