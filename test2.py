import matplotlib.pyplot as plt
from structured_optics import *
import numpy as np



b1 = Beam(nix=25e-6, 
          Dx=1024,
          waist = 6e-6,
          lamb = 1064e-9,
          pol_dim=2)


#b1.hg(1,1)
b1.Ex = hg(b1, 0,1, angle = np.pi/3) + hg(b1, 3,1, angle=2*np.pi/3)
#b1.lp(2,2,1.7,1.4)
#b1.Ey = lg(b1,2,2)

#b1.polarizer('D')
plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
plt.colorbar()

b1.propagate(b1.zr(), method='AS')

plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
plt.colorbar()

b1.propagate(-b1.zr(), method='fresnel')

plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
plt.colorbar()
plt.show()
