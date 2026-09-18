import matplotlib.pyplot as plt
from structured_optics import *
import numpy as np



b1 = Beam(nix=5e-3, 
          Dx=1024,
          waist = 0.5e-3,
          lamb = 633e-9,
          pol_dim=2)


b1.hg(0,0)
b2 = b1.copy()
b2.wiast = 7e-3
b2.hg(0,0)
b1 = b1*HWP(np.pi/6)

print(np.abs(overlap(b1, b2))**2)

base = hg_basis(b1, 2)
coefs = np.zeros((3,3))
coefs[1,1] = 1

b1.build_from_coefs_and_basis(coefs, base)

H = b1.dmd_holo(0.14, 0.14, sign = 1)



plt.figure()
plt.pcolormesh(b1.x, b1.y, H, cmap='inferno')
plt.colorbar()
plt.show()
exit()
