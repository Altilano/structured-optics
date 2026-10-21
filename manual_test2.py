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


b1.Ex = hg(b1, N=0, M=0) #- 1j*hg(b1, N=1, M=1)
b1 = b1*restrict_mask(HWP(np.pi/4), AngularSection(np.pi/2, np.pi))
b1 = b1*restrict_mask(HWP(np.pi), AngularSection(np.pi, 3*np.pi/2))
b1 = b1*restrict_mask(HWP(3*np.pi/4), AngularSection(3*np.pi/2,2*np.pi))
b1 = b1*Polarizer(np.pi/4)
b1 = b1*HWP(np.pi/4)
b1 = b1*np.exp(1j*np.min(b1.phase()))
#b1 = b1*HWP(np.pi/2)
print(b1.Power(0))
print(b1.Power(1))

plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.phase(0), cmap='jet')
plt.colorbar()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()

plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.phase(1), cmap='jet')
plt.colorbar()
plt.axis('equal')
plt.axis('off')
plt.tight_layout()
plt.show()
