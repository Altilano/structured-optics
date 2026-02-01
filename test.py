from struct_opt import *
import matplotlib.pyplot as plt
import numpy as np

b1 = Beam(nix=5e-3, 
          Dx=256,
          pol_dim=2)

b1.hg(n=2, m=0, pol_index=None)
#b1.crop(window=2)
print(b1.int_profile().shape)
print("Power Bessel 2:", b1.Power())
print("Center of mass Bessel 2:", b1.center_mass(0))
plt.pcolormesh(b1.x, b1.y, b1.int_section(0,np.pi)[1], cmap='gray')
plt.colorbar()
plt.show()