from struct_opt import *
import matplotlib.pyplot as plt
import numpy as np

b1 = Beam(nix=5e-3, 
          Dx=256,
          pol_dim=2)

b1.hg(n=2, m=0, pol_index=0)
b1.crop(window=2)
print(b1.int_profile().shape)
print("Power Bessel 2:", b1.get_Power())
print("Center of mass Bessel 2:", b1.center_mass(0))
plt.imshow(b1.int_profile()[0], cmap='gray')
plt.colorbar()
plt.show()