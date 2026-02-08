import structured_optics as so
import matplotlib.pyplot as plt
import numpy as np

b1 = so.Beam(nix=7e-3, 
          Dx=512,
          pol_dim=1,
          waist = 1e-3)


b1.frac_oam_qs(Ma=0.5, n_modes=10)
#b1.field = b1.field**2
"""b1.lens(10e-2)
b1.propagate(12e-2)
b1.stripe_v(0.2e-3, move = 0.5e-3)
H = b1.dmd_holo(0.08, 0.08)
H2 = b1.slm_holo(4, 3)"""

print(b1.int_profile().shape)
print("Power Bessel 2:", b1.Power())
print("Center of mass Bessel 2:", b1.center_mass())
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='gray')
#plt.imshow(H, cmap='gray')
plt.colorbar()
plt.show()