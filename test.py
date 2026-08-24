#import structured_optics as so
import matplotlib.pyplot as plt
from structured_optics import *
import structured_optics.slmdisplay as soslm
import numpy as np


"""slm = soslm.My_SLM(screen_index = 1)

print(slm.width, slm.height)
pixel_size = 0.31e-3
b2 = Beam(nix = pixel_size*slm.width,
          niy = pixel_size*slm.height,
          Dx = slm.width,
          Dy = slm.height,
          waist = 1e-2,
          lamb = 1064e-9)

b2.hg(2,2)

holo = b2.slm_holo(3, 4)
slm.update_image(holo)

time.sleep(5)

exit()"""

b1 = Beam(nix=15e-6, 
          Dx=512,
          waist = 6e-6,
          lamb = 1064e-9)

#b1.hg(1,1)
b1.field = hg(b1, 1,1) + hg(b1, 2,1)
basis = b1.hg_projector(3)
#b1.rotate(np.pi/3, order=3)

#b1.field = b1.field**2
#b1.lens(10e-2)
#print(b1.Power())
#b1.propagate(1*b1.zr(), method='fresnel')



"""b1.stripe_v(0.2e-3, move = 0.5e-3)
H = b1.dmd_holo(0.08, 0.08)
H2 = b1.slm_holo(4, 3)"""

l = 8
m = 1
n_core = 1.6
n_clad = 1.455
params = get_LP_params(l,m, n_core, n_clad, b1.waist, b1.lamb)

b1.lp(l, m, n_core, n_clad)
#b1.propagate(z = 1e-5)

"""print(b1.int_profile().shape)
print("Power Bessel 2:", b1.Power())
print("Center of mass Bessel 2:", b1.center_mass())"""
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
#plt.imshow(H, cmap='gray')
plt.colorbar()
plt.show()

