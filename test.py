#import structured_optics as so
import matplotlib.pyplot as plt
from structured_optics import *
import structured_optics.slmdisplay as soslm
import numpy as np
import time


slm = soslm.My_SLM(screen_index = 1)

print(slm.width, slm.height)
pixel_size = 0.31e-3
b2 = Beam(nix = pixel_size*slm.width,
          niy = pixel_size*slm.height,
          Dx = slm.width,
          Dy = slm.height,
          waist = 1e-2,
          lamb = 1064e-9)

b2.hg(2,2)


step = 0.1
strength = np.arange(-1,1.1, step)
strength2 = strength

coefs_zern = np.array([(2,-2), (2,2)])

for s in strength:
    for s2 in strength2:
        
        aux = b2.copy()
        aux.apply_zernike(coefs_zern[0, 0], coefs_zern[0,1], s)
        aux.apply_zernike(coefs_zern[1, 0], coefs_zern[1,1], s2)
        
        start = time.time()
        holo = aux.slm_holo(3,4)
        end  = time.time()
        
        slm.update_image(holo)
        
        print(end-start)
        time.sleep(0.1)

print("finished")
"""holo = b2.slm_holo(3, 4)
slm.update_image(holo)

time.sleep(0.1)"""




exit()


b1 = Beam(nix=20e-6, 
          Dx=1024,
          waist = 6e-6,
          lamb = 1064e-9)


b1.Ex = lg(b1, 1,0)
#b1.lp(2,2,1.7,1.4, angle=np.pi/3)

#b1.vslit(b1.waist/4, babinet=True)
print(b1.center_mass())
print(b1.radial_std())
print(b1.waist/np.sqrt(2))
#b1.propagate(1e-3, method='fresnel')



"""b1.stripe_v(0.2e-3, move = 0.5e-3)
H = b1.dmd_holo(0.08, 0.08)
H2 = b1.slm_holo(4, 3)"""

"""l = 3
m = 3
n_core = 1.6
n_clad = 1.455
params = get_LP_params(l,m, n_core, n_clad, b1.waist, b1.lamb)

b1.lp_hel(l, m, n_core, n_clad)"""


#b1.double_slit(1e-6, 3e-6)






plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
#plt.imshow(H, cmap='gray')
plt.colorbar()
plt.show()

exit()




b1.tilted_lens(4e-6, phi = 30*np.pi/180)
b1.propagate(z = 3.0e-6)

"""print(b1.int_profile().shape)
print("Power Bessel 2:", b1.Power())
print("Center of mass Bessel 2:", b1.center_mass())"""
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
#plt.imshow(H, cmap='gray')
plt.colorbar()
plt.show()

