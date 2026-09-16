import matplotlib.pyplot as plt
from structured_optics import *
import numpy as np



b1 = Beam(nix=5e-3, 
          Dx=1024,
          waist = 1e-3,
          lamb = 633e-9,
          pol_dim=2)


#b1.hg(1,1)
b1.Ex =  hg(b1, 2,1)
#b1.lp(2,2,1.7,1.4)
#b1.Ey = lg(b1,2,2)
#b1 = b1*HWP(np.pi/6)*HWP(-np.pi/6)
b1.apply(ZernikeMask([(2,-2)], [1]))

#b1.lp(1,1, 1.45, 1.4)

b1.propagate(b1.zr()/10, method='fres_c')


print(b1.Power(0), b1.Power(1))
print(b1.Power())
plt.figure()
plt.pcolormesh(b1.x, b1.y, b1.int_profile(), cmap='inferno')
plt.colorbar()
plt.show()
exit()
