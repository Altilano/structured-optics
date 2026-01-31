from beam import *
import numpy as np
import matplotlib.pyplot as plt


"""b1 = Space(4e-3, 512)
b1.waist = 1e-3
b1.hg(0,0)"""

b2 = Space(5e-3, 512)
b2.waist = 1e-3
b2.lg(3,6)


U = np.abs(b2.field)
U = U/np.amax(U)
phi = b2.phase()
A = np.arcsin(U)

c = 0.08
fx = c*b2.Dx/(2*b2.nix)
fy = c*b2.Dy/(2*b2.niy)
g = [fx, fy]
G = g[0]*b2.x + g[1]*b2.y


H = 0.5 + 0.5*np.sign(np.cos(phi + 2*np.pi*G) - np.cos(A))

plt.pcolormesh(b2.x, b2.y, H, cmap="gray")
plt.colorbar()
plt.show()
