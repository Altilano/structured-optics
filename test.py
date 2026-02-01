from struct_opt import *
import matplotlib.pyplot as plt
import numpy as np

b1 = Beam(5e-3, 256)

b1.HelIG(p=10, m=6, q=1)


plt.imshow(b1.phase(), cmap='gray')
plt.colorbar()
plt.show()