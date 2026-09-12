"""
Sets a beam space with 10mm x 10mm, creates a Laguerre-Gaussian Beam with waist = 1mm, wavelength = 1064nm and indices (l=2, p=0). 
Propagates it by 10cm and display intensity profile.
"""


#Used just to import from folder structured_optics directly
import os
import sys
# Add parent folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import structured_optics as so

# Create a beam on a 10mm x 10mm (2*5e-3) grid with 512x512 points
beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)

# Set the field to a Laguerre-Gaussian mode with l=2, p=0
beam.lg(l=2, p=0)

# Propagate 10 cm using Fresnel Convolution
beam.propagate(z=0.1, method='fres_c')


# Look at the intensity profile
import matplotlib.pyplot as plt
plt.pcolormesh(beam.x, beam.y, beam.int_profile(), cmap='inferno')
plt.colorbar()
plt.show()