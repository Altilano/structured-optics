"""
Sets a beam space with 10mm x 10mm and polarization dimension = 2.
Creates a Hermite-Gaussian Beam with waist = 1mm, wavelength = 1064nm and indices (n=1, m=0) in horizontal polarization Ex. 
Creates a Hermite-Gaussian Beam with same aprameters and indices (n=0, m=1) in vertical polarization Ey. 
Display intensity profile of horizontal polarization, vertical polarization and full intensity profile.
"""

#Used just to import from folder structured_optics directly
import os
import sys
# Add parent folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from structured_optics import *

# Create a beam on a 10mm x 10mm (2*5e-3) grid with 512x512 points
beam = Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9, pol_dim=2)

beam.Ex = hg(beam, 1,0)  #sets HG_10 into horizontal polarization
beam.Ey = hg(beam, 0,1)  #sets HG_01 into vertical polarization

import matplotlib.pyplot as plt

#Displays

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].pcolormesh(beam.x, beam.y, beam.int_profile(0), cmap='inferno')
axes[0].set_title('Figure 1: Horizontal Polarization')

axes[1].pcolormesh(beam.x, beam.y, beam.int_profile(1), cmap='inferno')
axes[1].set_title('Figure 2: Vertical Polarization')

axes[2].pcolormesh(beam.x, beam.y, beam.int_profile(), cmap='inferno')
axes[2].set_title('Figure 3: Full Beam intensity profile')

fig.tight_layout()
plt.show()
