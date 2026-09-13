"""
Sets a beam space with 10mm x 10mm, creates a Gaussian Beam with waist = 1mm and wavelength = 1064nm. 
Pass it by a double slit then propagates to the far field using fraunhofer.
"""

#Used just to import from folder structured_optics directly
import os
import sys
# Add parent folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from structured_optics import *

# Create a beam on a 10mm x 10mm (2*5e-3) grid with 512x512 points
beam = Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)

beam.hg(0,0)                            #gaussian beam
beam = beam*DoubleSlit(0.5e-4, 2e-4)    #apply double slit with slit size 50um and distance 200um between the centers of the slits.



#Displays near field and far field
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

axes[0].pcolormesh(beam.x, beam.y, beam.int_profile(), cmap='inferno')
axes[0].set_title('Figure 1: Near Field')

beam.propagate(100*beam.zr(), method='fraun')    #propagates 100 times the Rayleigh distance, using Fraunhofer.

axes[1].pcolormesh(beam.x, beam.y, beam.int_profile(), cmap='inferno')
axes[1].set_title('Figure 2: Far Field')

fig.tight_layout()
plt.show()