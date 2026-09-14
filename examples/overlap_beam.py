"""
Sets a beam space with 10mm x 10mm, copies it into beam 2, creates a Hermite-Gaussian Beam with waist = 1mm, wavelength = 1064nm and indices (n=1, m=0). 
Repeats for beam2 with indices (n=0, m=0).
Calculates modulus squared overlap between beam and beam2, and then between beam2 and beam2.
"""

#Used just to import from folder structured_optics directly
import os
import sys
# Add parent folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from structured_optics import *

# Create a beam on a 10mm x 10mm (2*5e-3) grid with 512x512 points
beam = Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9, pol_dim=2)

beam2 = beam.copy()  #create deep copy
beam3 = beam.copy()

beam.hg(1,0)  #n=1, m=0
beam2.hg(0,0) #n=0, m=0
beam3.lg(1,0) #l=1, p=0

print(f"Modulus squared overlap between gaussian and HG_10: {np.abs(overlap(beam, beam2))**2:.2f}     (expected ~0)")
print(f"Modulus squared overlap between HG_10 and itself:   {np.abs(overlap(beam2, beam2))**2:.2f}    (expected ~1)")
print(f"Modulus squared overlap between HG_10 and LG_10:   {np.abs(overlap(beam, beam3))**2:.2f}    (expected ~0.5)")
