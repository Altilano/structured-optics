# structured-optics

A Python library for generating, propagating, and analyzing structured light beams
(Hermite-Gaussian, Laguerre-Gaussian, Bessel, Ince-Gaussian, fiber LP modes, and more)
on a discretized transverse grid.

## Features

- **Mode generation**: Hermite-Gaussian, Laguerre-Gaussian, Bessel, Gaussian-Bessel,
  Ince-Gaussian (even/odd/helical), fractional OAM, fiber LP modes, and geometric
  apertures (circle, square, triangle).
- **Propagation**: Fresnel, Fraunhofer, and incoherent propagation methods.
- **Polarization**: vector beams with Ex/Ey/Ez components, Jones-matrix optics
  (waveplates, polarizers).
- **Optical elements**: lenses (spherical, astigmatic, tilted), slits, irises,
  and other amplitude/phase masks.
- **Analysis**: power, intensity and phase profiles, centroid, beam size (std),
  angular sections, and modal decomposition (HG/LG/Bessel basis projections).
- **Holography**: SLM and DMD hologram generation for experimental beam shaping.

## Installation

```bash
pip install structured-optics
```

## Quick start

```python
import structured_optics as so

# Create a beam on a 10mm x 10mm (2*5e-3) grid with 512x512 points
beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)

# Set the field to a Laguerre-Gaussian mode with l=2, p=0
beam.lg(l=2, p=0)

# Propagate 10 cm using Fresnel Convolution
beam.propagate(z=0.1, method='fres_c')

# Look at the intensity profile
import matplotlib.pyplot as plt
plt.pcolormesh(beam.x, beam.y, beam.int_profile())
plt.colorbar()
plt.show()
```

See the **API Reference** in the navigation for full documentation of every
class and function, generated automatically from the package's docstrings.

## Requirements

- Python 3.9+
- numpy
- scipy
- screeninfo

## License

This project is licensed under the BSD-3-Clause License.
