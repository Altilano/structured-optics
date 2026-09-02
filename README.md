# structured-optics

A Python library for generating, propagating, and analyzing structured light beams
(Hermite-Gaussian, Laguerre-Gaussian, Bessel, Ince-Gaussian, fiber LP modes, and more)
on a discretized transverse grid.

[![PyPI version](https://img.shields.io/pypi/v/structured-optics.svg)](https://pypi.org/project/structured-optics/)

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

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

<!-- TODO: confirm this is the real distribution name on PyPI -->

## Quick start

```python
from structured_optics import Beam

# Create a beam on a 5mm x 5mm grid with 256x256 points
beam = Beam(nix=5e-3, Dx=256, waist=1e-3, lamb=1064e-9)

# Set the field to a Laguerre-Gaussian mode with l=2, p=0
beam.lg(l=2, p=0)

# Propagate 10 cm and check total power
beam.propagate(z=0.1, method='fresnel')
print(beam.Power())

# Look at the intensity profile
import matplotlib.pyplot as plt
plt.imshow(beam.int_profile())
plt.show()
```

## Documentation

Full API documentation, including all mode types, propagation methods, and
optical elements, is available at:

<!-- TODO: link to your Read the Docs / GitHub Pages site -->
**https://structured-optics.readthedocs.io**

## Requirements

- Python 3.9+
- numpy
- scipy

<!-- TODO: confirm actual minimum versions and any other dependencies -->

## Contributing

Issues and pull requests are welcome at
[github.com/Altilano/structured-optics](https://github.com/Altilano/structured-optics).

## License

This project is licensed under the BSD-3-Clause License. See `LICENSE` for details.

## Citation

<!-- TODO: if this accompanies a paper, add a BibTeX entry here -->
