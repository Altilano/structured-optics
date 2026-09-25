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

## Architecture Overview

The core of the library is the `Beam` class in `struct_opt.py`. It handles the
beam's physical parameters, operator overloading, views into the field's
polarization components, methods for setting the field to a desired structured
mode, and propagation methods (currently, only scalar propagation).

`Beam` inherits from the `BeamDiagnostics` class, which contains the methods
that output the beam's physical parameters (phase, power, k-vector, Rayleigh
range, etc.), as well as field views and angular-section views of the field.
`BeamDiagnostics` lives in `diagnostics.py`.

The actual math behind each mode family lives in `modes.py`. This separation
was chosen so that a mode can be called both as a method on a `Beam` and as a
standalone function that returns the mode's field. The two are equivalent, but
the function form is better suited for simple mode superpositions:

```python
beam.hg(0, 0)
beam.Ex = hg(beam, 0, 0)
```

`masks.py` handles the various classes representing optical elements that
interact with a beam in a single plane — lenses, irises, Zernike aberrations,
slit apertures, and so on. These fall into two groups: **boolean masks**, which
output masks of 0s and 1s and represent opaque objects (invertible via
`.complementary`), and **spatial masks**, which represent any object that
imposes a phase, such as a lens applying a different phase at each `(x, y)`
point. Any mask can be applied to a beam via right-side multiplication:

```python
beam = beam * Lens(f=10e-2)
```

`polarization.py` contains functions for polarization elements such as
half-wave plates and polarizers. They inherit from the mask classes — even
though they aren't phase- or amplitude-based — so that they can also be
applied via right-side multiplication, just like a mask. These functions
currently only work when `pol_dim = 2`, and they modify `Ex` and `Ey`
accordingly.

`prop_methods.py` contains the actual propagation math; for now, it implements
only scalar propagation methods.

`bluestein.py` contains the Bluestein FFT method, used by the Bluestein
propagation method to compute the output field over any desired region of
interest.

`hologram.py` and `slmdisplay.py` contain helper functions for generating
holograms for use with Spatial Light Modulators and Digital Micromirror
Devices. The `My_SLM` class handles displaying a hologram on an SLM, which the
computer treats as an additional screen. This doesn't apply to DMDs, since
each DMD device can have its own display method.

`tools.py` contains helper functions for computing the overlap between two
beams and their projections onto a given basis — useful if you want to treat
these fields as vectors in a modal basis.

`utils.py` contains utility functions used throughout the rest of the package.
Look here if you want to understand how something works under the hood.

## Requirements

- Python 3.9+
- numpy
- scipy
- screeninfo

## License

This project is licensed under the BSD-3-Clause License.
