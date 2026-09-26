# Usage examples

A few end-to-end snippets to get you started. For more, see the
[`examples/`](https://github.com/Altilano/structured-optics/tree/main/examples)
folder in the repository.

## Generating a structured mode

```python
import structured_optics as so
import matplotlib.pyplot as plt

beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)
beam.lg(l=2, p=0)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].pcolormesh(beam.x, beam.y, beam.int_profile())
ax[0].set_title("Intensity")
ax[1].pcolormesh(beam.x, beam.y, beam.phase())
ax[1].set_title("Phase")
plt.show()
```

## Free-space propagation

```python
beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)
beam.hg(0, 0)

# Propagate 10 cm using Fresnel convolution
beam.propagate(z=0.1, method="fres_c")

plt.pcolormesh(beam.x, beam.y, beam.int_profile())
plt.colorbar()
plt.show()
```

## Applying optical elements with masks

Masks apply to a beam via right-side multiplication, so a simple imaging
system reads like the optical setup itself:

```python
from structured_optics.masks import Lens

beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)
beam.hg(0, 0)

# Pass through a lens, then propagate to its focal plane
beam = beam * Lens(f=10e-2)
beam.propagate(z=10e-2, method="fres_c")
```

## Polarization optics

```python
from structured_optics.polarization import HWP

beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9, pol_dim=2)
beam.hg(0, 0)

# Rotate the polarization state with a half-wave plate at 22.5 degrees
beam = beam * HWP(angle=22.5)
```

## Modal decomposition

```python
import structured_optics as so
from structured_optics import tools

beam = so.Beam(nix=5e-3, Dx=512, waist=1e-3, lamb=1064e-9)
beam.lg(l=2, p=0)

# Project the field onto an LG basis to recover its modal content
l, p, coeffs = tools.lg_proj(beam, N=4)
```


