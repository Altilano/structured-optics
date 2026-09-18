"""
Spatial optical elements that act on a Beam by multiplication.

SpatialMask produces a (Dy, Dx) array multiplied into the whole field
(lenses, Zernike aberrations, apertures/slits).


Usage:
    beam = beam * Lens(f=0.2) * Iris(radius=1e-3)
    # or, if you prefer a verb:
    beam = beam.apply(Lens(f=0.2))
"""
from abc import ABC, abstractmethod
import numpy as np

from .modes import circle, square, triangle
from .utils import zernikes_phase

__all__ = [ "Lens", "AstigmaticLens", "TiltedLens", "ZernikeMask",
            "Iris", "SquareAperture", "TriangleAperture", "HSlit", 
            "VSlit", "DoubleSlit", "CrossSlit"]


class Mask(ABC):
    @abstractmethod
    def apply(self, beam) -> object:
        """Apply this mask to `beam`, and return a copy."""
        raise NotImplementedError

    def __rmul__(self, beam):
        # enables `beam * SomeMask(...)`
        if hasattr(beam, "field"):
            return self.apply(beam)
        return NotImplemented



# ---------------------------------------------------------------- spatial --

class SpatialMask(Mask):
    """A mask defined by a complex transmittance T(x, y), applied elementwise
    to the whole field."""

    @abstractmethod
    def array(self, beam) -> np.ndarray:
        """Return an array broadcastable to beam.field.shape[-2:]."""
        raise NotImplementedError

    def apply(self, beam):
        result = beam.copy()
        result.field *= self.array(result)
        return result


class Lens(SpatialMask):
    def __init__(self, f: float, f0: tuple = (0, 0)):
        self.f, self.f0 = f, f0

    def array(self, beam):
        k = 2 * np.pi / beam.lamb
        x, y = beam.x - self.f0[0], beam.y - self.f0[1]
        return np.exp(-1j * k * (x**2 + y**2) / (2 * self.f))


class AstigmaticLens(SpatialMask):
    def __init__(self, fx: float, fy: float, f0: tuple = (0, 0)):
        self.fx, self.fy, self.f0 = fx, fy, f0

    def array(self, beam):
        k = 2 * np.pi / beam.lamb
        x, y = beam.x - self.f0[0], beam.y - self.f0[1]
        return np.exp(-1j * k * (x**2 / self.fx + y**2 / self.fy) / 2)


class TiltedLens(AstigmaticLens):
    def __init__(self, f: float, phi: float, f0: tuple = (0, 0), flip_axis: bool = False):
        fx, fy = f * np.cos(phi) ** 3, f * np.cos(phi)
        if flip_axis:
            fx, fy = fy, fx
        super().__init__(fx, fy, f0)


class ZernikeMask(SpatialMask):
    def __init__(self, coefs: tuple, strengths: tuple):
        self.coefs, self.strengths = coefs, strengths

    def array(self, beam):
        return zernikes_phase(beam, coefs=self.coefs, strengths=self.strengths)

class BooleanMask(SpatialMask):
    """Base class for hard-edged apertures. Subclasses implement bool_array;
    `complementary=True` inverts the aperture."""

    def __init__(self, complementary: bool = False):
        self.complementary = complementary

    @abstractmethod
    def bool_array(self, beam) -> np.ndarray:
        raise NotImplementedError

    def array(self, beam):
        m = np.asarray(self.bool_array(beam), dtype=bool)
        spatial_shape = beam.field.shape[-2:]
        try:
            m = np.broadcast_to(m, spatial_shape)
        except ValueError:
            raise ValueError(f"Mask must be broadcastable to spatial shape {spatial_shape}, got {m.shape}")
        return ~m if self.complementary else m


class Iris(BooleanMask):
    def __init__(self, radius=None, center=(0, 0), complementary=False):
        super().__init__(complementary)
        self.center, self.radius = center, radius

    def bool_array(self, beam):
        return circle(beam, self.center, self.radius)


class SquareAperture(BooleanMask):
    def __init__(self, side_length=None, center=(0, 0), complementary=False):
        super().__init__(complementary)
        self.center, self.side_length = center, side_length

    def bool_array(self, beam):
        return square(beam, self.center, self.side_length)


class TriangleAperture(BooleanMask):
    def __init__(self, center=(0, 0), side_length=None, complementary=False):
        super().__init__(complementary)
        self.center, self.side_length = center, side_length

    def bool_array(self, beam):
        return triangle(beam, self.center, self.side_length)


class HSlit(BooleanMask):
    def __init__(self, size, center=0, complementary=False):
        super().__init__(complementary)
        self.size, self.center = size, center

    def bool_array(self, beam):
        return np.abs(beam.x - self.center) <= self.size


class VSlit(BooleanMask):
    def __init__(self, size, center=0, complementary=False):
        super().__init__(complementary)
        self.size, self.center = size, center

    def bool_array(self, beam):
        return np.abs(beam.y - self.center) <= self.size


class DoubleSlit(BooleanMask):
    def __init__(self, size, dis, center=0, complementary=False):
        super().__init__(complementary)
        self.size, self.dis, self.center = size, dis, center

    def bool_array(self, beam):
        slit1 = np.abs(beam.x - (self.center - self.dis / 2)) <= self.size
        slit2 = np.abs(beam.x - (self.center + self.dis / 2)) <= self.size
        return slit1 | slit2


class CrossSlit(BooleanMask):
    def __init__(self, size, hsize=None, complementary=False):
        super().__init__(complementary)
        self.size, self.hsize = size, hsize if hsize is not None else size

    def bool_array(self, beam):
        vertical = np.abs(beam.x) < self.size
        horizontal = np.abs(beam.y) < self.hsize
        return vertical | horizontal










""" #for the future
    def get_turbulence_mvk(self, d, r0, l0, L0):
        deltax = d/self.Dx
        deltay = d/self.Dy
        del_fx = 1/(deltax*self.Dx)
        del_fy = 1/(deltay*self.Dy)
        freqx = fft.fftshift(2*np.pi*fft.fftfreq(self.Dx, d=del_fx))
        freqy = fft.fftshift(2*np.pi*fft.fftfreq(self.Dy, d=del_fy))
        fx, fy = np.meshgrid(freqx, freqy)
        fm = 5.92/l0/(2*np.pi)
        f0 = 1/L0
        phi_mvk = 0.023*(r0**(-5/3))*np.exp(-(fx**2+fy**2)/fm**2)/(fx**2+fy**2 + f0**2)**(11/6)
        cn = (np.random.randn(self.Dy, self.Dx) + 1j*np.random.randn(self.Dy, self.Dx))*np.sqrt(phi_mvk) * np.sqrt(del_fx*del_fy)
        self.mask = np.real(fft.ifft2(cn))
        return self"""