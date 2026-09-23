"""
Spatial optical elements that act on a Beam by multiplication.

SpatialMask produces a (Dy, Dx) array multiplied into the whole field
(lenses, Zernike aberrations, apertures/slits).


Usage:
    beam = beam * Lens(f=0.2) * Iris(radius=1e-3)  
    # First a Lens is applyed then an Iris.
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
        """Enable `beam * SomeMask(...)` syntax.
 
        Parameters
        ----------
        beam : Beam
            Left-hand operand of the multiplication.
 
        Returns
        -------
        object or NotImplemented
            The result of ``self.apply(beam)`` if `beam` looks like a
            `Beam` (i.e. it has a `.field` attribute); otherwise
            `NotImplemented`, so Python can fall back to other
            multiplication behavior.
        """

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
        """Multiply this mask's transmittance into a copy of `beam`.
 
        Parameters
        ----------
        beam : Beam
            The beam to apply the mask to.
 
        Returns
        -------
        Beam
            A copy of `beam` whose `.field` has been multiplied elementwise
            by `self.array(beam)`. The original `beam` is left unmodified.
        """
        result = beam.copy()
        result.field *= self.array(result)
        return result


class Lens(SpatialMask):
    """An ideal thin lens, applying a paraxial parabolic phase profile."""
    def __init__(self, f: float, f0: tuple = (0, 0)):
        """
        Parameters
        ----------
        f : float
            Focal length of the lens.
        f0 : tuple, optional
            (x, y) offset of the lens center relative to the beam's
            optical axis. Defaults to (0, 0).
        
        Reference
        ---------
            [1] Goodman, Joseph W., and Mary E. Cox. "Introduction to Fourier optics." (1969): 97-101.
        """

        self.f, self.f0 = f, f0

    def array(self, beam):
        k = beam.k()
        x, y = beam.x - self.f0[0], beam.y - self.f0[1]
        return np.exp(-1j * k * (x**2 + y**2) / (2 * self.f))


class AstigmaticLens(SpatialMask):
    """A lens with independent focal lengths along x and y (astigmatism)."""
    def __init__(self, fx: float, fy: float, f0: tuple = (0, 0)):
        """
        Parameters
        ----------
        fx : float
            Focal length along the x axis.
        fy : float
            Focal length along the y axis.
        f0 : tuple, optional
            (x, y) offset of the lens center. Defaults to (0, 0).
        """

        self.fx, self.fy, self.f0 = fx, fy, f0

    def array(self, beam):
        k = beam.k()
        x, y = beam.x - self.f0[0], beam.y - self.f0[1]
        return np.exp(-1j * k * (x**2 / self.fx + y**2 / self.fy) / 2)


class TiltedLens(AstigmaticLens):
    """A spherical lens viewed at an angle, modeled as an equivalent
    AstigmaticLens with effective (fx, fy) derived from the tilt angle."""
    def __init__(self, f: float, phi: float, f0: tuple = (0, 0), flip_axis: bool = False):
        """
        Parameters
        ----------
        f : float
            Nominal (untilted) focal length of the lens.
        phi : float
            Tilt angle, in radians.
        f0 : tuple, optional
            (x, y) offset of the lens center. Defaults to (0, 0).
        flip_axis : bool, optional
            If True, swap the roles of the tilted/untilted axes.
            Defaults to False.
        """

        fx, fy = f * np.cos(phi) ** 3, f * np.cos(phi)
        if flip_axis:
            fx, fy = fy, fx
        super().__init__(fx, fy, f0)


class ZernikeMask(SpatialMask):
    """A phase mask built from a weighted sum of Zernike polynomials,
    used to model optical aberrations."""
    def __init__(self, coefs: tuple, strengths: tuple):
        """
        Parameters
        ----------
        coefs : tuple
            Zernike term identifiers (indices/orders) to include.
        strengths : tuple
            Coefficient strengths corresponding to each term in `coefs`.
        
        Reference
        ---------
            [1] https://en.wikipedia.org/wiki/Zernike_polynomials
        """

        self.coefs, self.strengths = coefs, strengths

    def array(self, beam):
        return zernikes_phase(beam, coefs=self.coefs, strengths=self.strengths)

class BooleanMask(SpatialMask):
    """Base class for hard-edged apertures. Subclasses implement bool_array;
    `complementary=True` inverts the aperture."""

    def __init__(self, complementary: bool = False):
        """
        Parameters
        ----------
        complementary : bool, optional
            If True, invert the aperture (block where it would otherwise
            pass, and vice versa). Defaults to False.
        """

        self.complementary = complementary

    @abstractmethod
    def bool_array(self, beam) -> np.ndarray:
        """Return a boolean array, True where the aperture transmits."""
        raise NotImplementedError

    def array(self, beam):
        """Build the (possibly inverted) boolean transmittance array.
 
        Parameters
        ----------
        beam : Beam
            Beam whose spatial shape (`beam.field.shape[-2:]`) the mask
            must broadcast to.
 
        Returns
        -------
        np.ndarray
            Boolean array broadcast to the beam's spatial shape. Equal to
            `bool_array(beam)`, or its logical negation if
            `self.complementary` is True.
 
        Raises
        ------
        ValueError
            If `bool_array(beam)` cannot be broadcast to the beam's
            spatial shape.
        """
        m = np.asarray(self.bool_array(beam), dtype=bool)
        spatial_shape = beam.field.shape[-2:]
        try:
            m = np.broadcast_to(m, spatial_shape)
        except ValueError:
            raise ValueError(f"Mask must be broadcastable to spatial shape {spatial_shape}, got {m.shape}")
        return ~m if self.complementary else m


class Iris(BooleanMask):
    """A circular aperture."""
    def __init__(self, radius=None, center=(0, 0), complementary=False):
        """
        Parameters
        ----------
        radius : float, optional
            Radius of the circular aperture.
        center : tuple, optional
            (x, y) center of the circle. Defaults to (0, 0).
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """
        super().__init__(complementary)
        self.center, self.radius = center, radius

    def bool_array(self, beam):
        return circle(beam, self.center, self.radius)


class SquareAperture(BooleanMask):
    """A square aperture."""
    def __init__(self, side_length=None, center=(0, 0), complementary=False):
        """
        Parameters
        ----------
        side_length : float, optional
            Side length of the square aperture.
        center : tuple, optional
            (x, y) center of the square. Defaults to (0, 0).
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

        super().__init__(complementary)
        self.center, self.side_length = center, side_length

    def bool_array(self, beam):
        return square(beam, self.center, self.side_length)


class TriangleAperture(BooleanMask):
    """A triangular aperture."""
    def __init__(self, center=(0, 0), side_length=None, complementary=False):
        """
        Parameters
        ----------
        center : tuple, optional
            (x, y) center of the triangle. Defaults to (0, 0).
        side_length : float, optional
            Side length of the triangle.
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

        super().__init__(complementary)
        self.center, self.side_length = center, side_length

    def bool_array(self, beam):
        return triangle(beam, self.center, self.side_length)


class HSlit(BooleanMask):
    """A horizontal slit: passes a vertical strip of width `2 * size`
    centered at `center` along the x axis."""

    def __init__(self, size, center=0, complementary=False):
        """
        Parameters
        ----------
        size : float
            Half-width of the slit.
        center : float, optional
            Center position along x. Defaults to 0.
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

        super().__init__(complementary)
        self.size, self.center = size, center

    def bool_array(self, beam):
        return np.abs(beam.x - self.center) <= self.size


class VSlit(BooleanMask):
    """A vertical slit: passes a horizontal strip of width `2 * size`
    centered at `center` along the y axis."""

    def __init__(self, size, center=0, complementary=False):
        """
        Parameters
        ----------
        size : float
            Half-width of the slit.
        center : float, optional
            Center position along y. Defaults to 0.
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

        super().__init__(complementary)
        self.size, self.center = size, center

    def bool_array(self, beam):
        return np.abs(beam.y - self.center) <= self.size


class DoubleSlit(BooleanMask):
    """Two parallel slits (double-slit experiment), each of half-width
    `size`, separated by distance `dis`, centered at `center`."""

    def __init__(self, size, dis, center=0, complementary=False):
        """
        Parameters
        ----------
        size : float
            Half-width of each slit.
        dis : float
            Center-to-center distance between the two slits.
        center : float, optional
            Midpoint of the two-slit pair along x. Defaults to 0.
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

        super().__init__(complementary)
        self.size, self.dis, self.center = size, dis, center

    def bool_array(self, beam):
        slit1 = np.abs(beam.x - (self.center - self.dis / 2)) <= self.size
        slit2 = np.abs(beam.x - (self.center + self.dis / 2)) <= self.size
        return slit1 | slit2


class CrossSlit(BooleanMask):
    """A cross-shaped (plus-sign) aperture: the union of a vertical band
    and a horizontal band through the origin."""

    def __init__(self, size, hsize=None, complementary=False):
        """
        Parameters
        ----------
        size : float
            Half-width of the vertical band (along x). Also used as the
            default horizontal band half-width if `hsize` is omitted.
        hsize : float, optional
            Half-width of the horizontal band (along y). Defaults to
            `size` if not given.
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

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