"""
Spatial optical elements that act on a Beam by multiplication.

SpatialMask produces a (Dy, Dx) array multiplied into the whole field
(lenses, Zernike aberrations, apertures/slits).


Usage:
    beam = beam * Lens(f=0.2) * Iris(radius=1e-3)  
    # First a Lens is applyed then an Iris.

Any mask (spatial or polarization) can be restricted to a region defined by a
BooleanMask with `restrict_mask`; outside the region the beam is untouched:

    quarter = AngularSection(0, np.pi / 2)
    beam = beam * restrict_mask(QWP(angle=np.pi / 4), quarter)
"""
from abc import ABC, abstractmethod
import numpy as np

from .modes import circle, square, triangle
from .utils import zernikes_phase

__all__ = [ "Lens", "AstigmaticLens", "TiltedLens", "ZernikeMask",
            "Iris", "SquareAperture", "TriangleAperture", "HSlit", 
            "VSlit", "DoubleSlit", "CrossSlit", "AngularSection",
            "RegionMask", "restrict_mask"]


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
    """A thin spherical lens tilted about the y axis, modeled as an equivalent
    AstigmaticLens with effective (fx, fy) derived from the tilt angle and
    refractive index."""

    def __init__(self, f: float, phi: float, n: float = 1.5,
                 f0: tuple = (0, 0), flip_axis: bool = False):
        """
        Parameters
        ----------
        f : float
            Nominal (untilted) focal length of the lens.
        phi : float
            Tilt angle, in radians.
        n : float, optional
            Refractive index of the lens material. Defaults to 1.5.
        f0 : tuple, optional
            (x, y) offset of the lens center. Defaults to (0, 0).
        flip_axis : bool, optional
            If True, the tilt is about the x axis instead of y
            (swaps fx and fy). Defaults to False.
        """
        if n <= 1:
            raise ValueError("n must be greater than 1")

        denom = np.sqrt(n**2 - np.sin(phi)**2) - np.cos(phi)
        fy = f * (n - 1) / denom
        fx = fy * np.cos(phi) ** 2

        if flip_axis:
            fx, fy = fy, fx

        self.n = n
        self.phi = phi
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


class AngularSection(BooleanMask):
    """An angular sector (wedge) aperture.

    Transmits the points whose polar angle, measured counter-clockwise from
    the +x axis around `center`, lies in the half-open interval
    ``[ang_min, ang_max)``.
    """

    def __init__(self, ang_min: float, ang_max: float, center=(0, 0), complementary=False):
        """
        Parameters
        ----------
        ang_min : float
            Start angle of the section, in radians. Wrapped to [0, 2*pi).
        ang_max : float
            End angle of the section (exclusive), in radians. Wrapped to
            [0, 2*pi). If it is smaller than `ang_min` after wrapping, the
            section crosses the +x axis (e.g. ``(3*pi/2, pi/2)`` is the
            right half-plane). If the two wrapped angles are equal (e.g.
            ``(0, 2*pi)``), the section covers the whole plane.
        center : tuple, optional
            (x, y) vertex of the wedge. Defaults to (0, 0).
        complementary : bool, optional
            Invert the aperture. Defaults to False.
        """

        super().__init__(complementary)
        self.ang_min = ang_min % (2 * np.pi)
        self.ang_max = ang_max % (2 * np.pi)
        self.center = center

    def bool_array(self, beam):
        x, y = beam.x - self.center[0], beam.y - self.center[1]
        theta = np.mod(np.arctan2(y, x), 2 * np.pi)

        if self.ang_min < self.ang_max:
            return (theta >= self.ang_min) & (theta < self.ang_max)
        # Section crosses the +x axis (angle 0 == 2*pi).
        return (theta >= self.ang_min) | (theta < self.ang_max)


# ------------------------------------------------------------ composition --

class RegionMask(Mask):
    """Applies `mask` only inside the True region of a `BooleanMask`.

    Outside the region the beam passes through unchanged (the identity
    element of the wrapped mask: transmittance 1 for a spatial mask, the
    identity Jones matrix for a polarization mask), which is *not* the same
    as blocking the beam there. Works with any `Mask` -- spatial,
    polarization, or another `RegionMask` -- because it relies only on
    `mask.apply`. Prefer building it through `restrict_mask`.
    """

    def __init__(self, mask: Mask, region: BooleanMask):
        """
        Parameters
        ----------
        mask : Mask
            The mask to apply inside the region.
        region : BooleanMask
            Defines where `mask` acts: it is applied wherever
            ``region.array(beam)`` is True.

        Raises
        ------
        TypeError
            If `mask` is not a `Mask` or `region` is not a `BooleanMask`.
        """
        if not isinstance(mask, Mask):
            raise TypeError(f"mask must be a Mask, got {type(mask).__name__}")
        if not isinstance(region, BooleanMask):
            raise TypeError(f"region must be a BooleanMask, got {type(region).__name__}")
        self.mask, self.region = mask, region

    def apply(self, beam):
        """Apply `self.mask` to a copy of `beam`, but only inside the region.

        Parameters
        ----------
        beam : Beam
            The beam to apply the mask to.

        Returns
        -------
        Beam
            A copy of `beam` equal to ``self.mask.apply(beam)`` where the
            region is True and equal to `beam` elsewhere. The original
            `beam` is left unmodified.
        """
        modified = self.mask.apply(beam)
        inside = self.region.array(beam)
        result = beam.copy()
        result.field = np.where(inside, modified.field, beam.field)
        return result


def restrict_mask(mask: Mask, region: BooleanMask) -> RegionMask:
    """Restrict `mask` to the True region of a boolean mask.

    The returned mask applies `mask` where `region` is True and leaves the
    beam untouched elsewhere. It works for spatial masks (lenses, Zernike
    aberrations, apertures) and polarization masks (waveplates, polarizers).

    Parameters
    ----------
    mask : Mask
        The mask to apply inside the region.
    region : BooleanMask
        Defines the region where `mask` acts. For convenience, the two
        arguments may be given in either order when exactly one of them is
        a `BooleanMask`; if both are, the second is the region.

    Returns
    -------
    RegionMask
        A new mask, usable as ``beam * restrict_mask(...)``.

    Raises
    ------
    TypeError
        If neither argument is a `BooleanMask`, or if either is not a `Mask`.

    Examples
    --------
    Quarter-wave plate acting only in the upper half-plane:

    >>> upper = AngularSection(0, np.pi)
    >>> beam = beam * restrict_mask(QWP(angle=np.pi / 4), upper)

    Lens only outside a circular aperture (use ``complementary=True``):

    >>> beam = beam * restrict_mask(Lens(f=0.5), Iris(radius=1e-3, complementary=True))
    """
    if isinstance(mask, BooleanMask) and not isinstance(region, BooleanMask):
        mask, region = region, mask
    if not isinstance(region, BooleanMask):
        raise TypeError("one of the two masks must be a BooleanMask")
    return RegionMask(mask, region)



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