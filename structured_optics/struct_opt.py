import numpy as np
import copy
from scipy import fft
from contextlib import contextmanager

from .prop_methods import(  propagate_fresnel_conv, propagate_fresnel_fft, propagate_bluestein,
                            propagate_angular_spectrum, propagate_incoherent, propagate_fraunhofer,
                            suggest_propagation_method, estimate_bluestein_range)

from .modes import (hg, hg_astigmatic, lg, nbessel, gbessel, lg_prod, frac_oam, frac_oam_qs,
                    IG_even, IG_odd, HInceG, circle, square, triangle, lp, lp_hel)

from .utils import _build_from_coefs_and_basis
from .diagnostics import BeamDiagnostics

__all__ = ["Beam"]



class Beam(BeamDiagnostics):
    """
    Represents a monochromatic paraxial optical beam on a discretized transverse grid.

    A Beam stores a complex electric field over a 2D (x, y) region of interest,
    with 1, 2, or 3 polarization components (Ex, Ey, Ez). It provides methods to
    populate the field with standard mode families (Hermite-Gaussian, Laguerre-Gaussian,
    Bessel, Ince-Gaussian, fiber LP modes, etc.), apply optical elements (lenses,
    apertures, polarization optics), propagate the field, and analyze its properties
    (power, phase, centroid, mode decomposition).

    The spatial grid spans [-nix, nix] x [-niy, niy] with Dx x Dy points. A matching
    spatial-frequency grid (kx, ky) is precomputed for Fourier-based propagation.

    All input values in meters.

    Attributes
    ----------
    nix, niy : float
        Half-width of the region of interest in x and y.
    Dx, Dy : int
        Number of grid points in x and y.
    x, y : ndarray
        Sparse meshgrid arrays of spatial coordinates.
    kx, ky : ndarray
        Sparse meshgrid arrays of spatial frequency coordinates (angular, rad/unit length).
    field : ndarray, shape (pol_dim, Dy, Dx), complex128
        The transverse field. Index 0 = Ex, 1 = Ey, 2 = Ez (when present).
    x0, y0 : float
        Reference center coordinates used by mode generators and rotations.
    lamb : float
        Wavelength.
    waist : float
        Reference beam waist used by mode-generating methods.
    pol : int
        Number of polarization components stored (1 = scalar, 2 = Ex/Ey, 3 = Ex/Ey/Ez).
    """
    def __init__(self, nix: float,        
                 Dx: int,                 
                 niy: float=None,         
                 Dy: int=None,            
                 waist:float=1e-3,        
                 lamb:float = 1064e-9,    
                 x0:float = 0,            
                 y0:float = 0,            
                 pol_dim:int = 1) -> object: 
        """Initialize the spatial/spectral grids and an empty field array."""

        #Geometric properties
        self.nix = nix
        if niy is None:
            self.niy = nix
        else:
            self.niy = niy
        self.Dx = Dx
        if Dy is None:
            self.Dy = Dx
        else:
            self.Dy = Dy
        self.x, self.y = np.meshgrid(np.linspace(-self.nix, self.nix, self.Dx), np.linspace(-self.niy, self.niy, self.Dy), sparse=True)
        self.field = np.zeros((pol_dim, self.Dy, self.Dx), dtype='complex128')
        self.kx, self.ky = np.meshgrid(2*np.pi*fft.fftfreq(self.Dx, 2*self.nix/(self.Dx-1)), 2*np.pi*fft.fftfreq(self.Dy, 2*self.niy/(self.Dy-1)), sparse=True)
        self.x0 = x0
        self.y0 = y0
        

        #Physical properties
        self.lamb = lamb                 
        self.waist = waist   
        self.pol = pol_dim     


        
    @property
    def Ex(self):
        """ndarray: View of the field's x-polarization component (field[0])."""
        return self.field[0]
    
    @property
    def Ey(self):
        """ndarray: View of the field's y-polarization component (field[1])."""
        return self.field[1]

    @property
    def Ez(self):
        """ndarray: View of the field's z-polarization component (field[2])."""
        return self.field[2]

    @Ex.setter
    def Ex(self, mode):
        self.field[0] = mode

    @Ey.setter
    def Ey(self, mode):
        self.field[1] = mode

    @Ez.setter
    def Ez(self, mode):
        self.field[2] = mode



    #Auxialiary 

    def copy(self):                 
        """Return a deep copy of this Beam, including its field data."""
        return copy.deepcopy(self)
    
    def copy_clean(self, pol_dim = None):           
        """
        Return a deep copy of this Beam with the field reset to zero.

        Parameters
        ----------
        pol_dim : int, optional
            Number of polarization components for the new (zeroed) field.
            Defaults to this beam's current `pol`.

        Returns
        -------
        Beam
            A copy sharing this beam's grid/physical parameters but with an
            all-zero field of shape (pol_dim, Dy, Dx).
        """
        new = self.copy()
        if pol_dim is None:
            pol_dim = self.pol
        new.field = np.zeros((pol_dim, self.Dy, self.Dx), dtype='complex128')
        return new





    #Operator overloading

    def __mul__(self, other):
        """
        Multiply fields.

        If `other` is a Beam, multiplies the two fields element-wise. If `other` is a scalar
        (int/float/complex), scales the field globally. If `other` is a Mask
        (see masks.py / polarization.py), returns NotImplemented so Python
        falls back to `other.__rmul__(self)`, which applies the mask.
        """
        if isinstance(other, (type(self))):
            new = self.copy()
            new.field = self.field*other.field
            return new
        elif isinstance(other, (int, float, complex)):
            new = self.copy()
            new.field = other*self.field
            return new

        return NotImplemented   

    __rmul__ = __mul__
    
    def __add__(self, other: object):
        """Add another Beam's field to this one, element-wise (coherent superposition)."""
        if isinstance(other, (Beam, type(self))):
            new = self.copy()
            new.field = other.field + self.field
            return new
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other: object):
        """Subtract another Beam's field from this one, element-wise."""
        if isinstance(other, Beam):
            new = self.copy()
            new.field = self.field - other.field
            return new
        return NotImplemented

    def __rsub__(self, other: object):
        """Subtract this Beam's field from `other`'s field, element-wise."""
        if isinstance(other, Beam):
            new = self.copy()
            new.field = other.field - self.field
            return new
        return NotImplemented

    def __truediv__(self, other):
        """Divide the field by a numeric scalar."""
        if isinstance(other, (int, float, complex)):
            new = self.copy()
            new.field = self.field / other
            return new
        return NotImplemented
    





    #modes

    @contextmanager
    def rotated_grid(self, angle):
        """
        Temporarily rotate the beam's (x, y) coordinate grid about (x0, y0).

        Within the `with` block, `self.x` and `self.y` are replaced by grids
        rotated by `angle` (counter-clockwise, in radians) about the beam center.
        The original grids are restored automatically on exit, even if an
        exception occurs. Used internally by mode generators that accept an
        `angle` argument.

        Lose sparse property, so modes that can use this property to speed up calculation can be significantly slower. Eg: HG modes.

        Parameters
        ----------
        angle : float
            Rotation angle in radians.

        Yields
        ------
        None
        """
        old_x = self.x
        old_y = self.y
        try:
            c = np.cos(angle)
            s = np.sin(angle)
            X = self.x - self.x0
            Y = self.y - self.y0
            self.x = c*X + s*Y + self.x0
            self.y = -s*X + c*Y + self.y0
            yield
        finally:
            self.x = old_x
            self.y = old_y

    def _set_mode(self, mode, polarization=None):
        """
        Write a scalar transverse mode into the field, distributed across
        polarization components according to a (normalized) Jones-like vector.

        Parameters
        ----------
        mode : ndarray, shape (Dy, Dx)
            Complex scalar spatial mode profile.
        polarization : array_like, shape (pol,), optional
            Complex weights describing how the scalar mode is distributed among
            polarization components. Normalized to unit norm internally.
            Defaults to putting all amplitude into the first component
            (e.g. purely Ex-polarized).

        Returns
        -------
        Beam
            self, with `field` overwritten.

        Raises
        ------
        ValueError
            If `polarization` has the wrong shape or is the zero vector.
        """

        if polarization is None:
            polarization = np.zeros(self.pol, dtype=np.complex128)
            polarization[0] = 1

        polarization = np.asarray(polarization, dtype=np.complex128)
        if polarization.shape != (self.pol,):
            raise ValueError(
                f"polarization must have shape ({self.pol},)"
            )
        norm = np.linalg.norm(polarization)
        if norm == 0:
            raise ValueError("Polarization vector cannot be zero.")
        polarization /= norm

        self.field[:] = polarization[:, None, None] * mode[None, :, :]
        return self


    
    #returns the Beam object with the mode stored in field according to _set_mode
    def hg(self, n: int, m: int, z: float = 0, angle:float=0, polarization:list=None):
        """
        Set the field to a Hermite-Gaussian mode HG_{n,m}, analytically propagated to distance z.

        Parameters
        ----------
        n, m : int
            Mode indices along the (possibly rotated) x and y axes.
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
           [1] https://en.wikipedia.org/wiki/Gaussian_beam
        """
        return self._set_mode(hg(self, n, m, z, angle=angle),  polarization=polarization)


    def hg_astigmatic(self, n: int, m: int, wx: float, wy: float, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to an astigmatic Hermite-Gaussian mode with independent
        waists wx (x-axis) and wy (y-axis), analytically propagated to distance z.

        Parameters
        ----------
        n, m : int
            Mode indices along the (possibly rotated) x and y axes.
        wx, wy : float
            waists along x-axis and y-axis
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Arnaud, Jacques A., and Herwig Kogelnik. "Gaussian light beams with general astigmatism." 
            Applied optics 8.8 (1969): 1687-1693.
        """
        return self._set_mode(hg_astigmatic(self, n, m, wx, wy, z, angle=angle),  polarization=polarization)


    def lg(self, l: int, p: int, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a Laguerre-Gaussian mode LG_{l,p} (azimuthal index l,
        radial index p), analytically propagated to distance z.

        Parameters
        ----------
        l, p : int
            Mode indices, l for orbital angular momentum and p for radial order.
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
           [1] https://en.wikipedia.org/wiki/Gaussian_beam
        """
        return self._set_mode(lg(self, l, p, z, angle=angle),  polarization=polarization)


    def bessel(self, N: int, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to an ideal (non-diffracting) Bessel mode of order N, 
        analytically propagated to distance z.

        Parameters
        ----------
        N : int
            Order of the bessel mode.
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] https://www.rp-photonics.com/bessel_beams_and_bessel_gauss_beams.html
        """
        return self._set_mode(nbessel(self, N, z, angle=angle),  polarization=polarization)


    def gbessel(self, N: int, r0: int, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a Gaussian-apodized Bessel beam of order N at z=0.

        Parameters
        ----------
        N : int
            Bessel order.
        r0 : float
            Radius of the first intensity null, sets the transverse scale.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
        
        Reference
        ---------
            [1] https://www.rp-photonics.com/bessel_beams_and_bessel_gauss_beams.html
        """
        return self._set_mode(gbessel(self,N,r0, angle=angle),  polarization=polarization)


    def lg_prod(self, N: int, ls: tuple = None, centers: tuple = None, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a superposition/product of N Laguerre-Gaussian modes.

        Parameters
        ----------
        N : int
            Number of LG modes to combine.
        ls : tuple, optional
            Azimuthal indices for each of the N modes.
        centers : tuple, optional
            Transverse center offsets for each of the N modes.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
        
        Reference
        ---------
            [1] Mellado-Villaseñor, G., and B. M. Rodríguez-Lara. "Products of displaced Laguerre-Gaussian beams." 
            Physical Review A 113.2 (2026): 023514.
        """
        return self._set_mode(lg_prod(self, N, ls, centers, angle=angle),  polarization=polarization)


    def frac_oam(self, Ma: float, n_modes: int, beta: float = 0, theta_0: float = 0, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a fractional orbital-angular-momentum (OAM) beam,
        built from `n_modes` OAM components approximating a non-integer
        topological charge Ma.

        Parameters
        ----------
        Ma : float
            Desired fractional orbital angular momentum index.
        n_modes : int
            Number of integer-order LG modes included in the superposition.
        beta : float
            Angular parameter used in the complex expansion coefficients, in
            radians.
        theta_0 : float
            Reference angular position used in the expansion coefficients,
            in radians.
        z : float, optional
            Propagation distance along the optical axis, in meters.
            Default is 0.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Götte, Jörg B., et al. "Light beams with fractional orbital angular momentum and their vortex structure." 
            Optics express 16.2 (2008): 993-1006.
        """
        return self._set_mode(frac_oam(self, Ma, n_modes, beta, theta_0, z, angle=angle),  polarization=polarization)


    def frac_oam_qs(self, Ma: float, n_modes: int, beta: float = 0, theta_0: float = 0, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Quasi-stable variant of `frac_oam`: constructs a fractional-OAM beam
        using a mode expansion designed to propagate with reduced gouy phase
        differences, so quasi-stable propagation, compared to the standard construction.

        Parameters
        ----------
        Ma : float
            Desired fractional orbital angular momentum index.
        n_modes : int
            Number of integer-order LG modes included in the superposition.
        beta : float
            Angular parameter used in the complex expansion coefficients, in
            radians.
        theta_0 : float
            Reference angular position used in the expansion coefficients,
            in radians.
        z : float, optional
            Propagation distance along the optical axis, in meters.
            Default is 0.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Götte, Jörg B., et al. "Light beams with fractional orbital angular momentum and their vortex structure." 
            Optics express 16.2 (2008): 993-1006.
        """
        return self._set_mode(frac_oam_qs(self, Ma, n_modes, beta, theta_0, z, angle=angle),  polarization=polarization)


    def IG_even(self, p: int, m: int, q: float, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to an even-parity Ince-Gaussian mode IG^e_{p,m} with
        ellipticity parameter q, propagated to distance z.

        Parameters
        ----------
        p, m : int
            Ince-Gauss index with (p-m)%2 = 0
        q : float
            Ellipticity parameter
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Bandres, Miguel A., and Julio C. Gutiérrez-Vega. 
            "Ince–Gaussian modes of the paraxial wave equation and stable resonators." 
            Journal of the Optical Society of America A 21.5 (2004): 873-880.
        """
        return self._set_mode(IG_even(self, p, m, q, z, angle=angle),  polarization=polarization)


    def IG_odd(self, p: int, m: int, q: float, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to an odd-parity Ince-Gaussian mode IG^o_{p,m} with
        ellipticity parameter q, propagated to distance z.

        Parameters
        ----------
        p, m : int
            Ince-Gauss index with (p-m)%2 = 0
        q : float
            Ellipticity parameter
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.
        
        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Bandres, Miguel A., and Julio C. Gutiérrez-Vega. 
            "Ince–Gaussian modes of the paraxial wave equation and stable resonators." 
            Journal of the Optical Society of America A 21.5 (2004): 873-880.
        """
        return self._set_mode(IG_odd(self, p, m, q, z, angle=angle),  polarization=polarization)


    def HelIG(self, p: int, m: int, q: float, z: float = 0, helicity: int = 1, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to an helical Ince-Gaussian mode HelIG_{p,m} with
        ellipticity parameter q, propagated to distance z. Helical is a superposition of the kind IG^e +- 1j*IG^o.

        Parameters
        ----------
        p, m : int
            Ince-Gauss index with (p-m)%2 = 0
        q : float
            Ellipticity parameter
        z : float, optional
            Propagation distance from the beam waist at which the mode is evaluated.
        helicity : {-1, 1}, optional
            Defines the helicity +1 or -1. Must be one of the two.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.
        
        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Bandres, Miguel A., and Julio C. Gutiérrez-Vega. 
            "Ince–Gaussian modes of the paraxial wave equation and stable resonators." 
            Journal of the Optical Society of America A 21.5 (2004): 873-880.
        """
        return self._set_mode(HInceG(self, p, m, q, helicity=helicity, z=z, angle=angle),  polarization=polarization)


    def circle(self, center: tuple = None, radius: float = None, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a filled circular aperture (uniform amplitude) mode.

        Parameters
        ----------
        center : tuple, optional
            (x, y) center of the circle.
        radius : float, optional
            Radius of the circle.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
        """
        return self._set_mode(circle(self, center=center, radius=radius, angle=angle),  polarization=polarization)


    def square(self, center: tuple = None, side_length: float = None, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a filled square aperture (uniform amplitude) mode.

        Parameters
        ----------
        center : tuple, optional
            (x, y) center of the square.
        side_length : float, optional
            Side length of the square.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
        """
        return self._set_mode(square(self, center, side_length, angle=angle),  polarization=polarization)


    def triangle(self, center: tuple = None, side_length: float = None, angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a filled triangular aperture (uniform amplitude) mode.

        Parameters
        ----------
        center : tuple, optional
            (x, y) center of the triangle.
        side_length : float, optional
            Side length of the triangle.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
        """
        return self._set_mode(triangle(self, center, side_length, angle=angle),  polarization=polarization)


    def lp(self, l: int, m: int, n_core:float, n_clad:float , parity:str = "cos", angle:float=0, polarization:list=None) -> object:
        """
        Set the field to a linearly-polarized (LP) step-index fiber mode LP_{l,m}.

        Parameters
        ----------
        l : int
            Azimuthal mode index (l >= 0).
        m : int
            Radial mode index (m >= 1).
        n_core, n_clad : float
            Refractive indices of the fiber core and cladding.
        parity : {'cos', 'sin'}, optional
            Azimuthal parity of the mode.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
        
        Reference
        ---------
            [1] Mitschke, Fedor, and Fedor Mitschke. Fiber optics. Vol. 2. Berlin, Germany:: Springer, 2016.
        """
        return self._set_mode(lp(self, l, m, n_core, n_clad, parity=parity, angle=angle),  polarization=polarization)


    def lp_hel(self, l:int, m:int, n_core:float, n_clad:float, angle:float=0, polarization:list=None) -> object:
        """"
        Set the field to a helical LP fiber mode, formed as LP_cos +- 1j*LP_sin,
        carrying orbital angular momentum whose handedness is set by the sign of `l`.

        Parameters
        ----------
        l : int
            Azimuthal mode index; sign selects the helicity.
        m : int
            Radial mode index (m >= 1).
        n_core, n_clad : float
            Refractive indices of the fiber core and cladding.
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.

        Reference
        ---------
            [1] Mitschke, Fedor, and Fedor Mitschke. Fiber optics. Vol. 2. Berlin, Germany:: Springer, 2016.
        """
        return self._set_mode(lp_hel(self, l, m, n_core, n_clad, angle=angle),  polarization=polarization)
    



    #Linear Algebra with modes utils

    def build_from_coefs_and_basis(self, coefs:np.ndarray, basis:np.ndarray, pol_index:int = 0) -> object:
        """
        Build the field as a linear combination of basis modes weighted by coefs. Basis can be constructed with tools.py functions like hg_basis().

        Parameters
        ----------
        coefs : ndarray
            Complex expansion coefficients, one per basis mode.
        basis : ndarray
            Array of basis mode fields (e.g. from `hg_basis`/`lg_basis`/`bessel_basis`).
        pol_index : int, optional
            Polarization component into which the resulting field is written.

        Returns
        -------
        Beam
            self, with field set to the weighted sum of basis modes.
        """
        return _build_from_coefs_and_basis(self, coefs, basis, pol_index = pol_index)



    #Propagation 

    _PROPAGATORS = {
        'fres_c':   lambda beam, z, **kw: propagate_fresnel_conv(beam, z),
        'AS':       propagate_angular_spectrum,
        'fraun':    lambda beam, z, **kw: propagate_fraunhofer(beam, z),
        'fres_f':   lambda beam, z, **kw: propagate_fresnel_fft(beam, z),
        'blue':     propagate_bluestein,
        'blue_fix': lambda beam, z, **kw: propagate_bluestein(beam, z, Dx_out=beam.Dx, Dy_out=beam.Dy,**dict(zip(('x_out_range', 'y_out_range'), estimate_bluestein_range(beam, z, **kw)))),
        'inc':      lambda beam, z, **kw: propagate_incoherent(beam, z),
    }

    def propagate(self, z, method='fres_c', **kwargs):           
        """
        Parameters
        ----------
        z : float
            Propagation distance.
        method : {'auto', 'fresn_c', 'AS', 'fraun', 'fres_f', 'blue', 'blue_fix', 'inc'}, optional
            Diffraction model used for propagation. Default is 'fres_c'.
        renorm : bool, optional
            If True, renormalize total power to 1 after propagation.
        evanescent : bool, optional
            Optional in Angular Spectrum method. Default is False, but if True the code keeps the evanscent contribution to the field.
        x_out_range, y_out_range : tuple
            Necessary for bluestein method without fixed window range. tuple containing (x_min, x_max), (y_min, y_max) of the output window.
        Dx_out, Dy_out : int, optional
            Optional for bluestein method. Sets number of output samples along x and y.
        n_sigma : float, optional
            Optional in Bluestein Fix. How many standard deviations the bluestein_fix window range should be.

        Returns
        -------
        Beam
            self, with field propagated by distance z.

        References
        ----------
            [1] Goodman, Joseph W., and Mary E. Cox. "Introduction to Fourier optics." (1969): 97-101.
            [2] Hu, Yanlei, et al. "Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method." 
            Light: Science & Applications 9.1 (2020): 119.

        """
        if method == 'auto':
            method, _ = suggest_propagation_method(self, z)
            if method == 'none':
                return self
        try:
            propagator = self._PROPAGATORS[method]
        except KeyError:
            raise ValueError(f"Unable to propagate, invalid method {method!r}. " 
                             f"Valid options: {list(self._PROPAGATORS)}")

        self = propagator(self, z, **kwargs)
        return self


