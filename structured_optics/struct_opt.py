

import numpy as np
from scipy import fft, ndimage
from contextlib import contextmanager
from structured_optics.prop_methods import *
import copy
from structured_optics.utils import *
from structured_optics.modes import *
from structured_optics.algebra_utils import *
from structured_optics.hologram import *
from structured_optics.polarization import *



class Beam():
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
    def __init__(self, nix: float,        #Region of interest in x (from -nix to +nix)
                 Dx: int,                 #Number of points in x
                 niy: float=None,         #Region of interest in y (from -niy to +niy), equals nix if None
                 Dy: int=None,            #Number of points in y, equals Dx if None
                 waist:float=1e-3,        #Beam waist. Standard value 1mm
                 lamb:float = 1064e-9,    #Wavelength. Standard value 1064nm
                 x0:float = 0,            #Beam center x position
                 y0:float = 0,            #Beam center y position
                 pol_dim:int = 1) -> object: #Beam center y position
        """Initialize the spatial/spectral grids and an empty field array."""

        #Geometric properties
        self.nix = nix
        if niy == None:
            self.niy = nix
        else:
            self.niy = niy
        self.Dx = Dx
        if Dy == None:
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
        if pol_dim == None:
            pol_dim = self.pol
        new.field = np.zeros((pol_dim, self.Dy, self.Dx), dtype='complex128')
        return new





    #Operator overloading

    def __mul__(self, other):
        """
        Multiply fields.

        If `other` is a Beam, multiplies the two fields element-wise (e.g. for
        applying a transmission mask stored as a Beam). If `other` is a scalar
        (int/float/complex), scales the field globally.

        Raises
        ------
        TypeError
            If `other` is neither a Beam nor a numeric scalar.
        """
        if isinstance(other, (type(self))):
            new = self.copy()
            new.field = self.field*other.field
            return new
        elif isinstance(other, (int, float, complex)):
            new = self.copy()
            new.field = other*self.field
            return new

        raise TypeError(f"sorry, don't know how to multiply by {type(other).__name__}")
        
    __rmul__ = __mul__
    
    def __add__(self, other:object):       
        """Add another Beam's field to this one, element-wise (coherent superposition)."""
        if isinstance(other, (Beam, type(self))):
            new = self.copy()
            new.field = other.field + self.field
            return new
        raise TypeError(f"sorry, don't know how to add by {type(other).__name__}")
        
    __radd__ = __add__
        
    def __sub__(self, other:object):       
        """Subtract another Beam's field from this one, element-wise."""
        if isinstance(other, Beam):
            new = self.copy()
            new.field = self.field - other.field
            return new
        raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")
        
    def __rsub__(self, other:object):       
        """Subtract this Beam's field from `other`'s field, element-wise."""
        if isinstance(other, Beam):
            new = self.copy()
            new.field = other.field - self.field
            return new
        raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")

    def __truediv__(self, other):
        """Divide the field by a numeric scalar."""
        if isinstance(other, (int, float, complex)):
            new = self.copy()
            new.field = self.field / other
            return new

        raise TypeError(f"Cannot divide Beam by {type(other).__name__}")
    





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
            Target (possibly non-integer) OAM/topological charge.
        n_modes : int
            Number of integer-OAM components used in the expansion.
        beta : float, optional
            ???
        theta_0 : float, optional
            ???
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
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
            Target (possibly non-integer) OAM/topological charge.
        n_modes : int
            Number of integer-OAM components used in the expansion.
        beta : float, optional
            ???
        theta_0 : float, optional
            ???
        angle : float, optional
            Rotation of the mode's axes (radians) relative to the beam's x/y axes.
        polarization : array_like, optional
            Polarization weighting; see `_set_mode`.

        Returns
        -------
        Beam
            self, with field set to the requested mode.
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
        """
        return self._set_mode(HInceG(self, p, m, q, helicity=helicity, z=z, angle=angle),  polarization=polarization)


    def circle(self, center: tuple = (0, 0), radius: float = None, angle:float=0, polarization:list=None) -> object:
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
        return self._set_mode(circle(self, center, radius, angle=angle),  polarization=polarization)


    def square(self, center: tuple = (0, 0), side_length: float = None, angle:float=0, polarization:list=None) -> object:
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


    def triangle(self, center: tuple = (0, 0), side_length: float = None, angle:float=0, polarization:list=None) -> object:
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
        """
        return self._set_mode(lp_hel(self, l, m, n_core, n_clad, angle=angle),  polarization=polarization)
    





    #Linear Algebra with modes utils

    def hg_projector(self,N:int) -> tuple:
        """
        Project the beam's field onto the Hermite-Gaussian basis up to order N.

        Parameters
        ----------
        N : int
            Maximum combined mode order to project onto.

        Returns
        -------
        tuple
            (n, m, overlaps): arrays of HG mode indices n, m and the array of
            complex overlap coefficients between the beam and each basis mode.
        """
        return hg_proj(self, N)

    def lg_projector(self, N:int) -> tuple:
        """
        Project the beam's field onto the Laguerre-Gaussian basis up to order N.

        Parameters
        ----------
        N : int
            Maximum combined mode order to project onto.

        Returns
        -------
        tuple
            (l, p, overlaps): arrays of LG mode indices l, p and the array of
            complex overlap coefficients between the beam and each basis mode.
        """
        return lg_proj(self, N)
        
    def hg_basis(self, N:int, waist:float=None, norm1:bool = False) -> np.ndarray:
        """
        Build a Hermite-Gaussian basis set up to order N on this beam's grid.

        Parameters
        ----------
        N : int
            Maximum combined mode order.
        waist : float, optional
            Waist used to generate the basis modes. Defaults to `self.waist`.
        norm1 : bool, optional
            If True, normalize each basis mode to unit power.

        Returns
        -------
        ndarray
            Array of HG basis mode fields.
        """
        return hg_basis(self, N, waist, norm1)
    
    def lg_basis(self, N:int, waist:float=None, norm1:bool=False) -> np.ndarray:
        """
        Build a Laguerre-Gaussian basis set up to order N on this beam's grid.

        Parameters
        ----------
        N : int
            Maximum combined mode order.
        waist : float, optional
            Waist used to generate the basis modes. Defaults to `self.waist`.
        norm1 : bool, optional
            If True, normalize each basis mode to unit power.

        Returns
        -------
        ndarray
            Array of LG basis mode fields.
        """
        return lg_basis(self, N, waist, norm1)
    
    def bessel_basis(self, Nmax:int, waist:float=None, norm1:bool=False) -> np.ndarray:
        """
        Build a Bessel mode basis up to order Nmax on this beam's grid.

        Parameters
        ----------
        Nmax : int
            Maximum Bessel order.
        waist : float, optional
            Waist/scale used to generate the basis modes. Defaults to `self.waist`.
        norm1 : bool, optional
            If True, normalize each basis mode to unit power.

        Returns
        -------
        ndarray
            Array of Bessel basis mode fields.
        """
        return bessel_basis(self, Nmax, waist, norm1)

    def build_from_coefs_and_basis(self, coefs:np.ndarray, basis:np.ndarray, pol_index:int = 0) -> object:
        """
        Build the field as a linear combination of basis modes weighted by coefs.

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
        return build_from_coefs_and_basis(self, coefs, basis, pol_index = pol_index)
    
    



    #Beam physical atributes and its utilities

    def Power(self, pol_index=None) -> float:
        """Get the total Power of the field.
        
        Parameters
        ----------
        pol_index: int, optional
            Polarization component into which the resulting power is measured. Default is power of 
            all field components.

        Returns
        -------
        float
        """
        return np.sum(self.int_profile(pol_index))*(4*self.nix/self.Dx)*(self.niy/self.Dy)

    
    def zr(self) -> float:
        """
        Compute the Rayleigh range of the beam from its `waist` and `lamb`.

        Returns
        -------
        float
            Rayleigh range, zr = pi*waist**2/lamb.
        """
        return np.pi*self.waist**2/self.lamb
    
    def int_profile(self, pol_index=None) -> np.ndarray:
        """
        Get intensity profile.

        pol_index=None -> total intensity
        pol_index=0    -> Ex intensity
        pol_index=1    -> Ey intensity
        pol_index=2    -> Ez intensity

        Parameters
        ----------
        pol_index: int, optional
            Polarization component into which the resulting intensity profile is measured. Default is sum of 
            all intensity components.

        Return
        ------
        ndarray
            Array with intensity profile (x,y).
        """
        if pol_index is None:
            return np.sum(np.abs(self.field)**2, axis=0)
        return np.abs(self.field[pol_index])**2

    
    def phase(self, pol_index=None, twopi=False) -> np.ndarray:
        """
        Get phase profile.

        pol_index=None -> phase of all components
        pol_index=0    -> Ex
        pol_index=1    -> Ey
        pol_index=2    -> Ez 

        Parameters
        ----------
        pol_index: int, optional
            Polarization component into which the resulting phase profile is measured. Default is array with 
            all phase components.

        Return
        ------
        ndarray
            Array of phase profile (x,y).
        """
        if pol_index is None:
            field = self.field
        else:
            field = self.field[pol_index]
        p = np.angle(field)
        if twopi:
            p = np.mod(p, 2*np.pi)
        if self.pol == 1:
            return p[0]
        return p
    
    def center_mass(self, pol_index=None) -> tuple:
        """
        Compute the intensity-weighted center of mass of the field, in grid
        index coordinates.

        Parameters
        ----------
        pol_index : int, optional
            Polarization component to use; None uses total intensity.

        Returns
        -------
        tuple
            (row, col) center-of-mass indices, as returned by
            `scipy.ndimage.center_of_mass`.
        """
        return ndimage.center_of_mass(self.int_profile(pol_index))
  
    def std(self, pol_index=None) -> float:
        """
        Calculate the intensity-weighted radial standard deviation of the field
        about its center of mass.

        Parameters
        ----------
        pol_index : int, optional
            Polarization component to use; None uses total intensity.

        Returns
        -------
        float
            Intensity-weighted radial standard deviation.
        """
        I = self.int_profile(pol_index)
        c = self.center_mass(pol_index)
        x0 = self.x[0, int(round(c[1]))]
        y0 = self.y[int(round(c[0])), 0]
        r2 = (self.x - x0)**2 + (self.y - y0)**2
        return np.sqrt(np.average(r2, weights=I))
    
    def section(self, ang_min:float, ang_max:float, pol_index:int=0) -> np.ndarray:
        #Return field distribution of given section, defined by minimum angle and maximum angle
        return get_section(self, ang_min, ang_max, pol_index = pol_index)
    
    def int_section(self, ang_min:float, ang_max:float, pol_index:int=0)-> np.ndarray:
        #Get intensity profile of given section, defined by minimum angle and maximum angle
        return np.abs(self.section(ang_min, ang_max, pol_index = pol_index))**2
    
    def Power_section(self, ang_min:float, ang_max:float, pol_index:int=0)-> float:
        #Get power of given section
        return np.sum(np.abs(self.section(ang_min, ang_max, pol_index = pol_index))**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def norm_beam(self)-> object:                                       
        #normalize beam so that Total Power = 1 in region of interest
        self.field = self.field/np.sqrt(self.Power())
        return self
    
    def Max_int1(self)-> object:                                         
        #rescale the field such that the maximum Intensity point is equal to one.
        self.field = self.field/np.sqrt(np.max(self.int_profile()))
        return self
    
    def crop(self, center:tuple=None, std:float=None, window:float=2) -> object:
        #Crops a field by its std*window arround the center of mass
        return get_crop(self, center, std, window)
    

    






    #masks
    
    def lens(self, f:float, f0:tuple=(0,0))-> object:                                
        #apply a lens operator to the field, with lens center at f0 and focus lenght equal to f
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(-1j*k*(((self.x-f0[0])**2 + (self.y-f0[1])**2)/(2*f)))
        return self

    def astigmatic_lens(self, fx:float, fy:float, f0:tuple=(0,0))-> object:                      
        #apply a astigmatic lens with two focal axis, with focus fx and fy.
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(-1j*k*(((self.x-f0[0])**2)/fx + ((self.y-f0[1])**2)/fy)/2)
        return self
    
    def tilted_lens(self, f:float, phi:float, f0:tuple=(0,0), flip_axis:bool=False)-> object:                      
        #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi.
        fx = f*np.cos(phi)**3
        fy = f*np.cos(phi)
        if flip_axis == True:
            fx, fy = fy, fx
        return self.astigmatic_lens(fx,fy, f0)


    #boolean masks
    def _apply_bool_mask(self, mask, babinet=False):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.field.shape[-2:]:
            raise ValueError(f"Mask must have shape {self.field.shape[-2:]}, "f"got {mask.shape}")
        if babinet:
            mask = ~mask
        self.field *= mask
        return self
    
    def cross_slit(self, size, hsize=None, babinet=False):
        """Def cross stripe"""
        if hsize == None:
            hsize = size
        vertical = np.abs(self.x) < size
        horizontal = np.abs(self.y) < hsize
        mask = vertical | horizontal
        return self._apply_bool_mask(mask, babinet)

    def hslit(self, size, center=0, babinet=False):
        mask = np.abs(self.x - center) <= size
        return self._apply_bool_mask(mask, babinet)

    def vslit(self, size, center=0, babinet=False):
        mask = np.abs(self.y - center) <= size
        return self._apply_bool_mask(mask, babinet)
    
    def double_slit(self, size, dis, center=0, babinet=False):
        slit1 = np.abs(self.x - (center - dis/2)) <= size
        slit2 = np.abs(self.x - (center + dis/2)) <= size
        mask = slit1 | slit2
        return self._apply_bool_mask(mask, babinet)

    def iris(self, center=(0, 0), radius=None, babinet=False):
        mask = np.asarray(circle(self, center, radius), dtype=bool)
        return self._apply_bool_mask(mask, babinet)

    def triangle_slit(self, center=(0,0), side_length=None, babinet=False):
        mask = np.asarray(triangle(self, center, side_length), dtype=bool)
        return self._apply_bool_mask(mask, babinet)

    def square_slit(self, center=(0,0), side_length=None, babinet=False):
        mask = np.asarray(square(self, center, side_length), dtype=bool)
        return self._apply_bool_mask(mask, babinet)




    #polarization

    def _apply_jones(beam, J):
        """
        Apply a 2x2 Jones matrix to Beam's Ex, Ey components in place.

        J : (2,2) complex array-like, acts on (Ex, Ey).
        pol=1 -> no-op (scalar field, no polarization info)
        pol=2 -> standard Jones matrix application
        pol=3 -> acts only on Ex, Ey; Ez untouched
        """
        if beam.pol == 1:
            return beam

        J = np.asarray(J, dtype=complex)
        if J.shape != (2, 2):
            raise ValueError("Jones matrix must be 2x2")

        if beam.pol not in (2, 3):
            raise ValueError(f"Unsupported pol: {beam.pol}")

        Ex, Ey = beam.Ex, beam.Ey
        new_Ex = J[0, 0] * Ex + J[0, 1] * Ey
        new_Ey = J[1, 0] * Ex + J[1, 1] * Ey

        beam.Ex = new_Ex
        beam.Ey = new_Ey
        # Ez left alone 
        return beam

    def hwp(self, angle):
        self._apply_jones(J_hwp(angle))
        return self

    def qwp(self, angle):
        self._apply_jones(J_qwp(angle))
        return self

    def polarizer(self, angle=0, proj='H'):
        """Polarizer projector. Use proj = 'H', 'V', 'D', 'A', 'R', 'L' for 
        horizontal, vertical, diagonal, antidiagonal, right and left polarizers"""
        name = proj + 'PROJ'
        self._apply_jones(J_rot(eval(name), angle))
        return self
    




    
    #Propagation 
    def propagate(self, z, method='fresnel', renorm=False):           
        #propagate the field by a distance z
        if method == 'fresnel':
            self = propagate_fresnel(self, z)
        elif method == 'fraunhofer':
            self = propagate_fraunhofer(self, z)
        elif method == 'incoherent':
            self = propagate_incoherent(self, z)
        else:
            print('Unable to propagate, insert valid method.')
        if renorm == True:
            self.norm_beam()
        return self
    

    #Holograms

    def slm_holo(self, x_grating:int, y_grating:int, method:str = 'bessel1', input_beam:object= None, 
                 eps:float = 1e-12, max_range:int = 255)-> np.ndarray:
        #generates hologram for slm
        return slm_hologram(self, x_grating, y_grating, method = method, input_beam = input_beam, eps = eps, max_range = max_range)
    
    def dmd_holo(self, cx:float, cy:float, sign:int=1)-> np.ndarray:
        #generates hologram for dmd
        return dmd_hologram(self, cx, cy, sign)