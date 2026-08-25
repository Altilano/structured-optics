

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
    def __init__(self, nix: float,        #Region of interest in x (from -nix to +nix)
                 Dx: int,                 #Number of points in x
                 niy: float=None,         #Region of interest in y (from -niy to +niy), equals nix if None
                 Dy: int=None,            #Number of points in y, equals Dx if None
                 waist:float=1e-3,        #Beam waist. Standard value 1mm
                 lamb:float = 1064e-9,    #Wavelength. Standard value 1064nm
                 x0:float = 0,            #Beam center x position
                 y0:float = 0,
                 pol_dim = 1) -> object: #Beam center y position
        # Initiate an object with the necessary parameters for calculating transverse fields

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
        self.x, self.y = np.meshgrid(np.linspace(-self.nix, self.nix, self.Dx), np.linspace(-self.niy, self.niy, self.Dy))
        self.field = np.zeros((pol_dim, self.Dy, self.Dx), dtype='complex128')
        self.kx, self.ky = np.meshgrid(2*np.pi*fft.fftfreq(self.Dx, 2*self.nix/self.Dx), 2*np.pi*fft.fftfreq(self.Dy, 2*self.niy/self.Dy))
        self.x0 = x0
        self.y0 = y0
        

        #Physical properties
        self.lamb = lamb                 
        self.waist = waist   
        self.pol = pol_dim     

        
    @property
    def Ex(self):
        return self.field[0]
    
    @property
    def Ey(self):
        return self.field[1]

    @property
    def Ez(self):
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
        #Perform a deep copy of the Beam object
        return copy.deepcopy(self)
    
    def copy_clean(self, pol_dim = None):           
        #Perform a deep copy of the Beam object, but with field initialized to zero
        new = self.copy()
        if pol_dim == None:
            pol_dim = self.pol
        new.field = np.zeros((pol_dim, self.Dy, self.Dx), dtype='complex128')
        return new




    #Operator overloading

    def __mul__(self, other):
        #Multiply the transverse fields point by point if multiplied by another field, or globally if multiplied by number.
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
        #Add the transverse fields point by point
        if isinstance(other, (Beam, type(self))):
            new = self.copy()
            new.field = other.field + self.field
            return new
        raise TypeError(f"sorry, don't know how to add by {type(other).__name__}")
        
    __radd__ = __add__
        
    def __sub__(self, other:object):       
        #Subtract the transverse fields point by point
        if isinstance(other, Beam):
            new = self.copy()
            new.field = self.field - other.field
            return new
        raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")
        
    def __rsub__(self, other:object):       
        #Subtract the transverse fields point by point
        if isinstance(other, Beam):
            new = self.copy()
            new.field = other.field - self.field
            return new
        raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")

    def __truediv__(self, other):

        if isinstance(other, (int, float, complex)):
            new = self.copy()
            new.field = self.field / other
            return new

        raise TypeError(f"Cannot divide Beam by {type(other).__name__}")
    



    #modes

    @contextmanager
    def rotated_grid(self, angle):
        # Save original grid
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
            # Always restore original grid
            self.x = old_x
            self.y = old_y

    def _set_mode(self, mode, polarization=None):
        """Calculate/distribute a mode to the polarization components."""

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
        Get an HG mode at distance z.
        """
        return self._set_mode(hg(self, n, m, z, angle=angle),  polarization=polarization)


    def hg_astigmatic(self, n: int, m: int, wx: float, wy: float, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get an astigmatic HG mode at distance z.
        """
        return self._set_mode(hg_astigmatic(self, n, m, wx, wy, z, angle=angle),  polarization=polarization)


    def lg(self, l: int, p: int, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get an LG mode at distance z.
        """
        return self._set_mode(lg(self, l, p, z, angle=angle),  polarization=polarization)


    def bessel(self, N: int, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get a Bessel mode of order N at distance z.
        """
        return self._set_mode(nbessel(self, N, z, angle=angle),  polarization=polarization)


    def gbessel(self, N: int, r0: int, angle:float=0, polarization:list=None) -> object:
        """
        Get a Gaussian-Bessel beam of order N at z=0.
        r0 is the radius of the first intensity null.
        """
        return self._set_mode(gbessel(self,N,r0, angle=angle),  polarization=polarization)


    def lg_prod(self, N: int, ls: tuple = None, centers: tuple = None, angle:float=0, polarization:list=None) -> object:
        """
        Get a product/superposition of N LG modes.
        """
        return self._set_mode(lg_prod(self, N, ls, angle=angle),  polarization=polarization)


    def frac_oam(self, Ma: float, n_modes: int, beta: float = 0, theta_0: float = 0, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get a fractional OAM beam.
        """
        return self._set_mode(frac_oam(self, Ma, n_modes, beta, theta_0, z, angle=angle),  polarization=polarization)


    def frac_oam_qs(self, Ma: float, n_modes: int, beta: float = 0, theta_0: float = 0, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get a fractional OAM quasi-stable beam.
        """
        return self._set_mode(frac_oam_qs(self, Ma, n_modes, beta, theta_0, z, angle=angle),  polarization=polarization)


    def IG_even(self, p: int, m: int, q: float, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get an even Ince-Gaussian beam.
        """
        return self._set_mode(IG_even(self, p, m, q, z, angle=angle),  polarization=polarization)


    def IG_odd(self, p: int, m: int, q: float, z: float = 0, angle:float=0, polarization:list=None) -> object:
        """
        Get an odd Ince-Gaussian beam.
        """
        return self._set_mode(IG_odd(self, p, m, q, z, angle=angle),  polarization=polarization)


    def HelIG(self, p: int, m: int, q: float, z: float = 0, helicity: int = 1, angle:float=0, polarization:list=None) -> object:
        """
        Get a Ince-Gaussian beam with given helicity.
        """
        return self._set_mode(HInceG(self, p, m, q, helicity=helicity, z=z, angle=angle),  polarization=polarization)


    def circle(self, center: tuple = (0, 0), radius: float = None, angle:float=0, polarization:list=None) -> object:
        """
        Get a circle mode.
        """
        return self._set_mode(circle(self, center, radius, angle=angle),  polarization=polarization)


    def square(self, center: tuple = (0, 0), side_length: float = None, angle:float=0, polarization:list=None) -> object:
        """
        Get a square mode.
        """
        return self._set_mode(square(self, center, side_length, angle=angle),  polarization=polarization)


    def triangle(self, center: tuple = (0, 0), side_length: float = None, angle:float=0, polarization:list=None) -> object:
        """
        Get a triangle mode.
        """
        return self._set_mode(triangle(self, center, side_length, angle=angle),  polarization=polarization)


    def lp(self, l: int, m: int, n_core:float, n_clad:float , parity:str = "cos", angle:float=0, polarization:list=None) -> object:
        """
        Get an LP fiber mode. cos or sin, with l>=0 and m>=1
        """
        return self._set_mode(lp(self, l, m, n_core, n_clad, parity=parity, angle=angle),  polarization=polarization)


    def lp_hel(self, l:int, m:int, n_core:float, n_clad:float, angle:float=0, polarization:list=None) -> object:
        """
        Get a helical LP fiber mode. With positive and negative l, and m>=1.
        """
        return self._set_mode(lp_hel(self, l, m, n_core, n_clad, angle=angle),  polarization=polarization)
    





    #Linear Algebra with modes utils

    def hg_projector(self,N:int) -> tuple:
        #Project the beam into HG basis up to order N
        #returns tuple with n, m index and overlaps array
        return hg_proj(self, N)

    def lg_projector(self, N:int) -> tuple:
        #Project the beam into LG basis up to order N
        #returns tuple with l, p index and overlaps array
        return lg_proj(self, N)
        
    def hg_basis(self, N:int, waist:float=None, norm1:bool = False) -> np.ndarray:
        #Create a HG basis up to order N
        return hg_basis(self, N, waist, norm1)
    
    def lg_basis(self, N:int, waist:float=None, norm1:bool=False) -> np.ndarray:
        #Create a LG basis up to order N
        return lg_basis(self, N, waist, norm1)
    
    def bessel_basis(self, Nmax:int, waist:float=None, norm1:bool=False) -> np.ndarray:
        #Create a Bessel basis up to Nmax
        return bessel_basis(self, Nmax, waist, norm1)

    def build_from_coefs_and_basis(self, coefs:np.ndarray, basis:np.ndarray, pol_index:int = 0) -> object:
        #Build beam from given coefficients and basis
        return build_from_coefs_and_basis(self, coefs, basis, pol_index = pol_index)
    
    



    #Beam physical atributes and its utilities

    def Power(self) -> float:
        """Get the total Power of a field within the region of interest"""
        return np.sum(self.int_profile())*(4*self.nix/self.Dx)*(self.niy/self.Dy)

    
    def zr(self) -> float:
        #Get the Rayleigh range of the beam
        return np.pi*self.waist**2/self.lamb
    
    def int_profile(self, pol_index=None) -> np.ndarray:
        """
        Get intensity profile.

        pol_index=None -> total intensity
        pol_index=0    -> Ex intensity
        pol_index=1    -> Ey intensity
        pol_index=2    -> Ez intensity
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
        #Calculates the center of mass of intensities of a given field in given polarization
        return ndimage.center_of_mass(self.int_profile(pol_index))
  
    def std(self, pol_index=None) -> float:
        """Calculate intensity-weighted radial standard deviation."""
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

    def polarizer(self, proj='H'):
        """Polarizer projector. Use proj = 'H', 'V', 'D', 'A', 'R', 'L' for 
        horizontal, vertical, diagonal, antidiagonal, right and left polarizers"""
        name = proj + 'PROJ'
        self._apply_jones(eval(name))
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