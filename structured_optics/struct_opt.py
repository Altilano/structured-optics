

import numpy as np
from scipy import fft, ndimage
from structured_optics.prop_methods import *
import copy
from structured_optics.utils import *
from structured_optics.modes import *
from structured_optics.algebra_utils import *
from structured_optics.hologram import *



class Beam():
    def __init__(self, nix: float,        #Region of interest in x (from -nix to +nix)
                 Dx: int,                 #Number of points in x
                 niy: float=None,         #Region of interest in y (from -niy to +niy), equals nix if None
                 Dy: int=None,            #Number of points in y, equals Dx if None
                 sparse:bool =True,       #Whether to use sparse meshgrid for x and y
                 pol_dim:int=1,           #Number of polarization components, Ex, Ey, Ez
                 waist:float=1e-3,        #Beam waist. Standard value 1mm
                 lamb:float = 1064e-9,    #Wavelength. Standard value 1064nm
                 x0:float = 0,            #Beam center x position
                 y0:float = 0) -> object: #Beam center y position
        # Initiate an object with the necessary parameters for calculating transverse fields

        self.sparse = sparse
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
        self.pol_dim = pol_dim
        self.x, self.y = np.meshgrid(np.linspace(-self.nix, self.nix, self.Dx), np.linspace(-self.niy, self.niy, self.Dy), sparse=sparse)
        self.field = np.zeros((pol_dim, self.Dy, self.Dx), dtype='complex128')
        self.kx, self.ky = np.meshgrid(2*np.pi*fft.fftfreq(self.Dx, 2*self.nix/self.Dx), 2*np.pi*fft.fftfreq(self.Dy, 2*self.niy/self.Dy), sparse=sparse)
        self.lamb = lamb                 
        self.waist = waist          
        self.x0 = x0
        self.y0 = y0




    #Auxialiary 

    def copy(self):                 
        #Perform a deep copy of the Beam object
        return copy.deepcopy(self)
    
    def copy_clean(self):           
        #Perform a deep copy of the Beam object, but with field initialized to zero
        new = self.copy()
        new.field = np.zeros((self.pol_dim, self.Dy, self.Dx), dtype='complex128')
        return new




    #Operator overloading

    def __mul__(self, other):
        #Multiply the transverse fields point by point if multiplied by another field, or globally if multiplied by number.
        if isinstance(other, (type(self))):
            new = self.copy()
            new.field = self.field*other.field
            return new
        if isinstance(other, (int, float, complex)):
            new = self.copy()
            new.field = other*self.field
            return new

        else:
            raise TypeError(f"sorry, don't know how to multiply by {type(other).__name__}")
        
    __rmul__ = __mul__
    
    def __add__(self, other:object):       
        #Add the transverse fields point by point
        if isinstance(other, (Beam, type(self))):
            new = self.copy()
            new.field = other.field + self.field
            return new
        else:
            raise TypeError(f"sorry, don't know how to add by {type(other).__name__}")
        
    __radd__ = __add__
        
    def __sub__(self, other:object):       
        #Subtract the transverse fields point by point
        if isinstance(other, Beam):
            new = self.copy()
            new.field = self.field - other.field
            return new
        else:
            raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")
        
    __rsub__ = __sub__
    



    #modes
    #pol_index: polarization index to store the mode, if None uses all polarizations
    #returns the Beam object with the mode stored in field
    def hg(self, n:int, m:int, z:float=0, pol_index:int=None) -> object:      
        #get a HG mode at distance z of order n+m
        return hg_mode(self, n, m, z, pol_index)                    
    
    def lg(self,l:int,p:int, z:float=0, pol_index:int=None) -> object:        
        #get a LG mode at distance z of order abs(N) + 2M
        return lg_mode(self, l, p, z, pol_index)
    
    def bessel(self, N:int, z:float=0, pol_index:int=None) -> object:         
        #get a Bessel mode of order N at distance z
        return bessel_mode(self, N, z, pol_index)
    
    def gbessel(self, N:int, r0:int, pol_index:int=None) -> object:           
        #get a gaussian bessel beam of order N at z=0, r0 is the radius of the first intensity null
        return gbessel_mode(self, N, r0, pol_index)
    
    def lg_prod(self, N:int, ls:tuple=None, centers:tuple=None, pol_index:int=None) -> object:     
        #get a product superposition of N LG modes, with list of OAMs ls and list of centers
        return lg_prod_mode(self, N, ls, centers, pol_index)
    
    def frac_oam(self, Ma:float, n_modes:int, beta:float = 0, theta_0:float=0, z:float = 0, pol_index=None) -> object:        
        #get a fractional OAM beam, with OAM Ma (!= integer).
        return frac_oam_mode(self, Ma, n_modes, beta, theta_0, z, pol_index)
    
    def frac_oam_qs(self, Ma:float, n_modes:int, beta:float = 0, theta_0:float=0, z:float = 0, pol_index=None) -> object:        
        #get a fractional OAM quasi_stable beam, with OAM Ma (!= integer).
        #Quasi-stability is achieved by engeniring p to set the order to one of only 2 values.
        return frac_oam_qs_mode(self, Ma, n_modes, beta, theta_0, z, pol_index)
    
    def IG_even(self, p:int, m:int, q:float, z:float=0, pol_index:int=None) -> object:    
        #get an even Ince-Gaussian beam IG_p,m^e at distance z with ellipticity q
        return IG_even_mode(self, p, m, q, z, pol_index)
    
    def IG_odd(self, p:int, m:int, q:float, z:float=0, pol_index:int=None) -> object:     
        #get an odd Ince-Gaussian beam IG_p,m^o at distance z with ellipticity q
        return IG_odd_mode(self, p, m, q, z, pol_index)
                        
    def HelIG(self, p:int, m:int, q:float, z:float=0, helicity:int=1, pol_index:int=None) -> object:     
        #get a Hermite-Ince-Gaussian beam HIG_p,m at distance z with ellipticity q and given helicity (+1 or -1)
        return HInceG_mode(self, p, m, q, z, helicity, pol_index)
    
    #Holograms put dmd and slm hologram here



    #Linear Algebra with modes utils

    def hg_projector(self,N:int, completeness:bool = False) -> tuple:
        #Project the beam into HG basis up to order N
        #returns tuple with n, m index and overlaps array, if completeness=True also returns completeness value
        return hg_proj(self, N, completeness)

    def lg_projector(self, N:int, completeness:bool = False) -> tuple:
        #Project the beam into LG basis up to order N
        #returns tuple with l, p index and overlaps array, if completeness=True also returns completeness value
        return lg_proj(self, N, completeness)
        
    def hg_basis(self, N:int, waist:float=None, norm1:bool = False) -> np.ndarray:
        #Create a HG basis up to order N
        return hg_basis(self, N, waist, norm1)
    
    def lg_basis(self, N:int, waist:float=None, norm1:bool=False) -> np.ndarray:
        #Create a LG basis up to order N
        return lg_basis(self, N, waist, norm1)
    
    def bessel_basis(self, Nmax:int, waist:float=None, norm1:bool=False) -> np.ndarray:
        #Create a Bessel basis up to Nmax
        return bessel_basis(self, Nmax, waist, norm1)

    def build_from_coefs_and_basis(self, coefs:np.ndarray, basis:np.ndarray) -> object:
        #Build beam from given coefficients and basis
        return build_from_coefs_and_basis(self, coefs, basis)
    
    def mode_converter(self, N:int, theta:float=np.pi/4) -> object: 
        #Astigmatic mode converter that converts HG to LG modes and viceversa
        return astigmatic_mode_converter(self, N, theta)
    



    #Beam physical atributes and its utilities

    def Power(self, pol_index:int=None) -> float:
        #Get the total Power of a field within the region of interest
        if pol_index == None:
            return np.sum(self.int_profile())*(4*self.nix/self.Dx)*(self.niy/self.Dy)
        else:
            return np.sum(self.int_profile(pol_index))*(4*self.nix/self.Dx)*(self.niy/self.Dy)
    
    def zr(self) -> float:
        #Get the Rayleigh range of the beam
        return np.pi*self.waist**2/self.lamb
    
    def int_profile(self, pol_index:int=None) -> np.ndarray:
        #Get intensity profile of the field within Beam
        I = np.abs(self.field)**2
        if pol_index == None:
            return I
        else:
            return I[pol_index]
    
    def phase(self, twopi:bool=False) -> np.ndarray:
        #Get phase profile of the field. If twopi=True, phase is given in [0, 2pi], else in [-pi, pi]
        if twopi == False:
            return np.angle(self.field)
        if twopi == True:
            p = np.angle(self.field)
            neg = p<0
            p = p + neg*2*np.pi
            return p
    
    def center_mass(self, pol_index:int=0) -> tuple:
        #Calculates the center of mass of intensities of a given field in given polarization
        #pol_index: polarization index to use for the calculation, default 0
        if self.pol_dim >1:
            c = ndimage.center_of_mass(self.int_profile(pol_index))
        else:
            c = ndimage.center_of_mass(self.int_profile())
        return c
  
    def std(self, pol_index:int=0) -> float:
        #Calculate the standard deviation of intensities
        #pol_index: polarization index to use for the calculation, default 0
        if self.pol_dim >1:
            c = self.center_mass(pol_index)
            return np.sqrt(np.average((self.x-self.x[0,int(c[1])])**2+(self.y-self.y[int(c[0]),0])**2, weights=self.int_profile(pol_index)))
        else:
            c = self.center_mass()
            return np.sqrt(np.average((self.x-self.x[0,int(c[1])])**2+(self.y-self.y[int(c[0]),0])**2, weights=self.int_profile()))
    
    def section(self, ang_min:float, ang_max:float) -> np.ndarray:
        #Return field distribution of given section, defined by minimum angle and maximum angle
        return get_section(self, ang_min, ang_max)
    
    def int_section(self, ang_min:float, ang_max:float)-> np.ndarray:
        #Get intensity profile of given section, defined by minimum angle and maximum angle
        return np.abs(self.section(ang_min, ang_max))**2
    
    def Power_section(self, ang_min:float, ang_max:float)-> float:
        #Get power of given section
        return np.sum(np.abs(self.section(ang_min, ang_max))**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    def norm_beam(self)-> object:                                       
        #normalize beam so that Total Power = 1 in region of interest
        self.field = self.field/np.sqrt(self.Power())
        return self
    
    def Max_int1(self)-> object:                                         
        #rescale the field such that the maximum Intensity point is equal to one.
        self.field = self.field/np.sqrt(np.amax(self.int_profile()))
        return self
    
    def crop(self, center:tuple=None, std:float=None, window:float=2, pol_index:int=0)-> object:
        #Crops a field by its std*window arround the center of mass
        #Here pol_index is the field taken as reference for calcutale std and center
        return get_crop(self, center, std, window, pol_index)
    







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
    
    def tilted_lens(self, f:float, phi:float, f0:tuple=(0,0))-> object:                      
        #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi.
        fx = f*np.cos(phi)**3
        fy = f*np.cos(phi)
        return self.astigmatic_lens(fx,fy, f0)
    
    def tilted_lens_y(self, f:float, phi:float, f0:tuple=(0,0))-> object:                      
        #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi, this time the tilt is in y direction.
        fx = f*np.cos(phi)
        fy = f*np.cos(phi)**3
        return self.astigmatic_lens(fx,fy, f0)

    
    def stripe_v(self, size, move=0):
        """Def vertical stripe"""
        d = move*self.Dx/(2*self.nix)
        s = size*self.Dx/(2*self.nix)
        self.field[:,int(self.Dx/2 - s + d):int(self.Dx/2+s+d)] = 0
        return self
    
    def stripe_h(self, size):
        """Def horizontal stripe"""
        s = size*self.Dy/(2*self.niy)
        self.field[int(self.Dy/2 - s):int(self.Dy/2+s),:] = 0
        return self
    
    def stripe_cross(self, size, hsize=None):
        """Def cross stripe"""
        if hsize == None:
            hsize = size
        s = size*self.Dx/(2*self.nix)
        sh = hsize*self.Dy/(2*self.niy)
        self.field[int(self.Dy/2 - sh):int(self.Dy/2+sh),:] = 0
        self.field[:,int(self.Dx/2 - s):int(self.Dx/2+s)] = 0
        return self

    def slit(self, size):
        """Def 1 slit"""
        s = size*self.Dx/(2*self.nix)
        self.field[:,:int(self.Dx/2-s)] = 0
        self.field[:,int(self.Dx/2 + s):] = 0
        return self
    
    def double_slit(self, size, dis):
        """Def double slit"""
        s = size*self.Dx/(2*self.nix)
        d = dis*self.Dx/(2*self.nix)
        self.field[:,0:int((self.Dx-d-s)/2)] = 0
        self.field[:,int((self.Dx-d+s)/2):int((self.Dx+d-s)/2)] = 0
        self.field[:,int((self.Dx+d+s)/2):] = 0
        return self
    

    
    

    #Propagation 
    def propagate(self, z, method='conv', renorm=False):           #propagate the field by a distance z using fresnel integral with exp(-ikz)
        if method == 'conv':
            self = propagate_conv(self, z)
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