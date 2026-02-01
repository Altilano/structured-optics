import numpy as np
from scipy import fft, special, ndimage
from scipy.special import eval_hermite, factorial
from prop_methods import *
import copy
from utils import *
from modes import *



class Beam():
    def __init__(self, nix: float, 
                 Dx: int, 
                 niy: float=None, 
                 Dy: int=None, 
                 sparse:bool =True, 
                 pol_dim:int=1, 
                 waist:float=1e-3,
                 lamb:float = 1064e-9,
                 x0:float = 0,
                 y0:float = 0) -> object:
        """ Initiate an object with the necessary parameters for calculating transverse fields
        """
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
        self.lamb = lamb        #standard 1064 nanometer         
        self.waist = waist          #standard 1 milimeter
        self.x0 = x0
        self.y0 = y0

    def copy(self):
        return copy.deepcopy(self)
    
    def copy_clean(self):
        new = self.copy()
        new.field = np.zeros((self.pol_dim, self.Dy, self.Dx), dtype='complex128')
        return new

    def __mul__(self, other):
        """Multiply the transverse fields point by point if multiplied by another field, or globally if multiplied by number.
        Changes lamb by 1/lamb_new = 1/lamb_1 + 1/lamb_2. Doesn't change waist.
        
        """
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
    
    def __add__(self, other):
        if isinstance(other, (Beam, type(self))):
            new = self.copy()
            new.field = other.field + self.field
            return new
        else:
            raise TypeError(f"sorry, don't know how to add by {type(other).__name__}")
        
    __radd__ = __add__
        
    def __sub__(self, other):
        if isinstance(other, Beam):
            new = self.copy()
            new.field = self.field - other.field
            return new
        else:
            raise TypeError(f"sorry, don't know how to subtract by {type(other).__name__}")
        
    __rsub__ = __sub__

    def zr(self):
        return np.pi*self.waist**2/self.lamb
        
    def hg(self, N:int, M:int, z:float=0, pol_index:int=None):                                                        #get a HG mode at distance z of order N+M
        return hg_mode(self, N, M, z, pol_index)
    
    def lg(self,l:int,p:int, z:float=0, pol_index:int=None):                                    #get a LG mode at distance z of order abs(N) + 2M
        return lg_mode(self, l, p, z, pol_index)
    
    def bessel(self, N:int, z:float=0, norm:bool=False, pol_index:int=None):                       #get a Bessel mode of order N at distance z
        return bessel_mode(self, N, z, norm, pol_index)
    
    def gbessel(self, N:int, r0:int, pol_index:int=None):                                    #get a gaussian bessel beam of order N at z=0, M is the asymptotic radial period
        return gbessel_mode(self, N, r0, pol_index)
    
    def lg_prod(self, N:int, ls:tuple=None, centers:tuple=None, pol_index:int=None):                                   #get a superposition of N LG modes, with list of OAMs ls and list of centers
        return lg_prod_mode(self, N, ls, centers, pol_index)
    
    def frac_oam(self, Ma:float, n_modes:int, beta:float = 0, theta_0:float=0, z:float = 0, pol_index=None):        #get a fractional OAM beam, with OAM Ma (!= integer), by the method of LG supperpositions.
        return frac_oam_mode(self, Ma, n_modes, beta, theta_0, z, pol_index)
    
    def IG_even(self, p:int, m:int, q:float, z:float=0, pol_index:int=None):    #get an even Ince-Gaussian beam IG_p,m^e at distance z
        return IG_even_mode(self, p, m, q, z, pol_index)
    
    def IG_odd(self, p:int, m:int, q:float, z:float=0, pol_index:int=None):     #get an odd Ince-Gaussian beam IG_p,m^o at distance z
        return IG_odd_mode(self, p, m, q, z, pol_index)
                        
    def HelIG(self, p:int, m:int, q:float, z:float=0, helicity:int=1, pol_index:int=None):     #get a Hermite-Ince-Gaussian beam HIG_p,m^e at distance z
        return HInceG(self, p, m, q, z, helicity, pol_index)
    

    def hg_projector(self,N, completeness = False):
        overlaps = np.zeros((N+1, N+1), dtype='complex')
        auxiliary_beam  = self.copy_clean()
        for n in range(N+1):
            for m in range(N+1):
                auxiliary_beam.hg(n,m)
                overlaps[n,m] = overlap(self, auxiliary_beam)
        n, m =  np.meshgrid(np.arange(0, N+1, 1), np.arange(0, N+1, 1))
        comp = np.sum(np.abs(overlaps)**2)
        if completeness == True:
            return n, m, overlaps, comp
        else:
            return n, m, overlaps
    
    def hg_projector_adaptive(self, threshold=0.95, Nmax=50):
        """
        Adaptive HG expansion such that:
            - We only keep terms where n + m <= N (triangular basis)
            - All coefficients with n + m > N are forced to zero
            - N increases until the captured power exceeds the threshold.
        Returns:
            n_grid, m_grid, overlaps_matrix, N, cumulative_power
        """

        # To store coefficients, we use a full square matrix but force zeros outside triangle
        overlaps = np.zeros((1, 1), dtype=complex)

        cumulative_power = 0.0
        N = -1

        # Prepare auxiliary beam
        auxiliary_beam = self.copy_clean()

        while cumulative_power < threshold:
            N += 1
            if N > Nmax:
                print(f"Warning: maximum N={Nmax} reached before threshold.")
                break

            # Expand coefficient matrix to size (N+1)x(N+1)
            new_overlaps = np.zeros((N+1, N+1), dtype=complex)
            old_N = overlaps.shape[0]
            new_overlaps[:old_N, :old_N] = overlaps
            overlaps = new_overlaps

            # Compute new coefficients only in the new triangular band n+m = N
            for n in range(N+1):
                m = N - n
                if m < 0 or m > N:
                    continue

                auxiliary_beam.hg(n, m)
                overlaps[n, m] = overlap(self, auxiliary_beam)

            # Enforce triangular condition: zero out entries with n+m > N
            for n in range(N+1):
                for m in range(N+1):
                    if n + m > N:
                        overlaps[n, m] = 0.0

            # Compute cumulative power in triangular region
            cumulative_power = np.sum(np.abs(overlaps)**2)

        # Prepare meshgrid for return
        n_idx, m_idx = np.meshgrid(np.arange(N+1), np.arange(N+1), indexing='ij')

        return n_idx, m_idx, overlaps, N, cumulative_power
    
    def _precompute_1d_hg(self, Nmax, z=0):
        """
        Precompute 1D HG functions along x and y for orders 0..Nmax (vectorized).
        Returns:
            un: shape (Nmax+1, Dx)   (functions along x for orders n)
            um: shape (Nmax+1, Dy)   (functions along y for orders m)
        """
        Dx = self.Dx
        Dy = self.Dy
        nix = self.nix
        niy = self.niy

        # extract 1D coordinates assuming meshgrid(sparse=True)
        # self.x shape is (Dy, Dx); take first row; self.y shape is (Dy, Dx); take first col
        x1 = self.x[0, :]    # shape (Dx,)
        y1 = self.y[:, 0]    # shape (Dy,)

        # optical parameters at distance z
        zr = self.zr()
        q0 = 1j * zr
        q = -z + 1j * zr
        k = 2 * np.pi / self.lamb
        w = self.waist * np.sqrt(1 + (z / zr)**2)

        # prepare arrays of orders
        ns = np.arange(Nmax + 1)
        # factorials and phase factors vectorized
        fac = factorial(ns)
        two_pow = 2.0 ** ns
        # (-conj(q)/q)^(n/2) -> same scalar for each n but different power
        phase_base = -np.conjugate(q) / q
        # normalization constant factor (vectorized)
        Cn_vec = np.sqrt(np.sqrt(2.0 / np.pi) * q0 / (two_pow * fac * q * self.waist))

        # prepare evaluation points for Hermite polynomials
        X = np.sqrt(2.0) * (x1 - self.x0) / w    # shape (Dx,)
        Y = np.sqrt(2.0) * (y1 - self.y0) / w    # shape (Dy,)

        # Evaluate hermite polynomials for all orders at X and Y
        # eval_hermite(n, x) accepts vector x, so we vectorize across n by comprehension but still efficient
        # We'll build arrays un and um of shapes (Nmax+1, Dx) and (Nmax+1, Dy)
        un = np.empty((Nmax + 1, Dx), dtype=np.complex128)
        um = np.empty((Nmax + 1, Dy), dtype=np.complex128)

        # Vectorize eval_hermite across orders by broadcasting: use list comprehension (fast enough)
        for n in ns:
            Hn_x = eval_hermite(n, X)      # shape (Dx,)
            Hn_y = eval_hermite(n, Y)      # shape (Dy,)

            pref = Cn_vec[n] * (phase_base ** (n / 2.0))
            # longitudinal quadratic phase term (same shape for all orders)
            exp_x = np.exp(-1j * k * (x1 - self.x0)**2 / (2.0 * q))
            exp_y = np.exp(-1j * k * (y1 - self.y0)**2 / (2.0 * q))

            un[n, :] = pref * Hn_x * exp_x
            um[n, :] = pref * Hn_y * exp_y

        return un, um


    def hg_projector_vectorized(self, threshold=0.95, Nmax=80, z=0):
        """
        Vectorized triangular HG projector:
        - Only modes with n+m <= N are kept (triangular)
        - Adds triangular bands n+m = 0,1,2,... until captured power >= threshold
        - All coefficients with n+m > N are zeroed
        Returns:
        n_idx, m_idx, overlaps_matrix, N_reached, cumulative_power
        overlaps_matrix is shape (N_reached+1, N_reached+1) but allocated up to Nmax+1 internally.
        """
        # area element used in your get_Power/overlap (kept consistent)
        area_elem = (4.0 * self.nix / self.Dx) * (self.niy / self.Dy)

        # total power of the input beam (normalized per your get_Power)
        P_input = self.get_Power()

        # precompute 1D HG functions up to Nmax
        un, um = self._precompute_1d_hg(Nmax, z=z)   # un: (Nmax+1, Dx), um: (Nmax+1, Dy)

        # prepare overlaps matrix (square up to Nmax+1)
        overlaps = np.zeros((Nmax + 1, Nmax + 1), dtype=np.complex128)

        cumulative_power = 0.0
        N = -1

        # For efficient band processing we will compute bands where n+m == N
        # We will vectorize computation of all modes in the band at once.
        while cumulative_power < threshold and N < Nmax:
            N += 1

            # collect all pairs (n,m) with n+m == N and 0 <= n,m <= Nmax
            ns = np.arange(0, N+1, dtype=int)
            ms = N - ns   # array same length
            # number of modes in this band
            K = ns.size

            # build batch of modes for this band using outer product between um[ms] and un[ns]
            # um[ms] has shape (K, Dy), un[ns] has shape (K, Dx)
            # We want modes shape (K, Dy, Dx) = um[ms][:, :, None] * un[ns][:, None, :]
            um_batch = um[ms, :]            # shape (K, Dy)
            un_batch = un[ns, :]            # shape (K, Dx)
            modes = um_batch[:, :, None] * un_batch[:, None, :]   # (K, Dy, Dx)

            # compute overlap for each mode: integral of input_field * conj(mode)
            # vectorized: overlaps_vec = sum_{y,x} (input.field * conj(modes[k])) * area_elem
            # shapes: modes.conj() (K, Dy, Dx), self.field (Dy, Dx)
            # compute elementwise product and sum over (1,2)
            prod = modes.conj() * self.field[None, :, :]    # (K, Dy, Dx)
            overlaps_vec = np.sum(prod, axis=(1, 2)) * area_elem   # (K,)

            # compute each mode power (for normalization if desired)
            mode_powers = np.sum(np.abs(modes)**2, axis=(1, 2)) * area_elem   # (K,)

            # normalize overlaps consistent with your overlap() which divides by sqrt(P_input * P_mode)
            # avoid division by zero
            denom = np.sqrt(P_input * mode_powers)
            small = denom == 0
            denom[small] = 1.0
            overlaps_norm = overlaps_vec / denom

            # place normalized overlaps into overlaps matrix at (n,m)
            overlaps[ns, ms] = overlaps_norm

            # update cumulative_power over triangular region (incremental: add |overlaps_norm|**2 for this band)
            cumulative_power += np.sum(np.abs(overlaps_norm)**2)

            # continue loop until threshold reached or N reaches Nmax

        # finalize triangle: zero coefficients where n+m > N (already zeros by initial matrix)
        overlaps_tri = overlaps[:N+1, :N+1].copy()
        for ii in range(N+1):
            for jj in range(N+1):
                if ii + jj > N:
                    overlaps_tri[ii, jj] = 0.0

        # create index grids (matching your previous return)
        n_idx, m_idx = np.meshgrid(np.arange(N+1), np.arange(N+1), indexing='ij')

        return n_idx, m_idx, overlaps_tri, N, cumulative_power
    
    def lg_projector(self, N, completeness = False):
        overlaps = np.zeros((int(2*N)+1, int(N/2)+1), dtype='complex')
        auxiliary_beam  = Beam(self.nix, self.Dx, self.niy, self.Dy)
        auxiliary_beam.waist = self.waist
        for l in np.arange(-N, N+1, 1):
            for p in range(int(N/2)+1):
                auxiliary_beam.lg(l,p)
                overlaps[l+N,p] = overlap(self, auxiliary_beam)
        p, l = np.meshgrid(np.arange(0, int(N/2)+1, 1), np.arange(-N, N+1, 1))
        comp = np.sum(np.abs(overlaps)**2)
        if completeness == True:
            return l, p, overlaps, comp
        else:
            return l, p, overlaps
        
    def hg_basis(self, N, waist = None, norm1 = False):
        aux = self.copy_clean()
        if waist != None:
            aux.waist = waist
        basis = np.empty((N+1, N+1, self.Dy, self.Dx), dtype = 'complex')
        for n in range(N+1):
            for m in range(N+1):
                aux.hg(n, m)
                basis[n, m] = aux.field
        if norm1 == True:
            basis = basis/np.max(basis)
        return basis
    
    def lg_basis(self, N, waist=None, norm1=False):
        aux = self.copy_clean()
        if waist != None:
            aux.waist = waist
        basis = np.empty((N+1, N+1, self.Dy, self.Dx), dtype = 'complex')
        for n in range(N+1):
            for m in range(N+1):
                l = m-n
                p = min(n,m)
                aux.lg(l, p)
                basis[n, m] = aux.field
        if norm1 == True:
            basis = basis/np.max(basis)
        return basis
    
    def bessel_basis(self, Nmax, waist=None, norm = False, norm1=False):
        aux = self.copy_clean()
        if waist is not None:
            aux.waist = waist
        # basis[n] = Bessel beam of order n
        basis = np.empty((2*Nmax+1, aux.Dy, aux.Dx), dtype='complex')
        for n in range(2*Nmax+1):
            aux.bessel(n-Nmax, norm=norm)   # or aux.bessel(n, theta=...)
            basis[n] = aux.field
        if norm1:
            basis = basis / np.max(np.abs(basis))
        return basis
    
        
    def build_from_coefs(self, coefs):
        auxiliary_beam = Beam(self.nix, self.Dx, self.niy, self.Dy)
        auxiliary_beam.waist = self.waist
        N = len(coefs)
        for i in range(N):
            for j in range(N):
                self.hg(i,j)
                auxiliary_beam = auxiliary_beam + coefs[i,j]*self    
        self.field = auxiliary_beam.field
        return self

    def build_from_coefs_and_basis(self, coefs, basis):
        N = len(coefs)
        aux = self.copy_clean()
        for i in range(N):
            for j in range(N):
                aux.field = aux.field + coefs[i,j] * basis[i,j]
        self.field = aux.field
        return self
    
    def mode_converter(self, N): 
        n, m, overlaps = self.hg_projector(N)
        converter = np.exp(1j*(m-n)*np.pi/4)
        overlaps = overlaps*converter
        self.build_from_coefs(overlaps)
        return self
    
    def mode_converter_vectorized(self, threshold=0.95, Nmax=50):
        n, m, overlaps, N_reached, power = self.hg_projector_vectorized(threshold, Nmax)

        # Hermite-Gaussian phase converter
        converter = np.exp(1j * (m - n) * np.pi / 4)

        overlaps = overlaps * converter

        # Reconstruct beam
        self.build_from_coefs(overlaps)
        return self


    def get_Power(self):
        """Get the total Power of a field within Beam"""
        return np.sum(np.abs(self.field)**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    
    
    def section(self, ang_min, ang_max):
        """Return field distribution of given section, defined by minimum angle and maximum angle"""
        if ang_min > ang_max:
            ang_min, ang_max = ang_max, ang_min
        if ang_max <= np.pi and ang_min <= np.pi:
            sec = (np.arctan2(self.y,self.x)>=ang_min)*(np.arctan2(self.y,self.x)<ang_max)
        if ang_max> np.pi and ang_min <= np.pi:
            sec1 = (np.arctan2(self.y,self.x)>=ang_min)
            ang_max = ang_max-2*np.pi
            sec2 = (np.arctan2(self.y,self.x)<ang_max)
            sec = sec1 + sec2
        if ang_max>np.pi and ang_min > np.pi:
            ang_max = ang_max - 2*np.pi
            ang_min = ang_min - 2*np.pi
            sec = (np.arctan2(self.y,self.x)>=ang_min)*(np.arctan2(self.y,self.x)<ang_max)

        return self.field*sec
    
    def get_Power_section(self, ang_min, ang_max):
        """Get power of given section"""
        return np.sum(np.abs(self.section(ang_min, ang_max))**2*(4*self.nix/self.Dx)*(self.niy/self.Dy))
    

    
    def get_data_nsplit_detector(self, N, rot_angle=0):
        data = []
        d_ang = 2*np.pi/N
        for i in range(int(N/2)):
            pos_measure = self.get_Power_section(i*d_ang + rot_angle, (i+1)*d_ang + rot_angle)
            neg_measure = self.get_Power_section(i*d_ang + np.pi + rot_angle, (i+1)*d_ang + np.pi + rot_angle)
            data.append(pos_measure - neg_measure)
        data = int(N/2)*np.array(data)
        return data
    
    def int_profile(self):
        """ Get intensity profile of the field within Beam"""
        return np.abs(self.field)**2
    
    def phase(self, twopi=False):
        """Get phase profile of the field."""
        if twopi == False:
            return np.angle(self.field)
        if twopi == True:
            p = np.angle(self.field)
            neg = p<0
            p = p + neg*2*np.pi
            return p
    
    def norm_beam(self):                                        #normalize beam so that Total Power = sum(field**2 * Delta(x)*Delta(y)) = 1
        self.field = self.field/np.sqrt(self.get_Power())
        return self
    
    def Max_int1(self):                                         #rescale the field such that the maximum Intensity point is equal to one.
        self.field = self.field/np.sqrt(np.amax(np.abs(self.field)**2))
        return self

    def propagate(self, z, method='conv', renorm=False):           #propagate the field by a distance z using fresnel integral with exp(-ikz)
        if method == 'conv':
            self = propagate_conv(self, z)
        if method == 'dsum':
            self = propagate_dsum(self, z)
        if renorm == True:
            self.norm_beam()
        return self

    def lens(self, f, f0=(0,0)):                                #apply a lens operator to the field, with lens center at f0 and focus lenght equal to f
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(1j*k*(((self.x-f0[0])**2 + (self.y-f0[1])**2)/(2*f)))
        return self

    def astigmatic_lens(self, fx, fy, f0=(0,0)):                      #apply a astigmatic lens with two focal axis, with focus fx and fy.
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(-1j*k*(((self.x-f0[0])**2)/fx + ((self.y-f0[1])**2)/fy)/2)
        return self
    
    def tilted_lens(self, f, phi, f0=(0,0)):                      #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi.
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(-1j*k*(((self.x-f0[0])**2)/np.cos(phi)**2 + ((self.y-f0[1])**2))/(2*f*np.cos(phi)))
        return self
    
    def tilted_lens_y(self, f, phi, f0=(0,0)):                      #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi.
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(-1j*k*(((self.x-f0[0])**2) + ((self.y-f0[1])**2)/np.cos(phi)**2)/(2*f*np.cos(phi)))
        return self
    
    def tilted_lens_wag(self, f, phi, f0=(0,0)):                      #apply a astigmatic lens with two focal axis, with each focus given by a tilt phi.
        k = 2*np.pi/self.lamb
        self.field = self.field*np.exp(-1j*k*(((self.x-f0[0])**2)/np.cos(phi)**2 + ((self.y-f0[1])**2))/(2*f))
        return self
    
    def stripe_v(self, size):
        """Def vertical stripe"""
        s = size*self.Dx/(2*self.nix)
        self.field[:,int(self.Dx/2 - s):int(self.Dx/2+s)] = 0
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
    
    def apply_mask(self, mask):
        """Applies a mask to the field"""
        self.field = self.field*np.exp(-1j*mask.mask)
        return self
    
    def center_mass(self):
        """Calculates the center of mass of intensities of a given field"""
        return ndimage.center_of_mass(self.int_profile())
    
    def std(self):
        """Calculate the standard deviation of intensities"""
        c = self.center_mass()
        return np.sqrt(np.average((self.x-self.x[0,int(c[1])])**2+(self.y-self.y[int(c[0]),0])**2, weights=self.int_profile()))
    
    def crop(self, center=None, std=None, window=2):
        """Crops a field by its std*window arround the center of mass"""
        if center == None:
            center = self.center_mass()
        if std == None:
            std = self.std()
        xmin = int(center[1] - window*std*self.Dx/self.nix/2)
        xmax = int(center[1] + window*std*self.Dx/self.nix/2)
        ymin = int(center[0] - window*std*self.Dy/self.niy/2)
        ymax = int(center[0] + window*std*self.Dy/self.niy/2)
        self.x = self.x[:,xmin:xmax]
        self.y = self.y[ymin:ymax, :]
        self.field = self.field[ymin:ymax, xmin:xmax]
        self.Dx = len(self.x[0,:])
        self.Dy = len(self.y[:,0])
        self.nix = (self.x[0,-1] - self.x[0,0])/2
        self.niy = (self.y[-1,0] - self.y[0,0])/2
        self.x0 = center[1]
        self.y0 = center[0]
        return self

    



def overlap(first_beam, second_beam): # calculate overlap between two beams, only properly works if ni and D of beams are equal.
    return np.sum(first_beam.field*np.conjugate(second_beam.field))*4*(first_beam.nix/first_beam.Dx) \
        *(first_beam.niy/first_beam.Dy)/np.sqrt(first_beam.get_Power()*second_beam.get_Power())

def int_overlap(first_beam, second_beam):
    return np.sum(first_beam.int_profile()*second_beam.int_profile())*4*(first_beam.nix/first_beam.Dx) \
        *(first_beam.niy/first_beam.Dy)/(first_beam.get_Power()*second_beam.get_Power())