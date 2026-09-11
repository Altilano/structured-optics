import numpy as np
from scipy import fft


def propagate_fresnel(Beam, z):                                     
    #Propagate the field by a distance z using fresnel integral. convolution method
    k = 2*np.pi/Beam.lamb
    prop = np.exp(-1j * z * (Beam.kx ** 2 + Beam.ky ** 2) / (2 * k) + 1j*k*z)
    Beam.fourier_field = fft.fft2(Beam.field, axes=(-2,-1))
    Beam.field = fft.ifft2(prop*Beam.fourier_field, axes=(-2,-1))
    return Beam

def propagate_AS(Beam, z, evanescent=False):
    #Propagate the field by a distance z using Angular Spectrum method.
    k = 2*np.pi/Beam.lamb
    kz2 = k**2 - Beam.kx**2 - Beam.ky**2
    kz = np.sqrt(kz2.astype(complex))
    if not evanescent:
        not_evanescent = kz2 >= 0
        kz = kz*not_evanescent
    prop = np.exp(1j*kz*z + 1j*k*z)
    Beam.fourier_field = fft.fft2(Beam.field, axes=(-2,-1))
    Beam.field = fft.ifft2(prop*Beam.fourier_field, axes=(-2,-1))
    return Beam


def propagate_incoherent(Beam, z):
    #Propagates the INTENSITY by a distance z using incoherent propagation. Loses phase information.
    #without pupil function doesn't seems to work
    k = 2*np.pi/Beam.lamb
    if z != 0:
        h2 = np.ones_like(Beam.field)/(Beam.lamb*z)**2
    else:
        h2 = np.ones_like(Beam.field)
    h2_fourier = fft.fft2(h2)
    I_fourier = fft.fft2(Beam.int_profile())
    Beam.field = np.sqrt(fft.ifft2(h2_fourier*I_fourier))
    return Beam
    
    
def propagate_fraunhofer(Beam, z):
    #Propagate the field by a distance z using fraunhofer integral. Changes XY grid.

    if z < 2/Beam.lamb*Beam.waist**2:
        print("z small. Fraunhofer is probably not a good aproximation.") 
    if z != 0:
        k = 2*np.pi/Beam.lamb
        dx = Beam.x[0,1] - Beam.x[0,0]
        dy = Beam.y[1,0] - Beam.y[0,0]
        fx = np.fft.fftshift(np.fft.fftfreq(Beam.Dx, d=dx))
        fy = np.fft.fftshift(np.fft.fftfreq(Beam.Dy, d=dy))
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        Beam.x = Beam.lamb * z * FX
        Beam.y = Beam.lamb * z * FY
        Beam.nix = np.max(Beam.x)
        Beam.niy = np.max(Beam.y)
        C = np.exp(1j*k*z)/(1j*Beam.lamb*z)*np.exp(1j*k*(Beam.x**2 + Beam.y**2)/(2*z))
        Beam.fourier_field = fft.fft2(Beam.field, axes=(-2,-1))
        Beam.field = C*fft.fftshift(Beam.fourier_field)*dx*dy
    return Beam








def max_propagation_distance(Nx, Ny, dx, dy, wavelength, n=1.0):
    """
    Rough estimate of the maximum propagation distance for which the
    angular spectrum transfer function is adequately sampled, based on
    the Nyquist criterion applied to the quadratic phase of H.
 
    Returns (z_max_x, z_max_y): safe distances in x and y directions.
    """
    Lx = Nx * dx
    Ly = Ny * dy
    lam = wavelength / n
    z_max_x = Lx * dx / lam
    z_max_y = Ly * dy / lam
    return z_max_x, z_max_y
