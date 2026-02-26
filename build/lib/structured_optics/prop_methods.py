import numpy as np
from scipy import fft


def propagate_fresnel(Beam, z):                                     
    #Propagate the field by a distance z using fresnel integral. convolution method
    k = 2*np.pi/Beam.lamb
    prop = np.exp(-1j * z * (Beam.kx ** 2 + Beam.ky ** 2) / (2 * k) + 1j*k*z)
    Beam.fourier_field = fft.fft2(Beam.field)
    Beam.field = fft.ifft2(prop*Beam.fourier_field)
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
        Beam.fourier_field = fft.fft2(Beam.field)
        Beam.field = C*fft.fftshift(Beam.fourier_field)*dx*dy
    return Beam
