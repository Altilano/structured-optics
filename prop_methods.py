import numpy as np
from scipy import fft


def propagate_conv(space, z):                                     
    #propagate the field by a distance z using fresnel integral. convolution method
    k = 2*np.pi/space.lamb
    prop = np.exp(-1j * z * (space.kx ** 2 + space.ky ** 2) / (2 * k) - 1j*k*z)
    space.fourier_field = fft.fft2(space.field)
    space.field = fft.ifft2(prop*space.fourier_field)
    return space

