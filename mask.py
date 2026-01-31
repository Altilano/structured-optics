import numpy as np
from scipy import fft


class Mask():
    def __init__(self, space):
        self.nix = space.nix
        self.niy = space.niy
        self.Dx = space.Dx
        self.Dy = space.Dy
        self.x = space.x
        self.y = space.y
        self.mask = np.zeros((self.Dy, self.Dx), dtype='complex128')
    
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
        return self