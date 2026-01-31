import numpy as np
from scipy import fft
import tqdm

def propagate_conv(space, z):                                     #propagate the field by a distance z using fresnel integral. convolution method
    k = 2*np.pi/space.lamb
    prop = np.exp(-1j * z * (space.kx ** 2 + space.ky ** 2) / (2 * k) - 1j*k*z)
    space.fourier_field = fft.fft2(space.field)
    space.field = fft.ifft2(prop*space.fourier_field)
    return space

def propagate_dsum(space, z):                                    #propagate the field by a distance z using fresnel integral. direct sum method
    if z == 0:
        return space
    else:
        k = 2*np.pi/space.lamb
        dA = 2*space.nix/space.Dx * 2*space.niy/space.Dy 
        prog = tqdm.tqdm(range(space.field.shape[0]))
        def cexp(xi, yi, x, y, z, space):
            return np.exp(-1j*k*z)*np.exp(1j*k*((x-xi)**2 + (y-yi)**2)/(2*z))/(-1j*space.lamb*z)
        def f(xi, yi, x, y, z, space):
            return cexp(xi, yi, x, y, z, space)*space.field
        def F(x, y, z, space):
            integrand = f(space.x, space.y, x, y, z, space)
            return np.sum(integrand) * dA
        def F_grid(space, z):
            F_grid = np.zeros_like(space.field, dtype=complex)
            for i in prog:
                prog.set_description(f"Propagating row {i+1}")
                for j in range(space.field.shape[1]):
                    F_grid[i, j] = F(space.x[0, j], space.y[i, 0], z, space)
            return F_grid
        space.field = F_grid(space, z)
        return space


def propagate_conv2(space, z, scale=1):                                     #propagate the field by a distance z using fresnel integral. convolution method
    k = 2*np.pi/space.lamb

    L1 = 2*space.nix
    L2 = L1*scale

    dx = 2*space.nix/space.Dx
    dy = 2*space.niy/space.Dy
    factor =  (dx*dy)* np.exp(np.pi*1j * (space.x + space.y))

    prop = np.exp(1j * z * (space.kx ** 2 + space.ky ** 2) / (2 * k) - 1j*k*z)


    space.fourier_field = fft.fft2(space.field*np.exp(-1j*np.pi* (k/(2*np.pi*z))*(space.x**2 + space.y**2) ) * np.exp(1j*np.pi*(L1- L2)/L1 * (k/(2*np.pi*z))*(space.x**2 + space.y**2 )))
    space.fourier_field = space.fourier_field*factor*prop

    Uf = fft.ifft2(np.exp(- 1j * np.pi / (k/(2*np.pi*z)) * L1/L2 * (space.kx**2 + space.ky**2))  *  space.fourier_field)

    Uf = L1/L2 * np.exp(-1j *np.pi*(k/(2*np.pi*z))* (space.x**2 + space.y**2)   - 1j * np.pi*(k/(2*np.pi*z))* (L1-L2)/L2 * (space.x**2 + space.y**2)) * Uf *1j * (space.lamb*z)
    
    

    space.x = space.x*scale
    space.y = space.y*scale

    dx = dx*scale
    dy = dy*scale

    #space.kx, space.ky = np.meshgrid(2*np.pi*fft.fftfreq(space.Dx, 2*space.nix/space.Dx), 2*np.pi*fft.fftfreq(space.Dy, 2*space.niy/space.Dy), sparse=True)
    space.nix = (space.kx[0,1]-space.kx[0,0])*space.Dx/2*scale
    space.niy = (space.ky[1,0]-space.ky[0,0])*space.Dy/2*scale


    space.field = Uf
    return space
