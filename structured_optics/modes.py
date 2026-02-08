from structured_optics.utils import *
import numpy as np
from scipy import special


#modes implementations

def hg_mode(Beam, N, M, z, pol_index=None):                                                        
    #get a HG mode at distance z of order N+M
    zr = Beam.zr()
    q0 = 1j*zr
    q = -z + 1j*zr
    k = 2*np.pi/Beam.lamb
    w = Beam.waist*np.sqrt(1 + (z/zr)**2)
    Cn = np.sqrt(np.sqrt(2/np.pi)*q0/(2**N * special.factorial(N) * q * Beam.waist))
    un = Cn*(-np.conjugate(q)/q)**(N/2)*hermite(np.sqrt(2)*(Beam.x-Beam.x0)/w, N)*np.exp(-1j*k*(Beam.x-Beam.x0)**2/(2*q))
    Cm = np.sqrt(np.sqrt(2/np.pi)*q0/(2**M * special.factorial(M) * q * Beam.waist))
    um = Cm*(-np.conjugate(q)/q)**(M/2)*hermite(np.sqrt(2)*(Beam.y-Beam.y0)/w, M)*np.exp(-1j*k*(Beam.y-Beam.y0)**2/(2*q))
    F = un*um*np.exp(1j*k*z)
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])/np.sqrt(2)
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])/np.sqrt(3)
    else:
        Beam.field[pol_index] = F
    return Beam


def lg_mode(Beam,l,p, z, pol_index=None):                                    
    #get a LG mode at distance z of order abs(N) + 2M
    zr = Beam.zr()
    k = 2*np.pi/Beam.lamb
    w = Beam.waist*np.sqrt(1 + (z/zr)**2)
    R = (z**2 + zr**2)
    gouy = (np.abs(l) + 2*p + 1)*np.arctan(z/zr)
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    Cte = np.sqrt(2*special.factorial(p) / (np.pi*special.factorial(p+np.abs(l))))
    F = Cte*(1/w)*(np.sqrt(2)*r/w)**(np.abs(l)) * np.exp(-(r/w)**2) * laguerre(2*(r/w)**2, np.abs(l), p) *\
        np.exp(1j*(k*z + k*z*(r**2)/(2*R) + l*np.arctan2(Beam.y-Beam.y0,Beam.x-Beam.x0) - gouy))
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])/np.sqrt(2)
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])/np.sqrt(3)
    else:
        Beam.field[pol_index] = F
    return Beam

def bessel_mode(Beam, N, z, pol_index=None):                       
    #get a Bessel mode of order N at distance z
    k = 2*np.pi / Beam.lamb        # wavenumber
    j01 = special.jn_zeros(abs(N), 1)[0]  # first zero of J_N
    # enforce the first zero at r = self.waist
    kr = j01 / Beam.waist
    kz = np.sqrt(k**2 - kr**2)    
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    phi = np.arctan2(Beam.y-Beam.y0, Beam.x-Beam.x0)
    F = (np.exp(1j*kz*z) * special.jv(abs(N), kr*r) * np.exp(-1j*N*phi))
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam

def gbessel_mode(Beam, N, r0, pol_index):                                    
    #get a gaussian bessel beam of order N at z=0
    rad = 2*np.pi*Beam.waist**2/r0
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    F = special.jv(np.abs(N),rad*r/Beam.waist**2)*\
        np.exp(-1j*N*np.arctan2(Beam.y-Beam.y0, Beam.x-Beam.x0))*np.exp(-(r/Beam.waist)**2)
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam

def lg_prod_mode(Beam, N, ls, centers, pol_index):
    if centers is None:
        centers = np.zeros((N, 2))
    centers = np.asarray(centers)
    if ls is None:
        ls = np.ones(N, dtype=int)
    ls = np.asarray(ls)

    F = np.ones((Beam.Dy, Beam.Dx), dtype='complex128')
    for i in range(N):
        x0 = centers[i, 0]
        y0 = centers[i, 1]
        l = ls[i]
        sign = -1 if l < 0 else 1
        exp_pow = int(np.abs(l))
        arg = (Beam.x - x0) + 1j * sign * (Beam.y - y0)
        r2 = (Beam.x - x0)**2 + (Beam.y - y0)**2
        lg = arg**exp_pow * np.exp(-r2/N / ((Beam.waist)**2))
        F *= lg
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam

def frac_oam_mode(Beam, Ma, n_modes, beta, theta_0, z, pol_index):        
    #get a fractional OAM beam, with OAM Ma (!= integer), by the method of LG supperpositions.
    mu = Ma%1
    m = Ma-mu
    if mu >=0.5:
        n_min = np.ceil(Ma - n_modes / 2)
    else: 
        n_min = np.floor(Ma - n_modes / 2)
    n_max = n_min + n_modes - 1
    F = np.zeros((Beam.Dy, Beam.Dx), dtype='complex128')
    for l in np.arange(int(n_min), int(n_max)+1):
        coef = np.exp(-1j*mu*beta)*1j*np.exp(1j*(Ma-l)*theta_0)/(2*np.pi*(Ma-l))*np.exp(1j*(m-l)*beta)*(1-np.exp(1j*mu*2*np.pi))
        if Beam.pol_dim == 1:
            F += coef*(Beam.lg(l, 0, z = z).field) 
        else:
            F += coef*(Beam.lg(l, 0, z = z, pol_index=0).field[0])
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam

def frac_oam_qs_mode(Beam, Ma, n_modes, beta, theta_0, z, pol_index):
    #get a quasi-stable fractional oam mode.
    n_min = np.round(Ma-n_modes/2) 
    n_max = n_min + n_modes - 1
    mu = Ma%1
    m = Ma-mu
    F = np.zeros((Beam.Dy, Beam.Dx), dtype='complex128')
    for l in np.arange(int(n_min), int(n_max)+1):
        coef = np.exp(-1j*mu*beta)*1j*np.exp(1j*(Ma-l)*theta_0)/(2*np.pi*(Ma-l))*np.exp(1j*(m-l)*beta)*(1-np.exp(1j*mu*2*np.pi))
        p = np.floor((np.abs(Ma) + n_modes/2 -np.abs(l))/2)
        if Beam.pol_dim == 1:
            F += coef*(Beam.lg(l, p, z = z).field) 
        else:
            F += coef*(Beam.lg(l, p, z = z, pol_index=0).field[0])
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam



def IG_even_mode(Beam, p, m, q, z, pol_index):
    #Even Ince-Gaussian mode IG_p,m^e(x,y)
    
    xi, eta = cartesian_to_elliptic(Beam.x-Beam.x0, Beam.y-Beam.y0, q, Beam.waist, z, Beam.lamb)

    Ce_xi  = C_ince(1j*xi,  p, m, q)
    Ce_eta = C_ince(eta, p, m, q)

    zr = np.pi * Beam.waist**2 / Beam.lamb
    k = 2*np.pi/Beam.lamb
    wz = Beam.waist * np.sqrt(1 + (z * Beam.lamb / (np.pi * Beam.waist**2))**2)
    Rz = (z**2 + zr**2)
    r2 = (Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2
    gouy = (p+1)*np.arctan(z / zr)
    F = Ce_xi * Ce_eta * np.exp(-r2 / (wz**2))*np.exp(1j*(k*z + k*z*(r2)/(2*Rz) - gouy))
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam

def IG_odd_mode(Beam, p, m, q, z, pol_index):
    #Odd Ince-Gaussian mode IG_p,m^o(x,y)
    
    xi, eta = cartesian_to_elliptic(Beam.x-Beam.x0, Beam.y-Beam.y0, q, Beam.waist, z, Beam.lamb)
    So_xi  = S_ince(1j*xi,  p, m, q)
    So_eta = S_ince(eta, p, m, q)

    zr = np.pi * Beam.waist**2 / Beam.lamb
    k = 2*np.pi/Beam.lamb
    wz = Beam.waist * np.sqrt(1 + (z * Beam.lamb / (np.pi * Beam.waist**2))**2)
    Rz = (z**2 + zr**2)
    r2 = (Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2
    gouy = (p+1)*np.arctan(z / zr)
    F = So_xi * So_eta * np.exp(-r2 / (wz**2))*np.exp(1j*(k*z + k*z*(r2)/(2*Rz) - gouy))
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam

def HInceG_mode(Beam, p, m, q, z, helicity, pol_index):
    #Hermite-Ince-Gaussian mode HIG_p,m^e(x,y), combining even and odd Ince-Gaussian modes with given helicity.
    #For some reason IG_odd has a 3*pi/2 phase shift wrt IG_even, so the helicity sign is inverted here.
    xi, eta = cartesian_to_elliptic(Beam.x-Beam.x0, Beam.y-Beam.y0, q, Beam.waist, z, Beam.lamb)

    Ce_xi  = C_ince(1j*xi,  p, m, q)
    Ce_eta = C_ince(eta, p, m, q)

    So_xi  = S_ince(1j*xi,  p, m, q)
    So_eta = S_ince(eta, p, m, q)

    zr = np.pi * Beam.waist**2 / Beam.lamb
    k = 2*np.pi/Beam.lamb
    wz = Beam.waist * np.sqrt(1 + (z * Beam.lamb / (np.pi * Beam.waist**2))**2)
    Rz = (z**2 + zr**2)
    r2 = (Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2
    gouy = (p+1)*np.arctan(z / zr)
    F = (Ce_xi * Ce_eta - helicity*So_xi * So_eta) * np.exp(-r2 / (wz**2))*np.exp(1j*(k*z + k*z*(r2)/(2*Rz) - gouy))
    if pol_index == None:
        if Beam.pol_dim == 1:
            Beam.field = F
        elif Beam.pol_dim == 2:
            Beam.field = np.array([F, F])
        elif Beam.pol_dim == 3:
            Beam.field = np.array([F, F, F])
    else:
        Beam.field[pol_index] = F
    Beam.norm_beam()
    return Beam
