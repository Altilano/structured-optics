from structured_optics.utils import *
import numpy as np
from scipy.special import factorial, jn_zeros, jv
from scipy.optimize import brentq



#modes implementations

@rotated_mode
def hg(Beam, N, M, z=0):                                                        
    #get a HG mode at distance z of order N+M
    zr = Beam.zr()
    q0 = 1j*zr
    q = -z + 1j*zr
    k = 2*np.pi/Beam.lamb
    w = Beam.waist*np.sqrt(1 + (z/zr)**2)
    Cn = np.sqrt(np.sqrt(2/np.pi)*q0/(2**N * factorial(N) * q * Beam.waist))
    un = Cn*(-np.conjugate(q)/q)**(N/2)*herm(np.sqrt(2)*(Beam.x-Beam.x0)/w, N)*np.exp(-1j*k*(Beam.x-Beam.x0)**2/(2*q))
    Cm = np.sqrt(np.sqrt(2/np.pi)*q0/(2**M * factorial(M) * q * Beam.waist))
    um = Cm*(-np.conjugate(q)/q)**(M/2)*herm(np.sqrt(2)*(Beam.y-Beam.y0)/w, M)*np.exp(-1j*k*(Beam.y-Beam.y0)**2/(2*q))
    F = un*um*np.exp(1j*k*z)
    return F

@rotated_mode
def hg_astigmatic(Beam, N, M, wx, wy, z=0):                                                        
    #get a astigmatic HG mode at distance z of order N+M
    zrx = np.pi*wx**2/Beam.lamb
    q0x = 1j*zrx
    qx = -z + q0x
    k = 2*np.pi/Beam.lamb
    wxz = wx*np.sqrt(1 + (z/zrx)**2)
    Cn = np.sqrt(np.sqrt(2/np.pi)*q0x/(2**N * factorial(N) * qx * wx))
    un = Cn*(-np.conjugate(qx)/qx)**(N/2)*herm(np.sqrt(2)*(Beam.x-Beam.x0)/wxz, N)*np.exp(-1j*k*(Beam.x-Beam.x0)**2/(2*qx))
    
    zry = np.pi*wy**2/Beam.lamb
    q0y = 1j*zry
    qy = -z + q0y
    wyz = wy*np.sqrt(1 + (z/zry)**2)
    Cm = np.sqrt(np.sqrt(2/np.pi)*q0y/(2**M * factorial(M) * qy * wy))
    um = Cm*(-np.conjugate(qy)/qy)**(M/2)*herm(np.sqrt(2)*(Beam.y-Beam.y0)/wyz, M)*np.exp(-1j*k*(Beam.y-Beam.y0)**2/(2*qy))
    F = un*um*np.exp(1j*k*z)
    return F

@rotated_mode
def lg(Beam,l,p, z=0):                                    
    #get a LG mode at distance z of order abs(N) + 2M
    zr = Beam.zr()
    k = 2*np.pi/Beam.lamb
    w = Beam.waist*np.sqrt(1 + (z/zr)**2)
    R = (z**2 + zr**2)
    gouy = (np.abs(l) + 2*p + 1)*np.arctan(z/zr)
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    Cte = np.sqrt(2*factorial(p) / (np.pi*factorial(p+np.abs(l))))
    F = Cte*(1/w)*(np.sqrt(2)*r/w)**(np.abs(l)) * np.exp(-(r/w)**2) * laguerre(2*(r/w)**2, np.abs(l), p) *\
        np.exp(1j*(k*z + k*z*(r**2)/(2*R) + l*np.arctan2(Beam.y-Beam.y0,Beam.x-Beam.x0) - gouy))
    return F

@rotated_mode
def nbessel(Beam, N, z=0):                       
    #get a Bessel mode of order N at distance z
    k = 2*np.pi / Beam.lamb        # wavenumber
    j01 = jn_zeros(abs(N), 1)[0]  # first zero of J_N
    # enforce the first zero at r = Beam.waist
    kr = j01 / Beam.waist
    kz = np.sqrt(k**2 - kr**2)    
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    phi = np.arctan2(Beam.y-Beam.y0, Beam.x-Beam.x0)
    F = (np.exp(1j*kz*z) * jv(abs(N), kr*r) * np.exp(-1j*N*phi))
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def gbessel(Beam, N, r0):                                    
    #get a gaussian bessel beam of order N at z=0
    rad = 2*np.pi*Beam.waist**2/r0
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    F = jv(np.abs(N),rad*r/Beam.waist**2)*\
        np.exp(-1j*N*np.arctan2(Beam.y-Beam.y0, Beam.x-Beam.x0))*np.exp(-(r/Beam.waist)**2)
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def lg_prod(Beam, N, ls, centers):
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
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def frac_oam(Beam, Ma, n_modes, beta, theta_0, z=0):        
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
        F += coef*(lg(Beam,l, 0, z = z)) 

    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def frac_oam_qs(Beam, Ma, n_modes, beta, theta_0, z=0):
    #get a quasi-stable fractional oam mode.
    n_min = np.round(Ma-n_modes/2) 
    n_max = n_min + n_modes - 1
    mu = Ma%1
    m = Ma-mu
    F = np.zeros((Beam.Dy, Beam.Dx), dtype='complex128')
    for l in np.arange(int(n_min), int(n_max)+1):
        coef = np.exp(-1j*mu*beta)*1j*np.exp(1j*(Ma-l)*theta_0)/(2*np.pi*(Ma-l))*np.exp(1j*(m-l)*beta)*(1-np.exp(1j*mu*2*np.pi))
        p = np.floor((np.abs(Ma) + n_modes/2 -np.abs(l))/2)
        F += coef*(lg(Beam, l, p, z = z)) 

    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F


@rotated_mode
def IG_even(Beam, p, m, q, z=0):
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
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def IG_odd(Beam, p, m, q, z=0):
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
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def HInceG(Beam, p, m, q, helicity, z=0):
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
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def circle(Beam, center, radius):
    if radius == None:
        radius = Beam.waist
    #Create a circular mode
    center_x, center_y = center
    # Calculate distance from center for all points
    distances = np.sqrt((Beam.x - center_x)**2 + (Beam.y - center_y)**2)
    # Set points within radius to 1
    mask = distances <= radius
    F = np.zeros((Beam.Dy, Beam.Dx), dtype='complex')
    F[mask] = 1
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def square(Beam, center, side_length):
    if side_length == None:
        side_length = Beam.waist
    center_x, center_y = center
    
    # Calculate square boundaries
    half_side = side_length / 2.0
    
    # Create mask for points inside square
    mask = (np.abs(Beam.x - center_x) <= half_side) & (np.abs(Beam.y - center_y) <= half_side)
    
    # Set points inside square to 1
    F = np.zeros((Beam.Dy, Beam.Dx), dtype='complex')
    F[mask] = 1
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def triangle(Beam, center, side_length):
    if side_length == None:
        side_length = Beam.waist
    center_x, center_y = center
    
    # Height of equilateral triangle
    height = side_length * np.sqrt(3) / 2
    
    # Vertices for upward-pointing equilateral triangle
    # Top vertex
    v0_x = center_x
    v0_y = center_y + height * 2/3
    
    # Bottom left vertex
    v1_x = center_x - side_length / 2
    v1_y = center_y - height * 1/3
    
    # Bottom right vertex
    v2_x = center_x + side_length / 2
    v2_y = center_y - height * 1/3
    
    # Use barycentric coordinates to determine if point is inside triangle
    def sign(px, py, v1_x, v1_y, v2_x, v2_y):
        return (px - v2_x) * (v1_y - v2_y) - (v1_x - v2_x) * (py - v2_y)
    
    # Compute signs for all points
    d1 = sign(Beam.x, Beam.y, v0_x, v0_y, v1_x, v1_y)
    d2 = sign(Beam.x, Beam.y, v1_x, v1_y, v2_x, v2_y)
    d3 = sign(Beam.x, Beam.y, v2_x, v2_y, v0_x, v0_y)
    
    # Point is inside if all signs are same
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    
    mask = ~(has_neg & has_pos)
    F = np.zeros((Beam.Dy, Beam.Dx), dtype='complex')
    F[mask] = 1
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def lp(Beam, l, m, n_core, n_clad, parity="cos"):
    """Evaluate the scalar LP_l field Psi(r,phi)"""
    params = get_LP_params(l,m, n_core, n_clad, Beam.waist, Beam.lamb)

    r = np.sqrt(Beam.x**2 + Beam.y**2)
    PHI = np.arctan2(Beam.y, Beam.x)
    a = Beam.waist
    inside = r <= a
    F = np.zeros_like(r)
    F[inside] = jv(l, params['u'] * r[inside] / a)
    scale = jv(l, params['u']) / kv(l, params['v'])
    F[~inside] = scale * kv(l, params['v'] * r[~inside] / a)
    ang = np.cos(l * PHI) if parity == "cos" else np.sin(l * PHI)
    return F * ang

@rotated_mode
def lp_hel(Beam, l, m, n_core, n_clad):
    co = lp(Beam, np.abs(l), m, n_core, n_clad, parity = 'cos')
    si = lp(Beam, np.abs(l), m, n_core, n_clad, parity = 'sin')
    if l>=0:
        F = (co + 1j*si)/np.sqrt(2)
    else:
        F = (co - 1j*si)/np.sqrt(2)
    return F

