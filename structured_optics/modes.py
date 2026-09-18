from .utils import C_ince, S_ince, cartesian_to_elliptic, get_LP_params, herm, laguerre, rotated_mode
import numpy as np
from scipy.special import factorial, jn_zeros, jv, kv

__all__ = [ "hg", "hg_astigmatic", "lg", "nbessel", "gbessel", "lg_prod",
            "frac_oam", "frac_oam_qs", "IG_even", "IG_odd", "HInceG",
            "circle", "square", "triangle", "lp", "lp_hel"]




#modes implementations

@rotated_mode
def hg(Beam, N, M, z=0):                                                        
    """
    Generate a Hermite-Gaussian (HG) mode.

    The mode is evaluated at the propagation distance ``z`` using the
    paraxial Gaussian-beam solution. The transverse mode orders are ``N``
    and ``M`` along the x- and y-directions, respectively, giving a total
    transverse order of ``N + M``.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    N : int
        Hermite-Gaussian mode order along the x-axis.
    M : int
        Hermite-Gaussian mode order along the y-axis.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued transverse field with shape ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate an astigmatic Hermite-Gaussian (HG) mode.

    Unlike :func:`hg`, this function allows independent beam waists along
    the x- and y-directions. The two transverse dimensions therefore have
    independent Rayleigh ranges and beam-width evolution.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, wavelength,
        and beam center.
    N : int
        Hermite-Gaussian mode order along the x-axis.
    M : int
        Hermite-Gaussian mode order along the y-axis.
    wx : float
        Beam waist along the x-axis at ``z = 0``, in meters.
    wy : float
        Beam waist along the y-axis at ``z = 0``, in meters.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued transverse field with shape ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate a Laguerre-Gaussian (LG) mode.

    The mode is described in cylindrical coordinates by the azimuthal
    index ``l`` and radial index ``p``. Its transverse mode order is

    ``|l| + 2p``.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    l : int
        Azimuthal mode index. The sign determines the azimuthal phase
        winding and therefore the sign of the orbital angular momentum.
    p : int
        Radial mode index. Must be a non-negative integer.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued transverse field with shape ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate a normalized finite-grid Bessel mode.

    The transverse field is proportional to a Bessel function of the first
    kind. The radial wave number is chosen such that the first zero of
    ``J_|N|`` occurs at ``r = Beam.waist``.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, wavelength,
        beam waist, and beam center.
    N : int
        Bessel mode order. The magnitude determines the Bessel-function
        order, while the sign determines the azimuthal phase winding.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate a Gaussian-Bessel beam.

    The field consists of a Bessel function multiplied by a Gaussian
    envelope. This produces a spatially localized approximation to an
    ideal Bessel beam.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    N : int
        Bessel mode order. The sign determines the azimuthal phase winding.
    r0 : float
        Parameter controlling the radial spatial frequency of the Bessel
        component, in meters.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
    rad = 2*np.pi*Beam.waist**2/r0
    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    F = jv(np.abs(N),rad*r/Beam.waist**2)*\
        np.exp(-1j*N*np.arctan2(Beam.y-Beam.y0, Beam.x-Beam.x0))*np.exp(-(r/Beam.waist)**2)
    F = F/np.sqrt(np.sum(np.abs(F**2))*(4*Beam.nix/Beam.Dx)*(Beam.niy/Beam.Dy))
    return F

@rotated_mode
def lg_prod(Beam, N, ls, centers):
    """
    Generate a product of displaced LG-like modes.

    The resulting field is constructed as the product of ``N`` factors,
    each associated with an azimuthal index and transverse center. This
    provides a convenient way to construct composite fields containing
    multiple localized orbital-angular-momentum structures.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        and beam center.
    N : int
        Number of factors to multiply.
    ls : array_like of int or None
        Azimuthal indices for the individual factors. If ``None``, all
        indices are set to ``+1``.
    centers : array_like of shape (N, 2) or None
        ``(x, y)`` centers of the individual factors, in meters. If
        ``None``, all centers are set to ``(0, 0)``.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate a fractional orbital-angular-momentum field using an LG
    superposition.

    A fractional OAM value is represented as a finite superposition of
    integer-order Laguerre-Gaussian modes. The selected integer orders are
    centered around the integer part of the requested fractional OAM.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    Ma : float
        Desired fractional orbital angular momentum index.
    n_modes : int
        Number of integer-order LG modes included in the superposition.
    beta : float
        Angular parameter used in the complex expansion coefficients, in
        radians.
    theta_0 : float
        Reference angular position used in the expansion coefficients,
        in radians.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate a quasi-stable fractional orbital-angular-momentum field.

    The field is constructed as a finite superposition of
    Laguerre-Gaussian modes with integer azimuthal indices surrounding the
    requested fractional OAM value. Unlike :func:`frac_oam`, the radial
    index of each LG component is selected according to the fractional
    OAM index and the chosen number of modes.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    Ma : float
        Desired fractional orbital angular momentum index.
    n_modes : int
        Number of integer-order LG modes included in the superposition.
    beta : float
        Angular parameter used in the complex expansion coefficients, in
        radians.
    theta_0 : float
        Reference angular position used in the expansion coefficients,
        in radians.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate an even Ince-Gaussian (IG) mode.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    p : int
        Ince-Gaussian order.
    m : int
        Ince-Gaussian mode index.
    q : float
        Ellipticity parameter used to define the elliptic coordinates.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
    
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
    """
    Generate an odd Ince-Gaussian (IG) mode.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    p : int
        Ince-Gaussian order.
    m : int
        Ince-Gaussian mode index.
    q : float
        Ellipticity parameter used to define the elliptic coordinates.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
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
    """
    Generate a Hermite-Ince-Gaussian (HIG) mode.

    The HIG field is constructed as a complex combination of the even and
    odd Ince-Gaussian modes with a specified helicity.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, beam waist,
        wavelength, and beam center.
    p : int
        Ince-Gaussian order.
    m : int
        Ince-Gaussian mode index.
    q : float
        Ellipticity parameter used to define the elliptic coordinates.
    helicity : int or float
        Helicity of the HIG mode. The sign determines the relative phase
        between the even and odd Ince-Gaussian components.
    z : float, optional
        Propagation distance along the optical axis, in meters.
        Default is 0.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
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
def circle(Beam, center=None, radius=None):
    """
    Generate a circular binary mode.

    The field has unit amplitude inside the specified circular aperture and
    zero amplitude outside it.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid and default
        beam center and waist.
    center : tuple of float or None, optional
        ``(x, y)`` coordinates of the circle center, in meters. If ``None``,
        the center of ``Beam`` (``Beam.x0, Beam.y0``) is used.
    radius : float or None, optional
        Circle radius, in meters. If ``None``, ``Beam.waist`` is used.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized binary field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
    if radius is None:
        radius = Beam.waist
    #Create a circular mode
    if center is None:
        center = Beam.x0, Beam.y0
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
def square(Beam, center=None, side_length=None):
    """
    Generate a square binary mode.

    The field has unit amplitude inside the square and zero amplitude
    outside it.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid and default
        beam center and waist.
    center : tuple of float or None, optional
        ``(x, y)`` coordinates of the square center, in meters. If ``None``,
        the center of ``Beam`` (``Beam.x0, Beam.y0``) is used.
    side_length : float or None, optional
        Side length of the square, in meters. If ``None``, ``Beam.waist``
        is used.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized binary field with shape
        ``(Beam.Dy, Beam.Dx)``.
    """
    if side_length is None:
        side_length = Beam.waist
    if center is None:
        center = Beam.x0, Beam.y0
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
def triangle(Beam, center=None, side_length=None):
    """
    Generate an equilateral triangular binary mode.

    The triangle is centered at the specified position and oriented with
    one vertex pointing upward.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid and default
        beam center and waist.
    center : tuple of float or None, optional
        ``(x, y)`` coordinates of the triangle center, in meters. If
        ``None``, the center of ``Beam`` (``Beam.x0, Beam.y0``) is used.
    side_length : float or None, optional
        Side length of the equilateral triangle, in meters. If ``None``,
        ``Beam.waist`` is used.

    Returns
    -------
    numpy.ndarray
        Complex-valued normalized binary field with shape
        ``(Beam.Dy, Beam.Dx)``.

    Notes
    -----
    The triangle is defined by three vertices and evaluated using
    barycentric-coordinate sign tests.
    """
    if side_length is None:
        side_length = Beam.waist
    if center is None:
        center = Beam.x0, Beam.y0
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
    """
    Generate a scalar linearly polarized (LP) fiber mode.

    The transverse mode is calculated using the scalar approximation for a
    step-index optical fiber. The field is represented by a Bessel
    function inside the fiber core and a modified Bessel function in the
    cladding.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, where
        ``Beam.waist`` is used as the fiber-core radius and ``Beam.lamb``
        as the wavelength.
    l : int
        Azimuthal mode index.
    m : int
        Radial mode index.
    n_core : float
        Refractive index of the fiber core.
    n_clad : float
        Refractive index of the fiber cladding.
    parity : {"cos", "sin"}, optional
        Angular parity of the mode. ``"cos"`` generates a
        ``cos(l * phi)`` dependence, while any other value generates a
        ``sin(l * phi)`` dependence. Default is ``"cos"``.

    Returns
    -------
    numpy.ndarray
        Complex-valued scalar transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.

    Notes
    -----
    The normalized propagation parameters ``u`` and ``v`` are obtained
    using :func:`get_LP_params` inside 'structured_optics/utils.py'.
    """
    params = get_LP_params(l,m, n_core, n_clad, Beam.waist, Beam.lamb)

    r = np.sqrt((Beam.x-Beam.x0)**2 + (Beam.y-Beam.y0)**2)
    PHI = np.arctan2((Beam.y-Beam.y0), (Beam.x-Beam.x0))
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
    """
    Generate a helical scalar LP fiber mode.

    The helical mode is constructed as a complex superposition of the
    cosine- and sine-parity scalar LP modes. The sign of ``l`` determines
    the sign of the relative phase between these two components.

    Parameters
    ----------
    Beam : Beam
        Beam object defining the transverse computational grid, where
        ``Beam.waist`` is used as the fiber-core radius and ``Beam.lamb``
        as the wavelength.
    l : int
        Azimuthal mode index. Its sign determines the helicity of the
        resulting mode.
    m : int
        Radial mode index.
    n_core : float
        Refractive index of the fiber core.
    n_clad : float
        Refractive index of the fiber cladding.

    Returns
    -------
    numpy.ndarray
        Complex-valued scalar transverse field with shape
        ``(Beam.Dy, Beam.Dx)``.

    Notes
    -----
    For positive ``l``, the mode is constructed as

    ``(LP_cos + i LP_sin) / sqrt(2)``.

    For negative ``l``, the relative phase is reversed.
    """
    co = lp(Beam, np.abs(l), m, n_core, n_clad, parity = 'cos')
    si = lp(Beam, np.abs(l), m, n_core, n_clad, parity = 'sin')
    if l>=0:
        F = (co + 1j*si)/np.sqrt(2)
    else:
        F = (co - 1j*si)/np.sqrt(2)
    return F

