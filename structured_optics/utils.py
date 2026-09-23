import numpy as np
from scipy.special import hermite, genlaguerre, jv, j0, j1, kv, jn_zeros
from scipy.optimize import brentq
from functools import wraps
import math




#utils for modes

def rotated_mode(func):  #add angle to rotate modes analyticaly
    """
    Decorator that adds an `angle` keyword argument to a mode-generating
    method, rotating the mode analytically by temporarily rotating the
    beam's coordinate grid.

    Parameters
    ----------
    func : callable
        A method with signature `func(Beam, *args, **kwargs)` that sets
        `Beam`'s field to some mode (e.g. `Beam.hg`, `Beam.lg`).

    Returns
    -------
    callable
        A wrapped version of `func` that accepts an additional `angle`
        keyword (default 0, in radians). If `angle % (2*pi) != 0`, `func`
        is called inside `Beam.rotated_grid(angle)` (i.e. with the
        coordinate grid rotated by `angle` for the duration of the
        call); otherwise `func` is called normally with no rotation.
    """

    @wraps(func)
    def wrapper(Beam, *args, angle=0, **kwargs):
        if angle%(2*np.pi) != 0:
            with Beam.rotated_grid(angle):
                return func(Beam, *args, **kwargs)
        else:
            return func(Beam, *args, **kwargs)

    return wrapper


def herm(X, N):              #hermite polynomial
    """
    Evaluate the physicists' Hermite polynomial of order N.

    Parameters
    ----------
    X : array_like
        Points at which to evaluate the polynomial.
    N : int
        Polynomial order.

    Returns
    -------
    array_like
        `H_N(X)`, same shape as `X`.
    """
    HER = hermite(N)
    sn = HER(X)
    return sn

def laguerre(X, L, P):          #laguerre polynomial
    """
    Evaluate the generalized (associated) Laguerre polynomial.

    Parameters
    ----------
    X : array_like
        Points at which to evaluate the polynomial.
    L : int
        Passed as the `alpha` parameter of `scipy.special.genlaguerre`.
    P : int
        Passed as the degree `n` parameter of
        `scipy.special.genlaguerre`.

    Returns
    -------
    array_like
        `L_P^{(L)}(X)`, same shape as `X`.
    """
    LAG = genlaguerre(P, L)
    sn = LAG(X)
    return sn

def even_coeffs(p, q, kind):
    """
    Compute the eigenvalues and coefficient vectors of even Ince
    polynomials of order `p` (i.e. `C_p^m` for kind 'C', or the even-p
    'S' family), by solving the tridiagonal-like recurrence relation as
    an eigenvalue problem.

    Parameters
    ----------
    p : int
        Ince polynomial order. Must be even.
    q : float
        Ellipticity parameter of the Ince equation.
    kind : {'C', 'S'}
        Which family of Ince polynomials to build the recurrence
        matrix for ('C' for cosine-type, 'S' for sine-type).

    Returns
    -------
    eigvals_sorted : np.ndarray
        Eigenvalues, sorted in ascending order (for kind 'S', the
        zero eigenvalue -- corresponding to the trivial/degenerate
        solution -- is kept first, followed by the remaining
        eigenvalues sorted ascending).
    eigvecs_sorted : np.ndarray
        Matrix whose columns are the coefficient vectors `A_r`
        corresponding to each sorted eigenvalue, normalized so that
        the sum of each column's coefficients is positive.

    Raises
    ------
    ValueError
        If `p` is not even.
    """
    #Compute coefficients A_r of even Ince polynomials C_p^m.
    if p % 2 != 0:
        raise ValueError("p must be even for even coefs")

    N = p // 2 

    if kind =='C':
        M = np.zeros((N+1, N+1), dtype=float)
        # First equation
        M[0,0] = 0
        if N >= 1:
            M[0, 1] = (p/2 + 1) * q

        # Second equation
        if N >= 1:
            M[1, 0] = p * q
            M[1, 1] = 4
            if N >= 2:
                M[1, 2] = (p/2 + 2) * q

        # General recurrence
        for r in range(2,N):

            M[r, r-1]   = -(r -1 - p/2) * q
            M[r, r] = 4 * (r)**2
            M[r, r+1] = (p/2 + r + 1) * q
        # Last row
        if N >=2:
            M[N, N-1] = q
            M[N, N]   = 4 * (N)**2

    
    elif kind =='S':
        M = np.zeros((N+1, N+1), dtype=float)
        
        # First equation (r=1)
        if N >= 1:
            M[1, 1] = 4
        if N >= 2:
            M[1, 2] = (p/2 + 2) * q

        # General recurrence
        for r in range(2, N):
            M[r, r-1] = -(r - 1 - p/2) * q
            M[r, r] = 4 * (r)**2
            M[r, r+1] = (p/2 + r + 1) * q
        
        # Last row
        if N >= 2:
            M[N, N-1] = -(N - 1 - p/2) * q
            M[N, N] = 4 * (N)**2

    eigvals, eigvecs = np.linalg.eig(M)

    # Move the eigenvector with zero eigenvalue to the front (only for 'S' kind) and sort the rest.
    if kind == 'S':
        #find zero position
        zero_idx = np.argmin(np.abs(eigvals))
        
        mask = np.arange(len(eigvals)) != zero_idx
        
        non_zero_eigvals = eigvals[mask]
        non_zero_eigvecs = eigvecs[:, mask]
        
        #sort non-zero eigenvalues and eigenvectors
        sorted_idx = np.argsort(non_zero_eigvals)
        non_zero_eigvals_sorted = non_zero_eigvals[sorted_idx]
        non_zero_eigvecs_sorted = non_zero_eigvecs[:, sorted_idx]

        #combine zero eigenvalue and sorted non-zero ones
        eigvals_sorted = np.concatenate([[eigvals[zero_idx]], non_zero_eigvals_sorted])
        eigvecs_sorted = np.column_stack([eigvecs[:, zero_idx:zero_idx+1], non_zero_eigvecs_sorted])
    else:
        # For 'C' kind, just sort normally
        idx = np.argsort(eigvals)
        eigvals_sorted = eigvals[idx]
        eigvecs_sorted = eigvecs[:, idx]

    # Sign convention: ensure sum of coefficients is positive
    for i in range(eigvecs_sorted.shape[1]):
        if np.sum(eigvecs_sorted[:, i]) < 0:
            eigvecs_sorted[:, i] *= -1
        
    return eigvals_sorted, eigvecs_sorted


def odd_coeffs(p, q, kind):
    """
    Compute the eigenvalues and coefficient vectors of odd Ince
    polynomials of order `p` (i.e. `C_p^m`/`S_p^m` for odd p), by
    solving the corresponding recurrence relation as an eigenvalue
    problem.

    Parameters
    ----------
    p : int
        Ince polynomial order. Must be odd.
    q : float
        Ellipticity parameter of the Ince equation.
    kind : {'C', 'S'}
        Which family of Ince polynomials to build the recurrence
        matrix for ('C' for cosine-type, 'S' for sine-type).

    Returns
    -------
    eigvals_sorted : np.ndarray
        Eigenvalues, sorted in ascending order.
    eigvecs_sorted : np.ndarray
        Matrix whose columns are the coefficient vectors corresponding
        to each sorted eigenvalue, normalized so that the sum of each
        column's coefficients is positive.

    Raises
    ------
    ValueError
        If `p` is not odd.
    """
    
    if p % 2 != 1:
        raise ValueError("p must be odd for odd coefs")

    N = p // 2 
    M = np.zeros((N+1, N+1), dtype=float)

    if kind == 'C':
        # First equation
        M[0, 0] = q/2 * (p + 1) + 1
        if N >= 1:
            M[0, 1] = q/2 * (p + 3)

        # General recurrence (r >= 1)
        for r in range(1, N):
            M[r, r-1] = -q/2 * (2*r - p - 1)
            M[r, r]   = (2*r + 1)**2
            M[r, r+1] = q/2 * (p + 2*r + 3)
        # Last row
        if N >=1:
            M[N, N-1] = q   
            M[N, N]   = (2*N + 1)**2

    if kind == 'S':
        # First equation
        M[0, 0] = 1 - q/2 * (p + 1)
        if N >= 1:
            M[0, 1] = q/2 * (p + 3)

        # General recurrence (r >= 1)
        for r in range(1, N):
            M[r, r-1] = -q/2 * (2*r - p - 1)
            M[r, r]   = (2*r + 1)**2
            M[r, r+1] = q/2 * (p + 2*r + 3)
        # Last row
        if N >=1:
            M[N, N-1] = q   
            M[N, N]   = (2*N + 1)**2

    eigvals, eigvecs = np.linalg.eig(M)

    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    # Sign convention: ensure sum of coefficients is positive
    for i in range(eigvecs_sorted.shape[1]):
        if np.sum(eigvecs_sorted[:, i]) < 0:
            eigvecs_sorted[:, i] *= -1

    return eigvals_sorted, eigvecs_sorted
    

def C_ince(xi, p, m, q):
    """
    Evaluate the even (cosine-type) Ince polynomial `C_p^m(xi, q)`.

    Parameters
    ----------
    xi : array_like
        Points at which to evaluate the polynomial.
    p : int
        Ince polynomial order (p >= 0).
    m : int
        Ince polynomial degree (0 <= m <= p), with `p - m` even.
    q : float
        Ellipticity parameter of the Ince equation.

    Returns
    -------
    array_like
        `C_p^m(xi, q)`, same shape as `xi`.

    Raises
    ------
    ValueError
        If `p`, `m` are out of range, or if `p - m` is odd (`C_ince`
        is only defined for even `p - m`).
    """
    # Compute even Ince polynomial C_p^m(xi, q)
    if p<0 or m < 0 or m > p:
        raise ValueError("invalid p,m")
    if p % 2 == 0 and m % 2 == 0:
        eigvals, eigvecs = even_coeffs(p, q, kind='C')
        idx = m // 2 
        A = eigvecs[:, idx]
        s = sum(A[r] * np.cos(2*r*xi) for r in range(len(A)))
        return s
    if p % 2 == 1 and m % 2 == 1:
        eigvals, eigvecs = odd_coeffs(p, q, kind='C')
        idx = m // 2
        B = eigvecs[:, idx]
        s = sum(B[r] * np.cos((2*r+1)*xi) for r in range(len(B)))
        return s
    else:
        raise ValueError("C_ince is only defined for even (p-m)")
    
def S_ince(xi, p, m, q):
    """
    Evaluate the odd (sine-type) Ince polynomial `S_p^m(xi, q)`.

    Parameters
    ----------
    xi : array_like
        Points at which to evaluate the polynomial.
    p : int
        Ince polynomial order (p >= 0).
    m : int
        Ince polynomial degree (0 <= m <= p), with `p - m` even.
    q : float
        Ellipticity parameter of the Ince equation.

    Returns
    -------
    array_like
        `S_p^m(xi, q)`, same shape as `xi`.

    Raises
    ------
    ValueError
        If `p`, `m` are out of range, or if `p - m` is odd (`S_ince`
        is only defined for even `p - m`).
    """
    # Compute odd Ince polynomial S_p^m(xi, q)
    if p<0 or m < 0 or m > p:
        raise ValueError("invalid p,m")
    if p % 2 == 0 and m % 2 == 0:
        eigvals, eigvecs = even_coeffs(p, q, kind='S')
        idx = m // 2
        A = eigvecs[:, idx]
        s = sum(A[r] * np.sin(2*r*xi) for r in range(1,len(A)))
        return s
    if p % 2 == 1 and m % 2 == 1:
        eigvals, eigvecs = odd_coeffs(p, q, kind='S')
        idx = (m-1) // 2
        B = eigvecs[:, idx]
        s = sum(B[r] * np.sin((2*r+1)*xi) for r in range(len(B)))
        return s
    else:
        raise ValueError("S_ince is only defined for even (p-m)")
    

def cartesian_to_elliptic(x, y, q, w0, z, lamb):
    """
    Convert Cartesian coordinates (x, y) to elliptic coordinates
    (xi, eta), for an elliptic coordinate system whose foci scale with
    the beam's waist as it propagates (used for Ince-Gaussian modes).

    Parameters
    ----------
    x, y : array_like
        Cartesian coordinates.
    q : float
        Ellipticity parameter.
    w0 : float
        Beam waist at `z = 0`.
    z : float
        Propagation distance, used (with `lamb`) to compute the local
        beam width `w` and scale the focal distance `f` accordingly.
    lamb : float
        Wavelength.

    Returns
    -------
    xi, eta : array_like
        Elliptic radial coordinate `xi` (>= 0) and angular coordinate
        `eta`, same shape as `x`/`y`.
    """
    #Convert Cartesian (x,y) to elliptic coordinates (xi, eta) with elipticity q.

    f0 = w0*np.sqrt(q/2)
    w = w0 * np.sqrt(1 + (z * lamb / (np.pi * w0**2))**2)
    f = f0*w/w0
    r_plus  = np.sqrt((x + f)**2 + y**2)
    r_minus = np.sqrt((x - f)**2 + y**2)

    xi  = np.arccosh((r_plus + r_minus) / (2*f))
    eta = np.sign(y)*np.arccos((r_plus - r_minus) / (2*f))

    return xi, eta

def elliptic_to_cartesian(xi, eta, q, w0, z, lamb):
    """
    Convert elliptic coordinates (xi, eta) to Cartesian coordinates
    (x, y); the inverse of `cartesian_to_elliptic`.

    Parameters
    ----------
    xi : array_like
        Elliptic radial coordinate.
    eta : array_like
        Elliptic angular coordinate.
    q : float
        Ellipticity parameter.
    w0 : float
        Beam waist at `z = 0`.
    z : float
        Propagation distance, used (with `lamb`) to compute the local
        beam width `w` and scale the focal distance `f` accordingly.
    lamb : float
        Wavelength.

    Returns
    -------
    x, y : array_like
        Cartesian coordinates, same shape as `xi`/`eta`.
    """
    #Convert elliptic coordinates (xi, eta) to Cartesian (x,y) with elipticity q.
    f0 = w0*np.sqrt(q/2)
    w = w0 * np.sqrt(1 + (z * lamb / (np.pi * w0**2))**2)
    f = f0*w/w0
    x = f * np.cosh(xi) * np.cos(eta)
    y = f * np.sinh(xi) * np.sin(eta)
    return x, y







#utils for fiber modes

def v_number(n_core, n_clad, a, lamb):
    """
    Compute the normalized frequency (V-number) of a step-index fiber.

    Parameters
    ----------
    n_core : float
        Core refractive index.
    n_clad : float
        Cladding refractive index.
    a : float
        Core radius.
    lamb : float
        Wavelength (same length units as `a`).

    Returns
    -------
    float
        `V = (2*pi*a/lamb) * sqrt(n_core**2 - n_clad**2)`.
    """
    return (2 * np.pi * a / lamb) * np.sqrt(n_core**2 - n_clad**2)
 
 
def characteristic_eq(u, V, l):
    """
    Evaluate the LP-mode characteristic (eigenvalue) equation for a
    step-index fiber, whose roots `u` give the guided LP_l,m modes.

    Parameters
    ----------
    u : array_like
        Trial value(s) of the normalized transverse core wavenumber.
    V : float
        Fiber V-number (see `v_number`).
    l : int
        Azimuthal mode order.

    Returns
    -------
    array_like
        The characteristic function
        ``u * J_{l+1}(u)/J_l(u) - v * K_{l+1}(v)/K_l(v)``, where
        `v = sqrt(V**2 - u**2)`. Guided LP_l,m modes correspond to the
        zeros of this function in `u in (0, V)`.
    """
    v = np.sqrt(V**2 - u**2)
    return u * jv(l + 1, u) / jv(l, u) - v * kv(l + 1, v) / kv(l, v)
 
 
def _pole_positions(V, l):
    """
    Find the poles of `characteristic_eq` for azimuthal order `l` within
    `(0, V)` -- these are the zeros of `J_l`, which bracket the roots of
    the characteristic equation and are used to split the search
    interval in `find_LP_roots`.

    Parameters
    ----------
    V : float
        Fiber V-number.
    l : int
        Azimuthal mode order.

    Returns
    -------
    np.ndarray
        Zeros of the Bessel function `J_l` that are less than `V`.
        Internally searches an increasing number of zeros until enough
        are found (or a safety cap is hit).
    """
    n = 10
    while True:
        zeros = jn_zeros(l, n)
        if zeros[-1] > V or n > 5000:
            return zeros[zeros < V]
        n += 10

def find_LP_roots(V, l, m_max=6, n_samples=400):
    """Find up to m_max roots u (i.e. LP_l,1 ... LP_l,m_max) for azimuthal order l."""
    poles = _pole_positions(V, l)
    edges = np.concatenate(([0.0], poles, [V]))
    roots = []
    eps = 1e-8
    for i in range(len(edges) - 1):
        lo, hi = edges[i] + eps, edges[i + 1] - eps
        if hi <= lo:
            continue
        us = np.linspace(lo, hi, n_samples)
        fs = characteristic_eq(us, V, l)
        for j in range(len(us) - 1):
            if np.isnan(fs[j]) or np.isnan(fs[j + 1]):
                continue
            if fs[j] == 0:
                roots.append(us[j])
            elif fs[j] * fs[j + 1] < 0:
                try:
                    r = brentq(characteristic_eq, us[j], us[j + 1], args=(V, l))
                    roots.append(r)
                except (ValueError, RuntimeError):
                    pass
        if len(roots) >= m_max:
            break
    return roots[:m_max]

def effective_index(u, V, n_core, n_clad):
    """n_eff via normalized propagation constant b = (V^2-u^2)/V^2? use standard: n_eff^2 = n_clad^2 + (v/V)^2*(n_core^2-n_clad^2)"""
    v = np.sqrt(max(V**2 - u**2, 0.0))
    b = (v / V) ** 2
    return np.sqrt(n_clad**2 + b * (n_core**2 - n_clad**2))

def lp_cutoff_V(l, m):
    """Cutoff V-number for LP_lm: the m-th zero of J_{l-1} (l>0), or the
    (m-1)-th zero of J_1 for l=0 (LP01 itself has no cutoff, V_c=0)."""
    if l < 0 or m < 1:
        raise ValueError(f"l must be >= 0 and m must be >= 1 (got l={l}, m={m}).")
    if l == 0:
        if m == 1:
            return 0.0
        return jn_zeros(1, m - 1)[-1]
    return jn_zeros(l - 1, m)[-1]


def find_all_LP_modes(V, l_max=4, m_max=4):
    """Return dict {(l, m): (u, v)} for all supported LP_lm modes."""
    modes = {}
    for l in range(l_max + 1):
        us = find_LP_roots(V, l, m_max=m_max)
        if not us and l > 0:
            # no modes at this l -> higher l will have even fewer, stop scanning
            continue
        for m, u in enumerate(us, start=1):
            v = np.sqrt(max(V**2 - u**2, 0.0))
            modes[(l, m)] = (u, v)
    return modes

def get_LP_params(l, m, n_core, n_clad, a, lamb):
    """
    Return the (u, v, n_eff, ...) parameters of a single LP_lm mode.
 
    Raises ValueError, with an explanation, if:
      - l, m are not valid integers (l >= 0, m >= 1)
      - n_core <= n_clad (no guiding at all)
      - the fiber's V-number does not exceed the cutoff V-number for LP_lm
        (i.e. this mode is not supported by this fiber at this wavelength)
    """
    if not float(l).is_integer() or l < 0:
        raise ValueError(f"l must be a non-negative integer, got l={l}.")
    if not float(m).is_integer() or m < 1:
        raise ValueError(f"m must be a positive integer (m >= 1), got m={m}.")
    l, m = int(l), int(m)
 
    if n_core <= n_clad:
        raise ValueError(
            f"n_core ({n_core}) must be greater than n_clad ({n_clad}) for the "
            f"fiber to guide light at all; no LP modes exist."
        )
 
    V = v_number(n_core, n_clad, a, lamb)
    Vc = lp_cutoff_V(l, m)
 
    if V <= Vc:
        raise ValueError(
            f"LP{l}{m} does not exist for this fiber: V = {V:.4f}, but LP{l}{m} "
            f"only propagates once V > {Vc:.4f} (cutoff = the {m}-th zero of the "
            f"Bessel function J_{l-1}). To support LP{l}{m} you need to raise V, "
            f"e.g. increase the core radius, increase (n_core - n_clad), or "
            f"decrease the wavelength -- or choose a lower l and/or m."
        )
 
    us = find_LP_roots(V, l, m_max=m)
    if len(us) < m:
        raise ValueError(
            f"LP{l}{m} could not be located numerically even though V={V:.4f} "
            f"exceeds its cutoff {Vc:.4f} (only {len(us)} root(s) found for "
            f"l={l}). Try increasing n_samples in find_LP_roots."
        )
 
    u = us[m - 1]
    v = np.sqrt(max(V**2 - u**2, 0.0))
    n_eff = effective_index(u, V, n_core, n_clad)
    return {"l": l, "m": m, "u": u, "v": v, "V": V, "V_cutoff": Vc, "n_eff": n_eff}


#util for field angular section.
def get_section(Beam, ang_min, ang_max, pol_index=None):
    """
    Return the field restricted to the angular section
    [ang_min, ang_max).

    Angles are given in radians.
    """
    # Map angles to [0, 2π)
    ang_min = ang_min % (2 * np.pi)
    ang_max = ang_max % (2 * np.pi)

    theta = np.mod(np.arctan2(Beam.y, Beam.x), 2 * np.pi)

    # Normal section
    if ang_min < ang_max:
        sec = ((theta >= ang_min) & (theta < ang_max))
    # Section crosses 2π
    else:
        sec = ((theta >= ang_min) | (theta < ang_max))
    if pol_index is None:
        return Beam.field * sec[None, :, :]
    else:
        return Beam.field[pol_index]*sec[:,:]


#utils for holograms

def inv_sinc(A, n=10000):
    """
    Numerically invert the normalized sinc function on its first
    monotonic branch, `sinc(x/pi)` for `x` in `[0, pi]`.

    Parameters
    ----------
    A : array_like
        Value(s) of `sinc(x/pi)` to invert; expected to lie within the
        range of `sinc` over `x in [0, pi]` (i.e. `[sinc(1), 1]`, since
        `sinc` is decreasing there).
    n : int, optional
        Number of samples used to tabulate `sinc` before interpolating
        its inverse. Defaults to 10000.

    Returns
    -------
    array_like
        `x` such that `sinc(x/pi) ~= A`, found by linear interpolation
        on a tabulated grid.
    """
    #invert function sinc
    x = np.linspace(0, np.pi, n)
    y = np.sinc(x/np.pi)

    return np.interp(A, y[::-1], x[::-1])

def inv_J0(A, n=10000):
    """
    Numerically invert the Bessel function `J0` on its first monotonic
    branch, `x in [0, j01]` where `j01` is the first zero of `J0`.

    Parameters
    ----------
    A : array_like
        Value(s) of `J0(x)` to invert; expected to lie within
        `[J0(j01), J0(0)] = [0, 1]`.
    n : int, optional
        Number of samples used to tabulate `J0` before interpolating
        its inverse. Defaults to 10000.

    Returns
    -------
    array_like
        `x` such that `J0(x) ~= A`, found by linear interpolation on a
        tabulated grid over `[0, j01]`.
    """
    #invert bessel function J0
    j01 = 2.404825557695773
    x = np.linspace(0.0, j01, n)
    y = j0(x)  

    return np.interp(A, y[::-1], x[::-1])


def inv_J1(A, a=None, n=10000):
    """
    Numerically invert the Bessel function `J1` on its first monotonic
    branch, `x in [0, x1_max]` where `x1_max` is the location of `J1`'s
    first maximum.

    Parameters
    ----------
    A : array_like
        Normalized value(s) to invert; internally clipped to `[0, 1]`
        and scaled by `a` before inversion, so the effective target is
        `a * A`.
    a : float, optional
        Scale factor applied to `A`. Defaults to `J1(x1_max)` (i.e.
        `J1`'s maximum value), so that `A = 1` maps to the peak.
    n : int, optional
        Number of samples used to tabulate `J1` before interpolating
        its inverse. Defaults to 10000.

    Returns
    -------
    array_like
        `x` such that `J1(x) ~= a * A`, found by linear interpolation
        on a tabulated grid over `[0, x1_max]`.
    """
    #invert bessel function J1
    x1_max = 1.8411837813406593
    if a is None:
        a = j1(x1_max)  
    A = np.clip(A, 0.0, 1.0)
    x = np.linspace(0.0, x1_max, n)
    y = j1(x) 

    return np.interp(a * A, y, x)



#utils for aberration correction

def radial_poly(n, m, r):
    """
    Evaluate the Zernike radial polynomial `R_n^m(r)`.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Absolute value of the azimuthal frequency (0 <= m <= n),
        with `n - m` even.
    r : array_like
        Normalized radial coordinate(s) at which to evaluate the
        polynomial.

    Returns
    -------
    array_like
        `R_n^m(r)`, same shape as `r`.

    Raises
    ------
    ValueError
        If `n - m` is odd (the Zernike polynomial is then identically
        zero) or if `n`, `m` are otherwise invalid (`m > n`).
    """
    s = 0
    if (n - m) % 2 == 0 and m <= n:
        for k in range((n - m) // 2 + 1):
            s += (-1)**k * math.factorial(n-k) / (math.factorial(k) * math.factorial((n + m) // 2 - k) 
                                                     * math.factorial((n + m) // 2 - k))* r**(n - 2*k)
    elif (n - m) % 2 == 1 and m <= n:
        raise ValueError("Warning: (n - m) is odd, Zernike Polynomial is zero.")
    else:
        raise ValueError("Invalid n and m values for Zernike polynomials.")
    return s 

def zernike(n, m, r, phi):
    """
    Evaluate the (n, m) Zernike polynomial in polar coordinates.

    Parameters
    ----------
    n : int
        Radial order.
    m : int
        Signed azimuthal frequency. `m >= 0` selects the cosine
        ("even") term; `m < 0` selects the sine ("odd") term (using
        `|m|` for the radial part).
    r : array_like
        Normalized radial coordinate(s).
    phi : array_like
        Azimuthal angle(s), in radians.

    Returns
    -------
    array_like
        `R_n^{|m|}(r) * cos(m*phi)` if `m >= 0`, else
        `R_n^{|m|}(r) * sin(|m|*phi)`.
    """
    if m >= 0:
        return radial_poly(n, m, r) * np.cos(m*phi)
    else:
        return radial_poly(n, -m, r) * np.sin(-m*phi)
    
def zernikes_phase(beam, coefs, strengths,):
    """
    Build a complex phase-only transmittance from a weighted sum of
    Zernike polynomials, evaluated on `beam`'s own grid (normalized by
    its waist).

    Parameters
    ----------
    beam : object
        Beam instance supplying the coordinate grids `beam.x`,
        `beam.y` and the normalization radius `beam.waist`.
    coefs : array_like
        Array of shape (N, 2), where each row `(n, m)` selects one
        Zernike term (radial order `n`, signed azimuthal frequency
        `m`).
    strengths : array_like
        Length-N array of coefficient strengths, one per row of
        `coefs`.

    Returns
    -------
    np.ndarray
        Complex array `exp(1j * phase)`, where
        `phase = sum(strength * zernike(n, m, r, phi))` over all
        `(n, m, strength)` triples, `r = sqrt(x**2+y**2)/beam.waist`
        and `phi = arctan2(y, x)`.

    Raises
    ------
    ValueError
        If `coefs` does not have shape (N, 2), or if `coefs` and
        `strengths` have different lengths.
    """

    coefs = np.asarray(coefs, dtype=int)
    strengths = np.asarray(strengths, dtype=float)

    if coefs.ndim != 2 or coefs.shape[1] != 2:
        raise ValueError("coefs must have shape (N, 2), containing (n, m).")

    if len(coefs) != len(strengths):
        raise ValueError("coefs and strengths must have the same length.")

    r = np.sqrt(beam.x**2 + beam.y**2) / beam.waist
    phi = np.arctan2(beam.y, beam.x)

    phase = np.zeros_like(r, dtype=float)

    for (n, m), strength in zip(coefs, strengths):
        phase += strength * zernike(n, m, r, phi)
    return np.exp(1j * phase)