import numpy as np
from scipy import ndimage
from scipy.special import hermite, genlaguerre, jv, j0, j1, kv, jn_zeros
from scipy.optimize import brentq



#utils for modes

def herm(X, N):              #hermite polynomial
    HER = hermite(N)
    sn = HER(X)
    return sn

def laguerre(X, L, P):          #laguerre polynomial
    LAG = genlaguerre(P, L)
    sn = LAG(X)
    return sn

def even_coeffs(p, q, kind):
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
    #Convert elliptic coordinates (xi, eta) to Cartesian (x,y) with elipticity q.
    f0 = w0*np.sqrt(q/2)
    w = w0 * np.sqrt(1 + (z * lamb / (np.pi * w0**2))**2)
    f = f0*w/w0
    x = f * np.cosh(xi) * np.cos(eta)
    y = f * np.sinh(xi) * np.sin(eta)
    return x, y







#utils for fiber modes

def v_number(n_core, n_clad, a, lamb):
    return (2 * np.pi * a / lamb) * np.sqrt(n_core**2 - n_clad**2)
 
 
def characteristic_eq(u, V, l):
    v = np.sqrt(V**2 - u**2)
    return u * jv(l + 1, u) / jv(l, u) - v * kv(l + 1, v) / kv(l, v)
 
 
def _pole_positions(V, l):
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







#utils for Beam class parameters calculation

def overlap(first_beam:object, second_beam:object)->complex: 
    # calculate overlap between two beams, only properly works if ni and D of beams are equal.
    return np.sum(first_beam.field*np.conjugate(second_beam.field))*4*(first_beam.nix/first_beam.Dx) \
        *(first_beam.niy/first_beam.Dy)/np.sqrt(first_beam.Power()*second_beam.Power())

def int_overlap(first_beam:object, second_beam:object)->float:
    return np.sum(first_beam.int_profile()*second_beam.int_profile())*4*(first_beam.nix/first_beam.Dx) \
        *(first_beam.niy/first_beam.Dy)/(first_beam.Power()*second_beam.Power())


def get_section(Beam, ang_min, ang_max):
    #Return field distribution of given section, defined by minimum angle and maximum angle
    if ang_min > ang_max:
        ang_min, ang_max = ang_max, ang_min
    if ang_max <= np.pi and ang_min <= np.pi:
        sec = (np.arctan2(Beam.y,Beam.x)>=ang_min)*(np.arctan2(Beam.y,Beam.x)<ang_max)
    if ang_max> np.pi and ang_min <= np.pi:
        sec1 = (np.arctan2(Beam.y,Beam.x)>=ang_min)
        ang_max = ang_max-2*np.pi
        sec2 = (np.arctan2(Beam.y,Beam.x)<ang_max)
        sec = sec1 + sec2
    if ang_max>np.pi and ang_min > np.pi:
        ang_max = ang_max - 2*np.pi
        ang_min = ang_min - 2*np.pi
        sec = (np.arctan2(Beam.y,Beam.x)>=ang_min)*(np.arctan2(Beam.y,Beam.x)<ang_max)
    return Beam.field*sec


def get_crop(Beam, center=None, std=None, window=2):
    #Crops a field by its std*window arround the center of mass
    if center == None:
        center = Beam.center_mass()
    if std == None:
        std = Beam.std()
    xmin = int(center[1] - window*std*Beam.Dx/Beam.nix/2)
    xmax = int(center[1] + window*std*Beam.Dx/Beam.nix/2)
    ymin = int(center[0] - window*std*Beam.Dy/Beam.niy/2)
    ymax = int(center[0] + window*std*Beam.Dy/Beam.niy/2)
    Beam.x = Beam.x[:,xmin:xmax]
    Beam.y = Beam.y[ymin:ymax, :]
    Beam.field = Beam.field[ymin:ymax, xmin:xmax]
    Beam.Dx = len(Beam.x[0,:])
    Beam.Dy = len(Beam.y[:,0])
    Beam.nix = (Beam.x[0,-1] - Beam.x[0,0])/2
    Beam.niy = (Beam.y[-1,0] - Beam.y[0,0])/2
    Beam.x0 = center[1]
    Beam.y0 = center[0]
    return Beam

def rotate(matrix, angle, order=1):

    angle_degrees = angle*180/np.pi

    # Separate real and imaginary parts
    real_part = np.real(matrix)
    imag_part = np.imag(matrix)
    
    # Rotate both parts separately
    # cval=0 sets out-of-bounds values to zero
    real_rotated = ndimage.rotate(real_part, angle_degrees, 
                                   reshape=False, order=order, 
                                   cval=0.0, prefilter=True)
    
    imag_rotated = ndimage.rotate(imag_part, angle_degrees, 
                                   reshape=False, order=order, 
                                   cval=0.0, prefilter=True)
    
    # Recombine into complex matrix
    rotated_matrix = real_rotated + 1j * imag_rotated
    
    return rotated_matrix



#utils for holograms

def inv_sinc(A, n=10000):
    #invert function sinc
    x = np.linspace(0, np.pi, n)
    y = np.sinc(x/np.pi)

    return np.interp(A, y[::-1], x[::-1])

def inv_J0(A, n=10000):
    #invert bessel function J0
    j01 = 2.404825557695773
    x = np.linspace(0.0, j01, n)
    y = j0(x)  

    return np.interp(A, y[::-1], x[::-1])


def inv_J1(A, a=None, n=10000):
    #invert bessel function J1
    x1_max = 1.8411837813406593
    if a == None:
        a = j1(x1_max)  
    A = np.clip(A, 0.0, 1.0)
    x = np.linspace(0.0, x1_max, n)
    y = j1(x) 

    return np.interp(a * A, y, x)


