import numpy as np
from scipy import special


#utils for modes

def hermite(X, N):              #hermite polynomial
    HER = special.hermite(N)
    sn = HER(X)
    return sn

def laguerre(X, L, P):          #laguerre polynomial
    LAG = special.genlaguerre(P, L)
    sn = LAG(X)
    return sn

def even_coeffs(p, q, kind):
    """
    Compute coefficients A_r of even Ince polynomials C_p^m.
    """
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
    """
    Convert Cartesian (x,y) to elliptic coordinates (xi, eta) with elipticity q.
    """
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
    Convert elliptic coordinates (xi, eta) to Cartesian (x,y) with elipticity q.
    """
    f0 = w0*np.sqrt(q/2)
    w = w0 * np.sqrt(1 + (z * lamb / (np.pi * w0**2))**2)
    f = f0*w/w0
    x = f * np.cosh(xi) * np.cos(eta)
    y = f * np.sinh(xi) * np.sin(eta)
    return x, y



#utils for Beam class parameters calculation

