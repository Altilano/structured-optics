import numpy as np
from scipy import special, ndimage



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


def get_crop(Beam, center=None, std=None, window=2, pol_index:int=0):
    #Crops a field by its std*window arround the center of mass
    if center == None:
        center = Beam.center_mass(pol_index)
    if std == None:
        std = Beam.std(pol_index)
    xmin = int(center[1] - window*std*Beam.Dx/Beam.nix/2)
    xmax = int(center[1] + window*std*Beam.Dx/Beam.nix/2)
    ymin = int(center[0] - window*std*Beam.Dy/Beam.niy/2)
    ymax = int(center[0] + window*std*Beam.Dy/Beam.niy/2)
    Beam.x = Beam.x[:,xmin:xmax]
    Beam.y = Beam.y[ymin:ymax, :]
    if Beam.pol_dim >1:
        Beam.field = Beam.field[:,ymin:ymax, xmin:xmax]
    else:
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
    y = special.j0(x)  

    return np.interp(A, y[::-1], x[::-1])


def inv_J1(A, a=None, n=10000):
    #invert bessel function J1
    x1_max = 1.8411837813406593
    if a == None:
        a = special.j1(x1_max)  
    A = np.clip(A, 0.0, 1.0)
    x = np.linspace(0.0, x1_max, n)
    y = special.j1(x) 

    return np.interp(a * A, y, x)

