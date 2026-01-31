import numpy as np
from scipy.linalg import eigh_tridiagonal
import matplotlib.pyplot as plt

"""def ince_even_coeffs(p, m, eps):
    if (p - m) % 2 != 0:
        raise ValueError("p - m must be even")

    n = (p - m) // 2 + 1
    r = np.arange(n)

    diag = (2*r + m)**2
    off  = eps * np.ones(n - 1)

    eigvals, eigvecs = eigh_tridiagonal(diag, off)

    # Select eigenvector with dominant highest-order component
    target = (p - m) // 2
    idx = np.argmax(np.abs(eigvecs[target, :]))

    A = eigvecs[:, idx]
    return A / np.max(np.abs(A))"""

def ince_even_coeffs(p, m, eps):
    if p < m: # or (p - m) % 2 != 0:
        raise ValueError("Invalid (p,m)")

    n = (p - m)//2 + 1
    r = np.arange(n)

    diag = (2*r + m)**2
    off  = eps * np.ones(n - 1)

    _, eigvecs = eigh_tridiagonal(diag, off)

    target = (p - m)//2
    idx = np.argmax(np.abs(eigvecs[target, :]))

    A = eigvecs[:, idx]
    return A / np.max(np.abs(A))


def ince_odd_coeffs(p, m, eps):
    if p < m: # or (p - m) % 2 != 0:
        raise ValueError("Invalid (p,m)")

    n = (p - m)//2
    if m < 1:
        return np.zeros(1)
    else:
        r = np.arange(n)

        diag = (m + 2*r + 1)**2
        off  = eps * np.ones(n - 1)

        _, eigvecs = eigh_tridiagonal(diag, off)
        print(eigvecs)
        print(np.abs(eigvecs).sum(axis=0))
        target = n - 1
        idx = np.argmax(np.abs(eigvecs[:, target]))
        print(idx)

        B = eigvecs[:, idx]
        return B / np.max(np.abs(B))


def ince_even(xi, p, m, eps):
    A = ince_even_coeffs(p, m, eps)
    y = np.zeros_like(xi, dtype=complex)
    for k, ak in enumerate(A):
        y += ak * np.cos((2*k + m) * xi)
    return y


def ince_odd(xi, p, m, eps):
    B = ince_odd_coeffs(p, m, eps)
    y = np.zeros_like(xi, dtype=complex)
    for k, bk in enumerate(B):
        y += bk * np.sin((m + 2*k + 1) * xi)
    return y

def cartesian_to_elliptic(x, y, f):
    r1 = np.sqrt((x + f)**2 + y**2)
    r2 = np.sqrt((x - f)**2 + y**2)

    # --- xi: safe arccos ---
    arg = (r1 - r2) / (2*f)
    arg = np.clip(arg, -1.0, 1.0)
    xi = np.arccos(arg)

    # --- eta: MUST be complex ---
    eta = np.arccosh((r1 + r2) / (2*f) + 0j)

    return xi, eta


def ince_gauss_beam(x, y, p, m, w0, eps, helicity=+1):
    if p < m: # or (p - m) % 2 != 0:
        raise ValueError("Invalid Ince–Gaussian indices")

    f = np.sqrt(eps * w0**2 / 2)
    xi, eta = cartesian_to_elliptic(x, y, f)

    Ce_xi  = ince_even(xi, p, m, eps)
    Ce_eta = ince_even(1j * eta, p, m, eps)

    G = np.exp(-(x**2 + y**2) / w0**2)

    # p = m → no odd part (pure HG-like)
    field = Ce_xi * Ce_eta * G
    """if p == m:
        field = Ce_xi * Ce_eta * G
    elif (p - m)//2 < 1:
        field = Ce_xi * Ce_eta * G
    else:
        So_xi  = ince_odd(xi, p, m, eps)
        So_eta = ince_odd(1j * eta, p, m, eps)

        field = (Ce_xi * Ce_eta +
                 helicity * So_xi * So_eta) * G"""

    return field




"""def ince_even(xi, p, m, eps):
    A = ince_even_coeffs(p, m, eps)
    r = np.arange(len(A))

    y = np.zeros_like(xi, dtype=complex)
    for k, ak in enumerate(A):
        y += ak * np.cos((2*k + m) * xi)

    return y 


def ince_odd_coeffs(p, m, eps):
    if (p - m) % 2 != 1:
        raise ValueError("p - m must be odd")

    n = (p - m + 1) // 2
    r = np.arange(n)

    diag = (m + 2*r + 1)**2
    off  = eps * np.ones(n - 1)

    eigvals, eigvecs = eigh_tridiagonal(diag, off)

    # Physical branch: dominant highest-order harmonic
    target = (p - m - 1) // 2
    idx = np.argmax(np.abs(eigvecs[target, :]))

    B = eigvecs[:, idx]
    return B / np.max(np.abs(B))


def ince_odd(xi, p, m, eps):
    B = ince_odd_coeffs(p, m, eps)
    r = np.arange(len(B))

    y = np.zeros_like(xi, dtype=complex)
    for k, bk in enumerate(B):
        y += bk * np.sin((m + 2*k + 1) * xi)

    return y"""

"""def cartesian_to_elliptic(x, y, f):
    r1 = np.sqrt((x + f)**2 + y**2)
    r2 = np.sqrt((x - f)**2 + y**2)

    xi  = np.arccos((r1 - r2) / (2*f))
    eta = np.arccosh((r1 + r2) / (2*f))

    return xi, eta"""

"""def ince_gauss_beam(x, y, p, m, w0, f):
    
    #Generate Ince–Gauss beam at waist (z = 0)
    #parity = 'even' or 'odd'
    

    eps = 2 * f**2 / w0**2
    xi, eta = cartesian_to_elliptic(x, y, f)

    if (p-m) % 2 == 0:
        C_xi  = ince_even(xi, p, m, eps)
        C_eta = ince_even(1j * eta, p, m, eps)
    else:
        C_xi  = ince_odd(xi, p, m, eps)
        C_eta = ince_odd(1j * eta, p, m, eps)
    Ce_xi  = ince_even(xi, p, m, eps)
    Ce_eta = ince_even(1j * eta, p, m, eps)

    G = np.exp(-(x**2 + y**2) / w0**2)

    #field = C_xi * C_eta * G
    if p == m:
        field = Ce_xi * Ce_eta * G
    else:
        So_xi  = ince_odd(xi, p, m, eps)
        So_eta = ince_odd(1j * eta, p, m, eps)
        field = (Ce_xi * Ce_eta + So_xi * So_eta) * G
    return field / np.max(np.abs(field))"""



"""
# Grid
N = 500
L = 3e-3
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y)

# Beam parameters
w0 = 0.5e-3
f  = 0.0001* w0
p, m = 2, 0

IG = ince_gauss_beam(X, Y, 2, 2, w0, f) -  ince_gauss_beam(X, Y, 2, 1, w0, f)

plt.imshow(np.abs(IG)**2, extent=[-L, L, -L, L], cmap="inferno")
plt.colorbar(label="Amplitude")
plt.title(rf"Ince–Gauss mode $IG_{{{p},{m}}}^e$")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.show()"""
